#!/usr/bin/env python

import json
import argparse
import os
import sys
import logging
from collections import defaultdict, Counter
import re

import pandas as pd
import networkx as nx
from pyvis.network import Network
from jinja2 import Template
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure WordNet data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate an interactive graph from JSON data using PyVis with dynamic text scaling and unconnected entity statistics."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input JSON file.'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Path to the output HTML file. Defaults to "graph.html" in the script\'s directory.'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output.'
    )
    return parser.parse_args()

def setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            logging.info(f"Loaded data from {input_path}")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON file: {e}")
            sys.exit(1)

def clean_text(text):
    """
    Cleans the input text by removing extra whitespace.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()
    return text

def normalize_text(text, lemmatizer):
    """
    Normalizes the input text by:
    - Removing possessive suffixes ('s, s')
    - Removing special characters
    - Converting to lowercase
    - Lemmatizing to remove plurals
    """
    # Remove possessive suffixes
    text = re.sub(r"'s$|s'$", "", text)
    # Remove special characters, keep alphanumerics and spaces
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    # Lowercase
    text = text.lower()
    # Split into words, lemmatize each
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    # Rejoin
    normalized_text = ' '.join(lemmatized_words)
    return normalized_text

def process_data(data):
    entities = data.get('entities', [])
    relations = data.get('relations', [])
    text = data.get('text', '')
    entity_dict = {}
    id_to_label = {}
    entity_counter = Counter()
    relation_counter = Counter()
    unique_texts_per_label = defaultdict(set)

    lemmatizer = WordNetLemmatizer()

    for entity in entities:
        entity_id = str(entity['id'])
        label = entity['label']  # Keep the original case for labels
        # Extract the entity text using the provided offsets
        try:
            entity_text = text[entity['start_offset']:entity['end_offset']]
        except IndexError:
            logging.warning(f"Invalid offsets for entity ID {entity_id}. Skipping entity.")
            continue
        original_text = entity_text.strip()
        normalized_text = normalize_text(original_text, lemmatizer)
        if not normalized_text:
            logging.warning(f"Entity ID {entity_id} has empty normalized text. Skipping entity.")
            continue
        entity_dict[entity_id] = {
            'label': label,
            'original_text': original_text,
            'normalized_text': normalized_text
        }
        id_to_label[entity_id] = f"{normalized_text} <{label}>"
        entity_counter[label] += 1  # Count entity labels
        unique_texts_per_label[label].add(normalized_text)  # Track unique normalized texts per label

    valid_relations = []
    for relation in relations:
        from_id = str(relation['from_id'])
        to_id = str(relation['to_id'])
        rel_type = relation['type']  # Keep the original case for relation types

        if from_id in id_to_label and to_id in id_to_label:
            valid_relations.append({
                'source': id_to_label[from_id],
                'target': id_to_label[to_id],
                'relation': rel_type
            })
            relation_counter[rel_type] += 1  # Count relation types
        else:
            logging.warning(f"Relation with missing entities: {relation}")

    # Prepare unique text counts per label
    unique_text_counts = {label: len(texts) for label, texts in unique_texts_per_label.items()}

    return entity_dict, valid_relations, entity_counter, relation_counter, unique_text_counts

def create_graph(entity_dict, relations):
    G = nx.DiGraph()
    for entity_id, details in entity_dict.items():
        node_label = f"{details['normalized_text']} <{details['label']}>"
        G.add_node(node_label, label=details['label'], original_text=details['original_text'])
    for relation in relations:
        G.add_edge(
            relation['source'],
            relation['target'],
            relation=relation['relation']
        )
    return G

def assign_colors(G):
    labels = set(nx.get_node_attributes(G, 'label').values())
    color_palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    label_color_mapping = {label: color_palette[i % len(color_palette)] for i, label in enumerate(labels)}
    color_map = {node: label_color_mapping[data['label']] for node, data in G.nodes(data=True)}
    nx.set_node_attributes(G, color_map, 'color')
    return label_color_mapping

def generate_html(G, output_file, entity_counter, relation_counter, unique_text_counts):
    # Initialize PyVis network with responsive design settings
    net = Network(height='100vh', width='100%', directed=True, bgcolor='#ffffff', font_color='black')
    net.barnes_hut()
    # net.show_buttons(filter_=['physics'])  # Uncomment to show physics controls

    # Add nodes with attributes
    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=node,
            title=f"Original Text: {data['original_text']}<br>Label: {data['label']}",
            color=data.get('color', '#888'),
        )

    # Add edges with labels
    for source, target, data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=data['relation'],
            label=data['relation'],
            arrows='to',
            physics=True,
        )

    # Identify unconnected entities (nodes with degree 0)
    unconnected_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    unconnected_counts = Counter()
    for node in unconnected_nodes:
        label = G.nodes[node]['label']
        unconnected_counts[label] +=1

    # Generate statistics as HTML
    entity_stats = '<br>'.join([f"{label}: {count} entities, {unique_text_counts[label]} unique texts" for label, count in entity_counter.items()])
    relation_stats = '<br>'.join([f"{rel}: {count}" for rel, count in relation_counter.items()])
    if unconnected_counts:
        unconnected_stats = '<br>'.join([f"{label}: {count} unconnected entities" for label, count in unconnected_counts.items()])
    else:
        unconnected_stats = 'None'
    stats_html = f"""
    <h3>Entity Counts:</h3>
    <p>{entity_stats}</p>
    <h3>Relation Counts:</h3>
    <p>{relation_stats}</p>
    <h3>Unconnected Entity Counts:</h3>
    <p>{unconnected_stats}</p>
    """

    # Define a Jinja2 template with Flexbox for responsive layout
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Interactive Graph</title>
      <style>
        body, html {
          height: 100%;
          margin: 0;
          padding: 0;
          display: flex;
          flex-direction: row;
          font-family: Arial, sans-serif;
        }
        #graph-container {
          flex: 3;
          height: 100vh;
        }
        #stats-container {
          flex: 1;
          padding: 20px;
          overflow-y: auto;
          background-color: #f5f5f5;
          border-left: 1px solid #ddd;
        }
      </style>
    </head>
    <body>
      <div id="graph-container">
        {{ graph }}
      </div>
      <div id="stats-container">
        {{ stats }}
      </div>
    </body>
    </html>
    """)

    # Generate the graph HTML
    graph_html = net.generate_html(notebook=False)

    # Set dynamic scaling options using set_options
    scaling_options = {
        "nodes": {
            "font": {
                "size": 14,  # Default font size
                "face": "Tahoma"
            },
            "scaling": {
                "min": 10,
                "max": 30,
                "label": {
                    "enabled": True
                }
            }
        },
        "edges": {
            "font": {
                "size": 12,
                "align": "middle"
            },
            "scaling": {
                "min": 10,
                "max": 20,
                "label": {
                    "enabled": True
                }
            }
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 250,
                "springConstant": 0.001,
                "damping": 0.09,
                "avoidOverlap": 0
            },
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True
        }
    }

    net.set_options(json.dumps(scaling_options))

    # Render the final HTML with the graph and statistics
    final_html = template.render(graph=graph_html, stats=stats_html)

    # Write the final HTML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    logging.info(f"Graph saved to {output_file}")

    # Print statistics to console
    print("\nEntity Statistics:")
    for label, count in entity_counter.items():
        unique_count = unique_text_counts[label]
        print(f"  {label}: {count} entities, {unique_count} unique texts")

    print("\nRelation Statistics:")
    for rel_type, count in relation_counter.items():
        print(f"  {rel_type}: {count}")

    print("\nUnconnected Entity Statistics:")
    if unconnected_counts:
        for label, count in unconnected_counts.items():
            print(f"  {label}: {count} unconnected entities")
    else:
        print("  None")

def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    input_file = args.input_file

    # Determine the output file path
    if args.output:
        output_file = args.output
    else:
        script_directory = os.getcwd()
        output_file = os.path.join(script_directory, 'graph.html')

    data = read_json(input_file)
    entity_dict, relations, entity_counter, relation_counter, unique_text_counts = process_data(data)
    G = create_graph(entity_dict, relations)
    assign_colors(G)
    generate_html(G, output_file, entity_counter, relation_counter, unique_text_counts)

if __name__ == "__main__":
    main()
