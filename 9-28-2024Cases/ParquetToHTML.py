#!/usr/bin/env python

import os
import sys
import argparse
import logging
from collections import defaultdict, Counter
import re
import pandas as pd
import networkx as nx
from pyvis.network import Network
from jinja2 import Template
import nltk
from nltk.stem import WordNetLemmatizer
import json

# Ensure WordNet data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate an interactive graph from Parquet data by converting it to JSON-like format."
    )
    parser.add_argument(
        'input_dir',
        type=str,
        nargs='?',
        default=os.getcwd(),
        help='Path to the input directory containing the Parquet files. Defaults to the current working directory.'
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


def read_parquet_files(entities_file, relationships_file):
    try:
        entities_df = pd.read_parquet(entities_file)
        logging.info(f"Loaded entities from {entities_file}")
    except Exception as e:
        logging.error(f"Failed to read entities Parquet file: {e}")
        sys.exit(1)

    try:
        relationships_df = pd.read_parquet(relationships_file)
        logging.info(f"Loaded relationships from {relationships_file}")
    except Exception as e:
        logging.error(f"Failed to read relationships Parquet file: {e}")
        sys.exit(1)

    return entities_df, relationships_df


def convert_to_json_like_structure(entities_df, relationships_df):
    # Create a mapping from entity names to IDs
    name_to_id = {}
    entities = []
    for _, row in entities_df.iterrows():
        entity_id = str(row['id'])
        name = row.get('name') or row.get('description') or ''
        label = row.get('label') or row.get('type') or 'Entity'
        name = name.strip()
        name_to_id[name] = entity_id  # Map name to ID

        entity = {
            'id': entity_id,
            'label': label,
            'text': name,
            'start_offset': 0,  # Placeholder
            'end_offset': 0     # Placeholder
        }
        entities.append(entity)

    # Convert relationships DataFrame to list of dictionaries
    relations = []
    for _, row in relationships_df.iterrows():
        from_name = row.get('from_id') or row.get('source') or ''
        to_name = row.get('to_id') or row.get('target') or ''
        rel_type = row.get('type') or row.get('label') or row.get('description') or 'related_to'

        from_name = from_name.strip()
        to_name = to_name.strip()

        from_id = name_to_id.get(from_name)
        to_id = name_to_id.get(to_name)

        if from_id and to_id:
            relation = {
                'from_id': from_id,
                'to_id': to_id,
                'type': rel_type
            }
            relations.append(relation)
        else:
            logging.warning(f"Could not find IDs for relation: from '{from_name}' to '{to_name}'")

    # Combine into a single data dictionary
    data = {
        'entities': entities,
        'relations': relations,
        'text': ''  # Placeholder
    }

    return data


def clean_text(text):
    """
    Cleans the input text by removing extra whitespace.
    """
    text = re.sub(r'\s+', ' ', str(text))  # Replace multiple spaces with a single space
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
    if not text:
        return ''
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
    text = data.get('text', '')  # Placeholder
    entity_dict = {}
    id_to_label = {}
    entity_counter = Counter()
    relation_counter = Counter()
    unique_texts_per_label = defaultdict(set)

    lemmatizer = WordNetLemmatizer()

    for entity in entities:
        entity_id = str(entity['id'])
        label = entity['label']  # Keep the original case for labels
        original_text = clean_text(entity.get('text', ''))
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

    # Set dynamic scaling options using set_options BEFORE generating HTML
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

    # Now generate the graph HTML
    graph_html = net.generate_html(notebook=False)

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

    input_dir = args.input_dir

    # Adjust to include 'artifacts' directory if it exists
    artifacts_dir = os.path.join(input_dir, 'artifacts')
    if os.path.exists(artifacts_dir):
        input_dir = artifacts_dir

    # Define the Parquet file paths
    entities_file = os.path.join(input_dir, 'create_final_entities.parquet')
    relationships_file = os.path.join(input_dir, 'create_final_relationships.parquet')

    # Check if files exist
    if not os.path.exists(entities_file):
        logging.error(f"Entities file not found: {entities_file}")
        sys.exit(1)
    if not os.path.exists(relationships_file):
        logging.error(f"Relationships file not found: {relationships_file}")
        sys.exit(1)

    # Read the Parquet files
    entities_df, relationships_df = read_parquet_files(entities_file, relationships_file)

    # Convert Parquet data to JSON-like structure
    data = convert_to_json_like_structure(entities_df, relationships_df)

    # Determine the output file path
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(os.getcwd(), 'graph.html')

    # Process data and generate the graph
    entity_dict, relations, entity_counter, relation_counter, unique_text_counts = process_data(data)
    G = create_graph(entity_dict, relations)
    assign_colors(G)
    generate_html(G, output_file, entity_counter, relation_counter, unique_text_counts)


if __name__ == "__main__":
    main()
