import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import spacy

# Assuming all the functions from the main script are available
# You might need to import them or copy them into this file

# Import functions from the main script
from TokenDistV40 import (
    load_json,
    json_to_text,
    assign_encode_ids,
    identify_node_or_edge_and_add_key,
    build_graphs,
    calculate_detailed_distances,
    save_json,
    save_embeddings_to_excel,
    output_networkx_style
)


class JSONComparer:
    def __init__(self, json1_path, json2_path, threshold=0.8, output_dir='output_files'):
        self.json1_path = json1_path
        self.json2_path = json2_path
        self.threshold = threshold
        self.output_dir = output_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize attributes
        self.json_objects1 = None
        self.json_objects2 = None
        self.G1 = None
        self.G2 = None
        self.embeddings1 = None
        self.embeddings2 = None
        self.tokenized_texts1 = None
        self.tokenized_texts2 = None

    def process(self):
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load and process JSON files
        data1 = load_json(self.json1_path)
        data2 = load_json(self.json2_path)

        texts1, self.embeddings1, self.tokenized_texts1, self.json_objects1, _, _ = json_to_text(data1, 'json1')
        texts2, self.embeddings2, self.tokenized_texts2, self.json_objects2, _, _ = json_to_text(data2, 'json2')

        # Assign encode IDs
        self.json_objects1, self.json_objects2, similarity_matrix = assign_encode_ids(
            texts1, self.embeddings1, texts2, self.embeddings2, self.json_objects1, self.json_objects2, self.threshold)

        # Identify nodes and edges
        self.json_objects1 = identify_node_or_edge_and_add_key(self.json_objects1)
        self.json_objects2 = identify_node_or_edge_and_add_key(self.json_objects2)

        # Build graphs
        self.G1, self.G2 = build_graphs(self.json_objects1, self.json_objects2)

    def save_combined_embeddings(self):
        combined_embeddings = np.vstack(self.embeddings1 + self.embeddings2)
        np.save(os.path.join(self.output_dir, 'combined_embeddings_combined.npy'), combined_embeddings)

    def save_combined_tokenized_output(self):
        combined_tokenized_texts = self.tokenized_texts1 + self.tokenized_texts2
        with open(os.path.join(self.output_dir, 'combined_tokenized_output_combined.txt'), 'w') as f:
            for tokenized_text in combined_tokenized_texts:
                f.write(" ".join(tokenized_text) + "\n")

    def save_detailed_distances(self):
        detailed_distances = calculate_detailed_distances(self.json_objects1, self.json_objects2, self.threshold)
        save_json(detailed_distances, os.path.join(self.output_dir, 'detailed_distances.json'))

    def save_embeddings_and_similarity(self):
        save_embeddings_to_excel(
            self.embeddings1, self.embeddings2, self.json_objects1, self.json_objects2,
            'embeddings_and_similarity.xlsx', self.threshold
        )

    def save_networkx_style_output(self):
        output_networkx_style(self.G1, self.G2)

    def save_normalized_data(self):
        save_json(self.json_objects1, os.path.join(self.output_dir, 'normalized_data1.json'))
        save_json(self.json_objects2, os.path.join(self.output_dir, 'normalized_data2.json'))

    def generate_all_outputs(self):
        self.process()
        self.save_combined_embeddings()
        self.save_combined_tokenized_output()
        self.save_detailed_distances()
        self.save_embeddings_and_similarity()
        self.save_networkx_style_output()
        self.save_normalized_data()

# Usage example:
if __name__ == "__main__":
    comparer = JSONComparer(r"C:\Users\omar2\OneDrive\Desktop\Final Edit Project\graphTest1.1.json", r"C:\Users\omar2\OneDrive\Desktop\Final Edit Project\graphTest1.2.json", threshold=0.8)
    comparer.generate_all_outputs()