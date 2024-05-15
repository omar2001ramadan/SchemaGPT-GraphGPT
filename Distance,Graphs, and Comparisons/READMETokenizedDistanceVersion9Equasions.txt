
README
Overview
This project involves loading two JSON files, converting their contents into text for analysis, and building graphs based on the JSON objects. The script calculates various distances between the JSON objects and visualizes the graphs.

Features
Load JSON data from files.
Convert JSON entries to text.
Preprocess JSON objects to ensure required fields are present.
Extract verb-centered snippets from descriptions.
Build graphs based on JSON objects.
Visualize the graphs with enhanced label formatting.
Plot threshold analysis for optimal threshold determination.
Recommend optimal threshold based on aggregated metrics.
Calculate node distance, key distance, value distance, and graph distance between JSON objects.
Requirements
Python 3.x
Required libraries:
json
networkx
matplotlib
spacy
sklearn
nltk
numpy
Setup
Install the required libraries:

bash
Copy code
pip install networkx matplotlib spacy scikit-learn nltk numpy
Download the necessary NLTK resources:

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Load the spaCy NLP model:

python
Copy code
import spacy
nlp = spacy.load('en_core_web_sm')
Usage
Place your JSON files in the desired directory.
Run the script and provide the file paths for the two JSON files when prompted.
Script Details
Load JSON Data
The function load_json(file_path) loads JSON data from a file and handles errors if the file is not found.

Convert JSON to Text
The function json_to_text(data) converts JSON entries to text for analysis, extracting text, ids, and JSON objects.

Preprocess JSON Objects
The function preprocess_json_objects(json_objects) ensures all JSON objects have the required fields: name, type, and description.

Extract Verb-Centered Snippet
The function extract_verb_centered_snippet(description, window=2, max_gap=4) extracts snippets around the first two verbs found in the description using spaCy NLP.

Build Graph
The function build_graph(json_objects, suffix, color) builds a graph based on JSON objects, ensuring unique IDs and adding nodes and edges.

Visualize Graph
The function visualize_graph(G) visualizes a NetworkX graph with enhanced label formatting for readability.

Plot Threshold Analysis
The function plot_threshold_analysis(similarity_matrix) plots threshold analysis for determining the optimal threshold value.

Recommend Optimal Threshold
The function recommend_optimal_threshold(similarity_matrix) recommends the optimal threshold value based on aggregated metrics.

Calculate Distances
The function calculate_distances(json_objects1, json_objects2) calculates node distance, key distance, value distance, and graph distance between two sets of JSON objects.

Running the Script
Run the script:
bash
Copy code
python script.py
Enter the file paths for JSON 1 and JSON 2 when prompted.
The script will perform the following:
Load the JSON files.
Convert the JSON entries to text.
Preprocess the JSON objects.
Vectorize the text data using TF-IDF.
Calculate the similarity matrix using cosine similarity.
Plot threshold analysis and recommend an optimal threshold.
Build separate graphs for each JSON dataset.
Visualize the graphs.
Calculate and print distances between the JSON objects.
Example Output
plaintext
Copy code
Enter the file path for JSON 1: "path/to/json1.json"
Enter the file path for JSON 2: "path/to/json2.json"

Node Distance: 0.60
Key Distance: 0.14
Value Distance: 0.70
Graph Distance: 4.00
The output includes node distance, key distance, value distance, and graph distance between the two JSON files.