import os
import json
import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import openpyxl
import spacy
import re

# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

def normalize_text(text):
    """Normalize text by converting to lowercase, removing special characters, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

def process_json(json_data):
    """Normalize all string fields in the JSON data."""
    for entry in json_data:
        for key, value in entry.items():
            if isinstance(value, str):
                entry[key] = normalize_text(value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                entry[key] = [normalize_text(item) for item in value]
    return json_data

def load_and_normalize_json(file_path):
    """Load a JSON file from a path and normalize its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return process_json(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def compare_json_files(json1, json2):
    """Compare two JSON objects and return a detailed report of differences."""
    difference = DeepDiff(json1, json2, ignore_order=True, verbose_level=2)
    return json.dumps(difference, indent=4)

def find_json_files(directory):
    """Return a list of file paths for all JSON files in the given directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]

def main():
    directory = input("Enter the path to the folder containing JSON files: ")
    json_files = find_json_files(directory)
    num_files = len(json_files)
    results_df = pd.DataFrame(np.empty((num_files, num_files), dtype=object),
                              columns=[os.path.basename(file) for file in json_files],
                              index=[os.path.basename(file) for file in json_files])

    for i, base_file in enumerate(json_files):
        base_data = load_and_normalize_json(base_file)
        for j, compare_file in enumerate(json_files):
            if i != j:
                compare_data = load_and_normalize_json(compare_file)
                differences = compare_json_files(base_data, compare_data)
                results_df.iloc[i, j] = differences
            else:
                results_df.iloc[i, j] = 'Self'

    results_excel_path = os.path.join(directory, "comparison_results_matrix_normalized_verbose.xlsx")
    writer = pd.ExcelWriter(results_excel_path, engine='openpyxl')
    results_df.to_excel(writer, index=True)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = openpyxl.styles.Alignment(wrapText=True)
    writer.close()
    print(f"Comparison results matrix has been saved to '{results_excel_path}'.")

if __name__ == "__main__":
    main()
