import os
import json
import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import openpyxl
import spacy
import re
from jinja2 import Template

# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

def normalize_text(text):
    """Normalize text by converting to lowercase, removing special characters, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
    return lemmatized_text.strip()

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
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return process_json(data)

def compare_json_files(json1, json2):
    """Compare two JSON objects and return the differences, focusing on missing and additional objects."""
    differences = DeepDiff(json1, json2, ignore_order=True, verbose_level=2)
    key_changes = differences.get('type_changes', {})
    all_value_changes = differences.get('values_changed', {})

    # Flatten all changes to count each value change at any depth within an object
    detailed_value_changes = {}
    for change_key, change_info in all_value_changes.items():
        # Extract detailed changes from each value change dictionary
        if isinstance(change_info, dict):
            for sub_key, sub_change in change_info.items():
                detailed_key = f"{change_key} -> {sub_key}"
                detailed_value_changes[detailed_key] = sub_change
        else:
            detailed_value_changes[change_key] = change_info

    # Track the full object additions and deletions
    added_objects = {desc: value for desc, value in differences.get('iterable_item_added', {}).items()}
    removed_objects = {desc: value for desc, value in differences.get('iterable_item_removed', {}).items()}
    object_diffs = added_objects.copy()
    object_diffs.update(removed_objects)

    return key_changes, detailed_value_changes, object_diffs

def generate_html_report(key_changes, value_changes, object_diffs, base_file, compare_file):
    """Generate sections for an HTML report for the differences."""
    html_section = f"<h2>Comparison: {os.path.basename(base_file)} vs {os.path.basename(compare_file)}</h2><div>"
    
    if key_changes:
        html_section += "<h3>Key Changes</h3><ul>"
        for desc, details in key_changes.items():
            html_section += f"<li>{desc}: {details}</li>"
        html_section += "</ul>"

    if value_changes:
        html_section += "<h3>Value Changes</h3>"
        for desc, changes in value_changes.items():
            html_section += f"<div><strong>Change at {desc}:</strong><ul>"
            if isinstance(changes, dict) and 'old_value' in changes and 'new_value' in changes:
                change_details = f"from {changes.get('old_value', 'N/A')} to {changes.get('new_value', 'N/A')}"
                html_section += f"<li>{change_details}</li>"
            else:
                html_section += f"<li>{changes}</li>"  # Direct string display if not in expected dict format
            html_section += "</ul></div>"

    if object_diffs:
        html_section += "<h3>Object Differences</h3><ul>"
        for desc, details in object_diffs.items():
            file_presence = os.path.basename(compare_file) if 'added' in desc else os.path.basename(base_file)
            object_type = details.get('type', 'Unknown Type')
            object_id = details.get('id', 'Unknown ID')
            html_section += f"<li>{object_type} with ID {object_id} is {desc.split('_')[0]} in {file_presence}</li>"
        html_section += "</ul>"

    html_section += "</div>"
    return html_section



def main():
    directory = input("Enter the path to the folder containing JSON files: ")
    json_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]
    num_files = len(json_files)
    results_df = pd.DataFrame(np.empty((num_files, num_files), dtype=object),
                              columns=[os.path.basename(file) for file in json_files],
                              index=[os.path.basename(file) for file in json_files])

    all_html_sections = ""

    for i, base_file in enumerate(json_files):
        base_data = load_and_normalize_json(base_file)
        for j, compare_file in enumerate(json_files):
            if i != j:
                compare_data = load_and_normalize_json(compare_file)
                key_changes, value_changes, object_diffs = compare_json_files(base_data, compare_data)
                results_df.iloc[i, j] = f"Key Changes: {len(key_changes)}, Value Changes: {len(value_changes)}, Object Differences: {len(object_diffs)}"
                all_html_sections += generate_html_report(key_changes, value_changes, object_diffs, base_file, compare_file)

    # Write all sections to a single HTML file
    final_html_output = f"<!DOCTYPE html><html><head><title>Comprehensive JSON Differences Report</title></head><body><h1>Detailed JSON Comparison Results</h1>{all_html_sections}</body></html>"
    with open(os.path.join(directory, "comprehensive_report.html"), 'w') as f:
        f.write(final_html_output)

    results_excel_path = os.path.join(directory, "comparison_results_matrix.xlsx")
    writer = pd.ExcelWriter(results_excel_path, engine='openpyxl')
    results_df.to_excel(writer, index=True)
    writer.close()
    print(f"Comparison results matrix has been saved to '{results_excel_path}'. Comprehensive HTML report generated in '{directory}'.")

if __name__ == "__main__":
    main()
