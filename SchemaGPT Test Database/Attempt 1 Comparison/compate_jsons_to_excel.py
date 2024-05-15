import os
import json
import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import openpyxl  # Import the library

def load_json(file_path):
    """Load a JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def compare_json_files(json1, json2):
    """Compare two JSON objects and return their differences."""
    difference = DeepDiff(json1, json2, ignore_order=False, verbose_level=2)
    return difference

def custom_json_serializer(obj):
    """Custom JSON serializer for non-serializable objects."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: custom_json_serializer(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return str(obj)

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
        base_data = load_json(base_file)
        for j, compare_file in enumerate(json_files):
            if i != j:
                compare_data = load_json(compare_file)
                differences = compare_json_files(base_data, compare_data)
                results_df.iloc[i, j] = json.dumps(differences, indent=4, default=custom_json_serializer)
            else:
                results_df.iloc[i, j] = 'Self'

    results_excel_path = os.path.join(directory, "comparison_results_matrix.xlsx")
    # Convert DataFrame to Excel and wrap text
    writer = pd.ExcelWriter(results_excel_path, engine='openpyxl')
    results_df.to_excel(writer, index=True)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Apply text wrap to all cells
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = openpyxl.styles.Alignment(wrapText=True)
    
    writer.close()
    print(f"Comparison results matrix has been saved to '{results_excel_path}'.")

if __name__ == "__main__":
    main()
