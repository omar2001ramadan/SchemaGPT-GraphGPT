import os
import json
import pandas as pd
from deepdiff import DeepDiff

def load_json(file_path):
    """Load a JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def compare_json_files(json1, json2):
    difference = DeepDiff(json1, json2, ignore_order=False, verbose_level=2)
    return difference

# Adjust your serialization or output format
def print_differences(differences):
    if differences == {}:
        print("No differences found.")
    else:
        for key, value in differences.items():
            print(f"Difference type {key}: {value}")

# Use this function in your main logic to print differences more clearly


def custom_json_serializer(obj):
    """Custom JSON serializer for non-serializable objects."""
    if isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, dict):
        return {k: custom_json_serializer(v) for k, v in obj.items()}  # Recursively handle dictionary
    elif hasattr(obj, '__dict__'):
        return obj.__dict__  # Try to serialize objects by their __dict__ attribute
    elif hasattr(obj, 'tolist'):
        return obj.tolist()  # Convert numpy arrays to list
    else:
        return str(obj)  # As a last resort, convert to string

def find_json_files(directory):
    """Return a list of file paths for all JSON files in the given directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]

def main():
    directory = input("Enter the path to the folder containing JSON files: ")
    json_files = find_json_files(directory)

    results_df = pd.DataFrame(columns=["Base File", "Compared File", "Differences"])

    for base_file in json_files:
        base_data = load_json(base_file)
        for compare_file in json_files:
            if base_file != compare_file:
                compare_data = load_json(compare_file)
                differences = compare_json_files(base_data, compare_data)
                # Call the print_differences function to print the differences
                print(f"Differences between {base_file} and {compare_file}:")
                print_differences(differences)
                # Serialize differences for storage in DataFrame
                differences_json = json.dumps(differences, indent=4, default=custom_json_serializer)
                new_row = pd.DataFrame({
                    "Base File": [base_file],
                    "Compared File": [compare_file],
                    "Differences": [differences_json]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

    results_csv_path = os.path.join(directory, "comparison_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Comparison results have been saved to '{results_csv_path}'.")

if __name__ == "__main__":
    main()
