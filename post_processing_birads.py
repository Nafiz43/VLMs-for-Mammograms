import os
import json
import re

def extract_integer(value):
    """Extracts the first integer found in a string."""
    match = re.search(r'\d+', value)
    return int(match.group()) if match else 1

def process_json_files(repo_path):
    """Reads all JSON files in the repo, processes the BIRADS field, and updates the JSON files."""
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Process the BIRADS field if it exists
                    if "BIRADS" in data and isinstance(data["BIRADS"], str):
                        data["BIRADS"] = extract_integer(data["BIRADS"])
                    
                    # Save the processed JSON back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                    
                    # print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage
repo_directory = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/qwen_base"  # Change this to your repo path
process_json_files(repo_directory)
