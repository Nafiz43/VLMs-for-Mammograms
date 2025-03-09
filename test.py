import os
import re

import json


# def extract_number(s: str) -> int:
#     match = re.search(r'\d+', s)  # Find the first sequence of digits
#     return int(match.group()) if match else None  # Convert to int if found

# # Example usage
# input_str = "Img151"
# output = extract_number(input_str)
# print(output)  # Output: 151


import os
import glob

def remove_json_files(directory: str):
    json_files = glob.glob(os.path.join(directory, "*.json"))  # Get all .json files in the directory
    
    for file in json_files:
        os.remove(file)  # Remove each JSON file
        print(f"Deleted: {file}")

# Example usage
directory_path = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/fine_tuned_results"  # Change this to your directory path
remove_json_files(directory_path)
