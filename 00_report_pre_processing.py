import json
import re
import os

import os
import shutil

# repo_path = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/qwen_fine_tuned'
# repo_path = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS'


def remove_quotes(s: str) -> str:
    return s.replace('"', '')

def extract_number(s: str) -> int:
    match = re.search(r'\d+', s)  # Find the first sequence of digits
    return int(match.group()) if match else None  # Convert to int if found

def clean_repo(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if not file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Removing file: {file_path}")
                os.remove(file_path)
        
        # Remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                print(f"Removing empty directory: {dir_path}")
                shutil.rmtree(dir_path)



def parse_mammogram_report(report: str, img_id: str) -> dict:
    # Extract Breast Composition
    print(report)
    report = remove_quotes(report)
    breast_composition_match = re.search(r'Breast-Composition:\s*(.*)', report, re.IGNORECASE)
    print(breast_composition_match)
    if(breast_composition_match == None):
        breast_composition_match = re.search(r'Breast Composition:\s*(.*)', report, re.IGNORECASE)

    breast_composition = breast_composition_match.group(1).strip() if breast_composition_match else ""
    
    # Extract BIRADS
    birads_match = re.search(r'BIRADS:\s*(\d+)', report, re.IGNORECASE)
    birads = birads_match.group(1) if birads_match else 1
    birads = int(birads)
    
    # Extract Findings
    findings_match = re.search(r'Findings:\s*(.*?)(?=\n\*\*\*REPORT ENDS\*\*\*|$)', report, re.DOTALL | re.IGNORECASE)

    findings = findings_match.group(1).strip() if findings_match else ""
    
    # Construct JSON output
    result = {
        "IMG_ID": img_id,
        "Breast_Composition": breast_composition,
        "BIRADS": birads,
        "Findings": findings
    }    
    print(result)
    return result

def process_reports(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                report_content = file.read()
            img_id = os.path.splitext(filename)[0]  # Extract filename without extension
            img_id = "IMG"+str(extract_number(img_id)).zfill(3)
            print("Image ID:", img_id)
            # img_id = int(str(int(img_id)).zfill(3)) 
            json_data = parse_mammogram_report(report_content, img_id)
            
            json_filename = f"{img_id.upper()}.json"
            json_path = os.path.join(directory, json_filename)
            
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

# Example usage
# reports_directory = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"


process_reports(repo_path)
print("Done! txt files are converted to JSON")

# if os.path.exists(repo_path) and os.path.isdir(repo_path):
#     # clean_repo(repo_path)
#     print("Cleanup completed.")
# else:
#     print("Invalid repository path.")