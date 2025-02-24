import json
import re
import os

import os
import shutil

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


repo_path = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS'
if os.path.exists(repo_path) and os.path.isdir(repo_path):
    clean_repo(repo_path)
    print("Cleanup completed.")
else:
    print("Invalid repository path.")




def parse_mammogram_report(report: str, img_id: str) -> dict:
    # Extract Breast Composition
    breast_composition_match = re.search(r'Breast Composition:\s*(.*)', report, re.IGNORECASE)

    breast_composition = breast_composition_match.group(1).strip() if breast_composition_match else ""
    
    # Extract BIRADS
    birads_match = re.search(r'BIRADS:\s*(\d+)', report, re.IGNORECASE)
    birads = birads_match.group(1) if birads_match else ""
    
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
    
    return result

def process_reports(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                report_content = file.read()
            img_id = os.path.splitext(filename)[0]  # Extract filename without extension
            json_data = parse_mammogram_report(report_content, img_id)
            
            json_filename = f"{img_id.upper()}.json"
            json_path = os.path.join(directory, json_filename)
            
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

# Example usage
reports_directory = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"
process_reports(reports_directory)
