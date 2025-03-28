import os
import json
import re
import pandas as pd

# Define directories
base_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr'
ground_truth_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/GROUND_TRUTH_REPORTS'
subdirs = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']

# Create output directories if they don't exist
os.makedirs(os.path.join(ground_truth_dir, 'R'), exist_ok=True)
os.makedirs(os.path.join(ground_truth_dir, 'L'), exist_ok=True)

# Read annotations
breast_level_annotations = pd.read_csv('/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/breast-level_annotations.csv')
findings_annotations = pd.read_csv('/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/finding_annotations.csv')

# Function to extract BIRADS number
def extract_birads(value):
    match = re.search(r'(\d+)', str(value))
    return match.group(1) if match else None

# Function to interpret breast density
def interpret_breast_density(value):
    if isinstance(value, str):
        last_char = value.strip()[-1]
        return {
            'A': "Predominantly fibro fatty breast parenchyma (ACR A)",
            'B': "Fibro fatty and scattered glandular breast parenchyma (ACR B)",
            'C': "Heterogeneously dense breast parenchyma (ACR C)",
            'D': "Extremely dense breast parenchyma"
        }.get(last_char, value)
    return None

# Function to clean and interpret finding categories
def interpret_findings(value):
    if isinstance(value, str):
        findings = re.findall(r'[a-zA-Z0-9\s]+', value)
        findings = [f.strip() for f in findings if f.strip()]
        findings = list(set(findings))  # Remove duplicates
        return ', '.join(findings) if findings else "No Finding"
    return None

# Function to process images and create JSON with verbose output
def process_images(laterality):
    processed_count = 0
    for image_name in os.listdir(os.path.join(base_dir, f'{laterality}_CC')):
        if not image_name.lower().endswith('.png'):
            continue
        image_id = os.path.splitext(image_name)[0]
        img_path_cc = os.path.join(base_dir, f'{laterality}_CC', image_name)
        img_path_mlo = os.path.join(base_dir, f'{laterality}_MLO', image_name)

        # Get data from breast_level_annotations.csv
        breast_data = breast_level_annotations[breast_level_annotations['study_id'] == image_id]
        breast_birads = None
        if not breast_data.empty:
            breast_data = breast_data[(breast_data['laterality'] == laterality) & (breast_data['view_position'] == 'CC')]
            breast_birads = extract_birads(breast_data.iloc[0]['breast_birads']) if not breast_data.empty else None

        # Get data from findings_annotations.csv
        findings_data = findings_annotations[findings_annotations['study_id'] == image_id]
        breast_density = None
        cc_findings = []
        mlo_findings = []
        if not findings_data.empty:
            findings_data_cc = findings_data[(findings_data['laterality'] == laterality) & (findings_data['view_position'] == 'CC')]
            findings_data_mlo = findings_data[(findings_data['laterality'] == laterality) & (findings_data['view_position'] == 'MLO')]

            if not findings_data_cc.empty:
                breast_density = interpret_breast_density(findings_data_cc.iloc[0]['breast_density'])
                cc_findings = findings_data_cc['finding_categories'].dropna().apply(interpret_findings).tolist()
                cc_findings = list(set(cc_findings))  # Remove duplicates from cc_findings

            if not findings_data_mlo.empty:
                mlo_findings = findings_data_mlo['finding_categories'].dropna().apply(interpret_findings).tolist()
                mlo_findings = list(set(mlo_findings))  # Remove duplicates from mlo_findings
                
            # Determine final findings
            if "No Finding" in cc_findings and "No Finding" in mlo_findings:
                finding_categories = "No findings in both views"
            elif "No Finding" in cc_findings:
                finding_categories = f"No findings in CC view. {', '.join(mlo_findings)} in MLO view"
            elif "No Finding" in mlo_findings:
                finding_categories = f"{', '.join(cc_findings)} in CC view. No findings in MLO view"
            elif sorted(cc_findings) == sorted(mlo_findings):
                finding_categories = f"{', '.join(cc_findings)} in both views"
            else:
                finding_categories = f"{', '.join(cc_findings)} in CC view. {', '.join(mlo_findings)} in MLO view"

        # Create JSON output
        output_data = {
            "IMG_ID_CC": img_path_cc,
            "IMG_ID_MLO": img_path_mlo,
            "Breast_Composition": breast_density,
            "BIRADS": breast_birads,
            "Findings": finding_categories
        }

        # Save JSON file
        output_path = os.path.join(ground_truth_dir, laterality, f'{image_id}_{laterality}.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        processed_count += 1
        if processed_count % 20 == 0:
            print(f"Processed {processed_count} images for {laterality}")

# Process both laterality options
process_images('R')
process_images('L')


