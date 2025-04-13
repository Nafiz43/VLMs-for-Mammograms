# from _constant import *


# # source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

# # source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'
# source_file_dir_C =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_CC'
# source_file_dir_M =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_MLO'
# source_file_dir_reports =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/GROUND_TRUTH_REPORTS'
# # saving_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base/'


# img_files = list_png_files(source_file_dir)

# temp = 0
# prompt_technique = "base"
# prompt_template = """
# I will provide you with two mammogram images. First one is the top-view of a breast whereas the second one is the side-view. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.
# Please follow the below given JSON format for your response
# {
#     "IMG_ID_CC": "<Image_Filename>",
#     "IMG-ID-MLO": "<Image_Filename>",
#     "BREAST-COMPOSITION" "<Description of breast tissue composition>",
#     "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
#     "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations for both the views>"
# }

# """

# @click.command()
# @click.option(
#     "--model_name",
#     default="llama3.1:latest",
#     type=click.Choice(allowable_models),
#     help="name of the model to be used for processing",
# )
# @click.option(
#     "--reports_to_process", 
#     default=-1,  # Default value
#     type=int, 
#     help="An extra integer to be passed via command line"
# )

# def main(model_name, reports_to_process):
#     print(f"Received model_name: {model_name}")
#     print(f"Received value for reports_to_process: {reports_to_process}")

#     global data 

#     if(reports_to_process > 0):
#         # data = data.head(reports_to_process)
#         print(f"Processing only {reports_to_process} reports")
    
#     if(reports_to_process == -1):
#         reports_to_process = len(img_files)


#     for report in range(0, reports_to_process):
#         # report_id = source_file_dir + str(report+1).zfill(3)+'.png'
#         report_id = source_file_dir_reports + img_files[report]

#         print(report_id)
#         # image_id = 'IMG'+ str(report+1).zfill(3)
#         image_id =  img_files[report].replace('.png', '')

        
#         # query = 'image ID: ' + report_id
#         query = prompt_template+ 'image ID: '+  report_id

#         print("QUERY: ", query)

#         ollama = Ollama(model=model_name, temperature=temp)
#         logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
#         response = ollama.invoke(query)
#         print("RESPONSE: ",response)



#         json_match = re.search(r"\{.*\}", response, re.DOTALL)
#         if json_match in [None, ""]:
#             json_match = {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
#         else:
#             json_match = fix_json(json_match.group(0))
        
#         print(json_match)
#         # global saving_dir
#         #constructing the saving dir here
#         saving_dir = 'evaluated-vindr/'+model_name+'_/'
#         print(saving_dir)

#         image_saving_dir = saving_dir +image_id + '.json'

#         os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
#         with open(image_saving_dir, 'w') as json_file:
#             json.dump(json_match, json_file, indent=4)
        

#         print("Data has been written to", image_saving_dir)

#     print("\nTotal Reports Processed", reports_to_process)


# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
#     )
#     main()

import os
import json
import re
import logging
import click
from _constant import *  # Assumes that list_png_files, fix_json, and allowable_models are provided

# Directories for Left-side
source_file_dir_C_L = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_CC'
source_file_dir_M_L = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_MLO'
source_file_dir_reports_L = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/GROUND_TRUTH_REPORTS/L'

# Directories for Right-side
source_file_dir_C_R = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/R_CC'
source_file_dir_M_R = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/R_MLO'
source_file_dir_reports_R = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/GROUND_TRUTH_REPORTS/R'

temp = 0
prompt_technique = "base"
prompt_template = """
I will provide you with two mammogram images. First one is the top-view of a breast whereas the second one is the side-view. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.
Please follow the below given JSON format for your response
{
    "IMG_ID_CC": "<Image_Filename>",
    "IMG-ID-MLO": "<Image_Filename>",
    "BREAST-COMPOSITION" "<Description of breast tissue composition>",
    "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations for both the views>"
}
"""

def process_side(model_name, side, cc_dir, mlo_dir, report_dir, reports_to_process):
    logging.info(f"Processing {side} images")
    # Get list of png files in the CC directory (assumes corresponding filenames exist in the MLO and report dirs)
    img_files = list_png_files(cc_dir)
    if reports_to_process > 0:
        img_files = img_files[:reports_to_process]
        logging.info(f"Processing only {reports_to_process} reports for side {side}")

    for idx, filename in enumerate(img_files):
        # Build full paths for CC, MLO, and report files
        cc_image_path = os.path.join(cc_dir, filename)
        mlo_image_path = os.path.join(mlo_dir, filename)
        report_path = os.path.join(report_dir, filename)

        # Read report content if available. If not, continue or mark as missing.
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read().strip()
        else:
            logging.warning(f"Report file not found for: {report_path}")
            report_text = "No report available"

        image_id = filename.replace('.png', '')

        # Build the prompt query by appending image paths and report text
        query = (
            prompt_template +
            "\nImage_CC: " + cc_image_path +
            "\nImage_MLO: " + mlo_image_path +
            "\nReport: " + report_text +
            "\n"
        )

        logging.info(f"Processing {side} file {idx+1}/{len(img_files)}: {filename}")
        logging.debug(f"QUERY: {query}")

        # Invoke the model through Ollama with the constructed query
        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
        response = ollama.invoke(query)
        logging.info(f"RESPONSE: {response}")

        # Extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match is None or json_match.group(0) == "":
            json_data = {"IMG_ID_CC": "NA", "IMG_ID_MLO": "NA", "BREAST_COMPOSITION": "NA", "BIRADS": "NA", "FINDINGS": "NA"}
        else:
            json_data = fix_json(json_match.group(0))
        
        logging.debug(f"Parsed JSON: {json_data}")

        # Create saving directory based on model and side
        saving_dir = os.path.join('evaluated-vindr', f"{model_name}_{side}")
        os.makedirs(saving_dir, exist_ok=True)
        # Save the output JSON with the same base filename
        image_saving_path = os.path.join(saving_dir, image_id + '.json')
        with open(image_saving_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        
        logging.info(f"Data has been written to {image_saving_path}")
    logging.info(f"\nTotal Reports Processed for {side}: {len(img_files)}")


@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="Name of the model to be used for processing",
)
@click.option(
    "--reports_to_process",
    default=-1,
    type=int,
    help="Optional integer to limit the number of reports processed per side"
)
def main(model_name, reports_to_process):
    logging.info(f"Received model_name: {model_name}")
    logging.info(f"Received value for reports_to_process: {reports_to_process}")

    # Process the left side
    process_side(
        model_name,
        side='L',
        cc_dir=source_file_dir_C_L,
        mlo_dir=source_file_dir_M_L,
        report_dir=source_file_dir_reports_L,
        reports_to_process=reports_to_process if reports_to_process != -1 else -1,
    )

    # Process the right side
    process_side(
        model_name,
        side='R',
        cc_dir=source_file_dir_C_R,
        mlo_dir=source_file_dir_M_R,
        report_dir=source_file_dir_reports_R,
        reports_to_process=reports_to_process if reports_to_process != -1 else -1,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

