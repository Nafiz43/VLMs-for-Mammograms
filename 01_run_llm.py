import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
import sys
from pydantic import BaseModel
import random
import os
import json
import re

# source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'
# saving_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base/'

temp = 0
prompt_technique = "base"
prompt_template = """
I will provide you with a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.
Please follow the below given JSON format for your response:
{
    "IMG-ID": "<Image_Filename>",
    "BREAST-COMPOSITION": "<Description of breast tissue composition>",
    "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations>"
}

"""


allowable_models = ["meditron:latest", "jyan1/paligemma-mix-224:latest", "qwen2.5:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b",
          "deepseek-r1:7b", "deepseek-r1:70b", "qordmlwls/llama3.1-medical:latest", "mixtral:latest","llava:latest"]


# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

import json

def fix_json(json_input):
    """
    Ensures the input is a JSON string or a dictionary and always returns a dictionary.
    If input is a dictionary, return it as-is.
    If input is a valid JSON string, return parsed JSON as a dictionary.
    If input is an invalid JSON string, attempts to fix it by trimming trailing characters.
    """
    # If input is already a dictionary, return it directly
    if isinstance(json_input, dict):
        return json_input  

    # Ensure input is a string (or bytes), otherwise return error JSON
    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
    except json.JSONDecodeError:
        pass  # If invalid, proceed with fixing

    # Try trimming trailing characters
    for i in range(len(json_input), 0, -1):  
        try:
            parsed_json = json.loads(json_input[:i])  # Try parsing progressively shorter substrings
            if isinstance(parsed_json, dict):
                return parsed_json  # Ensure it's a dictionary
        except Exception as e:
            print(f"Unexpected error: {e}")  # Catch other unforeseen errors
            continue  # Keep trimming

    # If all attempts fail, return error JSON
    return {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}




@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="name of the model to be used for processing",
)
@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)

def main(model_name, reports_to_process):
    print(f"Received model_name: {model_name}")
    print(f"Received value for reports_to_process: {reports_to_process}")

    global data 

    if(reports_to_process > 0):
        # data = data.head(reports_to_process)
        print(f"Processing only {reports_to_process} reports")
    
    if(reports_to_process == -1):
        reports_to_process = 510


    # Your existing logic to handle logging
    # log_dir, log_file = "local_chat_history", f"{model_name+datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv"
    
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # log_path = os.path.join(log_dir, log_file)

    # if not os.path.isfile(log_path):
    #     with open(log_path, mode="w", newline="", encoding="utf-8") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["timestamp", "question", "answer","reason","model_name"])

    # cnt = 0
    # print(questions)


    for report in range(0, reports_to_process):
        report_id = source_file_dir + str(report+1).zfill(3)+'.png'
        print(report_id)
        image_id = 'IMG'+ str(report+1).zfill(3)
        
        # query = 'image ID: ' + report_id
        query = prompt_template+ 'image ID: '+  report_id

        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
        response = ollama.invoke(query)
        print(response)

        ###the following is a dummy response for testing ###

        # dummy_data = {
        #     "IMG_ID": "image_001.jpg",
        #     "Breast_Composition": "Dense tissue with scattered fibroglandular elements",
        #     "BIRADS": "2",
        #     "Findings": "No significant abnormalities or calcifications. Normal breast tissue."
        # }
        # dummy_data_str = json.dumps(dummy_data, indent=4)

        # response =dummy_data_str+"abdc"
        ### dummy response processing ENDS ###

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match in [None, ""]:
            json_match = {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
        else:
            json_match = fix_json(json_match.group(0))
        
        print(json_match)
        # global saving_dir
        #constructing the saving dir here
        saving_dir = 'evaluated/'+model_name+'_/'
        print(saving_dir)

        image_saving_dir = saving_dir +image_id + '.json'

        os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
        with open(image_saving_dir, 'w') as json_file:
            json.dump(json_match, json_file, indent=4)
        

        print("Data has been written to", image_saving_dir)

    print("\nTotal Reports Processed", reports_to_process)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
