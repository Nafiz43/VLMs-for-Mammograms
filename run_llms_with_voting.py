import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain.llms import Ollama
import sys
import random

# data = pd.read_csv('data/Labeled/labels_v2.csv')
data = pd.read_csv('data/labels.csv')

questions = pd.read_csv('data/PCL_Questions_V2.csv')
total_report_count = len(data)

# m_name = "llama3.2:latest"
temp = 0
prompt_technique = "base"
# prompt_template = "I am going to give you a radiology report. Then I am going to ask you several questions about it. I would like you to answer if a particular type of radiology study or procedure was performed. Please answer with a 1 if the study or procedure was performed. Please answer 0 if the study or procedure was not performed or was not documented in the report. Please answer only with a 1 or 0 without additional words including justification. "

allowable_models = ["meditron:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M"]

@click.command()

@click.option(
    "--model_1",
    default="llama3.2:3b-instruct-q4_K_M",
    type=click.Choice(allowable_models),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--model_2",
    default="llama3.2:3b-instruct-q4_K_M",
    type=click.Choice(allowable_models),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--model_3",
    default="llama3.2:3b-instruct-q4_K_M",
    type=click.Choice(allowable_models),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)


def main(model_1, model_2, model_3, reports_to_process):
    print(f"Received model_1: {model_1}")
    print(f"Received model_2: {model_2}")
    print(f"Received model_3: {model_3}")
    print(f"Received value for reports_to_process: {reports_to_process}")

    global data 

    if(reports_to_process > 0):
        data = data.head(reports_to_process)
        print(f"Processing only {reports_to_process} reports")


    # Your existing logic to handle logging
    log_dir, log_file = "local_chat_history", f"{model_1}_{model_2}_{model_3}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer", "model_name"])
    cnt = 0
    # print(questions)

    for index, row in data.iterrows():
        actual_annotation = ""
        for i in range(0, 39):
            
            query = prompt_template + 'REPORT: ' + row['Report_Text'].replace('\n', ' ### ').lower().split('attestation')[0] + '     TASK:' + str(questions.iloc[i]['Questions']) + '. AGAIN, answer either "1" or "0"'
            # print(query)
            ollama_model_1 = Ollama(model=model_1, temperature=temp, top_k=10, top_p=10)
            ollama_model_2 = Ollama(model=model_2, temperature=temp, top_k=10, top_p=10)
            ollama_model_3 = Ollama(model=model_3, temperature=temp, top_k=10, top_p=10)

             # Random response only for testing purpose; comment the next 3 lines if you are want to have actual respose from models
            response_model_1 = random.randint(0, 1)
            response_model_2 = random.randint(0, 1)
            response_model_3 = random.randint(0, 1)
             # Random response only for testing purpose

            # actual response from models; comment the next 3 lines if you are want to have random respose
            # response_model_1 = ollama_model_1(query)
            # response_model_2 = ollama_model_2(query)
            # response_model_3 = ollama_model_3(query)
            # actual response from models

            if(isinstance(response_model_1, int) and isinstance(response_model_2, int) and isinstance(response_model_3, int)):
                answer = 1 if (response_model_1 + response_model_2 + response_model_3) >= 2 else 0
            elif(isinstance(response_model_1, int) and isinstance(response_model_2, int)):
                answer = 1 if (response_model_1 + response_model_2) >= 1 else 0
            elif(isinstance(response_model_1, int) and isinstance(response_model_3, int)):
                answer = 1 if (response_model_1 + response_model_3) >= 1 else 0
            elif(isinstance(response_model_2, int) and isinstance(response_model_3, int)):
                answer = 1 if (response_model_2 + response_model_3) >= 1 else 0
            else:
                answer = "NA"

            # print("Voted Answer:", answer)
            # print("\n\n> Question:")
            # print(query)
            # print("\n> Answer:")
            # print(answer)

            with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, query, answer, (model_1+model_3+model_3)])

        progress_percentage = ((index+1) / len(data)) * 100
        print(f"Processed {index+1}/{len(data)} reports ({progress_percentage:.2f}% complete)", end="\r")
        # sys.stdout.flush()

    # print("\n")
    print("\nTotal Reports Processed", len(data))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
