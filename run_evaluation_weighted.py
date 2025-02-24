import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

file_containing_ground_truth = 'data/labels.csv'
local_history_directory = 'local_chat_history'

def process_metrics(ground_truth, llm_response):
    indexes_to_remove = {i for i, value in enumerate(llm_response) if value == 404}
    ground_truth = [value for i, value in enumerate(ground_truth) if i not in indexes_to_remove]
    llm_response = [value for i, value in enumerate(llm_response) if i not in indexes_to_remove]

    precision = round(precision_score(ground_truth, llm_response, average='weighted', zero_division=0), 2)
    recall = round(recall_score(ground_truth, llm_response, average='weighted',zero_division=0), 2)
    f1 = round(f1_score(ground_truth, llm_response, average='weighted',zero_division=0), 2)
    return precision, recall, f1

def calculate_metrics(ground_truth, llm_response):
    TP = TN = FP = FN = 0
    for gt, pred in zip(ground_truth, llm_response):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative

    # Calculate Precision, Recall, and F1 Score
    precision,recall,f1= process_metrics(ground_truth, llm_response)
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Weighted-Precision': precision,
        'Weighted-Recall': recall,
        'Weighted-F1-Score': f1
    }

def calculate_metrics_with_cases(report_index, ground_truth, llm_response, llm_reason, actual_question):
    TP = TN = FP = FN = 0
    
    # Loop through the ground truth and LLM responses
    for gt, pred, act in zip(ground_truth, llm_response, actual_question):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
            
            # fp_file_path.replace('.csv',"")
            if not os.path.exists(fp_file_path):
                open(fp_file_path, 'w').close()  # Create an empty file

            with open(fp_file_path, 'a') as file: 
                file.write(f"REPORT: {report_index}; QUESTION: {act}; REASON_BEHIND_POSITIVE: {llm_reason}\n")

        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
            # fn_file_path = fn_file_path.replace(".csv", "")
            if not os.path.exists(fn_file_path):
                open(fn_file_path, 'w').close()  # Create an empty file

            with open('results/'+cases_save_dir+'_false_negative_cases.txt', 'a') as file: 
                file.write(f"REPORT: {report_index}; QUESTION: {act}; REASON_BEHIND_NEGATIVE: {llm_reason}\n")

    precision,recall,f1= process_metrics(ground_truth, llm_response)
    # Return results as a dictionary
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Weighted-Precision': precision,
        'Weighted-Recall': recall,
        'Weighted-F1-Score': f1
    }


csv_files = [f for f in os.listdir(local_history_directory) if f.endswith('.csv')]
for csv_file in csv_files:
    print("Processing", os.path.join(local_history_directory, csv_file))

    file_containing_llm_response = os.path.join(local_history_directory, csv_file)
    cases_save_dir = file_containing_llm_response.replace('local_chat_history/', '')

    fp_file_path = 'results/'+cases_save_dir+'_false_positive_cases.txt'
    fn_file_path = 'results/'+cases_save_dir+'_false_negative_cases.txt'


    data = pd.read_csv(file_containing_ground_truth)
    llm_data = pd.read_csv(file_containing_llm_response)

    #Deciding how many documnets to process
    reports_to_process = -1
    # Parse arguments manually
    if "--reports_to_process" in sys.argv:
        idx = sys.argv.index("--reports_to_process")
        reports_to_process = int(sys.argv[idx + 1])

    # print(f"Reports to process: {reports_to_process}")
    total_num_of_questions = 39

    ### Identifying how many reports to process###
    valid_report_count = len(llm_data)//total_num_of_questions
    if(reports_to_process==-1):
        reports_to_process = valid_report_count


    data = data.head(reports_to_process)
    print("Processing Report Count: ", reports_to_process)
    #Cleaning the ground truth data


    # data = data.drop(columns=['Other'])
    # data = data.drop(columns=['Other.1'])
    data = data.drop(columns=['Modality'])
    data = data.drop(columns=['Exam Code'])
    data = data.drop(columns=['Completed'])
    data = data.drop(columns=['Exam Description'])
    data.fillna(0, inplace=True)

    #loading the LLM response
    llm_data = pd.read_csv(file_containing_llm_response)
    llm_data = llm_data.head(reports_to_process*total_num_of_questions)

    k=0
    llm_responses = []
    llm_reasonings = []
    ground_truths = []
    actual_questions = []

    vascular_diagonsis_ground_truth = []
    vascular_diagonsis_llm = []

    vascular_intervention_ground_truth = []
    vascular_intervention_llm = []

    non_vascular_intervention_ground_truth = []
    non_vascular_intervention_llm = []


    for index, row in data.iterrows():
        llm_response = []
        ground_truth = []
        actual_question = []

        ground_truth = row.iloc[2:].tolist()
        # print("Ground Truth: ", index, ground_truth)

        for i in range(total_num_of_questions*index, total_num_of_questions*(index+1)):
            llm_response.append(int(llm_data.answer[k]))
            actual_question.append(llm_data.question[k])
            llm_reasonings.append(llm_data.reason[k])
            k=k+1
        # print("LLM Response: ", llm_response)
        ground_truth = [int(num) for num in ground_truth]
        # print("Ground Truth: ", ground_truth)
        # print(len(ground_truth), len(llm_response))
        llm_responses.append(llm_response)
        ground_truths.append(ground_truth)
        actual_questions.append(actual_question)






    metrics_df = pd.DataFrame(columns=['Modality', 'Report', 'Weighted-Precision', 'Weighted-Recall', 'Weighted-F1-Score'])

    # print("LLM RESPONSES:", llm_responses)
    for report_index in range(len(llm_responses)):
        # print(f"Report {report_index}:")
        
        ###########Considering ALL Modalities#####
        llm_response = llm_responses[report_index]
        ground_truth = ground_truths[report_index]
        actual_question = actual_questions[report_index]
        llm_reasoning = llm_reasonings[report_index]
        metrics = calculate_metrics_with_cases(report_index+1, ground_truth, llm_response, llm_reasoning, actual_question)
        # print(metrics)
        
        new_row = pd.DataFrame({
            'Modality': ["All"],  # Wrap the string in a list
            'Report': [report_index], 
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Weighted-Precision': [metrics['Weighted-Precision']],  # Wrap in a list
            'Weighted-Recall': [metrics['Weighted-Recall']],  # Wrap in a list
            'Weighted-F1-Score': [metrics['Weighted-F1-Score']]  # Wrap in a list
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ###########Considering ALL Modalities#####

        ######Considering VascularDiagnosis ############
        llm_response = llm_responses[report_index][0:8]
        ground_truth = ground_truths[report_index][0:8]

        # print(vascular_diagonsis_ground_truth)

        vascular_diagonsis_ground_truth.extend(ground_truth)
        vascular_diagonsis_llm.extend(llm_response)

        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = pd.DataFrame({
            'Modality': ["VascularDiagnosis"],  # Wrap the string in a list
            'Report': [report_index], 
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Weighted-Precision': [metrics['Weighted-Precision']],  # Wrap in a list
            'Weighted-Recall': [metrics['Weighted-Recall']],  # Wrap in a list
            'Weighted-F1-Score': [metrics['Weighted-F1-Score']]  # Wrap in a list
        })

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering VascularDiagnosis ############


        ######Considering VascularIntervention ############
        llm_response = llm_responses[report_index][8:8+15]
        ground_truth = ground_truths[report_index][8:8+15]

        vascular_intervention_ground_truth.extend(ground_truth)
        vascular_intervention_llm.extend(llm_response)

        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = pd.DataFrame({
            'Modality': ["VascularIntervention"],  # Wrap the string in a list
            'Report': [report_index], 
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Weighted-Precision': [metrics['Weighted-Precision']],  # Wrap in a list
            'Weighted-Recall': [metrics['Weighted-Recall']],  # Wrap in a list
            'Weighted-F1-Score': [metrics['Weighted-F1-Score']]  # Wrap in a list
        })

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering VascularIntervention ############


        ######Considering NonVascularIntervention ############
        llm_response = llm_responses[report_index][8+15:8+15+16]
        ground_truth = ground_truths[report_index][8+15:8+15+16]
        non_vascular_intervention_ground_truth.extend(ground_truth)
        non_vascular_intervention_llm.extend(llm_response)


        metrics = calculate_metrics(ground_truth, llm_response)
        # print(metrics)
        
        new_row = pd.DataFrame({
            'Modality': ["NonVascularIntervention"],  # Wrap the string in a list
            'Report': [report_index], 
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Weighted-Precision': [metrics['Weighted-Precision']],  # Wrap in a list
            'Weighted-Recall': [metrics['Weighted-Recall']],  # Wrap in a list
            'Weighted-F1-Score': [metrics['Weighted-F1-Score']]  # Wrap in a list
        })

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        ######Considering NonVascularIntervention ############

    # Aggregated metrics calculation across reports
    aggregated_metrics = pd.DataFrame(columns=['Modality', 'TP', 'TN', 'FP', 'FN', 'Weighted-Precision', 'Weighted-Recall', 'Weighted-F1-Score'])

    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    # print("OUTSIDE",llm_responses)
    for modality in metrics_df['Modality'].unique():
        # print(modality)
        modality_data = metrics_df[metrics_df['Modality'] == modality]
        
        # Calculate aggregated TP, FP, FN, TN
        total_TP = modality_data['TP'].sum()
        total_TN = modality_data['TN'].sum()
        total_FP = modality_data['FP'].sum()
        total_FN = modality_data['FN'].sum()

        if(modality == "All"):
            # print("INSIDE",llm_responses)
            llm_res = np.concatenate(llm_responses)
            gnd_truth = np.concatenate(ground_truths)
            precision,recall,f1=process_metrics(llm_res, gnd_truth)
        elif(modality == "VascularDiagnosis"):
            # print(vascular_diagonsis_llm)
            # print(vascular_diagonsis_llm)
            llm_res = (vascular_diagonsis_llm)
            gnd_truth = (vascular_diagonsis_ground_truth)
            precision,recall,f1=process_metrics(llm_res, gnd_truth)
        elif(modality == "VascularIntervention"):
            llm_res = (vascular_intervention_llm)
            gnd_truth = (vascular_intervention_ground_truth)
            precision,recall,f1=process_metrics(llm_res, gnd_truth)
        elif(modality == "NonVascularIntervention"):
            llm_res = (non_vascular_intervention_llm)
            gnd_truth = (non_vascular_intervention_ground_truth)
            precision,recall,f1=process_metrics(llm_res, gnd_truth)
        
        new_row = pd.DataFrame({
                'Modality': [modality],
                'Report': 'ALL', 
                'TP': [total_TP],
                'TN': [total_TN],
                'FP': [total_FP],
                'FN': [total_FN],
                'Weighted-Precision': [precision],
                'Weighted-Recall': [recall],
                'Weighted-F1-Score': [f1]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)


    # print(metrics_df)
    save_file_path = file_containing_llm_response.replace('local_chat_history/', '')

    result_directory = 'results/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    metrics_df.to_csv(result_directory+save_file_path)
    print("Result saved in: "+ result_directory+save_file_path)

