import os
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from post_processing_birads import *
import pandas as pd
import os
def replace_none_with_one(lst):
    return [1 if x is None else x for x in lst]

def replace_values(lst):
        return [1 if x in {1, 2, 3} else 2 if x in {4, 5, 6} else x for x in lst]

master_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated_2/"
results_dir = "results_2"
os.makedirs(results_dir, exist_ok=True)

subdirs = [d for d in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, d))]
print(subdirs)

master_df = pd.DataFrame()


for t_dir in subdirs:
    print(t_dir)
    ground_truth_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"
    test_dir = master_dir +t_dir

    process_json_files(test_dir)
    # Lists to store BIRADS values
    ground_truth_birads = []
    test_birads = []

    ground_truth_findings = []
    test_findings = []

    ground_truth_breast_composition = []
    test_breast_composition = []


    # Get list of JSON files in ground-truth directory
    json_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".json"))

    # Iterate through each file
    for json_file in json_files:
        ground_truth_path = os.path.join(ground_truth_dir, json_file)
        test_path = os.path.join(test_dir, json_file)

        # Read ground-truth file
        with open(ground_truth_path, "r") as gt_file:
            gt_data = json.load(gt_file)
            gt_data = {key.lower(): value for key, value in gt_data.items()}

            ground_truth_birads.append((gt_data.get("birads")))  # Extract BIRADS value
            ground_truth_findings.append(str(gt_data.get("findings")))
            ground_truth_breast_composition.append(str(gt_data.get("breast_composition")))

        # Read test file
        with open(test_path, "r") as test_file:
            test_data = json.load(test_file)
            test_data = {key.lower(): value for key, value in test_data.items()}

            test_birads.append(test_data.get("birads"))  # Extract BIRADS value
            test_findings.append(str(test_data.get("findings")))
            test_breast_composition.append(str(test_data.get("breast-composition")))

    # clubbing birads into two categories
    ground_truth_birads = replace_values(ground_truth_birads)
    test_birads = replace_values(test_birads)
    # clubbing birads into two categories

    # Replacing None values with 1
    ground_truth_birads = replace_none_with_one(ground_truth_birads)
    test_birads = replace_none_with_one(test_birads)
    # Replacing None values with 1

    # Print the lists
    # print("Ground Truth BIRADS:", ground_truth_birads)
    # print("Test BIRADS:", test_birads)

    # print(len(ground_truth_birads))
    # print(len(test_birads))


    # Convert lists to a uniform data type (integers)
    cnt = 0
    for x in ground_truth_birads:
        # print(cnt,x)
        if(x==0):
            print(cnt, x)
        cnt = cnt +1

    ground_truth_birads = [int(x) for x in ground_truth_birads]
    test_birads = [int(x) for x in test_birads]

    # Generate classification report
    report = classification_report(ground_truth_birads, test_birads)
    print("Classification Report:\n", report)

    # Compute precision, recall, and F1-score separately
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        ground_truth_birads, test_birads, average='weighted'
    )

    # Print the individual scores


    # Generate confusion matrix
    cm = confusion_matrix(ground_truth_birads, test_birads)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(ground_truth_birads), yticklabels=np.unique(ground_truth_birads))

    # Labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    image_save_dir = test_dir.replace(master_dir, results_dir)
    plt.savefig(image_save_dir+".png")
    plt.show()

    # print(ground_truth_findings[0])
    # print(test_findings[0])

    # reference =   # List of lists
    # hypothesis =   # Single list

    # print("Printing here", ground_truth_findings[0], test_findings[0])

    bleu_score = 0
    findings_rouge_l_score = 0
    for i in range(0, len(ground_truth_findings)):
        # x = sentence_bleu([word_tokenize(ground_truth_findings[i])], word_tokenize(test_findings[i]), weights=(0.25, 0.25, 0.25, 0.25),
        #                        smoothing_function=SmoothingFunction().method1)
        # # print(x)
        # bleu_score += x

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(str(ground_truth_findings[i]), str(test_findings[i]))  # Use i instead of 0
        findings_rouge_l_score += scores['rougeL'].fmeasure  # Use .fmeasure explicitly

    # print("FINDINGS CATEGORY BLEU-4 Score:", round(bleu_score/len(ground_truth_findings), 2))

    # print(test_findings)
    # Ensure inputs are lists of strings
    assert isinstance(test_findings, list) and isinstance(ground_truth_findings, list), "Inputs must be lists"
    assert all(isinstance(x, str) for x in test_findings), "test_findings must contain only strings"
    assert all(isinstance(x, str) for x in ground_truth_findings), "ground_truth_findings must contain only strings"



    P, R, findings_F1 = bert_score.score(test_findings, ground_truth_findings, lang="en", model_type="microsoft/deberta-xlarge-mnli")


    # print(test_breast_composition[0])

    bleu_score = 0
    breast_composition_rouge_l_score = 0
    for i in range(0, len(ground_truth_breast_composition)):
        # print(ground_truth_breast_composition[i], test_breast_composition[i])
        # x = sentence_bleu([word_tokenize(ground_truth_breast_composition[i])], word_tokenize(test_breast_composition[i]), weights=(0.25, 0.25, 0.25, 0.25),
        #                        smoothing_function=SmoothingFunction().method1)
        # print(x)
        # bleu_score += x

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(str(ground_truth_breast_composition[i]), str(test_breast_composition[i]))  # Use i instead of 0
        breast_composition_rouge_l_score += scores['rougeL'].fmeasure  # Use .fmeasure explicitly

    # print("BREAST-COMPOSITION CATEGORY BLEU-4 Score:", round(bleu_score/len(test_breast_composition), 2))
    P, R, breast_composition_F1 = bert_score.score(ground_truth_breast_composition, test_breast_composition, lang="en", model_type="microsoft/deberta-xlarge-mnli")


    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    print("FINDINGS CATEGORY BERT Score:", round(findings_F1.mean().item(),2))

    print("FINDINGS CATEGORY ROUGE-L Score:", round(findings_rouge_l_score/len(ground_truth_findings), 2))

    print("-------------------------------------------------------------------")


    print("BREAST-COMPOSITION CATEGORY BERT Score:", round(breast_composition_F1.mean().item(), 2))

    print("BREAST-COMPOSITION CATEGORY ROUGE-L Score:", round(breast_composition_rouge_l_score/len(ground_truth_breast_composition), 2))




    data = {
        "Model Name": t_dir,
        "`BIRADS` Precision": [round(precision, 2)],
        "`BIRADS` Recall": [round(recall, 2)],
        "`BIRADS` F1-Score": [round(f1_score, 2)],
        "`Findings` BERT": [round(findings_F1.mean().item(), 2)],
        "`Findings` ROUGE-L": [round(findings_rouge_l_score / len(ground_truth_findings), 2)],
        "`Breast-Composition` BERT": [round(breast_composition_F1.mean().item(), 2)],
        "`Breast-Composition` ROUGE-L": [round(breast_composition_rouge_l_score / len(ground_truth_breast_composition), 2)]
    }

    df = pd.DataFrame(data)
    print(df)
    master_df = pd.concat([master_df, df], ignore_index=True)  # Resets the index


master_df.to_csv(results_dir+'/LLM_Performance.csv', index=False)
print(master_df)