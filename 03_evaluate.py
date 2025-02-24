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



# Define directories
ground_truth_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"
test_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base"


# Lists to store BIRADS values
ground_truth_birads = []
test_birads = []

ground_truth_findings = []
test_findings = []

ground_truth_breast_composition = []
test_breast_composition = []



def replace_values(lst):
    return [1 if x in {1, 2, 3} else 2 if x in {4, 5, 6} else x for x in lst]



# Get list of JSON files in ground-truth directory
json_files = sorted(f for f in os.listdir(ground_truth_dir) if f.endswith(".json"))

# Iterate through each file
for json_file in json_files:
    ground_truth_path = os.path.join(ground_truth_dir, json_file)
    test_path = os.path.join(test_dir, json_file)

    # Read ground-truth file
    with open(ground_truth_path, "r") as gt_file:
        gt_data = json.load(gt_file)
        gt_data = {key.lower(): value for key, value in gt_data.items()}

        ground_truth_birads.append((gt_data.get("birads")))  # Extract BIRADS value
        ground_truth_findings.append((gt_data.get("findings")))
        ground_truth_breast_composition.append(gt_data.get("breast_composition"))

    # Read test file
    with open(test_path, "r") as test_file:
        test_data = json.load(test_file)
        test_data = {key.lower(): value for key, value in test_data.items()}

        test_birads.append(test_data.get("birads"))  # Extract BIRADS value
        test_findings.append((test_data.get("findings")))
        test_breast_composition.append((test_data.get("breast_composition")))

# clubbing birads into two categories
ground_truth_birads = replace_values(ground_truth_birads)
test_birads = replace_values(test_birads)
# clubbing birads into two categories

# Print the lists
print("Ground Truth BIRADS:", ground_truth_birads)
print("Test BIRADS:", test_birads)

print(len(ground_truth_birads))
print(len(test_birads))


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
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")


# Generate confusion matrix
cm = confusion_matrix(ground_truth_birads, test_birads)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(ground_truth_birads), yticklabels=np.unique(ground_truth_birads))

# Labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
image_save_dir = test_dir.replace('/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/', 'result_figs/')
plt.savefig(image_save_dir)
plt.show()

print(ground_truth_findings[0])
print(test_findings[0])

# reference =   # List of lists
# hypothesis =   # Single list

bleu_score = 0
for i in range(0, len(ground_truth_findings)):
    x = sentence_bleu([word_tokenize(ground_truth_findings[0])], word_tokenize(test_findings[0]))
    # print(x)
    bleu_score += x




print("BLEU-4 Score:", round(bleu_score/len(ground_truth_findings), 2))

P, R, F1 = bert_score.score(test_findings, ground_truth_findings, lang="en", model_type="microsoft/deberta-xlarge-mnli")

print("BERT Score:", F1.mean().item())