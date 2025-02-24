import os
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Define directories
ground_truth_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"
test_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/qwen_base"


# Lists to store BIRADS values
ground_truth_birads = []
test_birads = []

# Get list of JSON files in ground-truth directory
json_files = sorted(f for f in os.listdir(ground_truth_dir) if f.endswith(".json"))

# Iterate through each file
for json_file in json_files:
    ground_truth_path = os.path.join(ground_truth_dir, json_file)
    test_path = os.path.join(test_dir, json_file)

    # Read ground-truth file
    with open(ground_truth_path, "r") as gt_file:
        gt_data = json.load(gt_file)
        ground_truth_birads.append((gt_data.get("BIRADS")))  # Extract BIRADS value

    # Read test file
    with open(test_path, "r") as test_file:
        test_data = json.load(test_file)
        test_birads.append(test_data.get("BIRADS"))  # Extract BIRADS value

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
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")


# Generate confusion matrix
cm = confusion_matrix(ground_truth_birads, test_birads)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(ground_truth_birads), yticklabels=np.unique(ground_truth_birads))

# Labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('abc.png')
plt.show()