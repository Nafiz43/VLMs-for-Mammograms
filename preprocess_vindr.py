import os
import shutil
import pandas as pd

# Define paths
data_dir = "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/"  # Change this to your main directory
csv_file = "/mnt/data1/raiyan/breast_cancer/datasets/vindr/breast-level_annotations.csv"  # Change this to your CSV file path
output_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/"  # Change this to your desired output directory

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Extract necessary columns
columns_needed = ['study_id', 'image_id', 'laterality', 'view_position']
df = df[columns_needed]

# Find image in the specific study_id directory
def find_image(study_id, image_id, root_dir):
    image_path = os.path.join(root_dir, study_id, f"{image_id}.png")
    return image_path if os.path.exists(image_path) else None

# Process each row in the DataFrame
for _, row in df.iterrows():
    study_id = str(row['study_id'])
    image_id = str(row['image_id'])
    laterality = row['laterality']
    view_position = row['view_position']
    
    # Find the image
    image_path = find_image(study_id, image_id, data_dir)
    if image_path:
        # Define the destination folder
        dest_folder = os.path.join(output_dir, f"{laterality}_{view_position}")
        os.makedirs(dest_folder, exist_ok=True)
        
        # Define the destination image path
        dest_image_path = os.path.join(dest_folder, f"{study_id}.png")
        
        # Copy and rename the image
        shutil.copy(image_path, dest_image_path)
        print(f"Copied {image_path} to {dest_image_path}")
    else:
        print(f"Image {image_id}.png not found in {study_id} directory!")

print("Processing complete.")

# import os
# from collections import defaultdict

# # Define the parent directory containing the 4 subdirectories
# parent_dir = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/"  # Change this to your actual path

# # Get the list of subdirectories
# subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# if len(subdirs) != 4:
#     print("Warning: There are not exactly 4 subdirectories!")

# # Dictionary to track file occurrences
# file_counts = defaultdict(int)

# # Iterate over each subdirectory and count occurrences of each file
# for subdir in subdirs:
#     for file in os.listdir(subdir):
#         if file.endswith(".png"):  # Only consider PNG files
#             file_counts[file] += 1

# # Find files that are not present in all 4 subdirectories
# missing_files = [file for file, count in file_counts.items() if count < 4]

# # Print missing files
# if missing_files:
#     print("Files not present in all 4 subdirectories:")
#     for file in missing_files:
#         print(file)
# else:
#     print("All files are present in all 4 subdirectories.")

