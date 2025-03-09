import os
import re

import json


# def extract_number(s: str) -> int:
#     match = re.search(r'\d+', s)  # Find the first sequence of digits
#     return int(match.group()) if match else None  # Convert to int if found

# # Example usage
# input_str = "Img151"
# output = extract_number(input_str)
# print(output)  # Output: 151


# import os
# import glob

# def remove_json_files(directory: str):
#     json_files = glob.glob(os.path.join(directory, "*.json"))  # Get all .json files in the directory
    
#     for file in json_files:
#         os.remove(file)  # Remove each JSON file
#         print(f"Deleted: {file}")

# # Example usage
# directory_path = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/fine_tuned_results"  # Change this to your directory path
# remove_json_files(directory_path)


import torch
from PIL import Image
import open_clip

# Load OpenCLIP model and preprocessing function
output = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

print(type(output), len(output))
model, preprocess, tokenizer = output  # Assuming it returns 3 values

tokenizer = open_clip.get_tokenizer("ViT-B-32")


# Load and preprocess the image
img_path = "/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG001.png"  # Change to your image path
image = Image.open(img_path).convert("RGB")
image = preprocess(image).unsqueeze(0)  # Add batch dimension

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
image = image.to(device)

# Generate image embedding
with torch.no_grad():
    image_embedding = model.encode_image(image)

# Convert to NumPy array if needed
image_embedding_np = image_embedding.cpu().numpy()

print("Image embedding shape:", image_embedding_np.shape)

print(image_embedding)

print()

print(image_embedding_np)

# {'ids': [['IMG315.png', 'IMG145.png', 'IMG062.png']], 'embeddings': None, 'documents': [[None, None, None]], 'uris': None, 'data': None, 'metadatas': [[{'BIRADS': 4, 'Breast_Composition': 'fibro fatty and glandular breast parenchyma (ACR-C).', 'Findings': 'Well-defined lobulated soft opacity seen in the upper quadrant of the breast suggests high suspicious malignant lesion (BIRADS 4c)\nskin nipple and pectoral muscle appear normal.\nno abnormal microcalcification seen.\ntwo enlarged axillary nodes seen'}, {'BIRADS': 2, 'Breast_Composition': 'fibro fatty and glandular breast parenchyma (ACR B).', 'Findings': 'a benign-looking popcorn calcification is seen in the central quadrant.(BIRADS 2)\nskin, nipple, and pectoral muscle appear normal.\nbenign-looking axillary adenopathy.'}, {'BIRADS': 1, 'Breast_Composition': 'fibro glandular breast parenchyma (ACR B).', 'Findings': 'few punctate discrete foci of calcifications seen. \nwell-defined oval soft opacity seen adjacent to the normally positioned nipple may suggest inverted nipple \nshadow or abnormal lesion. Ultrasound correlation is suggested. \nSkin appears normal. \nMultiple axillary adenopathy seen.'}]], 'distances': [[14.23488576043342, 15.333428473250907, 17.65541342191785]], 'included': [<IncludeEnum.distances: 'distances'>, <IncludeEnum.documents: 'documents'>, <IncludeEnum.metadatas: 'metadatas'>]}