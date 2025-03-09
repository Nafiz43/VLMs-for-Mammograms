import os
import json
from glob import glob
from PIL import Image
import open_clip

import torch

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import os
import json
from glob import glob
from PIL import Image
import open_clip


# Create database file at folder "chroma" or load into client if exists.
chroma_client = chromadb.PersistentClient(path="chroma")

# Instantiate image loader helper.
image_loader = ImageLoader()

# Instantiate multimodal embedding function.
multimodal_ef = OpenCLIPEmbeddingFunction()

# Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
multimodal_db = chroma_client.get_or_create_collection(name="multimodal_db", embedding_function=multimodal_ef, data_loader=image_loader)



output = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

print(type(output), len(output))
model, preprocess, tokenizer = output  # Assuming it returns 3 values

tokenizer = open_clip.get_tokenizer("ViT-B-32")





# Assuming OpenCLIPEmbeddingFunction is already defined/imported
embedding_function = OpenCLIPEmbeddingFunction()  # Initialize your embedding function

# Define the directory paths
image_directory = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/'
json_directory = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS'

# Initialize lists
ids = []
uris = []
metadatas = []
image_embeddings = []

# Initialize counter
processed_count = 0  

# Loop through images
for img_path in glob(os.path.join(image_directory, '*.png')):  
    img_name = os.path.basename(img_path)  
    json_path = os.path.join(json_directory, img_name.replace('.png', '.json'))  

    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            metadata = json.load(json_file)

        # Extract metadata fields
        metadata_entry = {
            'Breast_Composition': metadata.get('Breast_Composition', 'N/A'),
            'BIRADS': metadata.get('BIRADS', 'N/A'),
            'Findings': metadata.get('Findings', 'N/A')
        }

        # Load image
        image = Image.open(img_path).convert("RGB")

        image = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        image = image.to(device)

        # Generate image embedding
        with torch.no_grad():
            image_embedding = model.encode_image(image)

        # Convert to NumPy array if needed and flatten
        image_embedding_np = image_embedding.cpu().numpy()
        image_embedding_list = image_embedding_np.flatten().tolist()  # Convert to list

        # Store details
        ids.append(img_name)
        uris.append(img_path)
        metadatas.append(metadata_entry)
        image_embeddings.append(image_embedding_list)  # Store flattened embedding

        # Increment counter and print progress
        processed_count += 1
        print(f"Indexed {processed_count}: {img_name}")

        if processed_count == 5:
            break  # Stop after 5 images for testing

# Check data before adding to DB (Optional)
print("IDs:", ids)
print("URIs:", uris)
print("Embeddings:", image_embeddings)
print("Metadatas:", metadatas)

# Add records to the multimodal database

print(image_embeddings)
multimodal_db.add(
    ids=ids,
    uris=uris,
    embeddings=image_embeddings,  # Store OpenCLIP embeddings here
    metadatas=metadatas
)

# Final stats
print(f"Total images indexed: {processed_count}")
count = multimodal_db.count()  # Get the count of stored embeddings
print(f"Number of indexes (embeddings) in the collection: {count}")


# multimodal_db.persist('/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/chromaDB')

