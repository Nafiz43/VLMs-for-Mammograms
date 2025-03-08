# import argparse
# import os
# import shutil
# from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function
# from langchain_community.vectorstores.chroma import Chroma


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"


# def main():

#     # Check if the database should be cleared (using the --clear flag).
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     # Create (or update) the data store.
#     documents = load_documents()
#     chunks = split_documents(documents)
#     add_to_chroma(chunks)


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)


# def add_to_chroma(chunks: list[Document]):
#     # Load the existing database.
#     db = Chroma(
#         persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
#     )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = db.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         db.persist()
#     else:
#         print("✅ No new documents to add")


# def calculate_chunk_ids(chunks):

#     # This will create IDs like "data/monopoly.pdf:6:2"
#     # Page Source : Page Number : Chunk Index

#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks


# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# if __name__ == "__main__":
#     main()


import argparse
import os
import shutil
import json
import numpy as np
from PIL import Image
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import torch
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image
import torch
import clip
from torchvision import transforms

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


CHROMA_PATH = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/chroma"
JSON_DATA_PATH = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/GROUND-TRUTH-REPORTS"
IMAGE_DATA_PATH = "/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    add_to_chroma(documents)

# def load_documents():
#     documents = []
#     json_files = {os.path.splitext(f)[0]: os.path.join(JSON_DATA_PATH, f) for f in os.listdir(JSON_DATA_PATH) if f.endswith(".json")}
#     image_files = {os.path.splitext(f)[0]: os.path.join(IMAGE_DATA_PATH, f) for f in os.listdir(IMAGE_DATA_PATH) if f.endswith(".png")}

#     common_keys = json_files.keys() & image_files.keys()
#     print(f"🔍 Found {len(common_keys)} matching JSON & PNG pairs.")  # Total matching files

#     for idx, key in enumerate(common_keys):
#         with open(json_files[key], "r", encoding="utf-8") as f:
#             json_data = json.load(f)
        
#         image_embedding = get_image_embedding(image_files[key])
        
#         metadata = {"source": json_files[key], "image_source": image_files[key]}
#         document_content = json.dumps(json_data)  # Convert JSON data to a string format
        
#         document = Document(page_content=document_content, metadata=metadata)
#         document.metadata["image_embedding"] = image_embedding.tolist()
        
#         documents.append(document)
#         print(f"✅ Indexed {idx + 1}/{len(common_keys)}: {key}")  # Per-document indexing message
    
#     return documents

def load_documents():
    documents = []
    json_files = {os.path.splitext(f)[0]: os.path.join(JSON_DATA_PATH, f) for f in os.listdir(JSON_DATA_PATH) if f.endswith(".json")}
    image_files = {os.path.splitext(f)[0]: os.path.join(IMAGE_DATA_PATH, f) for f in os.listdir(IMAGE_DATA_PATH) if f.endswith(".png")}

    common_keys = json_files.keys() & image_files.keys()
    print(f"🔍 Found {len(common_keys)} matching JSON & PNG pairs.")  # Total matching files

    for idx, key in enumerate(common_keys):
        with open(json_files[key], "r", encoding="utf-8") as f:
            json_data = json.load(f)

        image_embedding = get_image_embedding(image_files[key])  # Get embedding as a NumPy array

        # image_embedding = torch.tensor(image_files[key])


        # Prepare metadata with relevant fields
        metadata = {
            "source": json_files[key],
            "image_source": image_files[key],
            "img_id": json_data.get("IMG-ID", ""),
            "breast_composition": json_data.get("BREAST-COMPOSITION", ""),
            "birads": json_data.get("BIRADS", ""),
            "findings": json_data.get("FINDINGS", ""),
            "image_embedding": image_embedding,  # Ensure it is stored as a list
        }

        # Store the JSON data as text in `page_content`
        document_content = json.dumps(json_data)  # Convert JSON to string

        # Create the document and append it to the list
        document = Document(page_content=document_content, metadata=metadata)
        documents.append(document)

        print(f"✅ Indexed {idx + 1}/{len(common_keys)}: {key}")  # Per-document indexing message
    
    return documents


# def get_image_embedding(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((224, 224))  # Resize to a standard size
#     image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
#     return image_array.flatten()  # Flatten the array for embedding

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().flatten()  # Convert to NumPy array

def add_to_chroma(documents):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    existing_items = db.get(include=["metadatas"])  # ✅ Fetch metadata (which contains IDs)
    existing_ids = set(item["source"] for item in existing_items["metadatas"])  # Extract document IDs


    print(f"📂 Existing documents in DB before insert: {len(existing_ids)}")

    new_documents = []
    for doc in documents:
        doc_id = doc.metadata["source"]
        if doc_id not in existing_ids:
            new_documents.append(doc)
            print(f"📌 Adding: {doc_id}")  # Print each document being added

    if new_documents:
        print(f"👉 Adding {len(new_documents)} new documents to ChromaDB")
        new_ids = [doc.metadata["source"] for doc in new_documents]
        db.add_documents(new_documents, ids=new_ids)
        db.persist()  # Ensure database is saved
        print("✅ Database update complete!")
    else:
        print("✅ No new documents to add (they might already exist).")
        
    print(f"📂 Total stored documents: {len(db.get(include=['documents'])['documents'])}")



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()

# def check_chroma_contents(db, sample_size=4):
#     # Fetch all stored documents
#     all_documents = db.get(include=["metadatas", "documents", "embeddings"])  

#     num_documents = len(all_documents["documents"])
#     print(f"📂 Total documents in ChromaDB: {num_documents}")

#     # Show a few sample documents
#     for i in range(min(sample_size, num_documents)):
#         print("\n🔹 Sample Document", i + 1)
#         print("Metadata:", all_documents["metadatas"][i])
#         print("Content (truncated):", all_documents["documents"][i][:500])  # Show first 500 chars
#         print("Embedding shape:", len(all_documents["embeddings"][i]))  # Check embedding size

# # Usage
# db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
# check_chroma_contents(db)
