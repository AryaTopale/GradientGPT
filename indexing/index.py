import os
import sys
import json
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocessing.chunking_2 import load_document, clean_text, chunk_text_parent_child
from indexing.embed import get_embeddings, EMBEDDING_DIM


def get_pinecone_credentials_from_json(file_path: str):
    """Load Pinecone API key from JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        with open(full_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get("pinecone_api_key")
    except FileNotFoundError:
        print(f"Error: Credentials file not found at {full_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {full_path}.")
        return None


def initialize_pinecone_indexes(pc, child_index_name: str, parent_index_name: str):
    """Create Pinecone indexes if they don't exist."""
    # Create child index (for searching)
    if child_index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {child_index_name}")
        pc.create_index(
            name=child_index_name,
            dimension=EMBEDDING_DIM,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Child index created successfully.")
    else:
        print(f"Child index '{child_index_name}' already exists.")

    # Create parent index (for retrieval)
    if parent_index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {parent_index_name}")
        pc.create_index(
            name=parent_index_name,
            dimension=EMBEDDING_DIM,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Parent index created successfully.")
    else:
        print(f"Parent index '{parent_index_name}' already exists.")
    
    return pc.Index(child_index_name), pc.Index(parent_index_name)


def process_and_upload_parent_child(file_path: str, child_index, parent_index, parent_dir: str):
    """
    Process a file with parent-child chunking and upload to Pinecone.
    Child chunks are indexed for search, parent chunks for retrieval.
    """
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    raw_text = load_document(file_path)
    if not raw_text:
        return
    
    cleaned_text = clean_text(raw_text)
    
    # Get parent and child chunks with mapping
    parent_chunks, child_chunks, child_to_parent_map = chunk_text_parent_child(
        cleaned_text,
        file_path,
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=500,
        child_chunk_overlap=50
    )
    
    if not child_chunks:
        print("No chunks generated. Skipping.")
        return
    
    # Save pre-processed chunks to JSON
    output_dir = os.path.join(parent_dir, "processed_chunks")
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(file_path)
    json_filename = os.path.splitext(base_filename)[0] + "_parent_child.json"
    output_path = os.path.join(output_dir, json_filename)
    
    data_to_save = {
        "source_file": file_path,
        "parent_chunk_count": len(parent_chunks),
        "child_chunk_count": len(child_chunks),
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "child_to_parent_mapping": child_to_parent_map
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Successfully saved pre-processed chunks to {output_path}")
    except Exception as e:
        print(f"Error saving chunks to {output_path}: {e}")
    
    # Upload child chunks (for searching)
    print(f"\nGenerating embeddings for {len(child_chunks)} child chunks...")
    batch_size = 32
    
    for i in tqdm(range(0, len(child_chunks), batch_size), desc="Uploading child chunks"):
        batch_chunks = child_chunks[i:i + batch_size]
        embeddings = get_embeddings(batch_chunks)
        
        vectors_to_upsert = []
        for j, text_of_chunk in enumerate(batch_chunks):
            child_idx = i + j
            parent_idx = child_to_parent_map[child_idx]
            
            vectors_to_upsert.append({
                "id": f"{os.path.basename(file_path)}-child-{child_idx}",
                "values": embeddings[j],
                "metadata": {
                    "text": text_of_chunk,
                    "source": os.path.basename(file_path),
                    "parent_id": f"{os.path.basename(file_path)}-parent-{parent_idx}",
                    "type": "child"
                }
            })
        
        if vectors_to_upsert:
            child_index.upsert(vectors=vectors_to_upsert)
    
    print(f"Child chunks uploaded to child index")
    
    # Upload parent chunks (for retrieval)
    print(f"\nGenerating embeddings for {len(parent_chunks)} parent chunks...")
    
    for i in tqdm(range(0, len(parent_chunks), batch_size), desc="Uploading parent chunks"):
        batch_chunks = parent_chunks[i:i + batch_size]
        embeddings = get_embeddings(batch_chunks)
        
        vectors_to_upsert = []
        for j, text_of_chunk in enumerate(batch_chunks):
            parent_idx = i + j
            
            vectors_to_upsert.append({
                "id": f"{os.path.basename(file_path)}-parent-{parent_idx}",
                "values": embeddings[j],
                "metadata": {
                    "text": text_of_chunk,
                    "source": os.path.basename(file_path),
                    "parent_id": f"{os.path.basename(file_path)}-parent-{parent_idx}",
                    "type": "parent"
                }
            })
        
        if vectors_to_upsert:
            parent_index.upsert(vectors=vectors_to_upsert)
    
    print(f"Parent chunks uploaded to parent index")
    print(f"Upload complete for {os.path.basename(file_path)}")


def main():
    """Main function to process and upload documents to Pinecone."""
    # Get Pinecone credentials
    PINECONE_API_KEY = get_pinecone_credentials_from_json("../assets.json")
    if not PINECONE_API_KEY:
        raise ValueError("Could not find 'pinecone_api_key' in the assets.json file.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Index names
    CHILD_INDEX_NAME = "sme-agent-child-chunks"
    PARENT_INDEX_NAME = "sme-agent-parent-chunks"
    
    # Create indexes
    child_index, parent_index = initialize_pinecone_indexes(pc, CHILD_INDEX_NAME, PARENT_INDEX_NAME)
    
    # Define data directories
    DATA_DIRECTORY = [
        "data/cs229-2018-autumn/extra-notes/",
        "data/cs229-2018-autumn/materials/",
        "data/cs229-2018-autumn/notes/",
        "data/cs229-2018-autumn/",
        "data/"
    ]
    
    # Process all PDF files
    all_pdf_files = []
    for directory in DATA_DIRECTORY:
        if not os.path.isdir(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping.")
            continue
        
        print(f"--- Scanning directory: {directory} ---")
        pdf_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith('.pdf')
        ]
        all_pdf_files.extend(pdf_files)
    
    if not all_pdf_files:
        print("No PDF files found in any directory.")
        return
    
    print(f"\nFound {len(all_pdf_files)} PDF files to process.")
    
    # Process each PDF file
    for file_path in all_pdf_files:
        try:
            process_and_upload_parent_child(file_path, child_index, parent_index, parent_dir)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            print("Skipping to next file...")
    
    print("\n--- All Files Processed. Indexing Complete ---")
    print("Waiting for 10 seconds for the indexes to update...")
    time.sleep(10)
    
    print("\nChild index statistics:")
    print(child_index.describe_index_stats())
    print("\nParent index statistics:")
    print(parent_index.describe_index_stats())


if __name__ == "__main__":
    main()