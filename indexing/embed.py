import os
import json
from sentence_transformers import SentenceTransformer
from typing import List

def get_hf_token_from_json(file_path: str):
    """Reads the Hugging Face token from a JSON credentials file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        
        with open(full_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get("hf_token")
    except FileNotFoundError:
        print(f"Warning: Credentials file not found at {full_path}. Proceeding without a token.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {full_path}.")
        return None
HF_TOKEN = get_hf_token_from_json("../assets.json")
print("Loading sentence-transformer model... (This may take a moment on first run)")
model = SentenceTransformer('all-mpnet-base-v2', token=HF_TOKEN)
EMBEDDING_DIM = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")
def get_embeddings(chunks: List[str]) -> List[List[float]]:
    return model.encode(chunks).tolist()

