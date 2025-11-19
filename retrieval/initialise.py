import os
import sys
import json
import google.generativeai as genai
from pinecone import Pinecone

def get_api_key(key_name: str, file_path: str = "../assets.json"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        with open(full_path, 'r') as f:
            credentials = json.load(f)
        api_key = credentials.get(key_name)
        if not api_key:
            raise ValueError(f"'{key_name}' not found in {full_path}")
        return api_key
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading credentials file: {e}")
        sys.exit(1)

def initialize_pinecone():
    print("Initializing Pinecone client...")
    pinecone_api_key = get_api_key("pinecone_api_key")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "sme-agent-data-science"
    
    if index_name not in pc.list_indexes().names():
        print(f"Error: Pinecone index '{index_name}' does not exist.")
        sys.exit(1)
    print("Successfully connected to Pinecone index.")
    return pc.Index(index_name)

def initialize_gemini():
    print("Initializing Gemini client...")
    gemini_api_key = get_api_key("gemini_api_key")
    genai.configure(api_key=gemini_api_key)
    model_name = 'gemini-2.5-pro'
    print(f"Using Gemini model: {model_name}")
    return genai.GenerativeModel(model_name)
