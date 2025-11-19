import os
import sys
import json
import google.generativeai as genai

def get_gemini_api_key(file_path: str = "../assets.json"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        
        with open(full_path, 'r') as f:
            credentials = json.load(f)
        
        api_key = credentials.get("gemini_api_key")
        if not api_key:
            print(f"Error: 'gemini_api_key' not found in {full_path}")
            return None
        return api_key
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading credentials file: {e}")
        return None

GEMINI_API_KEY = get_gemini_api_key()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("--- Checking for available Gemini Models ---")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"- {m.name}")
    print("-------------------------------------------\n")
    model_name = 'gemini-2.5-pro'
    if f"models/{model_name}" not in available_models:
         print(f"Warning: '{model_name}' not found in the list of available models.")
         print("Attempting to use it anyway, but it may fail.")
    model = genai.GenerativeModel(model_name) 
    print(f"Sending request to Gemini using model: '{model_name}'...")
    try:
        response = model.generate_content("Explain how AI works in detail.")
        print("\n--- Gemini's Response ---")
        print(response.text)
    except Exception as e:
        print(f"\nAn error occurred while calling the Gemini API: {e}")
else:
    print("Could not proceed without a Gemini API key.")

