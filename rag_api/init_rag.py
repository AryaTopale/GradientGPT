import os
import sys
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def get_api_key(key_name: str, file_path: str = "assets.json"):
    try:
        full_path = os.path.join(parent_dir, file_path)
        with open(full_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get(key_name)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading credentials file: {e}")
        return None

PINECONE_API_KEY = get_api_key("pinecone_api_key")
GEMINI_API_KEY = get_api_key("gemini_api_key")
HF_TOKEN = get_api_key("hf_token")

if not all([PINECONE_API_KEY, GEMINI_API_KEY, HF_TOKEN]):
    raise ValueError("Pinecone, Gemini, and Hugging Face API keys must be set in assets.json.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
print("Initializing LangChain components...")
embeddings = HuggingFaceEmbeddings(
    model_name='all-mpnet-base-v2',
    model_kwargs={'use_auth_token': HF_TOKEN}
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GEMINI_API_KEY
)
INDEX_NAME = "sme-agent-data-science"
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
print("Components initialized successfully.")

prompt_template_str = """
You are a Data Science instructor. Your task is to create a short assignment based on the user's request, using the provided context as your primary source of information.
The assignment should be well-structured, clear, and relevant to the topic.
CONTEXT:
{context}
USER'S REQUEST:
{question}
ASSIGNMENT:
"""
PROMPT = PromptTemplate(
    template=prompt_template_str, input_variables=["context", "question"]
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
print("RAG chain for assignment generation created.")
user_query = "Create an assignment on binary classification that includes 2-3 theoretical questions."
print(f"\nExecuting RAG pipeline for request: '{user_query}'")
response = qa_chain.invoke({"query": user_query})
print("\n--- RAG PIPELINE COMPLETE ---")
print("\n[Generated Assignment]")
print(response["result"])

