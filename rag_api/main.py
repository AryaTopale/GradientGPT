import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from rag_api.init_rag import qa_chain
except ImportError as e:
    print(f"Error: Could not import 'qa_chain' from 'initial_langchain/init_rag.py'. Make sure both files are in the 'initial_langchain' folder and that an '__init__.py' file exists in that folder. Details: {e}")
    sys.exit(1)

app = FastAPI(
    title="SME Agent API",
    description="An API for the Data Science Subject Matter Expert Agent",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    source_documents: list = []


# --- Define the API Endpoint ---
@app.post("/generate-assignment", response_model=QueryResponse)
def generate_assignment(request: QueryRequest):

    print(f"Received request: {request.query}")
    
    response = qa_chain.invoke({"query": request.query})
    
    source_docs_content = [doc.page_content for doc in response.get("source_documents", [])]

    return QueryResponse(
        result=response.get("result", "No result found."),
        source_documents=source_docs_content
    )

# --- Add a simple root endpoint for health checks ---
@app.get("/")
def read_root():
    return {"status": "SME Agent API is running"}

