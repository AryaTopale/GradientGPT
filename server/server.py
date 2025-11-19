import uvicorn
import os
import aiofiles
import json  # <-- ADD THIS
import re   # <-- ADD THIS
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles  # <-- IMPORT THIS
from pydantic import BaseModel
from typing import Dict, Any

# --- Add Project to Path ---
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End Path Addition ---

try:
    from planning.conversational_agent_graph import get_agent_graph
except ImportError as e:
    print(f"Error importing agent graph: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempting to import from project root: {project_root}")
    sys.exit(1)

# Initialize the FastAPI app and the agent graph
app = FastAPI()
graph = get_agent_graph()

# --- *** START OF DOWNLOAD FIX *** ---

# --- UPDATED PATH: Use 'current_dir' ---
# This will be inside the 'server' folder (e.g., .../server/generated_documents)
generated_docs_path = os.path.join(current_dir, 'generated_documents')
os.makedirs(generated_docs_path, exist_ok=True)
print(f"Serving downloadable files from: {generated_docs_path}")
# --- END UPDATED PATH ---

# Mount this directory to a URL path
# Any file in 'generated_docs_path' will be accessible via 'http://.../generated_documents/filename' 
app.mount("/generated_documents", StaticFiles(directory=generated_docs_path), name="download")

# --- *** END OF DOWNLOAD FIX *** ---


# Pydantic model for the chat request
class ChatRequest(BaseModel):
    message: str
    thread_id: str

# --- API Endpoints ---

@app.get("/")
async def get_index():
    """Serves the main HTML page."""
    # --- UPDATED PATH: Use 'current_dir' ---
    html_file_path = os.path.join(current_dir, 'index.html')
    if not os.path.exists(html_file_path):
        # Fallback just in case
        html_file_path = os.path.join(project_root, 'UI', 'index.html') 
    return FileResponse(html_file_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles a single chat message."""
    
    print(f"Received message for thread {request.thread_id}: {request.message}")

    config = {"configurable": {"thread_id": request.thread_id}}
    
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    final_message_object = None 
    
    try:
        final_state = await graph.ainvoke(inputs, config=config)
        
        if "messages" in final_state and final_state["messages"]:
            final_message_object = final_state["messages"][-1]
            
        if final_message_object:
            response_text = final_message_object.content     
            
            # --- *** START OF AGENT FLUFF & LOOP FIX *** ---
            
            # 1. Regex to find a JSON block containing "download_path"
            # This is the IDEAL path.
            json_match = re.search(r'\{.*"download_path":\s*".*".*\}', response_text, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                try:
                    # 2. Validate it's real JSON
                    json.loads(json_string) 
                    
                    # 3. Send *only* the clean JSON string.
                    print(f"Extracted JSON response for thread {request.thread_id}: {json_string}")
                    return JSONResponse(content={"reply": json_string})
                
                except json.JSONDecodeError:
                    print(f"Regex found a JSON-like string, but it failed to parse. Sending as text.")
                    pass
            
            # --- *** END OF AGENT FLUFF & LOOP FIX *** ---


            # --- *** START OF FINAL ROBUST HACKY FIX *** ---
            #
            # If no JSON packet was found, we look for the path
            # *anywhere* in the plain text.
            #
            # --- THIS IS THE LINE THAT IS FIXED ---
            # We add a space " " to [\w.\- ] to capture filenames with spaces.
            
            path_match = re.search(
                                    r'generated_documents[/\\][\w.\- ]+\.(pdf|md|docx|pptx)',
                                    response_text,
                                    re.IGNORECASE
                                )
            
            if path_match:
                # Use group(0) to get the ENTIRE matched path
                extracted_path = path_match.group(0)

                # Clean the path
                extracted_path = extracted_path.replace("\\", "/")
                if not extracted_path.startswith("/"):
                    extracted_path = "/" + extracted_path
                
                print(f"FINAL FIX: Found path with spaces '{extracted_path}'. Building JSON packet.")
                
                # Manually build the JSON packet the frontend expects
                json_data = {
                    "download_path": extracted_path,
                    "message": response_text  # Send the full agent message
                }
                # Send this new JSON string
                return JSONResponse(content={"reply": json.dumps(json_data)})
            
            # --- *** END OF FINAL ROBUST HACKY FIX *** ---


            # No JSON found AND no path found in plain text.
            # This is a normal text response.
            print(f"Sending text response for thread {request.thread_id}: {response_text}")
            return JSONResponse(content={"reply": response_text})
        else:
            print(f"No AI response generated for thread {request.thread_id}")
            return JSONResponse(content={"reply": "I'm sorry, I couldn't find a response."}, status_code=500)

    except Exception as e:
        print(f"Error during graph invocation for thread {request.thread_id}: {e}")
        return JSONResponse(content={"reply": f"An error occurred: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)