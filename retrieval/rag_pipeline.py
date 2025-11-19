import os
import sys

from indexing.embed import get_embeddings
from retrieval.initialise import initialize_pinecone, initialize_gemini

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
pinecone_index = initialize_pinecone()
llm = initialize_gemini()

def retrieve_context(query: str, top_k: int = 3):
    print(f"\n1. Retrieving context for query: '{query}'")
    query_embedding = get_embeddings([query])[0]
    query_results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    context = [match['metadata']['text'] for match in query_results['matches']]
    print(f"Found {len(context)} relevant chunks of context.")
    return "\n---\n".join(context)

def build_prompt(query: str, context: str) -> str:
    print("2. Building prompt...")
    prompt_template = """
    You are a helpful Data Science assistant. Answer the following question based ONLY on the context provided below. 
    If the context does not contain the answer, state clearly that the answer is not found in the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    return prompt_template.format(question=query, context=context)

def generate_answer(prompt: str) -> str:
    print("3. Generating final answer with Gemini...")
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

user_query = "What is a Loss Function?"
retrieved_context = retrieve_context(user_query)
final_prompt = build_prompt(user_query, retrieved_context)
final_answer = generate_answer(final_prompt)
    
print("\n--- RAG PIPELINE COMPLETE ---")
print("\n[Final Answer]")
print(final_answer)
