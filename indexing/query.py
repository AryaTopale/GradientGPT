import os
import sys
import json
from pinecone import Pinecone
from embed import get_embeddings, EMBEDDING_DIM
from index import get_pinecone_credentials_from_json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  

PINECONE_API_KEY = get_pinecone_credentials_from_json("../assets.json")
if not PINECONE_API_KEY:
    raise ValueError("Could not find 'pinecone_api_key' in the assets.json file.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to both indexes
CHILD_INDEX_NAME = "sme-agent-child-chunks"
PARENT_INDEX_NAME = "sme-agent-parent-chunks"

try:
    child_index = pc.Index(CHILD_INDEX_NAME)
    parent_index = pc.Index(PARENT_INDEX_NAME)
    print("Successfully connected to Pinecone indexes.")
    print("\nChild index stats:")
    print(child_index.describe_index_stats())
    print("\nParent index stats:")
    print(parent_index.describe_index_stats())
except Exception as e:
    print(f"Error connecting to Pinecone indexes: {e}")
    sys.exit(1)


def retrieve_context_parent_child(query: str, top_k: int = 3):
    """
    Retrieves context using parent-child chunking:
    1. Search child chunks for semantic similarity
    2. Return the corresponding parent chunks for full context
    """
    print(f"\n1. Embedding the query: '{query}'")
    query_embedding = get_embeddings([query])[0]
    
    print("2. Querying child chunks to find the most relevant small pieces...")
    child_results = child_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract parent IDs from child results
    parent_ids = []
    for match in child_results['matches']:
        parent_id = match['metadata'].get('parent_id')
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)
    
    print(f"3. Found {len(parent_ids)} unique parent chunks to retrieve")
    
    # Fetch parent chunks by ID
    if not parent_ids:
        print("No parent chunks found!")
        return ""
    
    print("4. Fetching full parent chunks for complete context...")
    parent_results = parent_index.fetch(ids=parent_ids)
    
    # Extract text from parent chunks
    context_chunks = []
    for parent_id in parent_ids:
        if parent_id in parent_results['vectors']:
            parent_text = parent_results['vectors'][parent_id]['metadata']['text']
            context_chunks.append(parent_text)
    
    print(f"5. Retrieved {len(context_chunks)} parent chunks with full context")
    
    return "\n\n---\n\n".join(context_chunks)


def retrieve_context_with_scores(query: str, top_k: int = 3):
    """
    Same as retrieve_context_parent_child but also returns scores and metadata.
    Useful for debugging and understanding retrieval quality.
    """
    print(f"\n1. Embedding the query: '{query}'")
    query_embedding = get_embeddings([query])[0]
    
    print("2. Querying child chunks to find the most relevant small pieces...")
    child_results = child_index.query(
        vector=query_embedding,
        top_k=top_k * 2,  # Get more child results to have options
        include_metadata=True
    )
    
    # Group child results by parent ID with their scores
    parent_to_children = {}
    for match in child_results['matches']:
        parent_id = match['metadata'].get('parent_id')
        if parent_id:
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append({
                'score': match['score'],
                'child_text': match['metadata']['text'][:200] + "..."
            })
    
    # Sort parents by max child score and take top_k
    sorted_parents = sorted(
        parent_to_children.items(),
        key=lambda x: max(child['score'] for child in x[1]),
        reverse=True
    )[:top_k]
    
    parent_ids = [parent_id for parent_id, _ in sorted_parents]
    
    print(f"3. Selected top {len(parent_ids)} parent chunks based on child relevance scores")
    
    # Fetch parent chunks by ID
    if not parent_ids:
        print("No parent chunks found!")
        return "", []
    
    print("4. Fetching full parent chunks for complete context...")
    parent_results = parent_index.fetch(ids=parent_ids)
    
    # Build context with metadata
    context_chunks = []
    metadata_list = []
    
    for parent_id, children in sorted_parents:
        if parent_id in parent_results['vectors']:
            parent_text = parent_results['vectors'][parent_id]['metadata']['text']
            context_chunks.append(parent_text)
            
            max_child_score = max(child['score'] for child in children)
            metadata_list.append({
                'parent_id': parent_id,
                'source': parent_results['vectors'][parent_id]['metadata'].get('source', 'Unknown'),
                'max_child_score': max_child_score,
                'num_matching_children': len(children)
            })
    
    print(f"5. Retrieved {len(context_chunks)} parent chunks with full context\n")
    
    # Print detailed information
    print("Retrieval Details:")
    print("-" * 80)
    for i, metadata in enumerate(metadata_list, 1):
        print(f"{i}. Source: {metadata['source']}")
        print(f"   Parent ID: {metadata['parent_id']}")
        print(f"   Best child match score: {metadata['max_child_score']:.4f}")
        print(f"   Number of matching children: {metadata['num_matching_children']}")
        print()
    
    return "\n\n---\n\n".join(context_chunks), metadata_list


def build_prompt(query: str, context: str) -> str:
    """
    Builds a prompt for the LLM with the retrieved context.
    """
    prompt_template = """
You are a helpful Data Science assistant. Answer the following question based ONLY on the context provided below. If the context does not contain the answer, state that you cannot answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt_template.format(question=query, context=context)


if __name__ == "__main__":
    # Example 1: Basic retrieval
    print("=" * 80)
    print("BASIC PARENT-CHILD RETRIEVAL")
    print("=" * 80)
    
    user_query = "What is binary classification?"
    retrieved_context = retrieve_context_parent_child(user_query, top_k=3)
    final_prompt = build_prompt(user_query, retrieved_context)
    
    print("\n" + "=" * 80)
    print("FINAL PROMPT FOR LLM")
    print("=" * 80)
    print(final_prompt)
    
    # Example 2: Retrieval with detailed scores
    print("\n\n" + "=" * 80)
    print("DETAILED RETRIEVAL WITH SCORES")
    print("=" * 80)
    
    user_query_2 = "Explain gradient descent"
    retrieved_context_2, metadata = retrieve_context_with_scores(user_query_2, top_k=2)
    
    print("\nRetrieved Context Preview:")
    print("-" * 80)
    for i, chunk in enumerate(retrieved_context_2.split("\n\n---\n\n"), 1):
        print(f"\n[Parent Chunk {i}]")
        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        print("-" * 80)