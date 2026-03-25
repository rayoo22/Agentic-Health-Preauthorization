import openai
import faiss
import numpy as np
import pickle
import os

def query_policy_rag(diagnosis: str, service: str, index_path: str = "policy_index.faiss", 
                     metadata_path: str = "chunks_metadata.pkl", top_k: int = 3):
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load chunks metadata
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Create query text
    query = f"Diagnosis: {diagnosis}. Requested Service: {service}"
    
    # Generate query embedding
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = np.array([response.data[0].embedding]).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve relevant chunks
    relevant_chunks = []
    for idx in indices[0]:
        relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

def format_policy_context(relevant_chunks: list) -> str:
    """Format retrieved policy chunks into context for OpenAI prompt"""
    
    if not relevant_chunks:
        return "No relevant policy information found."
    
    context = "Relevant Policy Information:\n\n"
    
    for i, chunk in enumerate(relevant_chunks, 1):
        context += f"--- Policy {chunk['policy_id']} (Section {chunk['chunk_id']}) ---\n"
        context += f"{chunk['text']}\n\n"
    
    return context