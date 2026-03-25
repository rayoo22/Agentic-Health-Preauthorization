import openai
import numpy as np
import faiss
import os

def generate_embeddings_and_build_index(chunks: list, output_path: str = "policy_index.faiss"):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = []
    for text in texts:
        response = openai.embeddings.create(
            model=os.getenv('EMBEDDING_MODEL'),
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index (dimension 1536 for ada-002)
    index = faiss.IndexFlatL2(1536)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, output_path)
    
    return index