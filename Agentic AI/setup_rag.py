"""
One-time setup script to build FAISS index from policy documents
Run this before running main.py for the first time
"""

from dotenv import load_dotenv
from RAG.read_policy_documents import load_and_reading_content
from RAG.chunking_documents import chunking_documents
from RAG.generate_embeddings import generate_embeddings_and_build_index
import pickle
import os

def build_rag_index():
    load_dotenv()
    
    print("Step 1: Loading policy documents...")
    documents = load_and_reading_content("Policy_Documents")
    print(f"   Loaded {len(documents)} policy documents")
    
    print("\nStep 2: Chunking documents by sections...")
    chunks = chunking_documents(documents)
    print(f"   Created {len(chunks)} chunks")
    
    print("\nStep 3: Generating embeddings and building FAISS index...")
    print("   (This will make OpenAI API calls - estimated cost: ~$0.01)")
    index = generate_embeddings_and_build_index(chunks, "policy_index.faiss")
    print("   [OK] FAISS index saved to policy_index.faiss")
    
    print("\nStep 4: Saving chunks metadata...")
    with open("chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("   [OK] Metadata saved to chunks_metadata.pkl")
    
    print("\n[SUCCESS] RAG setup complete! You can now run main.py")

if __name__ == "__main__":
    build_rag_index()
