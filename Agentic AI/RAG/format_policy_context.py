def format_policy_context(relevant_chunks: list) -> str:
    """Format retrieved policy chunks into context for OpenAI prompt"""
    
    if not relevant_chunks:
        return "No relevant policy information found."
    
    context = "Relevant Policy Information:\n\n"
    
    for i, chunk in enumerate(relevant_chunks, 1):
        context += f"--- Policy {chunk['policy_id']} (Section {chunk['chunk_id']}) ---\n"
        context += f"{chunk['text']}\n\n"
    
    return context