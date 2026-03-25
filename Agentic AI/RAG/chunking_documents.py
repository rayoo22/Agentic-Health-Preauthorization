def chunking_documents(documents_content: dict)-> list:
    chunked_documents = []

    for policy_id, content in documents_content.items():
        sections = content.split('###')

        for chunk_id, section in enumerate(sections):
            if section.strip():
                chunked_documents.append({
                    'policy_id': policy_id,
                    'chunk_id': chunk_id,
                    'text':section.strip()
                })
        return chunked_documents
