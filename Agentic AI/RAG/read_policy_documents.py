import os
from pathlib import Path

"_____________loading and reading the contents of policy documents ______________"
def load_and_reading_content(policy_document_path)->dict:

    rag_folder_path = Path(policy_document_path)

    rag_files = os.listdir(rag_folder_path)

    print(rag_files)

    documents_content = {}

    for filename in rag_files:
        if filename:
            file_path = os.path.join(rag_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                policy_id = filename.split('_')[0]
                #print(policy_id)
                documents_content[policy_id] = content
    return(documents_content)

"_________________chunking of the content____________________"
