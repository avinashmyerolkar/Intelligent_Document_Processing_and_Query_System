import os
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from entity_extraction import openai_chat_completion_response
from pdfminer.high_level import extract_text


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

data_directory_path="/home/avinash_m_yerolkar/frontend/data"

def pdf_files_to_chunks(input_directory_path: str) -> List[Document]:
    # Get all PDF file paths in the directory
    file_paths = [
        os.path.join(input_directory_path, filename)
        for filename in os.listdir(input_directory_path)
        if filename.lower().endswith('.pdf')
    ]
    
    documents = []
    # Initialize text splitter once
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    for file_path in file_paths:
        try:
            # Load and process PDF file
            loader = PyPDFLoader(file_path=file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Split documents into chunks
    final_docs = text_splitter.split_documents(documents)
    
    return final_docs




NER_RESULTS = {
  "Equipment_name": ["Fitness Tracker"],
  "Domain": ["electronics"],
  "Manufacturer": ["FitTech"]
}

#"Model_numbers": ["FT-B100", "FT-A200", "FT-P300"],
def update_text_metadata(input_documents: List[Document])-> List[Document]:
    """
        Update metadata in a list of documents.
    """
    for doc in input_documents:
        source_path = doc.metadata['source']
        filename = os.path.basename(source_path)
        filename_without_extension = os.path.splitext(filename)[0]

        new_metadata = {
            'Equipment_name': NER_RESULTS["Equipment_name"][0],
            'Domain': NER_RESULTS["Domain"][0],  # Corrected key name
            'Manufacturer': NER_RESULTS["Manufacturer"][0]
        }
        # Replace the old metadata dictionary with the new one
        doc.metadata = new_metadata
    return input_documents 
#  'Model_numbers': NER_RESULTS["Model_numbers"],
#####################################################
if __name__=="__main__":
    NER_RESULTS = {
    "Equipment_name": ["Fitness Tracker"],
    "Domain": ["electronics"],
    "Model_numbers": ["FT-B100", "FT-A200", "FT-P300"],
    "Manufacturer": ["FitTech"]
    }
    chunks_without_metadata = pdf_files_to_chunks(input_directory_path=data_directory_path)
    chunks_with_metadata = update_text_metadata(input_documents=chunks_without_metadata, ner_result=NER_RESULTS)
    print(chunks_with_metadata)
    print("********************")










