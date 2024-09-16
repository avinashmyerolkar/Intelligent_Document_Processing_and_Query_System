
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from entity_extraction import openai_chat_completion_response
from utils import extract_text_from_pdf
from pdfminer.high_level import extract_text


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
def pdf_loader(pdf_file_path):
    """
    Take pdf file path and return list of chunks in langchian doc 
    loader format
    """
    loader = PyPDFLoader(file_path=pdf_file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    return chunks


def entity_extractor(pdf_path):
    extracted_text = extract_text(pdf_path)
    ner_result = openai_chat_completion_response(final_prompt=extracted_text)
    return ner_result


def restructuring_ner_dict(ner_dict):
    for key, value in ner_dict.items():
        if isinstance(value, list):
            ner_dict[key] = ', '.join(value)
    return ner_dict


def update_chunks_metadata(chunks,ner_dict):
    for chunk in chunks:
        new_metadata = {
            'Equipment_name': ner_dict["Equipment_name"],
            'Domain': ner_dict["Domain"],  # Corrected key name
            'Model_numbers' : ner_dict['Model_numbers'],
            'Manufacturer': ner_dict["Manufacturer"]
        }
        # Replace the old metadata dictionary with the new one
        chunk.metadata = new_metadata

    return chunks
    

if __name__=="__main__":
    file_path = ["/home/avinash_m_yerolkar/frontend/data/CleanBot_Robotic_Vacuum_Cleaner_FAQ.pdf",
    "/home/avinash_m_yerolkar/frontend/data/FitTech_Fitness_Tracker_FAQ.pdf"]

    documents = []
    for file in file_path:
        print(file)
        load_result = pdf_loader(pdf_file_path=file)
        print("chunk are", load_result)
        print("type of chunks", type(load_result))
        print("len of chunks", len(load_result))
        ner_result =  entity_extractor(pdf_path=file)
        print("ner prefinal", ner_result)
        updated_ner_result = restructuring_ner_dict(ner_dict=ner_result)
        print("ner final", updated_ner_result)
        updated_chunks = update_chunks_metadata(chunks=load_result,ner_dict=updated_ner_result)
        print("updated_chunks are", updated_chunks)
        print("type of updated_chunks", type(updated_chunks))
        print("len of updated_chunks", len(updated_chunks))  
        documents.extend(updated_chunks)
        print("<<<<<<<<<Next>>>>>>>>>>>>")








