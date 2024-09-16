"""
MAIN function which will take collection_name and file_path of which embedding to be done
#from api_keys import gpt_api_key
"""
import os
import json
import uuid
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.document import Document
from dotenv import dotenv_values
from langchain_community.embeddings import HuggingFaceEmbeddings

config = dotenv_values('config.env')
gpt_api_key = config['OPENAI_API_KEY']
PATH_PARSISTENT_DIR = config['PATH_PARSISTENT_DIR']



model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


path_persist_directory = "/home/avinash_m_yerolkar/Intelligent_Document_Processing_and_Query_System/Chroma_DB"
def pdb_chunks_to_vectorstore(vector_store_name, text_chunks):
    """ This function will take vector store name and chunked text from user 
        andstore them in the specified vector_store mentined by user itself
    """
    # Provide the path to your chroma_db
    chroma_client = chromadb.PersistentClient('/home/avinash_m_yerolkar/Intelligent_Document_Processing_and_Query_System/Chroma_DB')
    existing_collections = [collection.name for collection in chroma_client.list_collections()]
    
    if vector_store_name not in existing_collections:
        chroma_collection = chroma_client.create_collection(name=vector_store_name, metadata={"name": vector_store_name})
    else:
        chroma_collection = chroma_client.get_collection(name=vector_store_name)
    
    
    index = Chroma.from_documents(documents=text_chunks, embedding=embedding_model, collection_name=vector_store_name,
                                  persist_directory=path_persist_directory,collection_metadata={"hnsw:space": "cosine"})
    index.persist() 
    index = None     
    
    return f"Embedding Stored in {vector_store_name}"





if __name__== "__main__":
    pass