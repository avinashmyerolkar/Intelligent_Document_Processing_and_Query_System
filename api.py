from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import time
import shutil
from chunks_to_db import pdb_chunks_to_vectorstore
from chat_knowledge_base import chatvectorstore
from chat_knowledge_base import customize_template
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferMemory
from dotenv import dotenv_values

#---------------------------------------------------------------------
from preprocessing_chunk import pdf_loader
from preprocessing_chunk import entity_extractor
from preprocessing_chunk import restructuring_ner_dict
from preprocessing_chunk import update_chunks_metadata
#---------------------------------------------------------------------

config = dotenv_values('config.env')
gpt_api_key = config['OPENAI_API_KEY']
PORT=config['PORT']

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="ls__d11166dd5aa54910985d8de1c982c906"
os.environ["LANGCHAIN_PROJECT"]="Gen-AI-Chatbot"


app = Flask(__name__)



CORS(app) 
@app.route("/indexname", methods=['POST'])
def embed_specific_pdb():
    DOWNLOAD_DIR = 'download_dir'
    
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # Parse the form data to get the index_name
    index_name = request.form.get('index_name')

    if not index_name:
        return jsonify({"error": "index_name is required"}), 400

    # Check if the post request has the files part
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved_files = []
    try:
        for file in files:
            if file.filename == '':
                continue
            
            # Save each file to the specified directory
            file_path = os.path.join(DOWNLOAD_DIR, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

        file_paths = [
            os.path.join(DOWNLOAD_DIR, filename)
            for filename in os.listdir(DOWNLOAD_DIR)
            if filename.lower().endswith('.pdf')
        ]

        documents = []
        for file in file_paths:
            load_result = pdf_loader(pdf_file_path=file)
            ner_result = entity_extractor(pdf_path=file)
            updated_ner_result = restructuring_ner_dict(ner_dict=ner_result)
            updated_chunks = update_chunks_metadata(chunks=load_result, ner_dict=updated_ner_result)
            documents.extend(updated_chunks)

        pdb_chunks_to_vectorstore(vector_store_name=index_name, text_chunks=documents)

    finally:
        # Remove the directory and its contents after processing
        shutil.rmtree(DOWNLOAD_DIR)

    return jsonify({"message": "Files successfully uploaded", "files": saved_files}), 200





CORS(app) 
@app.route("/chatvectorstore",methods=['POST'])
def chat_specified_vectorstore():
    try:
        print("**************************************")
        response = request.get_json()
        print("dictionary",response)
        index_name = response["index_name"]
        question = response["question"]
        user_prompt = response['base_prompt']
        custom_template = customize_template(user_prompt)
        prompt = PromptTemplate(input_variables=["history", "context", "question"],template=custom_template)
        memory_stored = ConversationBufferMemory(memory_key="history", input_key="question")
        response = chatvectorstore(query = question ,
                                   memory_storage=memory_stored ,
                                   user_index_name= index_name,
                                   custom_prompt = prompt)
        print("ChatVectorStore Response:", response)  # Debugging line
        
        if response is None or 'result' not in response:
            return jsonify({"error": "Failed to retrieve the result from the chatvectorstore."}), 500
        
        result = response['result']
        return jsonify({"LLM_Response_With_Org_Data": response})
         
    
    except Exception as e:
        print("Exception ",e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return jsonify("Failed1")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, debug=True)














