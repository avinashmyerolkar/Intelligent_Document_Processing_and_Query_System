# 3. fucntion to chat with specific vectorstore based on user input
import os
import chromadb
import warnings
warnings.filterwarnings("ignore")

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import dotenv_values
config = dotenv_values('config.env')
gpt_api_key = config['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] =gpt_api_key
#embedding_model = OpenAIEmbeddings(api_key=gpt_api_key)
llmo = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=gpt_api_key)
path_persist_directory = "/home/avinash_m_yerolkar/Intelligent_Document_Processing_and_Query_System/Chroma_DB"

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


metadata_field_info = [
    AttributeInfo(
        name="Equipment_name",
        description="The equipment names like Smart TV, Robotic Vaccume, Washing Machine, Laptop, Refrigerator, Thermostat, Fitness Tracker, Digital Camera, Wireless Earbuds, Smartphone",
        type="string",
    ),
    AttributeInfo(
        name="Domain",
        description="The name of the Domain like electronics, mechanical, software",
        type="string",
    ),
    AttributeInfo(
        name="Manufacturer",
        description="The name of the Manufacturer like ViewMax, CleanBot, CleanPro, CompuTech, CoolTech, EcoControl, FitTech, PhotoPro, SoundWave, TechMobile ",
        type="string",
    )
    ]

document_content_description = """
The document document given is  related to various consumer products, 
each structured with sections such as 'Product Overview,' 'Technical Specifications,' 'Key Features,' 'Setup and Installation,' 'Usage Instructions,' 'Maintenance and Care,' 'Troubleshooting,' 'Warranty Information,' and 'Customer Support.

"""



def chatvectorstore(query, user_index_name , memory_storage ,custom_prompt, path_to_db = path_persist_directory ):
    """ Based on userinput this function will connect to specific knowledge base,
        with whih user can question
        """
    chroma_client = chromadb.PersistentClient(path_to_db)
    collection_objects = chroma_client.list_collections()
    collection_names = [collection.name for collection in collection_objects]
   
    for specific_store in collection_names:
        if user_index_name == specific_store:
            connect_vectorstore = Chroma(collection_name=specific_store, embedding_function=embedding_model, persist_directory=path_persist_directory)
        else:
            pass
    
    retriever_self_query_obj = SelfQueryRetriever.from_llm(
    llmo,
    connect_vectorstore,
    document_content_description,
    metadata_field_info,
    search_kwargs={'k':5}
    )
    
    # creating chain to club all things together
    qa_chain = RetrievalQA.from_chain_type(llm=llmo, 
                                  chain_type="stuff", 
                                  retriever=retriever_self_query_obj,
                                  chain_type_kwargs={"prompt": custom_prompt, "memory": memory_storage},
                                  return_source_documents=False,
                                  )
    
    
    # qa_chain_response = qa_chain.invoke(query)     
    response = qa_chain.invoke(query)                          
    return response                  



static_part ="""
Use the following pieces of retrieved context, delimited by % , to answer the question. 
If the answer is present in the provided context below as delimited by %, provide a relevant responses . 
If the answer is strictly not present in the provided context delimited by "%", ask for a more contextually rich question.
/n/n
Chat History:
{history}

Question:
{question}

Context:
%{context}%

Answer:

    """


def customize_template(user_prompt):
    """when user give prompt , this template will get 
    embedded in our prompt"""
    custom_template = f"{user_prompt}{static_part}"
    return custom_template


if __name__=="__main__":
    pass


##############################################################################
