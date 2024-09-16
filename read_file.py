from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import dotenv_values
config = dotenv_values('config.env')
gpt_api_key = config['OPENAI_API_KEY']


embeddings = OpenAIEmbeddings(openai_api_key=gpt_api_key)
text_splitter = SemanticChunker(embeddings=embeddings)

#file_paths = [os.path.join(input_directory_path, filename) for filename in os.listdir(input_directory_path) if filename.endswith('.pdf')]

single_pdf_file_path="/home/avinash_m_yerolkar/workspace/test/downloaded_files/TechMobile_Smartphone_FAQ.pdf"
loader = PyPDFLoader(single_pdf_file_path)
pages = loader.load_and_split()

docs = text_splitter.create_documents([pages])
print(docs[0].page_content)

docs = text_splitter.create_documents()

print(docs)



from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def text_files_to_chunks(input_directory_path:str) -> List[Document]:
  """
  convert files to chunks 
  """
  file_paths = [os.path.join(input_directory_path, filename) for filename in os.listdir(input_directory_path) if filename.endswith('.pdf')]
  documents = []
  for i in file_paths:
    loader = TextLoader(file_path=i)
    documents.extend(loader.load())
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10,chunk_overlap=0,separators=['####'])

  final_docs = text_splitter.split_documents(documents)
  return final_docs


def update_text_metadata(input_documents: List[Document])-> List[Document]:
    """
        Update metadata in a list of documents.
    """
    for doc in input_documents:
        source_path = doc.metadata['source']
        filename = os.path.basename(source_path)
        fund_name = filename.split("%")[0]
        year = filename.split("%")[1].split("-")[3].split('.')[0]
        month = filename.split("%")[1].split("-")[2]
        new_metadata = {
            'year': year,
            'fund_name': fund_name,  # Corrected key name
            'month_name': month
        }
        # Replace the old metadata dictionary with the new one
        doc.metadata = new_metadata
    return input_documents
