import chromadb
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import glob
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()

client = chromadb.PersistentClient(path=r"/mnt/c/Users/lokes/Desktop/Smartsense/chroma_embedding/")
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,    
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
)
db = Chroma(client = client, embedding_function=embeddings)

def process_pdf_file(filepath):
    pdf_loader = PyPDFLoader(filepath)
    pdf_docs = pdf_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=15)
    chunked_documents = text_splitter.split_documents(pdf_docs)
    db.add_documents(chunked_documents)
    print(f"Added {len(chunked_documents)} documents from {filepath} to the database\n\n\n")


def add_files(path_to_folder):
    for filepath in glob.glob(path_to_folder, recursive=True):
        print(f"Processing {filepath}")
        if filepath.endswith('.pdf'):
            process_pdf_file(filepath)
add_files("/mnt/c/Users/lokes/Desktop/Smartsense/chroma_data/**")