import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}

# Vector Embeddings
embeddings = OpenAIEmbeddings()

# Loading Documents
def loader(folder):
    docs = []
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = doc_type
        docs.append(doc)
    return docs

def creat_vectorstore(documents, embeddings):
    # Check if a Chroma Datastore already exists - if so, deleting the collection
    if os.path.exists(os.environ['db_name']):
        Chroma(persist_directory=os.environ['db_name'], embedding_function=embeddings).delete_collection()
    # Create a new Chroma Datastore
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=os.environ['db_name'])
    return vectorstore

def chunking(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Loading Documents
documents = loader(folders)
# Chunking Documents
doc_chunks = chunking(documents)
# Document Types
doc_types = set(chunk.metadata['doc_type'] for chunk in doc_chunks)
print(f"Document types found: {', '.join(doc_types)}")
# Create the Chroma vectorstore
vector_DB = creat_vectorstore(doc_chunks, embeddings)
# Finding the dimensions of the embeddings
collection = vector_DB._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")