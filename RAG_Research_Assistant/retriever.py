import streamlit as st
import faiss
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from data_ingest import transform_papers_to_documents

def prepare_document_retrieval(papers):
    """Prepare document retrieval using FAISS and Cohere embeddings."""
    if not papers:
        st.warning("No papers found. Try different keywords.")
        return None
    
    # Transform papers to LangChain documents
    documents = transform_papers_to_documents(papers)
    
    # Initialize embedding model
    embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(
        len(embedding_model.embed_query("Research paper embedding"))
    )
    
    # Create vector store
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    return vector_store.as_retriever()

def format_documents_with_citations(docs):
    """Format documents to include citation information."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        citation_id = doc.metadata.get("citation_id", f"[{i+1}]")
        title = doc.metadata.get("title", "Unknown Title")
        authors = doc.metadata.get("authors", "Unknown Authors")
        year = doc.metadata.get("year", "Unknown Year")
        
        formatted_docs.append(
            f"{doc.page_content}\n"
            f"Citation: {citation_id} {title} by {authors} ({year})"
        )
    
    return "\n\n".join(formatted_docs)
