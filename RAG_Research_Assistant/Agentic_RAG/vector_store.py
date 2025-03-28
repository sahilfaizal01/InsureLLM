import os
from typing import List, Dict
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class PaperVectorStore:
    def __init__(self, persist_directory='./paper_db'):
        """
        Initialize vector database for storing research papers
        Args:
            persist_directory (str): Path to store vector database
        """
        # Use Chroma's built-in embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collection with the embedding function
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            embedding_function=self.embedding_function
        )
    
    def add_papers(self, papers: List[Dict]):
        """
        Add papers to vector database
        Args:
            papers (List[Dict]): List of papers with metadata
        Returns:
            bool: True if papers were added successfully
        """
        if not papers:
            return False
            
        ids = []
        documents = []
        metadatas = []
        
        for paper in papers:
            # Generate a unique ID
            paper_id = str(hash(paper.get('title', 'Unknown Title')))
            
            # Skip if paper already exists
            if self._paper_exists(paper_id):
                continue
                
            ids.append(paper_id)
            
            # Prepare document text
            document_text = (
                f"Title: {paper.get('title', 'No Title')} "
                f"Authors: {', '.join(paper.get('authors', []))} "
                f"Abstract: {paper.get('abstract', 'No Abstract')}"
            )
            
            documents.append(document_text)
            
            # Prepare metadata
            metadata = {
                'title': paper.get('title', 'Unknown'),
                'year': str(paper.get('year', 'Unknown')),
                'authors': ', '.join(paper.get('authors', [])),
                'url': paper.get('url', ''),
                'venue': paper.get('venue', 'Unknown'),
                'citation_count': str(paper.get('citation_count', 0))
            }
            
            metadatas.append(metadata)
        
        # Add to collection if there are new papers
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return True
            
        return False
    
    def _paper_exists(self, paper_id):
        """
        Check if a paper already exists in the database
        Args:
            paper_id (str): Unique paper ID
        Returns:
            bool: True if paper exists
        """
        try:
            self.collection.get(ids=[paper_id])
            return True
        except:
            return False
    
    def search_papers(self, query: str, top_k: int = 5):
        """
        Search papers in vector database
        Args:
            query (str): Search query
            top_k (int): Number of results to return
        Returns:
            List of matching papers
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Process and return results
        matched_papers = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                matched_papers.append({
                    'id': results['ids'][0][i],
                    'title': results['metadatas'][0][i].get('title', 'No Title'),
                    'authors': results['metadatas'][0][i].get('authors', 'Unknown').split(', '),
                    'year': results['metadatas'][0][i].get('year', 'Unknown'),
                    'url': results['metadatas'][0][i].get('url', ''),
                    'venue': results['metadatas'][0][i].get('venue', 'Unknown'),
                    'citation_count': int(results['metadatas'][0][i].get('citation_count', '0')),
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i]
                })
                
        return matched_papers