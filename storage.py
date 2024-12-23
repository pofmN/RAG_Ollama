import os
from typing import List, Dict, Any, Optional
import chromadb
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from get_context_online import get_online_context

class DocumentStore:
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize the document store with ChromaDB.
        
        Args:
            db_path (str): Path to store the ChromaDB files
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
    def get_collection(self, collection_name: str = "document_embeddings") -> Collection:
        """
        Get or create a ChromaDB collection.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Collection: ChromaDB collection object
        """
        try:
            return self.client.get_or_create_collection(
                name=collection_name
            )
        except Exception as e:
            st.error(f"Error accessing ChromaDB collection: {str(e)}")
            raise

    def store_embeddings(
        self,
        chunks: List[str],
        collection_name,
        batch_size: int = 100
    ) -> None:
        """
        Store document chunks and their embeddings in ChromaDB.
        
        Args:
            chunks (List[str]): List of text chunks to embed and store
            collection_name (str): Name of the collection
            batch_size (int): Number of chunks to process at once
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")

            collection = self.get_collection(collection_name)
            total_chunks = len(chunks)
            
            # Process chunks in batches to avoid memory issues
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Generate embeddings for the batch
                with st.spinner(f"Generating embeddings for chunks {i+1} to {min(i+batch_size, total_chunks)}..."):
                    embeddings = [
                        st.session_state.embedding_model.encode(chunk).tolist()
                        for chunk in batch_chunks
                    ]
                
                # Store the batch in ChromaDB
                collection.add(
                    documents=batch_chunks,
                    embeddings=embeddings,
                    ids=[f"chunk_{j}" for j in range(i, i + len(batch_chunks))],
                    metadatas=[{"index": j} for j in range(i, i + len(batch_chunks))]  # Changed from metadata to metadatas
                )
                
            st.success(f"Successfully stored {total_chunks} chunks in the database")
            
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")
            raise

    def retrieve_relevant_chunks(
        self,
        query: str,
        collection_name,
        top_k: int = 2,
        # threshold: float = 0.3  # We'll use this as a similarity threshold
    ) -> List[str]:
        """
        Retrieve relevant document chunks for a given query.
        
        Args:
            query (str): User query
            collection_name (str): Name of the collection
            top_k (int): Number of results to retrieve
            threshold (float): Minimum similarity threshold (0 to 1, higher is better)
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = st.session_state.embedding_model.encode(query).tolist()
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Extract and return the relevant chunks
            print(results['documents'][0])
            return results['documents'][0]
            
        except Exception as e:
            st.error(f"Error retrieving relevant chunks: {str(e)}")
            raise

        
    def clear_collection(self, collection_name: str = "document_embeddings") -> None:
        """
        Clear all documents from a collection.
        
        Args:
            collection_name (str): Name of the collection to clear
        """
        try:
            collection = self.get_collection(collection_name)
            collection.delete()
            st.success(f"Collection '{collection_name}' cleared successfully")
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            raise

# Initialize the document store
@st.cache_resource
def initialize_document_store() -> DocumentStore:
    """
    Initialize and cache the document store instance.
    
    Returns:
        DocumentStore: Initialized document store
    """
    return DocumentStore()

# Helper functions to use in the main UI
def store_document_chunks(chunks: List[str]) -> None:
    """
    Store document chunks using the document store.
    
    Args:
        chunks (List[str]): List of text chunks to store
    """
    doc_store = initialize_document_store()
    if st.session_state.language == "English":
        doc_store.store_embeddings(chunks, collection_name="document_embeddings_en")
    elif st.session_state.language == "Vietnamese":
        doc_store.store_embeddings(chunks, collection_name="document_embeddings_vi")

def get_relevant_chunks(query: str, top_k: int = 1) -> List[str]:  # Increased top_k to 3
    """
    Get relevant chunks for a query.
    
    Args:
        query (str): User query
        top_k (int): Number of results to retrieve
        
    Returns:
        List[str]: List of relevant text chunks
    """
    doc_store = initialize_document_store()
    if st.session_state.language == "English":
        return doc_store.retrieve_relevant_chunks(
            query, 
            collection_name="document_embeddings_en", 
            top_k=top_k,
            #threshold=0.0  # Adjust threshold as needed
        )
    elif st.session_state.language == "Vietnamese":
        return doc_store.retrieve_relevant_chunks(
            query, 
            collection_name="document_embeddings_vi", 
            top_k=top_k,
            #threshold=0.0  # Adjust threshold as needed
        )