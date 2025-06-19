from typing import List, Optional
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_TOKENS,
    TEMPERATURE,
    LOCAL_PDF_PATH,
    SYSTEM_PROMPT
)

class RAGComponents:
    def __init__(self):
        # Initialize embeddings with GPU support if available
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize text splitter with smaller chunk size to handle token limits
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Reduced chunk size to handle token limits
            chunk_overlap=50,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Load and process PDF
        self.loader = PyPDFLoader(LOCAL_PDF_PATH)
        self.pages = self.loader.load()
        
        # Split documents into smaller chunks
        self.texts = self.text_splitter.split_documents(self.pages)
        
        # Create FAISS index
        self.db = FAISS.from_documents(self.texts, self.embeddings)
        
    def get_relevant_documents(self, query, k=3):
        """Retrieve relevant documents for a query"""
        try:
            docs = self.db.similarity_search(query, k=k)
            # Truncate documents to ensure they fit within token limits
            truncated_docs = []
            for doc in docs:
                content = doc.page_content
                if len(content) > 500:  # Limit document length
                    content = content[:500] + "..."
                truncated_docs.append(Document(page_content=content))
            return truncated_docs
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def process_text(self, text: str) -> List[Document]:
        """Process text into chunks and create documents."""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a FAISS vector store from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents in the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        return self.vector_store.similarity_search(query, k=k)

    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return SYSTEM_PROMPT 