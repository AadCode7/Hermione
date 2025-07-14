# Hermione Chatbot

A character-based chatbot emulating Hermione Granger from the Harry Potter series using a Retrieval-Augmented Generation (RAG) approach.

## Features

- Character-based chatbot using LLAMA2 for text generation
- PDF-based knowledge retrieval using Hugging Face embeddings
- Streamlit-based web interface
- Character conditioning for consistent responses

## Prerequisites

- Python 3.8 or higher
- Hugging Face API key
- A PDF of the first Harry Potter book 

## Installation

1. Clone this repository
2. Create a `.env` file in the root directory with your Hugging Face API key
3. Install the required dependencies


## How It Works

The Hermione Chatbot uses a Retrieval-Augmented Generation (RAG) approach to provide contextually relevant and in-character responses. Here's how it works:

1. **PDF Processing**: The application reads a local PDF file (named `harry1.pdf`) using the `pdf_processor.py` module. This module extracts text from the PDF, which is then used as the knowledge base for the chatbot.

2. **Text Chunking**: The extracted text is split into smaller chunks to facilitate efficient retrieval. This is done using Langchain's text splitter, which ensures that the chunks are of a manageable size for processing.

3. **Embedding Generation**: Each text chunk is converted into a vector embedding using the Hugging Face model `sentence-transformers/all-mpnet-base-v2`. These embeddings capture the semantic meaning of the text, allowing for efficient similarity searches.

4. **Vector Store Creation**: The embeddings are stored in a FAISS vector database. This allows the chatbot to quickly retrieve the most relevant text chunks based on the user's query.

5. **Question Understanding**: When a user asks a question, the chatbot uses the same embedding model to convert the question into a vector. This vector is then used to search the FAISS database for the most similar text chunks.

6. **Context Retrieval**: The retrieved text chunks are used as context for the LLM (Language Model). This context helps the LLM understand the question in the context of the Harry Potter universe and generate a response that is both accurate and in-character.

7. **Answer Generation**: The LLM, conditioned with a system prompt to impersonate Hermione Granger, generates a response based on the retrieved context and the user's question. The response is designed to be informative, witty, and true to Hermione's character.

This RAG approach ensures that the chatbot's responses are not only contextually relevant but also grounded in the knowledge extracted from the PDF, providing a more engaging and accurate interaction.
