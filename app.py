import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pdf_processor import PDFProcessor
from rag_components import RAGComponents
from config import (
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    CHARACTER_CONFIG,
    PAGE_TITLE,
    PAGE_ICON,
    HUGGINGFACE_API_KEY
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_components():
    # Initialize RAG components
    rag = RAGComponents()
    
    # Initialize text generation pipeline with smaller batch size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline(
        "text-generation",
        model=LLM_MODEL,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        model_kwargs={"low_cpu_mem_usage": True}
    )
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Process PDF and create vector store
    documents = rag.process_text(pdf_processor.text)
    rag.create_vector_store(documents)
    
    return rag, generator, pdf_processor

# Load components
try:
    rag, generator, pdf_processor = load_components()
except Exception as e:
    st.error(f"Error loading components: {str(e)}")
    st.stop()

# Title and description
st.title(PAGE_TITLE)
st.markdown(f"""
    Chat with {CHARACTER_CONFIG['name']} about the Harry Potter series!
    The chatbot uses the first Harry Potter book as its knowledge base.
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Hermione a question about Harry Potter"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get relevant context from RAG
    context_docs = rag.similarity_search(prompt)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    # Prepare the prompt for the LLM
    system_prompt = rag.get_system_prompt()
    full_prompt = f"""System: {system_prompt}

Context from Harry Potter book:
{context}

User: {prompt}
Assistant: Let me help you with that. """
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Generate with smaller max length to avoid memory issues
                outputs = generator(
                    full_prompt,
                    max_new_tokens=min(MAX_TOKENS, 100),
                    temperature=TEMPERATURE,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                # Extract the assistant's response
                response = outputs[0]['generated_text']
                response = response.split("Assistant: Let me help you with that.")[-1].strip()
                
                # Clean up the response
                if not response:
                    response = "I apologize, but I need to gather my thoughts on that. Could you please rephrase your question?"
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.error("I apologize, but I'm having trouble processing that request. Could you try asking something else?") 