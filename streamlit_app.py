import os
import numpy as np
import PyPDF2
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import tempfile
import pickle

# Configuration
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Streamlit page config
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_embedding_model():
    """Load the embedding model (cached for performance)"""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm_model(model_path):
    """Load the LLM model (cached for performance)"""
    try:
        return Llama(model_path=model_path, n_ctx=2048, n_threads=6, verbose=False)
    except Exception as e:
        st.error(f"Error loading LLM model: {e}")
        return None

@st.cache_data
def load_pdf(pdf_file):
    """Load and extract text from PDF"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def split_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embeddings(chunks, model):
    """Generate embeddings for text chunks"""
    return np.array(model.encode(chunks, show_progress_bar=False)).astype("float32")

def build_faiss_index(embeddings):
    """Build FAISS index for similarity search"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query, embed_model, index, chunks, k=3):
    """Search for relevant chunks"""
    query_embedding = embed_model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(context, question, llm):
    """Generate answer using LLM"""
    prompt = f"""
Use the following context to answer the question at the end. If the answer is not contained
within the text below, say "I don't have enough information to answer this question."

### Context:
{context}

### Question:
{question}

### Answer:
"""
    output = llm(prompt, max_tokens=512, stop=["###", "Question:"])
    return output['choices'][0]['text'].strip()

def main():
    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "LLM Model Path",
        value="model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        help="Path to your trained Mistral model file"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.sidebar.error("⚠️ Model file not found! Please check the path.")
        st.stop()
    
    # Load models
    with st.spinner("Loading embedding model..."):
        embed_model = load_embedding_model()
    
    with st.spinner("Loading LLM model..."):
        llm = load_llm_model(model_path)
    
    if llm is None:
        st.error("Failed to load LLM model. Please check the model path and file.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload your PDF document to analyze"
    )
    
    if uploaded_file is not None:
        # Process document
        with st.spinner("Processing document..."):
            # Load PDF
            text = load_pdf(uploaded_file)
            if text is None:
                st.stop()
            
            # Split text
            chunks = split_text(text)
            
            # Generate embeddings
            embeddings = get_embeddings(chunks, embed_model)
            
            # Build index
            index = build_faiss_index(embeddings)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embed_model = embed_model
            st.session_state.llm = llm
        
        st.success(f"✅ Document processed! Found {len(chunks)} text chunks.")
        
        # Display document info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Characters", len(text))
        with col2:
            st.metric("Text Chunks", len(chunks))
        
        # Q&A Interface
        st.header("Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        # Answer button
        if st.button("Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Searching and generating answer..."):
                    # Search for relevant context
                    relevant_chunks = search_index(
                        question, 
                        st.session_state.embed_model, 
                        st.session_state.index, 
                        st.session_state.chunks
                    )
                    context = "\n---\n".join(relevant_chunks)
                    
                    # Generate answer
                    answer = generate_answer(context, question, st.session_state.llm)
                
                # Display results
                st.subheader("Answer:")
                st.write(answer)
                
                # Show relevant context in expander
                with st.expander("View Relevant Context"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.text_area(f"Relevant Chunk {i+1}", chunk, height=150)
            else:
                st.warning("Please enter a question.")
        
        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            st.header("Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {q[:50]}..."):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")

if __name__ == "__main__":
    main()
