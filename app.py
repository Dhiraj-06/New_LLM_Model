# LLM Document Processing System
# This script implements a retrieval-augmented generation (RAG) system to answer questions 
# based on the content of a PDF document. It uses an embedding model to understand the semantic
# meaning of text, a vector database (FAISS) for efficient searching, and a large language
# model (Llama) to generate human-like answers.

import os
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- Configuration ---
# Suppress a specific warning from the Hugging Face library regarding symlinks.
# This is often needed in environments like Windows where symlink support can be inconsistent.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Main Application Logic ---

def load_pdf(path):
    """
    Loads and extracts all text from a specified PDF file.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: A single string containing all the extracted text from the PDF.
             Returns None if the file is not found or an error occurs.
    """
    print(f"Loading PDF from: {path}")
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            print(f"Successfully extracted text from {len(reader.pages)} pages.")
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the PDF: {e}")
        return None
    return text

def split_text(text, chunk_size=300, overlap=50):
    """
    Splits a long text into smaller, overlapping chunks. This is necessary because
    embedding models have a limit on the amount of text they can process at once.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The desired character length of each chunk.
        overlap (int): The number of characters to overlap between consecutive chunks
                       to maintain context.

    Returns:
        list[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_embeddings(chunks, model):
    """
    Converts a list of text chunks into numerical vectors (embeddings) using a
    sentence-transformer model.

    Args:
        chunks (list[str]): The list of text chunks.
        model: The loaded sentence-transformer model.

    Returns:
        np.ndarray: A numpy array of embeddings for each chunk.
    """
    print("Generating embeddings for text chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    """
    Builds a FAISS index for efficient similarity searching of embeddings.

    Args:
        embeddings (np.ndarray): The array of text embeddings.

    Returns:
        faiss.Index: A FAISS index containing the embeddings.
    """
    print("Building FAISS index for fast searching...")
    dimension = embeddings.shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built successfully with {index.ntotal} vectors.")
    return index

def search_index(query, embed_model, index, chunks, k=3):
    """
    Searches the FAISS index for the most relevant text chunks for a given query.

    Args:
        query (str): The user's question.
        embed_model: The sentence-transformer model to embed the query.
        index (faiss.Index): The FAISS index to search.
        chunks (list[str]): The original list of text chunks.
        k (int): The number of top relevant chunks to retrieve.

    Returns:
        list[str]: A list of the most relevant text chunks.
    """
    print("Searching for relevant chunks...")
    query_embedding = embed_model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype("float32")
    
    # D contains distances, I contains indices of the nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    # Return the chunks corresponding to the found indices
    return [chunks[i] for i in indices[0]]

def generate_answer(context, question, llm):
    """
    Generates an answer to a question based on the provided context using a
    local Large Language Model (LLM).

    Args:
        context (str): The relevant text retrieved from the document.
        question (str): The user's original question.
        llm: The loaded Llama language model.

    Returns:
        str: The generated answer.
    """
    print("Generating answer with LLM...")
    # This prompt template is crucial for guiding the LLM to use the provided context.
    prompt = f"""
Use the following context to answer the question at the end. If the answer is not contained
within the text below, say "I don't have enough information to answer this question."

### Context:
{context}

### Question:
{question}

### Answer:
"""
    output = llm(
        prompt,
        max_tokens=512,  # Max tokens to generate
        stop=["###", "Question:"] # Stop generation at these tokens
    )
    return output['choices'][0]['text'].strip()

def main():
    """
    The main function that orchestrates the entire process.
    """
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # Update this to the absolute path of your policy document PDF
    pdf_path = "data/Arogya_Sanjeevani_Policy.pdf" 
    # Update this to the absolute path of your downloaded Llama model file
    model_path = "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    # 1. Load and process the document
    text = load_pdf(pdf_path)
    if text is None:
        print("Exiting due to PDF loading failure.")
        return
    chunks = split_text(text)

    # 2. Load embedding model and create vector embeddings
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = get_embeddings(chunks, embed_model)

    # 3. Build the search index
    index = build_faiss_index(embeddings)

    # 4. Load the local LLM
    print("Loading language model (this may take a moment)...")
    try:
        # Set verbose=False to suppress detailed model loading logs from llama_cpp
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=6, verbose=False)
    except Exception as e:
        print(f"Error loading the language model: {e}")
        print("Please ensure the model_path is correct and the file is not corrupted.")
        return
    
    print("\n✅ System ready! You can now ask questions about the document.")
    
    # 5. Start interactive query loop
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower().strip() == 'exit':
            print("Exiting...")
            break
        
        # Retrieve relevant context
        relevant_chunks = search_index(query, embed_model, index, chunks)
        context = "\n---\n".join(relevant_chunks)
        
        # Generate the final answer
        answer = generate_answer(context, query, llm)
        print("\n✅ Answer:", answer)

if __name__ == "__main__":
    main()
