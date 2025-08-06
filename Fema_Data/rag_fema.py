# rag_pipeline.py


import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import fitz
import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re

load_dotenv()  # Load OPENAI_API_KEY from .env
# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

def load_pdfs_from_folder(folder_path):
    """Extracts text from all PDFs in a folder, returns a list of documents."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append(text)
    return documents

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF (helper for process_folder)."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# --- Chunking ---

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Chunks a string into overlapping segments of chunk_size (chars)."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= text_length:
            break
        start += chunk_size - chunk_overlap
    return chunks

def preprocess_text(text):
    """Simple text pre-processing: lowercasing and whitespace cleanup."""
    text = text.lower()
    text = ' '.join(text.split())
    return text

# --- Embedding ---

def create_embeddings(texts, model="text-embedding-3-small"):
    """
    Creates embeddings for a string or list of strings using OpenAI.
    Returns one or a list of embedding vectors (list of floats).
    """
    if isinstance(texts, str):
        input_text = [texts]
    else:
        input_text = texts

    response = client.embeddings.create(
        model=model,
        input=input_text
    )

    if isinstance(texts, str):
        return response.data[0].embedding
    return [item.embedding for item in response.data]

# --- Vector Store ---

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vec = np.array(query_embedding)
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in similarities[:k]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results

# --- Folder Processing ---

def process_folder(folder_path, chunk_size=1000, chunk_overlap=200):
    """
    Ingest every PDF in the folder and add chunks & embeddings to the vector store.
    """
    store = SimpleVectorStore()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            preprocessed = [preprocess_text(chunk) for chunk in chunks]
            if not preprocessed:
                continue
            embeddings = create_embeddings(preprocessed)
            for i, (chunk, embedding) in enumerate(zip(preprocessed, embeddings)):
                store.add_item(
                    text=chunk,
                    embedding=embedding,
                    metadata={"source": filename, "chunk_index": i}
                )
    print(f"Total chunks in vector store: {len(store.texts)}")
    return store

# --- Query Answering ---

def answer_query(store, question, k=3):
    """
    Search for the top-k most relevant chunks and return them for LLM synthesis.
    """
    query_embedding = create_embeddings(question)
    hits = store.similarity_search(query_embedding, k=k)
    return hits

def synthesize_answer_with_llm(chunks, question, model="gpt-4o-mini"):
    """
    Given the relevant text chunks and a question, use LLM to generate a concise answer.
    """
    import openai
    context = "\n\n".join([item['text'] for item in chunks])
    llm_prompt = (
        f"Answer the following question using ONLY the provided context. Cite specific steps or guidance mentioned.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = openai.OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": llm_prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message.content

# --- High-level Helpers for API/server ---

def load_fema_store(folder_path, chunk_size=1000, chunk_overlap=200):
    """Build and return a search-ready vector store for the given folder of PDFs."""
    return process_folder(folder_path, chunk_size, chunk_overlap)

def answer_fema_question(store, question, k=3):
    """
    Full pipeline: search vector store, call LLM, return answer + sources.
    """
    hits = answer_query(store, question, k)
    answer = synthesize_answer_with_llm(hits, question)
    sources = [{"source": h["metadata"]["source"], "similarity": h["similarity"]} for h in hits]
    return {"answer": answer, "sources": sources, "raw_chunks": hits}


