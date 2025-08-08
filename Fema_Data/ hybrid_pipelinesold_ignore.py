import os
import fitz
import numpy as np
import json
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import re
import networkx as nx
import heapq
import hashlib
from collections import Counter
from bert_score import score as bertscore_score
import torch
import matplotlib.pyplot as plt
import pandas as pd

# === SETUP ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assert os.getenv("OPENAI_API_KEY") is not None, "❌ OPENAI_API_KEY not loaded from .env"


# === CACHING SYSTEM ===
def get_cache_key(text, operation):
    """Generate cache key for operations"""
    return hashlib.md5(f"{operation}:{text[:500]}".encode()).hexdigest()

def load_cache(cache_file):
    """Load cache from disk"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache, cache_file):
    """Save cache to disk"""
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

# === DOC INGEST AND UTILITIES ===

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "".join([page.get_text() for page in doc])

def extract_texts_from_folder(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder_path, fname)
            text = extract_text_from_pdf(path)
            texts.append({"text": text, "filename": fname})
    return texts

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# --- Optimized Embedding ---
def create_embeddings_cached(texts, model="text-embedding-3-small", cache_file="embeddings_cache.pkl"):
    """Create embeddings with caching"""
    cache = load_cache(cache_file)
    
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []
    
    for i, text in enumerate(texts):
        cache_key = get_cache_key(text, "embedding")
        if cache_key in cache:
            embeddings.append(cache[cache_key])
        else:
            embeddings.append(None)  # Placeholder
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # Embed only uncached texts
    if texts_to_embed:
        print(f"Embedding {len(texts_to_embed)} new texts...")
        batch_size = 100
        new_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                print(f"⏳ Calling OpenAI Embedding API for {len(batch)} texts...")
                response = client.embeddings.create(model=model, input=batch)
                print("✅ Embedding API responded.")
            except Exception as e:
                print(f"❌ Embedding API failed: {e}")
                raise
            new_embeddings.extend([item.embedding for item in response.data])
        
        for i, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
            cache_key = get_cache_key(text, "embedding")
            cache[cache_key] = embedding
            embeddings[indices_to_embed[i]] = embedding
        
        save_cache(cache, cache_file)
    
    # Handle single query case - return first embedding directly
    if len(texts) == 1:
        return embeddings[0]
    
    return embeddings

# Original embedding function (for backward compatibility)
def create_embeddings(texts, model="text-embedding-3-small"):
    if not texts: return []
    if isinstance(texts, str):
        texts = [texts]
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    
    # For single query, return just the embedding, not wrapped in a list
    if len(texts) == 1:
        return all_embeddings[0]
    
    return all_embeddings

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
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        if not self.vectors: return []
        
        # Ensure query_embedding is a numpy array and flatten it
        if isinstance(query_embedding, list):
            query_vec = np.array(query_embedding).flatten()
        else:
            query_vec = np.array(query_embedding).flatten()
            
        similarities = []
        for i, vec in enumerate(self.vectors):
            if filter_func and not filter_func(self.metadata[i]): continue
            
            # Ensure vec is also flattened
            vec_flat = np.array(vec).flatten()
            
            # Calculate similarity
            dot_product = np.dot(query_vec, vec_flat)
            query_norm = np.linalg.norm(query_vec)
            vec_norm = np.linalg.norm(vec_flat)
            
            # Avoid division by zero
            if query_norm == 0 or vec_norm == 0:
                sim = 0.0
            else:
                sim = dot_product / (query_norm * vec_norm)
            
            # Ensure sim is a scalar
            if isinstance(sim, np.ndarray):
                if sim.size == 1:
                    sim = float(sim.item())
                else:
                    print(f"Warning: similarity array has size {sim.size}, taking first element")
                    sim = float(sim.flat[0])
            else:
                sim = float(sim)
                
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

# --- LLM Response ---
def generate_response(query, context_chunks, model="gpt-4o-mini"):
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    sys_msg = "You are a helpful FEMA assistant. Answer the question based only on the provided context."
    print("⏳ Calling OpenAI Chat Completion API...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )
        print("✅ Chat completion successful.")
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        raise
    return response.choices[0].message.content

# ========== 1. OPTIMIZED STANDARD RAG ==========

def standard_rag_cached(query, folder_path, chunk_size=1000, chunk_overlap=200, k=5):
    """Standard RAG with caching"""
    cache_file = f"{os.path.basename(folder_path)}_standard_cache.pkl"
    
    if os.path.exists(cache_file):
        print("Loading cached standard RAG data...")
        with open(cache_file, 'rb') as f:
            store = pickle.load(f)
    else:
        print("Building standard RAG cache...")
        docs = extract_texts_from_folder(folder_path)
        all_chunks = []
        all_metadata = []
        
        for doc in docs:
            chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": doc["filename"], "chunk_index": idx})
        
        embeddings = create_embeddings_cached(all_chunks)
        store = SimpleVectorStore()
        
        for text, emb, meta in zip(all_chunks, embeddings, all_metadata):
            store.add_item(text, emb, meta)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(store, f)
    
    query_embedding = create_embeddings_cached(query)
    retrieved = store.similarity_search(query_embedding, k=k)
    answer = generate_response(query, retrieved)
    
    return {
        "query": query,
        "response": answer,
        "retrieved_chunks": retrieved
    }

# Original standard_rag (for backward compatibility)
def standard_rag(query, folder_path, chunk_size=1000, chunk_overlap=200, k=5):
    # Aggregate all docs
    docs = extract_texts_from_folder(folder_path)
    all_chunks = []
    all_metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": doc["filename"], "chunk_index": idx})
    embeddings = create_embeddings(all_chunks)
    store = SimpleVectorStore()
    for text, emb, meta in zip(all_chunks, embeddings, all_metadata):
        store.add_item(text, emb, meta)
    query_embedding = create_embeddings(query)
    retrieved = store.similarity_search(query_embedding, k=k)
    answer = generate_response(query, retrieved)
    return {
        "query": query,
        "response": answer,
        "retrieved_chunks": retrieved
    }

# ========== 2. HIERARCHICAL RAG (UNCHANGED) ==========

def generate_page_summary(page_text):
    sys_prompt = "Summarize the following page text for FEMA context."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Please summarize this text:\n\n{page_text[:6000]}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def hierarchical_rag(query, folder_path, chunk_size=1000, chunk_overlap=200,
                     k_summaries=3, k_chunks=5, regenerate=False):
    # Build summary and detailed index across all files
    print("=== ENTERED HIERARCHICAL RAG ===")
    summary_store_file = f"{os.path.basename(folder_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(folder_path)}_detailed_store.pkl"

    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        # Extract all pages from all PDFs
        print(f"Extracting pages from: {folder_path}")
        all_pages = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(folder_path, fname)
                pdf = fitz.open(path)
                for i, page in enumerate(pdf):
                    text = page.get_text()
                    if len(text.strip()) > 50:
                        all_pages.append({"text": text, "metadata": {"source": fname, "page": i+1}})
        # Summaries
        print(f"Summarising pages")
        summaries = []
        for page in all_pages:
            summary_text = generate_page_summary(page["text"])
            summary_metadata = page["metadata"].copy()
            summary_metadata.update({"is_summary": True})
            summaries.append({"text": summary_text, "metadata": summary_metadata})

        # Detailed chunks
        print(f"Chunking text")
        detailed_chunks = []
        for page in all_pages:
            page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)
            for i, chunk in enumerate(page_chunks):
                meta = page["metadata"].copy()
                meta.update({"chunk_index": i, "is_summary": False})
                detailed_chunks.append({"text": chunk, "metadata": meta})
        # Vectorize using the cached embedding function
        summary_store = SimpleVectorStore()
        for summary in summaries:
            emb = create_embeddings_cached(summary["text"])  # Use cached version
            summary_store.add_item(summary["text"], emb, summary["metadata"])
        detailed_store = SimpleVectorStore()
        for chunk in detailed_chunks:
            emb = create_embeddings_cached(chunk["text"])  # Use cached version
            detailed_store.add_item(chunk["text"], emb, chunk["metadata"])
        # Save
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    # Retrieval using cached embeddings
    query_embedding = create_embeddings_cached(query)  # Use cached version
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding).flatten()
        
    summary_results = summary_store.similarity_search(query_embedding, k=k_summaries)
    relevant_sources = [(r["metadata"]["source"], r["metadata"]["page"]) for r in summary_results]

    def page_filter(metadata):
        return (metadata["source"], metadata.get("page")) in relevant_sources

    detailed_results = detailed_store.similarity_search(query_embedding, k=k_chunks * len(relevant_sources), filter_func=page_filter)
    answer = generate_response(query, detailed_results)
    return {
        "query": query,
        "response": answer,
        "retrieved_chunks": detailed_results,
        "summary_chunks": summary_results
    }

# ========== 3. OPTIMIZED GRAPH RAG ==========

def extract_concepts_simple(text):
    """Extract concepts using simple NLP instead of LLM calls"""
    # Simple keyword extraction
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)  # Capitalized phrases
    words.extend(re.findall(r'\b(?:emergency|disaster|FEMA|evacuation|response|planning|hazard|risk|safety|alert|warning|flood|fire|earthquake|hurricane|tornado|preparedness|mitigation|recovery|assistance|relief|shelter|communications|infrastructure|resources|personnel|coordination|training|exercise|drill|assessment|vulnerability|resilience|continuity|operations|logistics|supplies|equipment|transportation|medical|search|rescue|damage|debris|restoration|utilities|power|water|food|clothing|temporary|housing|family|community|volunteer|donation|funding|federal|state|local|tribal|nonprofit|private|sector|public|information|media|social|outreach|education|awareness|prevention|protection|security|threat|incident|command|system|management|leadership|decision|making|communication|interoperability|coordination|collaboration|partnership|mutual|aid|agreement|protocol|procedure|policy|guidance|standard|regulation|compliance|documentation|reporting|tracking|monitoring|evaluation|lessons|learned|improvement|capability|capacity|building|development|sustainability|long|term|short|term|immediate|urgent|critical|essential|priority|high|medium|low|level|phase|stage|pre|post|during|before|after|initial|ongoing|final|complete|partial|full|scale|large|small|major|minor|significant|substantial|limited|extensive|comprehensive|detailed|specific|general|overall|total|individual|collective|joint|unified|integrated|coordinated|systematic|strategic|tactical|operational|technical|administrative|financial|legal|regulatory|environmental|health|safety|welfare|wellbeing)\b', text, re.IGNORECASE))
    
    # Get most common concepts
    counter = Counter(words)
    return [word for word, count in counter.most_common(10)]

def build_knowledge_graph_fast(all_chunks):
    """Faster graph building without LLM calls"""
    graph = nx.Graph()
    texts = [chunk["text"] for chunk in all_chunks]
    
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = create_embeddings_cached(texts)
    
    print("Extracting concepts and building graph nodes...")
    for i, chunk in enumerate(all_chunks):
        concepts = extract_concepts_simple(chunk["text"])  # No LLM call!
        graph.add_node(i, text=chunk["text"], concepts=concepts, 
                       embedding=embeddings[i], metadata=chunk["metadata"])
    
    print("Computing similarities and adding edges...")
    # Vectorized similarity computation
    embeddings_array = np.array(embeddings)
    similarity_matrix = np.dot(embeddings_array, embeddings_array.T) / (
        np.linalg.norm(embeddings_array, axis=1)[:, None] * np.linalg.norm(embeddings_array, axis=1)[None, :]
    )
    
    # Add edges based on similarity threshold
    threshold = 0.7
    for i in range(len(all_chunks)):
        for j in range(i+1, len(all_chunks)):
            if similarity_matrix[i, j] > threshold:
                concepts_i = set(graph.nodes[i]["concepts"])
                concepts_j = set(graph.nodes[j]["concepts"])
                shared = concepts_i.intersection(concepts_j)
                if shared:
                    graph.add_edge(i, j, weight=similarity_matrix[i, j], 
                                   shared_concepts=list(shared))
    
    return graph, embeddings

# Original concept extraction (for backward compatibility)
def extract_concepts(text):
    sys_msg = "Extract the key concepts and entities as a JSON array of 5-10 strings."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Extract key concepts from:\n\n{text[:3000]}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
        if isinstance(result, list):
            return result
        return []
    except Exception:
        # fallback pattern
        matches = re.findall(r'\[(.*?)\]', response.choices[0].message.content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []

def build_knowledge_graph(all_chunks):
    graph = nx.Graph()
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = create_embeddings(texts)
    for i, chunk in enumerate(all_chunks):
        concepts = extract_concepts(chunk["text"])
        graph.add_node(i, text=chunk["text"], concepts=concepts, embedding=embeddings[i], metadata=chunk["metadata"])
    for i in range(len(all_chunks)):
        concepts_i = set(graph.nodes[i]["concepts"])
        for j in range(i+1, len(all_chunks)):
            concepts_j = set(graph.nodes[j]["concepts"])
            shared = concepts_i.intersection(concepts_j)
            if shared:
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                concept_score = len(shared) / min(len(concepts_i), len(concepts_j)) if concepts_i and concepts_j else 0
                edge_weight = 0.7 * sim + 0.3 * concept_score
                if edge_weight > 0.6:
                    graph.add_edge(i, j, weight=edge_weight, shared_concepts=list(shared))
    return graph, embeddings

def traverse_graph(query, graph, embeddings, top_k=3, max_depth=3):
    query_embedding = create_embeddings_cached(query)
    # Ensure query_embedding is 1D array
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    
    similarities = []
    for i, emb in enumerate(embeddings):
        emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
        sim = np.dot(query_embedding, emb_array) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb_array))
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    start_nodes = [node for node, _ in similarities[:top_k]]
    visited = set()
    results = []
    queue = []
    idx_map = {node: sim for node, sim in similarities}
    for node in start_nodes:
        heapq.heappush(queue, (-idx_map.get(node, 1), node))
    while queue and len(results) < (top_k * 3):
        _, node = heapq.heappop(queue)
        if node in visited: continue
        visited.add(node)
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "metadata": graph.nodes[node]["metadata"]
        })
        if len(visited) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"]) 
                         for neighbor in graph.neighbors(node)
                         if neighbor not in visited]
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))
    return results

def graph_rag_fast(query, folder_path, chunk_size=1000, chunk_overlap=200, top_k=3):
    """Fast Graph RAG without LLM concept extraction"""
    cache_file = f"{os.path.basename(folder_path)}_graph_cache.pkl"
    
    if os.path.exists(cache_file):
        print("Loading cached graph...")
        with open(cache_file, 'rb') as f:
            graph, embeddings = pickle.load(f)
    else:
        print("Building knowledge graph...")
        docs = extract_texts_from_folder(folder_path)
        all_chunks = []
        
        for doc in docs:
            chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({"text": chunk, "metadata": {"source": doc["filename"], "chunk_index": idx}})
        
        graph, embeddings = build_knowledge_graph_fast(all_chunks)  # No LLM calls!
        
        print("Saving graph cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump((graph, embeddings), f)
    
    relevant_chunks = traverse_graph(query, graph, embeddings, top_k)
    answer = generate_response(query, relevant_chunks)
    
    return {
        "query": query,
        "response": answer,
        "relevant_chunks": relevant_chunks,
        "graph": graph # Return graph for visualization
    }

# Original graph_rag (for backward compatibility)
def graph_rag(query, folder_path, chunk_size=1000, chunk_overlap=200, top_k=3):
    docs = extract_texts_from_folder(folder_path)
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({"text": chunk, "metadata": {"source": doc["filename"], "chunk_index": idx}})
    graph, embeddings = build_knowledge_graph(all_chunks)
    relevant_chunks = traverse_graph(query, graph, embeddings, top_k)
    answer = generate_response(query, relevant_chunks)
    return {
        "query": query,
        "response": answer,
        "relevant_chunks": relevant_chunks,
        "graph": graph
    }

# Add near top, after imports
with open('reference/val.json', 'r', encoding='utf-8') as f:
    VAL_QUESTIONS = json.load(f)

def get_reference_answer(query):
    """Fetch ideal_answer from val.json for the given query."""
    for item in VAL_QUESTIONS:
        if item["question"].strip().lower() == query.strip().lower():
            return item["ideal_answer"]
    return None

def openai_similarity_score(answer, ref_answer, model='text-embedding-3-small'):
    # Use your existing cached embedding function!
    vec1, vec2 = create_embeddings_cached([answer, ref_answer])
    # If either embedding is missing, return 0.
    if vec1 is None or vec2 is None:
        return 0.0
    arr1, arr2 = np.array(vec1), np.array(vec2)
    sim = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2) + 1e-8)
    return float(sim)

def bert_f1_score(answer, ref_answer, model_type='bert-base-uncased'):
    if not answer or not ref_answer:
        return 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # BERTScore expects batch lists
    P, R, F1 = bertscore_score([answer], [ref_answer], lang='en', model_type=model_type, device=device, verbose=False)
    return float(F1[0])

def validate_answers(query, results):
    reference = get_reference_answer(query)
    if not reference:
        return {
            "standard": {"embedding": 0.0, "bert_f1": 0.0},
            "hierarchical": {"embedding": 0.0, "bert_f1": 0.0},
            "graph": {"embedding": 0.0, "bert_f1": 0.0},
            "best": None
        }
    scores = {}
    for strat in ["standard", "hierarchical", "graph"]:
        cand = results.get(strat, {}).get("response") or ""
        scores[strat] = {
            "embedding": openai_similarity_score(cand, reference),
            "bert_f1": bert_f1_score(cand, reference)
        }
    # Decide best by your preferred metric
    best = max(scores, key=lambda k: scores[k]["bert_f1"])
    scores["best"] = best
    return scores

def plot_graph_rag(graph):
    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(graph, k=0.25)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, width=[2*v for v in edge_weights.values()], alpha=0.5)
    nx.draw_networkx_labels(graph, pos, {n: n for n in graph.nodes}, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={e: f"{d['weight']:.2f}" for e, d in graph.edges.items()})
    plt.title("Graph RAG Knowledge Graph (nodes are text chunks/concepts)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_bertscore_heatmap(reference_answers, standard_outputs, hierarchical_outputs, graph_outputs):
    all_outputs = [reference_answers, standard_outputs, hierarchical_outputs, graph_outputs]
    labels = ["Reference", "Standard RAG", "Hierarchical RAG", "Graph RAG"]
    matrix = np.zeros((4, 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def mean_bertscore_f1(pred, refs):
        P, R, F1 = bertscore_score(pred, refs, lang='en', model_type='bert-base-uncased', device=device, verbose=False)
        return F1.mean().item()
    for i in range(4):
        for j in range(4):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = mean_bertscore_f1(all_outputs[i], all_outputs[j])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap='Blues')
    plt.xticks(ticks=np.arange(4), labels=labels)
    plt.yticks(ticks=np.arange(4), labels=labels)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center', color='black', fontsize='x-large')
    plt.title("Average BERTScore F1 Similarity Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def update_results_csv(question, results, csv_path="reference/fema_evaluation.csv", trace_id=None, start_time=None, end_time=None):
    import pandas as pd
    import os

    # Make sure file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    # Load the CSV with fallback encoding
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # Locate the row where the 'Question' column matches
    matched = df['Question'].str.strip().str.lower() == question.strip().lower()
    if not matched.any():
        print(f"⚠️ No matching question found in CSV for: {question}")
        return

    idx = df.index[matched][0]  # If multiple, just use the first

    try:
        # --- RAG responses ---
        df.at[idx, 'Standard RAG Answer'] = results['standard'].get('response', "")
        df.at[idx, 'Hierarchical RAG Answer'] = results['hierarchical'].get('response', "")
        df.at[idx, 'Graph-based RAG Answer'] = results['graph'].get('response', "")
        if 'hybrid' in results:
            df.at[idx, 'Hybrid RAG Answer'] = results['hybrid'].get('response', "")

        # --- Similarity and BERT Scores ---
        val = results.get('validation', {})
        if val:
            df.at[idx, 'Standard RAG Similarity Score'] = val['standard'].get('embedding', "")
            df.at[idx, 'Standard RAG Bert Score'] = val['standard'].get('bert_f1', "")
            df.at[idx, 'Hierarchical RAG Similarity Score'] = val['hierarchical'].get('embedding', "")
            df.at[idx, 'Hierarchical RAG Bert Score'] = val['hierarchical'].get('bert_f1', "")
            df.at[idx, 'Graph-based RAG Similarity Score'] = val['graph'].get('embedding', "")
            df.at[idx, 'Graph-based RAG Bert Score'] = val['graph'].get('bert_f1', "")
            if 'hybrid' in val:
                df.at[idx, 'Hybrid RAG Similarity Score'] = val['hybrid'].get('embedding', "")
                df.at[idx, 'Hybrid RAG Bert Score'] = val['hybrid'].get('bert_f1', "")

            # --- Best method and score ---
            best_method = val.get('best')
            best_bert_score = val.get(best_method, {}).get('bert_f1') if best_method else None
            df.at[idx, 'Best RAG'] = best_method
            df.at[idx, 'Best Bert Score'] = best_bert_score

        # --- Traceability fields ---
        df.at[idx, 'TraceId'] = trace_id
        df.at[idx, 'Start Time'] = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None
        df.at[idx, 'End Time'] = end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None

        # --- Save back to CSV ---
        df.to_csv(csv_path, index=False)
        print(f"✅ Results for '{question}' updated in {csv_path}.")

    except Exception as e:
        print(f"❌ Error while updating CSV for '{question}': {e}")
        
# --- New function to combine RAG techniques ---
def hybrid_rag(query, folder_path, chunk_size=1000, chunk_overlap=200, k=5, k_summaries=3, k_chunks=5, top_k_graph=3):
    """
    A combined RAG pipeline that leverages all three techniques:
    1. Initial retrieval using Standard RAG (dense retrieval).
    2. Contextual filtering using Hierarchical RAG (page-level relevance).
    3. Context expansion using Graph RAG (conceptual connections).
    4. Re-ranking the combined set of chunks and generating the final response.
    """
    print("=== ENTERED HYBRID RAG ===")
    
    # --- Step 1: Initial Standard RAG Retrieval ---
    # This retrieves a broad set of chunks based on semantic similarity.
    print("Step 1: Performing initial retrieval with Standard RAG...")
    standard_results = standard_rag_cached(query, folder_path, chunk_size, chunk_overlap, k=k*2) # Retrieve more to filter later
    initial_chunks = standard_results['retrieved_chunks']

    # --- Step 2: Contextual Filtering with Hierarchical RAG ---
    # This filters the initial chunks, keeping only those from pages/documents the hierarchical model deems relevant.
    print("Step 2: Filtering chunks with Hierarchical RAG...")
    # This block re-uses the setup from hierarchical_rag without regenerating the whole thing.
    summary_store_file = f"{os.path.basename(folder_path)}_summary_store.pkl"
    if not os.path.exists(summary_store_file):
        # Fallback if no summary store exists, re-run hierarchical setup
        hierarchical_rag(query, folder_path, regenerate=True)
    with open(summary_store_file, 'rb') as f:
        summary_store = pickle.load(f)

    query_embedding = create_embeddings_cached(query)
    summary_results = summary_store.similarity_search(query_embedding, k=k_summaries)
    relevant_sources = [(r["metadata"]["source"], r["metadata"]["page"]) for r in summary_results]
    
    filtered_chunks = []
    for chunk in initial_chunks:
        if (chunk["metadata"]["source"], chunk["metadata"].get("page")) in relevant_sources:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        print("Warning: Hierarchical filtering removed all chunks. Falling back to initial retrieval.")
        filtered_chunks = initial_chunks
    
    # --- Step 3: Context Expansion with Graph RAG ---
    # This finds related chunks in the knowledge graph to enrich the context.
    print("Step 3: Expanding context with Graph RAG...")
    graph_cache_file = f"{os.path.basename(folder_path)}_graph_cache.pkl"
    if not os.path.exists(graph_cache_file):
        # Fallback if no graph cache exists, re-run graph setup
        graph_rag_fast(query, folder_path, regenerate=True)
    with open(graph_cache_file, 'rb') as f:
        graph, embeddings = pickle.load(f)

    # Use the filtered chunks as starting points for graph traversal
    initial_node_indices = [idx for idx, emb in enumerate(embeddings) if any(c['text'] == graph.nodes[idx]['text'] for c in filtered_chunks)]
    if not initial_node_indices:
        print("Warning: No initial nodes found in the graph. Using direct similarity search.")
        expanded_chunks_from_graph = traverse_graph(query, graph, embeddings, top_k=top_k_graph)
    else:
        # Traverse the graph from the filtered chunks to find related information
        expanded_chunks_from_graph = traverse_graph(query, graph, embeddings, top_k=len(initial_node_indices))

    # Combine all unique chunks
    combined_chunks_text = {c["text"]: c for c in filtered_chunks}
    for chunk in expanded_chunks_from_graph:
        combined_chunks_text[chunk["text"]] = chunk
    
    final_chunks = list(combined_chunks_text.values())
    
    # Re-rank the final chunks by similarity to the query
    query_embedding = create_embeddings_cached(query)
    reranked_chunks = sorted(final_chunks, key=lambda c: np.dot(query_embedding, np.array(create_embeddings_cached(c["text"]))) / (np.linalg.norm(query_embedding) * np.linalg.norm(create_embeddings_cached(c["text"])) + 1e-8), reverse=True)

    # Take the top 'k' chunks for final answer generation
    final_context = reranked_chunks[:k]
    
    # --- Step 4: Final Generation ---
    answer = generate_response(query, final_context)
    
    return {
        "query": query,
        "response": answer,
        "retrieved_chunks": final_context,
    }


# ========== 4. OPTIMIZED COMPARATORS ==========

def compare_all_rag_fast(query, folder_path, **kwargs):
    """Optimized version with caching and simplified processing"""
    print("Running optimized RAG comparison...")
    results = {
        "standard": standard_rag_cached(query, folder_path, **kwargs),
        "hierarchical": hierarchical_rag(query, folder_path, **kwargs),
        "graph": graph_rag_fast(query, folder_path, **kwargs),
        "hybrid": hybrid_rag(query, folder_path, **kwargs)
    }
    # Add validation scores
    results["validation"] = validate_answers(query, results)
    return results

def compare_all_rag(query, folder_path, **kwargs):
    """Original comparator (for backward compatibility)"""
    print("== RUNNING STANDARD ONLY ==")
    return {
        "standard": standard_rag(query, folder_path, **kwargs),
        "hierarchical": hierarchical_rag(query, folder_path, **kwargs),
        "graph": graph_rag(query, folder_path, **kwargs),
        

    }
