from sentence_transformers import SentenceTransformer
import faiss
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and split info.txt into chunks
def load_and_chunk(file_path, chunk_size=512):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks

chunks = load_and_chunk("info.txt")

# Generate embeddings
embeddings = model.encode(chunks)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load TinyLlama or Phi-2 model
model_name = "TinyLlama/TinyLlama-1.1B"  # Use a small model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

@app.post("/chat")
def chat(query: str):
    # Embed query
    query_embedding = model.encode(query).reshape(1, -1)

    # Search FAISS index
    _, I = index.search(query_embedding, 5)  # Top 5 results
    context = " ".join([chunks[i] for i in I[0]])

    # Generate response
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response}
