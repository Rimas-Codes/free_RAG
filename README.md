📘 Simple RAG Pipeline (Free & Local)
A lightweight, fully local Retrieval‑Augmented Generation (RAG) pipeline built in Google Colab using:

SentenceTransformers for embeddings

FAISS for vector search

FLAN‑T5 for answer generation

No API keys

No paid services

Runs on CPU or GPU

This project demonstrates how to build a minimal but functional RAG system from scratch — ideal for learning, experimenting, or adapting into larger applications.

🚀 Features
🔍 Semantic search using FAISS (inner‑product / cosine similarity)

📚 Context retrieval from your own documents

🧠 FLAN‑T5 generation (encoder‑decoder model, free & open‑source)

🧩 End‑to‑end RAG pipeline: Retrieve → Build Prompt → Generate

💻 Works on CPU (GPU optional)

📝 Clean, readable code designed for learning

📂 Project Structure
Code
simple_rag_free.ipynb     # Main Colab notebook
data/                     # (Optional) Your text files or chunks
README.md                 # Project documentation
🛠️ Installation
Inside Colab:

bash
!pip install transformers sentence-transformers faiss-cpu
🧠 Models Used
🔹 Embeddings
sentence-transformers/all-MiniLM-L6-v2  
Fast, lightweight, great for semantic search.

🔹 LLM (Generator)
google/flan-t5-base  
Open‑source, instruction‑tuned, works well for short factual answers.

🧮 How It Works
1. Embed your documents
python
embeddings = embedder.encode(documents, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index.add(embeddings)
2. Retrieve top‑k relevant chunks
python
scores, indices = index.search(query_emb, k)
3. Build a prompt
python
prompt = f"""
Answer the question using ONLY the context below...

Context:
{context}

Question: {query}

Answer below:
"""
4. Generate an answer (FLAN‑T5)
python
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
🧪 Example
Question:

What is the difference between supervised and unsupervised learning?

Retrieved Context:

Supervised learning uses labelled data

Unsupervised learning finds patterns in unlabelled data

Generated Answer:

Supervised learning trains on labelled input‑output pairs, while unsupervised learning discovers hidden patterns in unlabelled data.

⚠️ Notes & Tips
FLAN‑T5 must be loaded using AutoModelForSeq2SeqLM, not the text-generation pipeline.

If running on CPU, replace .to("cuda") with .to("cpu").

Avoid ending prompts with a colon (Answer:) — T5 sometimes outputs nothing.

FAISS inner‑product search requires normalized vectors for cosine similarity.

📜 License
MIT License — free to use, modify, and share.

🤝 Contributing
Pull requests are welcome!
Feel free to open issues for improvements or questions.
