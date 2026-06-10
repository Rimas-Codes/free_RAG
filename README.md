# 🔍 Simple RAG — 100% Free, No API Key Required

A beginner-friendly implementation of **Retrieval-Augmented Generation (RAG)** that runs entirely for free inside Google Colab — no OpenAI key, no paid APIs, no cloud credits needed.

> Based on the guide: [Simple RAG Explained – machinelearningplus.com](https://machinelearningplus.com/gen-ai/simple-rag-explained-a-beginners-guide-to-retrieval-augmented-generation/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## What is RAG?

**Retrieval-Augmented Generation** grounds an LLM's answers in your own documents rather than relying solely on what the model learned during training. This reduces hallucinations and lets you query any custom knowledge base.

```
User Question
     │
     ▼
[ Embed Question ] ──► [ Vector DB Search ] ──► Top-K Chunks
                                                      │
                                                      ▼
                                          [ LLM: context + question ]
                                                      │
                                                      ▼
                                                   Answer
```

The three stages are:

1. **Index** — split documents into chunks, embed each chunk, store in a vector database
2. **Retrieve** — embed the user's query, find the most similar chunks by cosine similarity
3. **Generate** — pass the retrieved chunks as context to an LLM and produce a grounded answer

---

## Free Stack (vs. the Paid Original)

| Component | Original (paid) | This repo (free) |
|-----------|----------------|------------------|
| Embeddings | OpenAI `text-embedding-ada-002` | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | External (OpenAI-managed) | `FAISS` — local, in-memory |
| LLM | `gpt-3.5-turbo` | `google/flan-t5-base` via HuggingFace |
| Cost | ~$0.01–0.10 per run | **$0.00** |

---

## Quickstart

### Run in Google Colab (recommended)

1. Open `simple_rag_free.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run **Cell 1** to install dependencies (takes ~1 min)
3. **Runtime → Restart session** after installation
4. Run all remaining cells from the top

### Run locally

```bash
git clone https://github.com/your-username/simple-rag-free.git
cd simple-rag-free

pip install sentence-transformers faiss-cpu "transformers==4.44.2" accelerate matplotlib
jupyter notebook simple_rag_free.ipynb
```

> **Note on `transformers` version:** Versions 4.52+ removed `text2text-generation` from the pipeline registry. Either pin to `transformers==4.44.2`, or use the direct `AutoModelForSeq2SeqLM` approach described in the notebook's troubleshooting section.

---

## Notebook Walkthrough

| Step | Description |
|------|-------------|
| 0 | Install dependencies |
| 1 | Import libraries |
| 2 | Define the knowledge base (customisable) |
| 3 | Chunk documents with sliding window + overlap |
| 4 | Embed chunks using `all-MiniLM-L6-v2` |
| 5 | Build a FAISS cosine similarity index |
| 6 | Retriever function — embed query, search index |
| 7 | Load `flan-t5-base` LLM (no API key) |
| 8 | Full RAG pipeline: Retrieve → Prompt → Generate |
| 9 | Ask questions and see answers with sources |
| 10 | Visualise retrieval scores with matplotlib |
| 11 | Instructions for using your own documents |

---

## Using Your Own Documents

In **Step 2**, replace the `documents` list with your own text:

```python
documents = [
    "Your first paragraph or document here...",
    "Your second paragraph here...",
    # add as many as you like
]
```

Or load from a file:

```python
with open("my_document.txt", "r") as f:
    raw_text = f.read()
documents = raw_text.split("\n\n")  # split on blank lines
```

Then re-run from Step 3 onwards to rebuild the index.

---

## Improving Answer Quality

The default `flan-t5-base` model is intentionally small so it runs on CPU. For better answers:

- **Better model, same approach:** Swap `flan-t5-base` → `flan-t5-large` or `flan-t5-xl` (needs GPU — use *Runtime → Change runtime type → T4 GPU* in Colab)
- **Free HuggingFace Inference API:** Use a larger model like `mistralai/Mistral-7B-Instruct-v0.1` via the HuggingFace API with a free token — no local GPU needed
- **Tune chunking:** Adjust `chunk_size` and `overlap` in `chunk_text()` to match your document style
- **Tune retrieval:** Increase `k` (number of retrieved chunks) for broader context

---

## Key Concepts Demonstrated

- **Semantic chunking** — why size and overlap matter for retrieval quality
- **Dense embeddings** — how text becomes a searchable vector
- **Cosine similarity search** — how FAISS finds the most relevant passages
- **Prompt engineering** — how retrieved context is injected to ground the LLM
- **Hallucination reduction** — the model is constrained to answer only from retrieved docs

---

## Requirements

```
sentence-transformers
faiss-cpu
transformers==4.44.2
accelerate
matplotlib
```

Python 3.9+ recommended.

---

## References

- [Simple RAG Explained – machinelearningplus.com](https://machinelearningplus.com/gen-ai/simple-rag-explained-a-beginners-guide-to-retrieval-augmented-generation/)
- [sentence-transformers documentation](https://www.sbert.net/)
- [FAISS documentation](https://faiss.ai/)
- [google/flan-t5-base on HuggingFace](https://huggingface.co/google/flan-t5-base)

---

## License

MIT
