# RAG Complete Book 📖

A hands-on RAG course where **every number is real**. Each post builds one technique
from scratch, runs it, and reads the actual output — then shows the production-library
version. Each post ships with a Colab notebook you can run top to bottom.

One tiny corpus (a fictional pet-care handbook) runs through the whole series, so every
post's villain becomes the next post's demo.

## The ladder

| # | Post | Level | One metaphor | Practice |
|---|------|-------|--------------|----------|
| **FOUNDATION** | | | | |
| 00 | [What is RAG?](blog/00_naive_rag.md) | 🐣 Novice | The open-book exam | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/00_naive_rag.ipynb) |
| 01 | [Chunk smart](blog/01_chunking.md) | ✂️ Apprentice | Index cards | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/01_chunking.ipynb) |
| **RETRIEVAL** | | | | |
| 2a | [What is BM25?](blog/2a_bm_25.md) | 🔎 Journeyman | The detective | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/2a_bm25.ipynb) |
| 2b | [What is vector search?](blog/2b_vector_search.md) | 🗺️ Journeyman | The map of meaning | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/2b_vector_search.ipynb) |
| 2c | [What is hybrid search?](blog/2c_hybrid_search.md) | 🤝 Journeyman | A committee of two experts | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/2c_hybrid_search.ipynb) |
| 03 | [How does a reranker work?](blog/03_reranking.md) | 🎯 Adept | The senior librarian | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/03_reranking.ipynb) |
| 04 | [Query transforms](blog/04_query_transforms.md) | 🧭 Expert | The tourist and the phrasebook | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/04_query_transforms.ipynb) |
| 05 | [Contextual retrieval](blog/05_contextual_retrieval.md) | 🧩 Master | A return address on every card | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/05_contextual_retrieval.ipynb) |
| **PRODUCTION** | | | | |
| 06 | [Semantic cache](blog/06_semantic_cache.md) | ⚡ Grandmaster | The receptionist who remembers | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/06_semantic_cache.ipynb) |
| 07 | [Agentic RAG](blog/07_agentic_rag.md) | 🥷 Ninja | The researcher | [▶ Colab](https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/07_agentic_rag.ipynb) |

Further reading and sources: [resources](blog/resources.md).

## Practice in Colab

Every post has a matching notebook in [`colab/`](colab/) — runnable top to bottom, all
models local except the LLM calls. See the [notebook guide](colab/README.md).

Notebooks that call an LLM read `OPENAI_API_KEY` from Colab's **Secrets** panel (🔑 in
the left sidebar) — add it once, never paste it again.

## Structure

```
blog/    the ten posts (markdown) + resources.md
colab/   one standalone notebook per post
```
