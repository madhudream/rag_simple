# Colab notebooks — companion code for `simple/secondblog/`

One standalone notebook per blog post. Each runs top-to-bottom in Google Colab with no
setup beyond its first `%pip` cell — no API keys, all models run locally.

| Notebook | Blog post | What you build |
|---|---|---|
| `00_naive_rag.ipynb` | `00_naive_rag.md` | Naive RAG from scratch (Retrieve → Augment → Generate), then the LlamaIndex one-liner. Needs an OpenAI API key. |
| `01_chunking.ipynb` | `01_chunking.md` | Chunking: dilution vs orphans, fixed-size + overlap from scratch, hit@1 scoreboard, then SentenceSplitter. API key for one demo. |
| `2a_bm25.ipynb` | `2a_bm_25.md` | BM25 from scratch (4 ideas), then FastEmbed's production BM25 |
| `2b_vector_search.ipynb` | `2b_vector_search.md` | Embeddings + cosine + vector search from scratch, then Qdrant dense search |
| `2c_hybrid_search.ipynb` | `2c_hybrid_search.md` | RRF fusion from scratch, then Qdrant hybrid (dense + sparse + RRF) in one query |
| `03_reranking.ipynb` | `03_reranking.md` | Bi- vs cross-encoder, two rescue demos, timing, the pool limit, then FlashRank |
| `04_query_transforms.ipynb` | `04_query_transforms.md` | Multi-query + HyDE: failed paraphrases, dialect rewrites, RRF pool, answer flip. Needs API key. |
| `05_contextual_retrieval.ipynb` | `05_contextual_retrieval.md` | Anthropic contextual retrieval: twin chunks, LLM stamps, flipped rankings, per-doc compromise. Needs API key. |
| `06_semantic_cache.ipynb` | `06_semantic_cache.md` | Semantic cache: ~600× hit speedup, threshold sweep, the overlap, wrong-serve demo, warm/freeze fixes. Needs API key. |
| `07_agentic_rag.ipynb` | `07_agentic_rag.md` | Agentic RAG with the OpenAI Agents SDK: 3 tools, 5 traces (routing, self-translation, multi-hop, holiday trap). Needs API key. |

Structure inside every notebook: **from-scratch cells first** (the blog's ideas, one per
cell), **PRODUCTION section last** (the real-library version in its own block).

Read the blog post alongside — the notebooks print the same numbers the posts discuss.
