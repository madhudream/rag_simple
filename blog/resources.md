# Resources & further reading

External material referenced by or complementary to this series.

## Deep dives on individual topics

- **Approximate Nearest Neighbor (ANN) search** — how HNSW, IVF, trees, and hashing make
  vector search fast at scale (referenced in post 2b):
  <https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work>
- **How does a Reranker work?** — bi-encoder vs cross-encoder, ColBERT and late
  interaction (post 03 follows a similar structure):
  <https://outcomeschool.com/blog/how-does-a-reranker-work>
- **How does an Embedding Cache work?** — caching text→vector calls, the complement to
  post 06's semantic cache:
  <https://outcomeschool.com/blog/how-does-an-embedding-cache-work>
- **Agentic RAG** — the concept overview post 07's explanation follows:
  <https://outcomeschool.com/blog/agentic-rag>
- **How does LangChain work?** — chains, memory, agents in one framework:
  <https://outcomeschool.com/blog/how-does-langchain-work>

## Primary sources

- **Contextual Retrieval** (Anthropic, 2024) — the technique behind post 05, including
  contextual BM25 and the ~67% retrieval-failure reduction figure:
  <https://www.anthropic.com/news/contextual-retrieval>
- **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, 2009) — the RRF formula and
  the k=60 constant used in posts 2c and 04.
- **OpenAI Agents SDK** — the agent framework used in post 07:
  <https://github.com/openai/openai-agents-python>
- **Okapi BM25** (Robertson & Walker) — the ranking function built from scratch in
  post 2a.
- **HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels** (Gao et al.,
  2022) — the hypothetical-document technique in post 04.
- **Interactive tokenizer** — watch text become tokens (post 2a's aside):
  <https://platform.openai.com/tokenizer>

## Libraries used throughout

- **FastEmbed** — local embeddings, sparse BM25 vectors, and cross-encoders:
  <https://github.com/qdrant/fastembed>
- **Qdrant** — vector database used for dense, sparse, and hybrid search:
  <https://qdrant.tech>
- **FlashRank** — local ONNX cross-encoder reranking: <https://github.com/PrithivirajDamodaran/FlashRank>
- **LlamaIndex** — the production framework in the course code: <https://www.llamaindex.ai>
