export interface Rung {
  slug: string;        // content collection id = filename without .md
  num: string;         // display number
  title: string;
  metaphor: string;
  level: string;
  emoji: string;
  group: 'FOUNDATION' | 'RETRIEVAL' | 'PRODUCTION';
  notebook: string | null;   // filename in colab/
  needsKey: boolean;
}

const COLAB = 'https://colab.research.google.com/github/madhudream/rag_simple/blob/main/colab/';

export const ladder: Rung[] = [
  { slug: '00_naive_rag', num: '00', title: 'What is RAG?', metaphor: 'The open-book exam', level: 'Novice', emoji: '🐣', group: 'FOUNDATION', notebook: '00_naive_rag.ipynb', needsKey: true },
  { slug: '01_chunking', num: '01', title: 'Chunk smart', metaphor: 'Index cards', level: 'Apprentice', emoji: '✂️', group: 'FOUNDATION', notebook: '01_chunking.ipynb', needsKey: true },
  { slug: '2a_bm_25', num: '2a', title: 'What is BM25?', metaphor: 'The detective', level: 'Journeyman', emoji: '🔎', group: 'RETRIEVAL', notebook: '2a_bm25.ipynb', needsKey: false },
  { slug: '2b_vector_search', num: '2b', title: 'What is vector search?', metaphor: 'The map of meaning', level: 'Journeyman', emoji: '🗺️', group: 'RETRIEVAL', notebook: '2b_vector_search.ipynb', needsKey: false },
  { slug: '2c_hybrid_search', num: '2c', title: 'What is hybrid search?', metaphor: 'A committee of two experts', level: 'Journeyman', emoji: '🤝', group: 'RETRIEVAL', notebook: '2c_hybrid_search.ipynb', needsKey: false },
  { slug: '03_reranking', num: '03', title: 'How does a reranker work?', metaphor: 'The senior librarian', level: 'Adept', emoji: '🎯', group: 'RETRIEVAL', notebook: '03_reranking.ipynb', needsKey: false },
  { slug: '04_query_transforms', num: '04', title: 'Query transforms', metaphor: 'The tourist and the phrasebook', level: 'Expert', emoji: '🧭', group: 'RETRIEVAL', notebook: '04_query_transforms.ipynb', needsKey: true },
  { slug: '05_contextual_retrieval', num: '05', title: 'Contextual retrieval', metaphor: 'A return address on every card', level: 'Master', emoji: '🧩', group: 'RETRIEVAL', notebook: '05_contextual_retrieval.ipynb', needsKey: true },
  { slug: '06_semantic_cache', num: '06', title: 'Semantic cache', metaphor: 'The receptionist who remembers', level: 'Grandmaster', emoji: '⚡', group: 'PRODUCTION', notebook: '06_semantic_cache.ipynb', needsKey: true },
  { slug: '07_agentic_rag', num: '07', title: 'Agentic RAG', metaphor: 'The researcher', level: 'Ninja', emoji: '🥷', group: 'PRODUCTION', notebook: '07_agentic_rag.ipynb', needsKey: true },
];

export const colabUrl = (notebook: string) => COLAB + notebook;
export const rungBySlug = (slug: string) => ladder.find((r) => r.slug === slug);
export const rungIndex = (slug: string) => ladder.findIndex((r) => r.slug === slug);
