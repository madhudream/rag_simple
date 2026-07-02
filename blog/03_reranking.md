# How does a reranker work? Two-stage retrieval explained from scratch

Post 2c ended with a hybrid retriever that finds a good *pool* of candidates. This post
is about a problem that survives even good retrieval: **the order inside the pool.**
Watch our own retriever from the last posts handle the holiday-refund question:

```
Q: "I booked boarding for a public holiday and want to cancel. Will I get my money back?"

retriever (bi-encoder) top-2:
  0.655  doc 3: Refunds for cancelled boarding are issued within 5 business days
  0.522  doc 4: Bookings made for public holidays are non-refundable
```

Both relevant documents retrieved — great recall! But the **rule** sits above the
**exception**, and post 01 already showed what an LLM does when the rule arrives without
the exception on top: it answers *"Yes, you'll get a refund"* to a non-refundable
booking. Retrieval found the right pages; it just **ordered them wrong.**

The fix is a **reranker** — a second, slower, smarter model that re-reads the shortlist
and reorders it. By the end of this post you'll know exactly how it works, because we
run one on the failures from every previous post and read the real numbers.

*(Further reading on rerankers is linked in [resources](resources.md). What this post
adds to the standard explanation: our running handbook corpus, executable code for
every step, and the measured numbers.)*

We'll cover it in four parts:

1. **Setup** — the librarian, where a reranker sits, and the two-stage idea
2. **The core** — bi-encoder vs cross-encoder: reading apart vs reading together
3. **Reranking from scratch** — two rescue missions, real scores, and the price tag
4. **Rerankers in the real world** — the pool limit, ColBERT, and production numbers

## The one metaphor to hold onto: the senior librarian

You walk into a huge library and ask about healthy cooking. Two staff members handle it:

1. **The junior helper sprints.** They run the aisles and bring back 50 books that have
   "healthy" or "cooking" somewhere on the cover. Fast, rough, cover-level judgment.
2. **The senior librarian reads.** She sits down with your *question in mind*, opens
   each of the 50 books, reads a few pages of each, and re-stacks the pile so the truly
   best matches are on top.
3. **She never fetches.** The librarian only reorders what the junior brought — if the
   right book stayed on the shelf, she can't rank it.
4. **She's slow — that's why the junior goes first.** She'd take a week to read the
   whole library; reading 50 books takes minutes.

The junior helper is your retriever (posts 2a–2c). The senior librarian is the
**reranker**. The rest of this post is those four rules made precise.

## Where a reranker sits

```
Question
   │
   ▼
[ Retriever ]   → fetches many candidates        (fast, rough — the junior)
   │
   ▼
[ Reranker ]    → re-reads and reorders them      (slow, precise — the librarian)
   │
   ▼
[ LLM ]         → answers from the top documents  (post 00's open-book exam)
   │
   ▼
Answer
```

And the numbers that make this shape necessary — the **two-stage idea**:

```
Corpus (1,000,000 chunks)
        │
        │  Stage 1: retriever — BM25 / vectors / hybrid
        ▼
   ~100 candidates          (rough shortlist, milliseconds)
        │
        │  Stage 2: reranker — reads each candidate WITH the question
        ▼
   top 3–5 documents        (carefully ordered, tens of milliseconds)
```

Stage 1 trades precision for speed. Stage 2 trades speed for precision — and stays
affordable *only because* stage 1 shrank a million to a hundred.

## Why stage 1 can't just be more careful

You already know both stage-1 machines and both of their blind spots:

- **BM25 (post 2a)** matches literal tokens — fast, but *semantic-blind*: `puppy` never
  finds `dog`.
- **Vector search (post 2b)** compresses each chunk into **one** position on the
  meaning-map — fast (positions precomputed), but compression *loses fine detail*. The
  rule and the exception both live in the "boarding refunds" neighborhood; which one
  *answers your question* is exactly the detail one point on a map can't hold.

Both share one structural limit: **they judge the document without ever reading it
together with the question.** That limit has a name — and removing it is the whole trick.

## The core idea — bi-encoder vs cross-encoder

This is the most important section of the post. Every retriever from posts 2a–2c is a
**bi-encoder**: two things encoded *separately*, compared afterward.

```
BI-ENCODER (stage 1 — fast retrieval)

  Question ──► [encoder] ──► [vector A]
                                   ╲
                                    compare (cosine) ──► score
                                   ╱
  Document ──► [encoder] ──► [vector B]      ← precomputed, ONCE, before any query
```

The superpower is that right side: every document's vector is computed **before any
question exists**. At query time you embed one question and compare cheap vectors — a
million comparisons in milliseconds (post 2b's ANN). The weakness is the same fact read
backwards: the document's meaning was frozen *before knowing what would be asked*.

A **cross-encoder** removes the separation:

```
CROSS-ENCODER (stage 2 — reranking)

  Question + Document ──► [ one model reads BOTH together ] ──► relevance score
```

One model, both texts in the same pass, attention flowing between every word of the
question and every word of the document. It doesn't produce embeddings at all — just a
single number: *how well does this document answer this question?* Nothing is
precomputable (the input includes the question), so every query pays full model cost per
candidate. Hence: shortlist only.

| Aspect | Bi-encoder | Cross-encoder |
|---|---|---|
| Reads question & document | Separately | **Together** |
| Output | One vector per text | One relevance score per **pair** |
| Precomputable | Yes — index the corpus once | No — question is part of the input |
| Speed | Very fast (vector compare) | Slow (full model per pair) |
| Precision | Good; loses fine detail | High; catches fine detail |
| Role | Stage 1: search everything | Stage 2: reorder the shortlist |

## Reranking from scratch

FastEmbed — the same library from posts 2a/2b — also runs cross-encoders locally
(`pip install fastembed`):

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

def rerank(query, doc_ids):
    scores = list(reranker.rerank(query, [corpus[i] for i in doc_ids]))
    return sorted(zip(scores, doc_ids), reverse=True)
```

That's the entire step: score every (question, candidate) pair, sort. The corpus is the
handbook plus the error-code twins — every failure this series has collected in one
place. Now, two rescue missions.

### Rescue 1 — the rule/exception flip (post 01's villain)

```
Q: "I booked boarding for a public holiday and want to cancel. Will I get my money back?"

BEFORE (bi-encoder order)                    AFTER (cross-encoder rerank)
  0.655  rule      (refunds in 5 days)          2.316  exception (holidays non-refundable)
  0.522  exception (holidays non-refundable)   -1.652  rule      (refunds in 5 days)
  0.393  closing times                         -8.435  grooming refunds
  0.391  grooming refunds                      -8.872  closing times
  0.304  error E-4042 doc                     -10.658  error E-4042 doc
```

Read what happened. The bi-encoder ranked the rule first because the query's words
("refund", "cancel", "boarding") overlap its position best — a *cover-level* judgment.
The cross-encoder read each document **with the question** and understood that for a
*holiday* booking, the exception is the answer: it scores **2.316**, while everything
merely-related plunges below zero. The librarian read the pages.

(About those scores: cross-encoders output raw logits — any real number, often negative.
Only the *order* means anything. A big gap, like 2.3 vs −1.7, reads as confidence.)

### Rescue 2 — the identifier twins (post 2b's villain)

```
Q: "error E-4042"

BEFORE (bi-encoder order)                    AFTER (cross-encoder rerank)
  0.523  E-4043 doc   ← wrong twin              8.380  E-4042 doc   ← the right one
  0.512  E-4044 doc   ← wrong twin              6.830  E-4043 doc
  0.504  E-4042 doc   ← the right one, THIRD    6.528  E-4044 doc
```

Post 2b showed vector search confusing the twins (one changed character barely moves a
compressed embedding). Post 2c fixed it with hybrid fusion. Here's a *second* fix: the
cross-encoder reads the raw text of query and document together, so the literal token
`E-4042` is right there in its attention — no compression ever happened. Two different
cures for the same disease; production systems often use both.

### The price tag

*Librarian rule 4: she's slow.* Measured on the same 12 documents, same machine:

```
bi-encoder  (query vs 12 precomputed vectors):    1.1 ms
cross-encoder (12 full query+doc reads)      :    9.0 ms    (8× slower)
```

8× on twelve documents, and the bi-encoder's cost barely grows with corpus size (vectors
are precomputed; ANN search is sub-linear). The cross-encoder's cost is *per pair, at
query time, always*. On a million documents: milliseconds versus **hours**. That ratio
is the entire reason two-stage retrieval exists — and why the reranker only ever sees a
shortlist.

## The pool limit — what a reranker can never fix

*Librarian rule 3: she never fetches.* Honesty section. Take the Saturday-pickup
question from post 00 — the one where retrieval drifts, because "Saturday" ≠ "weekends":

```
Q: "Until what time can I pick up my dog on a Saturday?"

full bi-encoder ranking:
  rank 1  0.561  complimentary bath doc
  rank 2  0.487  daycare hours doc
  rank 3  0.365  grooming booking doc
  rank 4  0.349  late fee doc
  rank 5  0.306  boarding closing times  ← the right doc (weekends = Saturday)
  ...
```

Rerank the **top-3 pool** — the right doc isn't in it:

```
rerank([bath, daycare, grooming]) →  bath, daycare, grooming     (garbage, reordered)
```

The librarian diligently re-sorted three wrong books. **A reranker cannot rank what
retrieval didn't fetch.** Widen the pool to top-6 so the right doc gets in:

```
rerank(top-6 pool) →
  -4.182  complimentary bath doc      ← still first (!)
  -7.947  daycare hours doc
  -8.774  boarding closing times      ← lifted rank 5 → 3, but not to the top
  ...
```

Two honest lessons in one output. First: pool width is the hard ceiling — that's why
production retrieves generously (top-20, top-100) before reranking down. Second:
even in the pool, this small cross-encoder (L-6, 80 MB) still got fooled by surface
overlap ("pickup", dogs) — every score is negative, the model isn't confident in any of
them. Bigger rerankers do better (the course uses L-12; nothing here says "No" *is*
on a page — low scores across the board are themselves a useful signal that retrieval
failed). Rerankers sharpen order; they don't create relevance.

## The middle ground — ColBERT, in one minute

Between "one vector per document" (fast, coarse) and "read everything together" (slow,
sharp) sits **late interaction** — ColBERT. It keeps a vector **per word** of each
document (precomputed, like a bi-encoder), and at query time compares query words to
document words individually (fine-grained, like a cross-encoder). You can deploy it as a
better reranker, or — with a much bigger index — as retriever and reranker in one:

```
Option A:  Question → [ Retriever ] → [ ColBERT reranks ] → [ LLM ]
Option B:  Question → [ ColBERT does both ]               → [ LLM ]
```

Costs more storage (a vector per word, not per chunk), recovers much of the detail the
single-vector compression threw away. It deserves a post of its own; for this series,
know it exists and where it sits on the speed/precision line.

## Reranking in production

In my course code the reranker is **FlashRank** — a cross-encoder that runs on local
ONNX, no GPU, no API (`pip install flashrank`):

```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")   # ~120 MB, local

passages = [{"id": i, "text": corpus[i]} for i in pool_ids]
result = ranker.rerank(RerankRequest(query=question, passages=passages))
```

Real output on the holiday question's pool:

```
  0.9092  doc 4: Bookings made for public holidays are non-refundable   ← decisive
  0.0457  doc 3: Refunds for cancelled boarding are issued within 5 business days
  0.0002  doc 1: The boarding facility closes at 7 pm on weekdays and...
```

(FlashRank reports probabilities instead of raw logits — same model family, friendlier
numbers: 0.909 vs 0.046 is the librarian being *sure*.)

The course wires this into LlamaIndex as a postprocessor: **retrieve 20, rerank, keep
4** — the generous-pool-then-shrink pattern this post has been building toward. And the
measured results on the full corpus (1,421 docs, 50 golden questions) are the honest
version of every claim above:

- **Faithfulness 0.865 → 0.909** — best in the whole course to that point. Better-ordered
  context means the LLM's answers stick to the retrieved truth.
- **Recall 0.73 → 0.73, exactly flat.** The pool limit, measured at scale: the missing
  documents weren't in the top-20, so no amount of reordering could surface them.
- **Latency 1.36 s → 2.56 s.** The librarian's fee, paid on every query.

One number improved, one number physically couldn't, one number got worse — and all
three follow from the same mechanism you now understand.

## The whole post in four lines

The librarian's rules, now with their engineering names:

1. **The junior sprints** → stage-1 retriever (bi-encoder / BM25 / hybrid): precomputed,
   fast, cover-level judgment over the whole corpus.
2. **The librarian reads** → cross-encoder: question + document in one model pass;
   caught the exception (2.316) and the exact token (8.380) that separate encodings
   missed.
3. **She never fetches** → rerankers only reorder the pool: recall stayed 0.73, and no
   pool means no rescue.
4. **She's slow, so shortlist** → 8× cost per document at query time: retrieve ~20–100
   wide, rerank, keep the top 3–5.

Fetch wide and fast, then read the shortlist carefully. That's a reranker.

*(Every snippet and every output block in this post was executed for real — bi-encoder
retrieval is `all-MiniLM-L6-v2` via FastEmbed, the from-scratch reranker is
`Xenova/ms-marco-MiniLM-L-6-v2` via FastEmbed's `TextCrossEncoder`, the production one
is FlashRank's `ms-marco-MiniLM-L-12-v2`, all run locally. Timings are one machine,
averaged over 10 runs — your absolute numbers will differ; the ratio is the lesson. The
course-scale metrics are from one measured run on my corpus, not a law.)*

---

**Next up: query transforms** — the reranker just showed us its ceiling: it can't fix a
bad pool. So the next post attacks the pool itself — rewriting the *question* before
retrieval (multi-query, HyDE) so the right documents get fetched in the first place.
"Saturday" becomes "weekend", and the missing doc walks into the pool on its own.
