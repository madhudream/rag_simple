# Query transforms: fix the pool by fixing the question

Post 03 ended on a hard limit: a reranker can only reorder what retrieval fetched. And
we left a question dying on that limit — watch the *whole pipeline* fail on it, LLM
answer included:

```
Q: "Until what time can I pick up my dog on a Saturday?"

dense retrieval, full ranking:
  rank 1  0.561  complimentary bath doc
  rank 2  0.487  daycare hours doc
  rank 3  0.365  grooming booking doc
  rank 4  0.349  late fee doc
  rank 5  0.306  boarding closing times   ← the right doc ("weekends" = Saturday)

pool = top-3 → context = [bath, daycare, grooming] → LLM answer:

  "The last pickup time is 8 pm."        ← WRONG (that's daycare; boarding closes 5 pm)
```

The right document exists, says "5 pm on weekends", and never gets fetched — because
the question says **"Saturday"** and the document says **"weekends"**. No reranker can
fix that. This post fixes it the only way left: **change the question before searching.**
By the end, the same pipeline answers "5 pm" — and you'll have watched the failed
attempt in between, because the first thing everyone tries doesn't work.

We'll cover it in four parts:

1. **Setup** — the dialect gap: why the right question retrieves the wrong docs
2. **Multi-query** — translate the question, ask it several ways, pool the results
3. **HyDE** — search with a *fake answer* instead of the question (and its danger)
4. **Transforms in the real world** — the bill, the measured wins, and what eventually replaces them

## The one metaphor to hold onto: the tourist and the phrasebook

You're abroad and ask a local, "Where can I grab a cab?" Blank stare. The local phrase
is *taxi rank* — same intent, different dialect, zero match. Four rules:

1. **Retrieval matches phrasing, not intent.** Your users write questions in *user
   dialect* ("Saturday", "grab a cab", "my card got rejected"). Your documents are
   written in *author dialect* ("weekends", "taxi rank", "transaction declined").
   Embeddings bridge small gaps (post 2b) — but not all of them.
2. **Repeating yourself louder isn't translating.** Rephrasing the question in the
   *same* dialect retrieves the same wrong documents.
3. **A phrasebook asks several ways.** Translate the question into the corpus's dialect
   — a few variants — and pool what comes back. One of them will speak like the author.
4. **Or sketch the answer and match it.** Describe what the answer *would look like*
   and search with that — answers resemble answers. But a sketched answer contains
   *invented* details, and invented details are dangerous.

Rules 3 and 4 are the two techniques of this post: **multi-query** and **HyDE**.

## First attempt — naive paraphrasing (watch it fail)

*Phrasebook rule 2: repeating yourself louder isn't translating.*

The obvious move: ask an LLM to rephrase the question a few ways, retrieve for each,
merge. Here's what a generic "rewrite this question 3 ways" prompt produced, for real:

```
 - What is the latest pickup time for my dog on Saturdays?
 - By what time must my dog be collected on a Saturday?
 - What is the cutoff time for retrieving my dog on Saturdays?

retrieval per query (top-3 doc ids):
  [2, 7, 0]   ← original
  [2, 7, 8]   ← rewrite 1
  [2, 7, 0]   ← rewrite 2
  [2, 7, 8]   ← rewrite 3
```

Four beautifully varied sentences — every one still says *dog*, *Saturday*, *pickup*.
Same dialect, same embedding neighborhood, same wrong documents, four times. The fused
result is identical garbage to the baseline. **Paraphrasing is not translating.** The
rewrites have to *change the vocabulary*, not the word order.

## Multi-query, done right: translate into the corpus's dialect

*Phrasebook rule 3: ask several ways — in the author's words.*

The fix is in the rewrite prompt. Tell the LLM what dialect the corpus speaks:

```python
REWRITE_PROMPT = """You are helping search a pet care center's policy handbook.
Handbooks use general policy language: categories instead of specifics
(a specific day becomes "weekday" or "weekend", a pet becomes the service area
like "boarding" or "daycare", times become "closing time" or "opening hours").

Rewrite the question below 3 different ways as the HANDBOOK would phrase the
underlying policy. Generalize the specifics. One rewrite per line, no numbering.

Question: {q}"""
```

Real rewrites this produced:

```
 - Until what time can a pet be picked up on a weekend?
 - What is the latest pickup time for a service area on a weekend?
 - By what time must pickup be completed on a weekend?
```

**"Saturday" became "weekend."** That's the translation. Now retrieve top-3 for the
original *and* each rewrite, and watch the second rewrite do the thing no previous post
could:

```
retrieval per query (top-3 doc ids):
  [2, 7, 0]   ← original                                   (doc 1 nowhere)
  [2, 7, 0]   ← "…pet be picked up on a weekend?"
  [7, 8, 1]   ← "…latest pickup time … on a weekend?"      ← doc 1 WALKED IN
  [7, 8, 0]   ← "…pickup be completed on a weekend?"
```

One translated question put the boarding-hours document into its own top-3. Now merge
the four ranked lists — and you already own the right tool: **RRF from post 2c**, the
exact same `rrf_fuse`, because "merge several ranked lists by position" is the same
problem whether the lists come from two search engines or four phrasings:

```python
def multi_query_search(question, n=3, top_k=5):
    rewrites = rewrite(question, n)                       # 1 LLM call
    lists = [vector_search(q, 3) for q in [question] + rewrites]
    return rrf_fuse(lists, top=top_k)                     # post 2c, verbatim
```

```
fused pool (top-5):  [7, 0, 2, 8, 1]     ← doc 1 IS IN THE POOL
```

Rank 5 of 5 — no prize for style, but *present*, which is the one thing post 03 proved
money can't buy downstream. Now finish the pipeline: hand that pool to the LLM as
context (post 00), same question, same prompt:

```
context = baseline pool  [2, 7, 0]        → "The last pickup time is 8 pm."      WRONG
context = fused pool     [7, 0, 2, 8, 1]  → "On a Saturday, you can pick up
                                             your dog until 5 pm."               RIGHT
```

Read what happened in that second line, because it's subtle: the *embedding* never
figured out Saturday = weekend — the fused pool still ranks the right doc last. But the
**generating LLM** connects Saturday to "weekends" instantly once the document is in
front of it. Retrieval's job was never to rank it first; it was to *get it in the room*.
Query transforms fixed the pool, and the pool fixed the answer.

## HyDE: search with a fake answer

*Phrasebook rule 4: sketch the answer and match it.*

**HyDE** (Hypothetical Document Embeddings) flips the direction. Instead of translating
the question, ask the LLM to *write a fake answer document* — then embed the fake and
search with **its** vector. The logic: on the meaning-map, answers live near answers. A
made-up policy sentence lands closer to real policy sentences than a question does.

```python
def hyde_search(question, top_k=3):
    fake = llm(f"Write one sentence that could appear in a pet care center's "
               f"policy handbook and would answer this question. "
               f"Invent plausible specifics.\n\nQuestion: {question}")
    return vector_search_with_text(fake, top_k)     # embed the FAKE, not the question
```

The fake document it actually wrote:

```
"Saturday dog pick-ups are available until 6:00 p.m., and all pets not
 collected by then will be held for the next business day."
```

Study that sentence. Two things are true at once:

- **The style is right** — it reads exactly like a handbook line, and that's the trick
  working as designed.
- **The facts are invented.** There is no 6 p.m. rule. HyDE *manufactures a
  hallucination on purpose* and searches with it. On our corpus the fake still said
  "Saturday" (not "weekend"), so it retrieved the same wrong documents — the trick
  helps only when the fake happens to resemble the true document's wording.

And that invented "6:00 p.m." is the danger in one image: the fake is supposed to be a
*search key* and nothing more, but implementations that let it anywhere near the
generation context leak fabricated facts into answers. The course measured exactly
that (numbers below): HyDE matched multi-query on recall — and dragged faithfulness
from 0.909 down to 0.815. Grounding, poisoned by the search step itself.

## The bill

Transforms are the first technique in this series that costs **an LLM call per query,
every query, forever.** Measured:

```
baseline:    1 retrieval                  =     1 ms
multi-query: 1 LLM call + 4 retrievals    =   870 ms
```

Not 4× — **hundreds of ×**, because the LLM call dwarfs everything (retrieval was
always ~1 ms; the rewrite is ~all of the 870). You pay it on cache misses, on typos, on
every question nobody has asked before. Keep that number in mind for the last section.

## Query transforms in production

In my course code, multi-query is LlamaIndex's `QueryFusionRetriever` — generate N
rewrites, retrieve for each, RRF-fuse, all in one object:

```python
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [index.as_retriever(similarity_top_k=3)],
    num_queries=4,                    # 1 original + 3 LLM rewrites
    mode="reciprocal_rerank",         # RRF — post 2c's formula, again
    similarity_top_k=5,
)
```

Real output on our Saturday question (OpenAI embeddings this time):

```
  0.0664  doc 8: A late pickup fee of 15 dollars applies...
  0.0650  doc 7: Daycare drop off starts at 6:30 am...
  0.0492  doc 2: Dogs staying longer than three nights...
  0.0161  doc 1: The boarding facility closes at 7 pm on weekdays and 5 pm on...  ← in the pool
```

Same shape as our from-scratch run: the right doc rides the rewrites into the pool.
(HyDE is one import away too — `HyDEQueryTransform` — same trade-offs.)

And the course-scale measurements (full corpus, 50 golden questions, on top of
hybrid + rerank), which turn this post's demos into numbers:

- **Recall 0.73 → 0.78.** After *two straight posts* of recall stuck flat at 0.73 —
  rerank reordered, couldn't fetch — the pool finally moved. Post 03's diagnosis
  ("it's a pool problem") proven by the cure working.
- **Multi-query vs HyDE:** identical recall, 0.78 each. But HyDE dropped faithfulness
  0.909 → 0.815 — the fake-answer leak — while multi-query held it. The course chose
  multi-query.
- **Latency 2.56 s → 6.50 s.** The bill, at scale: rewrite generation plus four
  retrievals, per query. Recall got 5 points cheaper than they look.

One honest spoiler, because this series doesn't pretend its winners stay winners: the
*next* post's technique (contextual retrieval) closed the same vocabulary gap **at
index time — once per document instead of once per query** — and the course then
dropped multi-query entirely: same recall, better precision, latency back to 2.56 s.
Query transforms are the right tool when you can't touch the index; when you can,
fixing the documents beats fixing every question.

## The whole post in four lines

The phrasebook's rules, now with their engineering names:

1. **Retrieval matches phrasing, not intent** → the dialect gap: "Saturday" query,
   "weekends" document, rank 5, wrong answer.
2. **Rephrasing ≠ translating** → naive paraphrases kept the user's vocabulary and
   retrieved the same wrong docs four times.
3. **Ask several ways, pool the lists** → multi-query + RRF (post 2c's fuse, reused):
   one rewrite said "weekend", the doc walked into the pool, the answer flipped 8 pm →
   5 pm.
4. **Or sketch the answer** → HyDE: style-matched fake documents as search keys — equal
   recall, but invented specifics that cost the course 0.09 of faithfulness.

Translate the question, pool the results, and let the LLM read what finally got fetched.
That's query transforms.

*(Every snippet and every output block in this post was executed for real — rewrites and
answers from `gpt-5.4-mini` at temperature 0, retrieval `all-MiniLM-L6-v2` via FastEmbed,
fusion k = 60. LLM rewrites vary between runs — you may get different phrasings, and a
run where the fused pool ranks differ; the pattern (translated rewrite → doc enters pool
→ answer flips) is what reproduces. Course metrics are one measured run on my corpus.)*

---

**Next up: contextual retrieval** — this post fixed the question at query time, and paid
an LLM call per query forever. The next post fixes the *documents* at index time
instead: prepend each chunk with LLM-written context about what it discusses, once, and
the vocabulary gap closes before any question arrives. It's the technique that made the
course's multi-query obsolete — and produced its biggest single jump in quality.
