# Semantic cache: answer repeated questions in a millisecond

Five posts of quality work — chunking, hybrid search, reranking, transforms, stamps —
and the pipeline finally answers well. Now look at what each answer *costs*: an
embedding call, a vector search, an LLM call, one to three seconds, real money. And
then look at your query logs: half of your users are asking **questions someone already
asked, in different words.**

```
Q1: "What time does the boarding facility close on weekends?"
    MISS → full pipeline
    A: The boarding facility closes at 5 pm on weekends.
    time: 0.92 s

Q2: "How late is boarding open on Saturdays?"
    CACHE HIT (matched Q1 @ 0.816)
    A: The boarding facility closes at 5 pm on weekends.
    time: 0.001 s                                          ← 629× faster, $0
```

Different words, same question, same answer — served from memory in a millisecond,
zero LLM tokens. That's a **semantic cache**, this post's technique, and the last pure
component before this series goes agentic. As always: built from scratch, every number
real — including the section where the cache confidently serves a **wrong** answer,
because that failure mode is the whole design problem.

We'll cover it in four parts:

1. **Setup** — why exact-match caching fails, and the receptionist who remembers
2. **The cache from scratch** — ~20 lines, a real hit, a 629× speedup
3. **The threshold** — the collision line, and the overlap no threshold can fix
4. **In the real world** — measured economics, two war-story bugs, staleness

## The one metaphor to hold onto: the receptionist who remembers

A good office receptionist answers "where's the bathroom?" fifty times a day — and
stopped walking people there personally years ago. Four rules:

1. **Visitors repeat each other, in different words.** "Where's the restroom?" /
   "Which way to the toilets?" / "Bathroom?" — one question, many phrasings. An exact-
   text cache (a dict keyed on the query string) catches *none* of these repeats.
2. **She matches meaning, not wording.** You already own the tool for that: post 2b's
   map. Embed the incoming *question*, compare it to remembered questions by cosine.
   Note what's new: every previous post embedded *documents* to find answers — the
   cache embeds *questions* to find **other questions**.
3. **"Close enough" needs a line.** Too generous and she hands you the wrong printout
   — a *grooming* customer gets the *boarding* refund policy. Too strict and she walks
   everyone to the office anyway. That line is the threshold, and it's the entire
   design problem.
4. **She changes the economics, not the answers.** A hit costs a millisecond and
   nothing; a miss runs the full pipeline and remembers the result. Quality was set by
   posts 00–05; the cache decides how often you *pay* for it.

## The cache from scratch

Twenty lines. Store `(question, embedding, answer)` triples; on lookup, embed the new
question and take the best cosine against everything remembered:

```python
class SemanticCache:
    def __init__(self, threshold=0.80):
        self.threshold = threshold
        self.entries = []                      # (query_text, query_emb, answer)

    def lookup(self, question):
        if not self.entries:
            return None
        q = embed(question)
        score, best = max((cosine(q, e), (t, a)) for t, e, a in self.entries)
        if score >= self.threshold:
            return {"cached_q": best[0], "answer": best[1], "score": score}
        return None

    def store(self, question, answer):
        self.entries.append((question, embed(question), answer))

def cached_rag(question, cache):
    hit = cache.lookup(question)
    if hit:
        return hit["answer"]                   # 1 embedding, N cosines, NO pipeline
    answer = rag_answer(question)              # posts 00–05, the expensive path
    cache.store(question, answer)
    return answer
```

Run it and you get the hit from the top of this post: first question misses (0.92 s,
full pipeline, stored), the paraphrase matches at **0.816** and returns in **0.001 s**.
At scale you'd store the embeddings in the same vector database as everything else —
a cache lookup is just one more nearest-neighbor search (post 2b's ANN) — but the
logic never gets bigger than this.

## What the receptionist hears

The whole system rides on one number: the cosine between the new question and a
remembered one. Here's what our stored question actually "sounds like" to the embedder
— five incoming queries, real similarities:

```
stored: "What time does the boarding facility close on weekends?"

  0.871  "Boarding closing time on Sunday?"                ← paraphrase, hits at 0.80
  0.816  "How late is boarding open on Saturdays?"         ← paraphrase, hits at 0.80
  0.724  "When does dog boarding shut on the weekend?"     ← paraphrase… MISSES at 0.80
  0.454  "What are the daycare drop off hours?"            ← different question, safely far
  0.409  "How long do boarding refunds take?"              ← different question, safely far
```

Two things to read there. The good news: real paraphrases score high, unrelated
questions score low — the map works. The bad news is line three: a genuine paraphrase
at 0.724 **misses** at threshold 0.80. That's a *false miss* — it costs you a pipeline
run, not a wrong answer. Keep that asymmetry in mind; it decides everything below.

## The collision line — and the overlap no threshold can fix

Time for the honesty section, and this one has teeth. Post 05's refund twins come back
one last time — as *questions*:

```
cosine("How long do boarding refunds take?",
       "How long do grooming refunds take?")  =  0.698
```

Two different questions with two different answers (5 days vs 10), similarity 0.698.
Now sweep the threshold over four paraphrase pairs that SHOULD hit and four
different-question pairs that MUST NOT:

```
threshold | good hits | wrong hits
     0.70 |    2/4    |    0/4
     0.75 |    2/4    |    0/4
     0.80 |    1/4    |    0/4
     0.85 |    0/4    |    0/4
```

Safe everywhere — but look at how much it costs: at 0.80 we catch **one paraphrase in
four**. So loosen it? Here are the raw pair similarities, sorted:

```
  0.816  SHOULD hit   (Saturday hours paraphrase)
  0.775  SHOULD hit   (grooming booking paraphrase)
  0.698  MUST NOT hit (boarding vs grooming refunds)   ←
  0.658  SHOULD hit   (late fee paraphrase)            ←
  0.582  SHOULD hit   (free bath paraphrase)           ←
```

Stare at the arrows: a **must-not-hit pair scores higher than two should-hit pairs.**
The distributions *overlap*. With this embedder on this traffic there is **no threshold
that catches the late-fee paraphrase without also poisoning grooming refunds.** Drop to
0.65 to catch more paraphrases and here's what actually happens — real run:

```
stored: "How long do boarding refunds take?"
        A: Refunds for cancelled boarding are issued within 5 business days.

ask:    "How long do grooming refunds take?"
        CACHE HIT @ 0.698 (matched the boarding question)
        served: "Refunds for cancelled boarding are issued within 5 business days."
                 ^ WRONG — grooming refunds take 10 business days
```

A wrong answer, served instantly, with total confidence, from a component whose whole
job was *saving money* — and unlike an LLM hallucination it will repeat **identically,
for every user, until the entry is evicted.** This is why the threshold decision is
asymmetric: a false miss wastes one pipeline run; a false hit manufactures a
systematically repeating wrong answer. **When in doubt, set the line high and eat the
misses.** And measure the overlap on *your* traffic — the course's golden questions
were topically distinct, so its collision line sat comfortably low; ours has twins, so
it doesn't. The line is a property of your data, not of caching.

## Semantic cache in production

The course measured its cache on the full pipeline (post 05's best config underneath):
50 golden questions warmed into the cache, then 50 *LLM-generated paraphrases* fired at
it, threshold swept:

```
threshold   0.80    0.84    0.88    0.92
hit-rate    1.00    0.98    0.90    0.66
wrong hits     0       0       0       0
```

Zero wrong hits even at 0.80 — because, as just established, that's a property of that
topically-distinct question set, not a law. With 100% of paraphrases served from
cache, the end-to-end numbers:

- **Latency 2.62 s → 0.21 s (−92%).** The headline. A cache hit costs one embedding
  and a nearest-neighbor lookup.
- **Faithfulness 0.917 → 0.920 — held.** The cache serves answers the full pipeline
  produced; quality passes through.
- **Recall on cached answers actually *rose* (0.76 → 0.85)** — a lucky quirk worth
  understanding: the paraphrase gets served the *original* question's answer, and the
  original's phrasing retrieved better than the paraphrase's would have.
- **Answer relevancy dipped slightly (0.703 → 0.679)** — same quirk, other side: the
  served answer was phrased for the original question, not yours.

Two war-story bugs from building it, both instructive:

- **The self-collapsing warm-up.** First version warmed the cache by calling
  `answer()` on each golden question with a low threshold — and the questions started
  hitting *each other*: question 2 matched question 1's fresh entry, got served its
  answer, stored nothing. The "warmed" cache held **one entry**. Warming must store
  **unconditionally**, bypassing lookup.
- **The self-polluting eval.** Measuring hit-rates with a cache that stores on every
  miss means the *measurement mutates the cache* — later questions in the eval hit
  entries created by earlier ones. The fix is a `frozen` flag: during evaluation, look
  up but never store.

Both bugs are the same lesson: a cache is *state*, and state leaks into everything
that touches it unless you're explicit about when writes happen.

Two closing distinctions. First, don't confuse this with an **embedding cache** —
that one caches text→vector calls so you never pay to re-embed the same string (a good
explainer is linked in [resources](resources.md)); a semantic cache stores *answers*
keyed by question meaning. Complementary, not
competing. Second, **staleness**: a cached answer is frozen at the moment it was
generated. Update the handbook — boarding now closes at 6 pm — and the cache keeps
serving 5 pm, instantly and confidently, forever. Tie cache invalidation to corpus
version, or every document update quietly becomes a lie factory.

## The whole post in four lines

The receptionist's rules, now with their engineering names:

1. **Visitors repeat each other** → exact-match caches catch none of it; embed the
   *question* and match by meaning (post 2b's map, pointed at queries).
2. **Match meaning, not wording** → ~20 lines: store (question, embedding, answer);
   lookup = best cosine ≥ threshold. Hit: 0.001 s, 629× faster, $0.
3. **"Close enough" needs a line** → and the distributions can overlap: our must-not
   pair (0.698) outscored two true paraphrases. False miss = wasted pipeline run;
   false hit = a wrong answer that repeats identically forever. Set the line high;
   measure on your traffic.
4. **Economics, not answers** → course-measured: −92% latency, faithfulness held,
   zero wrong hits *on that distinct question set*. Warm unconditionally, freeze
   during eval, invalidate on corpus updates.

Remember the question, not the words. That's a semantic cache.

*(Every snippet and every output block in this post was executed for real — embeddings
`all-MiniLM-L6-v2` via FastEmbed, answers `gpt-5.4-mini` at temperature 0, timings one
machine and one run; your absolute numbers will differ, the 600×-order gap won't. The
wrong-answer serve at threshold 0.65 is a genuine unedited output. Course metrics are
one measured run on my corpus.)*

---

**Next up: agentic RAG** — the finale. Every post so far built one fixed pipeline:
retrieve, maybe rerank, generate, always the same moves in the same order. 2026-style
RAG doesn't work like that. It hands the *LLM* the tools — search, rerank, re-query,
even "answer from cache" — and lets it decide what to call, judge whether what came
back is enough, and try again when it isn't. The pipeline becomes a loop with a brain.
