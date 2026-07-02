# What is hybrid search? BM25 + vectors, fused from scratch

Two posts, two searchers, two opposite blind spots:

- **BM25** (post 2a) found `error E-4042` instantly — but scored a flat **0.000** for
  `puppy playing outside` against documents literally about dogs playing outside. It
  matches words, not meaning.
- **Vector search** (post 2b) nailed `puppy` → dog — but asked for `E-4042`, it ranked
  the **wrong error document first**. Identifiers are noise to a meaning-map.

| Query looks like | BM25 | Vector |
|---|---|---|
| `puppy playing outside` (synonyms, meaning) | ✗ nothing | ✓ |
| `E-4042` (exact identifier) | ✓ | ✗ wrong doc |

Real users send you *both* kinds of query, all day, interleaved. You don't get to pick a
favorite. **Hybrid search** runs both searchers on every query and fuses their rankings —
and in this post we build the fusion from scratch, run it, and watch each searcher rescue
the other, with real numbers.

We'll cover it in four parts:

1. **Setup** — two experts, one committee, and a corpus with both kinds of query
2. **Two steps to fusion** — Step 1: add the scores (fails). Step 2: score by position (RRF)
3. **Hybrid assembled** — from-scratch code, two rescue missions, hand-checked math
4. **Hybrid in the real world** — real Qdrant hybrid code, and when hybrid *loses*

## The one metaphor to hold onto: a committee of two experts

Post 2a gave us a **detective** who matches literal clues. Post 2b gave us a **map guide**
who navigates by meaning. Hybrid search puts them on a committee. Four rules:

1. **Ask both experts, independently.** Same query to each; neither sees the other's work.
2. **Each returns a ranked shortlist.** Best candidate first.
3. **Never compare their raw scores.** The detective scores in "clue points," the guide in
   "map closeness" — different units. Only compare *positions* on each list.
4. **A document high on both lists wins.** Found by only one expert? Still counts — just
   less.

Those four rules are the whole algorithm. Rules 3 and 4 become the two steps below: first
we'll try the naive fusion and watch it break rule 3, then build the fusion that follows it.

## Our corpus: everyday sentences + a support-ticket wing

Same five sentences as the last two posts, plus four new documents that look like real
support content — including three near-twins that differ only in an error code (exactly
the trap vector search fell into last post):

```python
corpus = [
    "The cat sat on the warm windowsill in the sun",                       # doc 0
    "A dog chased the cat around the yard",                                # doc 1
    "Dogs are loyal and love to play fetch in the park",                   # doc 2
    "The park has a pond where ducks swim every morning",                  # doc 3
    "She planted tomatoes and basil in her garden",                        # doc 4
    "Error E-4042 refund transaction declined by the payment gateway",     # doc 5
    "Error E-4043 refund transaction succeeded but receipt email failed",  # doc 6
    "Error E-4044 refund transaction pending manual review",               # doc 7
    "How refunds work a general overview of the refund process",           # doc 8
]
```

Both searchers are exactly the code from the previous posts — the from-scratch BM25
(`bm25_search`, returning only documents with score > 0) and the from-scratch vector
search (`vector_search`, cosine over `all-MiniLM-L6-v2` embeddings). Run each alone on
our two problem queries. Real output:

```
Query: 'E-4042'
BM25:
  1.908  doc 5: Error E-4042 refund transaction declined by the payment gateway
Vector:
  0.468  doc 6: Error E-4043 refund transaction succeeded but receipt email failed
  0.465  doc 5: Error E-4042 refund transaction declined by the payment gateway
  0.451  doc 7: Error E-4044 refund transaction pending manual review

Query: 'puppy playing outside'
BM25:
  (nothing — every score 0.000)
Vector:
  0.423  doc 2: Dogs are loyal and love to play fetch in the park
  0.335  doc 1: A dog chased the cat around the yard
  0.221  doc 0: The cat sat on the warm windowsill in the sun
```

Read the two failures closely, because hybrid has to fix both:

- **`E-4042`, vector:** the wrong twin wins by a whisker — 0.468 vs 0.465. To the
  meaning-map the three error docs are the same neighborhood, and the one character that
  matters can't move the needle. Meanwhile **BM25 retrieves exactly one document** — the
  only one containing the literal token `e-4042`. Rare token, huge IDF, laser focus.
- **`puppy playing outside`, BM25:** empty-handed — no query word appears anywhere. The
  vector guide, of course, walks straight to the dog documents.

## Step 1 — Try the obvious fusion: add the scores. Watch it fail.

*This is the step that breaks committee rule 3.*

The first fusion anyone thinks of:

```
final(doc) = bm25_score(doc) + cosine(doc)
```

Looks reasonable. Now look at the real score ranges from our own runs:

```
BM25 top score for 'error E-4042'   : 2.963
BM25 top score for 'garden tomatoes': 4.015     ← same algorithm, different query!
Vector top score  (cosine, always)  : ≤ 1.0
```

See the problem forming: **BM25's whole opinion lives on a 0-to-4-ish scale (and it's
unbounded — bigger corpus, bigger scores), while vector search's whole opinion lives
inside 0-to-1.** Now watch the addition go wrong. Suppose the two experts *disagree*
about two documents:

```
              BM25 says      vector says      naive sum
doc X:          2.9      +      0.10        =    3.00   ← wins
doc Y:          1.2      +      0.99        =    2.19
```

Vector search is *screaming* that doc Y is the answer — 0.99 is nearly its maximum
possible score. BM25 is lukewarm on doc X. But 2.9 is numerically bigger than 0.99 can
ever compensate for, so **the louder unit wins, not the righter expert.** Adding meters
to dollars.

Can't we rescale? Divide BM25 by its max? The max *moves* — 2.963 for one query, 4.015
for the next, something else on your corpus. There is no constant that lines the two
scales up everywhere. Score fusion is a dead end.

So Step 2 throws the raw scores away entirely, and keeps the one thing that means the
same in any unit system: **the order**. "My #1" is "my #1" whether you score in clue
points or map closeness.

## Step 2 — The fix: score by position (Reciprocal Rank Fusion)

*Committee rules 3 + 4: compare positions, reward documents high on both lists.*

**Reciprocal Rank Fusion (RRF)** gives a document points from each list based on its
*position* there, then sums:

```
RRF(doc) = Σ over lists  1 / (k + rank)        (k = 60, rank starts at 1)
```

Do the head math for one list (k = 60):

```
rank 1  →  1 / (60 + 1)  = 1/61 ≈ 0.01639
rank 2  →  1 / (60 + 2)  = 1/62 ≈ 0.01613
rank 3  →  1 / (60 + 3)  = 1/63 ≈ 0.01587
rank 10 →  1 / (60 + 10) = 1/70 ≈ 0.01429
```

Not on a list at all → 0 points from that list.

**Why the k = 60?** It's a softener. Without it (k = 0), rank 1 would earn `1/1 = 1.0`
and rank 2 only `1/2 = 0.5` — the top spot would crush everything, and one expert's #1
would be nearly unbeatable. With k = 60, rank 1 earns 0.01639 and rank 2 earns 0.01613 —
a nudge, not a knockout. The constant comes from the original RRF paper, works well
everywhere, and nobody tunes it much.

Now the payoff property, straight from the arithmetic: **appearing on two lists roughly
doubles your points.**

```
on BOTH lists, ranks 1 and 2:  1/61 + 1/62 ≈ 0.0325
on ONE list, rank 1:           1/61        ≈ 0.0164
```

An okay-for-both-experts document beats a loved-by-one document. That's rule 4, made of
fractions.

### Watch RRF fuse two real lists, by hand

Before any code, run the whole algorithm on paper — using the *actual* ranked lists our
two searchers produced for `E-4042` earlier in this post:

```
BM25's list  : [doc 5]                      (it retrieved exactly one document)
Vector's list: [doc 6, doc 5, doc 7, doc 8, doc 3]
```

Go document by document. Each one collects points from every list it appears on, and
nothing from lists it's missing from:

```
        from BM25's list        from vector's list      TOTAL
doc 5   rank 1 → 1/61 = 0.01639  rank 2 → 1/62 = 0.01613  0.03252
doc 6   not on it   →  0         rank 1 → 1/61 = 0.01639  0.01639
doc 7   not on it   →  0         rank 3 → 1/63 = 0.01587  0.01587
doc 8   not on it   →  0         rank 4 → 1/64 = 0.01562  0.01562
doc 3   not on it   →  0         rank 5 → 1/65 = 0.01538  0.01538
```

Sort by total, and the fused ranking is:

```
1. doc 5  (0.03252)   ← the RIGHT answer
2. doc 6  (0.01639)   ← vector search's wrong pick, demoted to second
3. doc 7, then doc 8, then doc 3
```

Stare at the doc 5 vs doc 6 rows until they click, because this is the entire magic of
hybrid search in two lines of arithmetic:

- **doc 6** was vector search's *favorite* — rank 1! — but only ONE expert vouched for
  it. One list, one fraction: 0.01639.
- **doc 5** was nobody's perfect pick on the vector side (rank 2) — but BOTH experts put
  it high. Two lists, two fractions: 0.01639 + 0.01613 = 0.03252. **Two hands beat one.**

That's committee rule 4 doing the rescue. No score from BM25 (1.908) or cosine (0.465)
ever entered the arithmetic — only positions did.

## Hybrid search from scratch

Now the code is just the table above, automated. Ask both searchers (rule 1), take each
one's top-5 ids in order (rule 2), award points per position and sum (rules 3 + 4):

```python
from collections import Counter

def rrf_fuse(rankings, k=60, top=4):
    # points = the running totals column from the hand-worked table:
    #   {doc_id: sum of 1/(60+rank) across lists}
    points = Counter()
    for ranking in rankings:                 # rankings = [bm25's id list, vector's id list]
        for rank, doc_id in enumerate(ranking, start=1):   # rank 1 = first on the list
            points[doc_id] += 1 / (k + rank)
    return points.most_common(top)

def hybrid_search(query, top=4):
    bm25_ids = [i for score, i in bm25_search(query)[:5]]    # detective's shortlist
    vec_ids  = [i for score, i in vector_search(query)[:5]]  # map guide's shortlist
    return rrf_fuse([bm25_ids, vec_ids], top=top)
```

One line deserves a note: **`points.most_common(top)`**. `Counter` is usually used for
counting occurrences, and `most_common` normally means "highest counts first." Here we've
repurposed it as a points ledger, so `most_common(4)` simply means **"the 4 documents
with the highest point totals, sorted best-first"** — it returns `(doc_id, total)` pairs,
exactly the sorted TOTAL column of our table. Nothing is being "counted."

The whole pipeline:

```
                     query: "E-4042"
                    ┌───────┴────────┐
                    ▼                ▼
              BM25 (3a)        vector search (2b)
                    │                │
              ranked ids       ranked ids
                 [5]           [6, 5, 7, 8, 3]
                    └───────┬────────┘
                            ▼
                    RRF: points by position
                            ▼
              doc 5: 0.03252   ← final ranking
              doc 6: 0.01639
              doc 7: 0.01587
```

Run it on both problem queries. Real output — each result shows the two shortlists that
went *in*, then the fused ranking that came *out*:

```
Query: 'E-4042'
  BM25 ranked ids  : [5]                the detective's shortlist
  Vector ranked ids: [6, 5, 7, 8, 3]    the map guide's shortlist
  ── fused ──
  0.03252  doc 5: Error E-4042 refund transaction declined by the payment gateway
  0.01639  doc 6: Error E-4043 refund transaction succeeded but receipt email failed
  0.01587  doc 7: Error E-4044 refund transaction pending manual review
```

This is *exactly* the table we hand-worked a page ago — same lists in, same totals out
(0.03252, 0.01639, 0.01587). The code confirms the paper. The right document wins even
though vector search had it second, because it's the only document on **both** lists.

```
Query: 'puppy playing outside'
  BM25 ranked ids  : []                  the detective found NOTHING (all scores 0)
  Vector ranked ids: [2, 1, 0, 3, 4]
  ── fused ──
  0.01639  doc 2: Dogs are loyal and love to play fetch in the park
  0.01613  doc 1: A dog chased the cat around the yard
  0.01587  doc 0: The cat sat on the warm windowsill in the sun
```

Trace the points for this one — with BM25's list empty, every document collects from the
vector list alone:

```
doc 2: vector rank 1 → 1/61 = 0.01639     (+ nothing from BM25)
doc 1: vector rank 2 → 1/62 = 0.01613     (+ nothing from BM25)
doc 0: vector rank 3 → 1/63 = 0.01587     (+ nothing from BM25)
```

The fused ranking is just the vector ranking, untouched. **Both failures fixed** — one
by the two-hands-beat-one rescue, one by graceful fallback.

## What the results teach us

- **The `E-4042` rescue.** Vector search put the wrong twin first — but BM25's laser
  retrieval (one document, the literal token match) meant the right doc scored from
  *both* lists while the imposter scored from one. Two mediocre-looking fractions beat
  one good one: 0.03252 > 0.01639. Committee rule 4, live.
- **The `puppy` rescue.** BM25 came back empty, so its list contributed nothing — and
  fusion **degraded gracefully into pure vector search**. Nothing broke, no special
  case: an empty list just awards no points. When one expert shrugs, the other's opinion
  simply stands.
- **Notice what we never did:** compare 1.908 (BM25's score) with 0.465 (a cosine).
  Raw scores never crossed the searcher boundary — only ranks did.

## Hybrid search in production: Qdrant does all of this in one query

You already know both halves ship in one database. From post 2a: Qdrant stores each
document's **sparse** BM25 vector. From post 2b: it stores the **dense** embedding and
does ANN search over it. Hybrid is those two side by side in the same collection, fused
server-side with RRF — the same formula we just wrote. Real code, real output
(`pip install "qdrant-client[fastembed]"`):

```python
from qdrant_client import QdrantClient, models

DENSE  = "sentence-transformers/all-MiniLM-L6-v2"   # post 2b's model
SPARSE = "Qdrant/bm25"                              # post 2a's model

client = QdrantClient(":memory:")                   # real server: QdrantClient(url=...)

# one collection, BOTH vector types
client.create_collection(
    collection_name="hybrid_demo",
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

# each document stored twice-in-one: its dense embedding AND its sparse BM25 vector
client.upsert(
    collection_name="hybrid_demo",
    points=[
        models.PointStruct(
            id=i,
            vector={
                "dense":  models.Document(text=t, model=DENSE),
                "sparse": models.Document(text=t, model=SPARSE),
            },
            payload={"text": t},
        )
        for i, t in enumerate(corpus)
    ],
)

# one query = ask both experts (prefetch), fuse with RRF — our whole post, server-side
hits = client.query_points(
    collection_name="hybrid_demo",
    prefetch=[
        models.Prefetch(query=models.Document(text="E-4042", model=DENSE),  using="dense",  limit=5),
        models.Prefetch(query=models.Document(text="E-4042", model=SPARSE), using="sparse", limit=5),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)
```

Real output, both problem queries:

```
Query: 'E-4042'
  0.8333  Error E-4042 refund transaction declined by the payment gateway   ← right doc first
  0.7500  Error E-4043 refund transaction succeeded but receipt email failed
  0.5833  Error E-4044 refund transaction pending manual review

Query: 'puppy playing outside'
  1.0000  Dogs are loyal and love to play fetch in the park                 ← right doc first
  0.3333  A dog chased the cat around the yard
  0.2500  The cat sat on the warm windowsill in the sun
```

Same rescues as our from-scratch run. Map the code to the committee: the two `Prefetch`
entries are "ask both experts, top-5 shortlists each" (rules 1 + 2), and
`FusionQuery(fusion=models.Fusion.RRF)` is the points-by-position table (rules 3 + 4).

One delightful detail in the `puppy` result: the winner scored **1.0000**, which in
Qdrant's RRF means *both* lists had it at rank 1 — production BM25 wasn't empty-handed
like ours! Why? Qdrant's BM25 applies **stemming** (post 2a mentioned it): `playing` is
reduced to `play`, which literally appears in doc 2. Our bare-bones tokenizer couldn't
make that hop; a production one can. The fallback rescue still matters — `puppy` and
`outside` matched nothing — but it's a nice reminder that real BM25 is a slightly softer
detective than our from-scratch one.
The absolute scores differ from ours (0.8333 vs 0.03252) because Qdrant's RRF uses a
smaller k than 60 — a different softener, same recipe. Rank fusion only promises an
*order*, and the order is what matches.

In my course code this same setup hides behind two LlamaIndex settings
(`enable_hybrid=True`, `fastembed_sparse_model="Qdrant/bm25"`) — what you now know is
that those flags create exactly this collection and run exactly this two-prefetch,
RRF-fused query.

## Is hybrid always better? Measure it

Honesty section, as always. Hybrid's promise is **robustness across a mixed query
stream** — not a guaranteed win on every metric for every corpus. One measured example
from my course: on an identifier-heavy corpus (lots of form numbers and codes — BM25's
home turf), BM25 *alone* hit recall ≈ 0.85 while hybrid landed ≈ 0.73: fusing in the
weaker dense results *diluted* the specialist. Hybrid still won precision (≈ 0.85) —
fewer irrelevant chunks — but if I had shipped hybrid assuming "fusion always wins," I'd
have shipped a recall regression.

Rules of thumb, from that experience:

- **Mixed real-world queries (names + codes + natural language)** → hybrid is the safe
  default. Each expert covers the other's blind spot, as this post demonstrated.
- **Corpus and queries that live entirely on one expert's turf** → the specialist alone
  may beat the committee. Fusion averages opinions; averaging in a worse opinion costs.
- Either way: **measure on your own data.** Recall and precision per method, side by
  side, before choosing.

## The whole post in four lines

The committee's rules, now with their math names:

1. **Ask both experts** → run BM25 and vector search on every query, independently.
2. **Ranked shortlists** → keep each searcher's top-k ids, in order.
3. **Never compare raw scores** → BM25 is unbounded, cosine is ≤ 1; only *ranks* are
   comparable across searchers.
4. **High on both lists wins** → RRF: `Σ 1/(60 + rank)` per list; two appearances ≈
   double points; absent = zero.

Two searchers, opposite blind spots, one rank-based vote. That's hybrid search.

*(Every snippet and every output block in this post was executed for real — BM25 and
vector search are the exact from-scratch implementations of posts 2a and 3b, embeddings
from `all-MiniLM-L6-v2` run locally via FastEmbed, k = 60; the production snippet ran
against an in-memory Qdrant (`QdrantClient(":memory:")`). Different embedding models
shuffle the cosine scores, but the rescue pattern — and the reason ranks fuse where
scores can't — holds everywhere.)*

---

**Next up: reranking** — hybrid search now retrieves a good *pool* of candidates, but
the order within that pool is still rough. Next we add a slower, smarter model — a
cross-encoder — that reads the query and each candidate *together* and re-sorts the pool,
so the best answer doesn't just make the list, it tops it.
