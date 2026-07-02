# What is vector search? Semantic search explained from scratch

In the last post we built BM25 and watched it fail in one specific way: search `puppy`
and it will never find a document that only says `dog`. Different tokens, zero score,
end of story. Here's that failure, live, against the same five sentences we used last
time — the full BM25 scorer from the previous post, given a query with no exact word
overlap:

```
Query: 'puppy playing outside'
  doc 0: 0.000
  doc 1: 0.000
  doc 2: 0.000
  doc 3: 0.000
  doc 4: 0.000
```

Every score is zero. Two documents are literally about dogs playing outside, and BM25
cannot see them, because no document contains the exact strings `puppy`, `playing`, or
`outside`. This post builds the tool that fixes it: **vector search** — search by
*meaning*, not by matching words. And just like last time, we build it from scratch,
run every line, and read the real numbers.

We'll cover it in four parts:

1. **Setup** — where keyword search hits its wall, and the map-of-meaning idea
2. **The three ideas** — embeddings, cosine similarity, search as nearest neighbors
3. **Vector search assembled** — from-scratch code over the same five sentences, real rankings
4. **Vector search in the real world** — production models, searching at scale, and where *it* fails

## The one metaphor to hold onto: a map of meaning

Last post's detective matched literal clues. Vector search works differently — think of
a **giant map where every piece of text gets a location, and texts that mean similar
things live close together.** Three rules:

1. **Every sentence gets coordinates.** Turning text into its coordinates is called
   *embedding* it.
2. **Similar meaning = nearby on the map.** "A dog chased the cat" and "Dogs love to
   play fetch" sit in the same neighborhood; "She planted tomatoes" is across town.
3. **Search = drop the query onto the map and grab its nearest neighbors.** No word
   matching anywhere — just distance.

Hold onto the map. Every piece of math below is one of these rules made precise.

## Idea 1 — Embeddings: give every text coordinates

*Map rule 1: every sentence gets coordinates.*

Start tiny. Suppose we describe any text with just **two numbers**: how much it's about
*animals*, and how much it's about *plants*. Score a few words by hand:

```python
vec = {
    "dog":    [0.9, 0.1],   # very animal, barely plant
    "puppy":  [0.8, 0.1],   # very animal, barely plant
    "tomato": [0.1, 0.9],   # barely animal, very plant
}
```

A list of numbers like this is a **vector**, and each number is one **dimension**. Two
dimensions means we can draw it:

```
 plant-ness
    1.0 ┤          ● tomato [0.1, 0.9]
        │
        │
        │
        │
    0.1 ┤                        ● puppy [0.8, 0.1]  ● dog [0.9, 0.1]
        └──────────────────────────────────────────────
         0.1                    0.8   0.9      animal-ness →
```

Look at the picture: `dog` and `puppy` land almost on top of each other, `tomato` is far
away. **Position captures meaning.** That's the entire trick.

An **embedding** is exactly this — a vector that captures what a text *means* — except:

- Nobody scores the numbers by hand. A **neural network** does it, trained on billions of
  sentences until texts that appear in similar contexts get similar coordinates.
- Two dimensions can't capture much. Real models use hundreds or thousands — same idea,
  just a map we can no longer draw.

Here's a real one. FastEmbed (the same library that gave us BM25 sparse vectors last
post) can also run dense embedding models locally — here, the small classic
`all-MiniLM-L6-v2`:

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

emb = list(model.embed(["A dog chased the cat around the yard"]))[0]
print("length:", len(emb))
print("first 6 numbers:", [round(float(v), 3) for v in emb[:6]])
```

Real output:

```
length: 384
first 6 numbers: [0.068, -0.0, 0.021, 0.028, 0.023, -0.035]
```

One sentence in, **384 numbers** out — its coordinates on a 384-dimensional map. Unlike
our hand-made axes, no single dimension means anything readable ("dimension 217 = 0.021"
tells you nothing alone). The meaning lives in the *position as a whole*, and the only
thing we'll ever do with these numbers is compare positions. Which needs Idea 2.

## Idea 2 — Cosine similarity: measure closeness

*Map rule 2: similar meaning = nearby.*

We need one number that says how close two vectors are. The standard choice is **cosine
similarity**: it measures whether two vectors **point in the same direction** from the
origin.

- **+1** — same direction, same meaning
- **0** — unrelated (perpendicular)
- **−1** — opposite direction

The from-scratch version is three lines:

```python
import numpy as np

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Read it piece by piece, with numbers small enough to do in your head. Take two vectors
that point the **same way** but have different sizes — `b` is exactly `a` doubled:

```python
a = [3, 4]
b = [6, 8]      # = a, doubled — same direction, twice as long
```

**Step 1 — `np.dot`: multiply slot by slot, add it up.**

```
dot(a, b) = 3×6 + 4×8 = 18 + 32 = 50
```

The dot product is big when the vectors rise and fall together — slot 1 big where slot 1
is big, slot 2 big where slot 2 is big. But `50` on its own is ambiguous: is it big
because the vectors *agree*, or just because they're *long*? We need to strip the length
out.

**Step 2 — `np.linalg.norm`: the length of a vector.** Plain Pythagoras — a vector
`[3, 4]` is an arrow 3 across and 4 up, so its length is:

```
norm(a) = √(3² + 4²) = √25 = 5
norm(b) = √(6² + 8²) = √100 = 10
```

**Step 3 — divide, and the length cancels.**

```
cosine(a, b) = 50 / (5 × 10) = 1.0     ← same direction, perfect score
```

Now watch *why* the division kills magnitude. Double `b` again to `[12, 16]`:

```
dot(a, [12,16]) = 3×12 + 4×16 = 100    ← dot DOUBLED (longer vector)
norm([12,16])   = 20                    ← length DOUBLED too
cosine          = 100 / (5 × 20) = 1.0  ← unchanged
```

Make a vector twice as long and its dot product doubles — but so does its norm, and the
division cancels the two exactly. Whatever survives the division is the part length
can't touch: **direction**. And direction is where the meaning lives — we don't want a
document to score higher just because its vector happens to be "longer".

One more, for the "unrelated" case — `c = [4, -3]` is perpendicular to `a`:

```
dot(a, c) = 3×4 + 4×(−3) = 12 − 12 = 0  → cosine = 0, no agreement
```

That's the whole function: agreement (dot), divided by the two lengths (norms), leaving
pure direction. Run it on our hand-made toy vectors first:

```python
cosine(vec["dog"], vec["puppy"])    # -> 1.000
cosine(vec["dog"], vec["tomato"])   # -> 0.220
```

Exactly what the picture showed: `dog` and `puppy` point the same way (1.000 — well,
0.9997 before rounding), `tomato` points somewhere else entirely (0.220). Now the real
test — the same words through the **real 384-dimensional model**:

```python
words = ["puppy", "dog", "tomato", "car", "automobile"]
wembs = dict(zip(words, model.embed(words)))
# wembs maps each word to its 384-number vector:
#   wembs = {
#     "puppy": [-0.08, 0.035, 0.0, ...381 more],
#     "dog":   [...384 numbers],
#     ...
#   }
cosine(wembs["puppy"], wembs["dog"])        # ?
cosine(wembs["puppy"], wembs["tomato"])     # ?
cosine(wembs["car"],   wembs["automobile"]) # ?
cosine(wembs["car"],   wembs["tomato"])     # ?
```

Real output:

```
cosine('puppy'     , 'dog'       ) = 0.804
cosine('puppy'     , 'tomato'    ) = 0.342
cosine('car'       , 'automobile') = 0.865
cosine('car'       , 'tomato'    ) = 0.364
```

Read those numbers — this is the moment keyword search could never reach:

- `puppy` / `dog` → **0.804**. Different strings, zero shared characters that matter —
  and the model *knows they're nearly the same thing*, because it has seen them used the
  same way billions of times.
- `car` / `automobile` → **0.865**. The classic synonym pair, solved.
- Cross-topic pairs (`puppy`/`tomato`, `car`/`tomato`) → ~0.35. Far apart, as they
  should be.

No dictionary of synonyms anywhere. The *positions* encode it.

## Idea 3 — Search = nearest neighbors on the map

*Map rule 3: drop the query onto the map, grab the closest documents.*

Now assemble search itself, and it's almost nothing: embed every document once, embed
the query, rank by cosine. Here's the whole pipeline:

```
query: "puppy playing outside"
   │  embed (one vector, 384 numbers)
   ▼
[0.016, 0.024, 0.066, ...384 numbers]
   │  cosine against every document's vector
   ▼
doc 0: 0.221   doc 1: 0.335   doc 2: 0.423   doc 3: 0.169   doc 4: 0.082
   │  sort by similarity
   ▼
1.  (0.423)  "Dogs are loyal and love to play fetch in the park"
2.  (0.335)  "A dog chased the cat around the yard"
```

Same five sentences as last post, full from-scratch implementation:

```python
corpus = [
    "The cat sat on the warm windowsill in the sun",   # doc 0
    "A dog chased the cat around the yard",             # doc 1
    "Dogs are loyal and love to play fetch in the park",  # doc 2
    "The park has a pond where ducks swim every morning",  # doc 3
    "She planted tomatoes and basil in her garden",     # doc 4
]

doc_embs = list(model.embed(corpus))          # embed every document ONCE

def vector_search(query):
    q = list(model.embed([query]))[0]         # embed the query
    scored = [(cosine(q, e), corpus[i]) for i, e in enumerate(doc_embs)]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored

for q in ["puppy playing outside", "growing vegetables at home", "waterbirds"]:
    print(f"\nQuery: {q!r}")
    for score, text in vector_search(q):
        print(f"  {score:.3f}  {text}")
```

Real output:

```
Query: 'puppy playing outside'
  0.423  Dogs are loyal and love to play fetch in the park
  0.335  A dog chased the cat around the yard
  0.221  The cat sat on the warm windowsill in the sun
  0.169  The park has a pond where ducks swim every morning
  0.082  She planted tomatoes and basil in her garden

Query: 'growing vegetables at home'
  0.416  She planted tomatoes and basil in her garden
  0.190  The park has a pond where ducks swim every morning
  0.043  A dog chased the cat around the yard
  0.022  Dogs are loyal and love to play fetch in the park
  -0.066  The cat sat on the warm windowsill in the sun

Query: 'waterbirds'
  0.394  The park has a pond where ducks swim every morning
  0.161  Dogs are loyal and love to play fetch in the park
  0.126  The cat sat on the warm windowsill in the sun
  0.118  She planted tomatoes and basil in her garden
  -0.045  A dog chased the cat around the yard
```

## What the results teach us

Every query above is one BM25 would have scored **0.000 across the board** — not a
single query word appears in a single document. Read what the map did instead:

- **`puppy playing outside`** → "Dogs are loyal and love to **play fetch in the park**"
  first, "A **dog** chased the cat around the **yard**" second. It connected `puppy`→dogs,
  `playing`→play/fetch, `outside`→park/yard. Three synonym hops at once, no synonym list.
- **`growing vegetables at home`** → the garden document, by a huge margin (0.416 vs
  0.190 for second place). `tomatoes` and `basil` are *kinds of* things you grow;
  `garden` is *at home*. That's world knowledge baked into positions.
- **`waterbirds`** — one word, appearing nowhere — → "ducks swim in a pond." The model
  knows a duck is a waterbird.
- Note the **negative scores** (−0.066, −0.045): vectors pointing slightly *away* from
  each other. In practice read anything near or below zero as "unrelated."

One more real measurement — the pairwise similarity between all five *documents*:

```
doc0 vs doc1: 0.333    doc1 vs doc2: 0.317    doc2 vs doc3: 0.268
doc0 vs doc2: 0.195    doc1 vs doc3: 0.058    doc2 vs doc4: 0.071
doc0 vs doc3: 0.087    doc1 vs doc4: 0.133    doc3 vs doc4: 0.126
doc0 vs doc4: 0.187
```

Sketch those numbers as the map (2-D flattening of 384-D, positions approximate):

```
   ┌────────────────────────────────────────────────────┐
   │   the pet neighborhood                             │
   │    ● doc0 cat / windowsill                         │
   │        ● doc1 dog chased cat        outdoors       │
   │            ● doc2 dogs / fetch / park              │
   │                       ● doc3 ducks / pond / park   │
   │                                                    │
   │                                  ● doc4 garden     │
   │                                    (across town)   │
   └────────────────────────────────────────────────────┘
```

The cat and dog documents cluster (0.333, 0.317), the park connects doc2 to doc3
(0.268), and the garden sits far from everything (0.07–0.19). The corpus *organized
itself by meaning* — nobody told it cats and dogs go together.

## Searching the map at scale

*Map rule 4 — the one we haven't needed yet: at scale, don't visit every address.*

Our `vector_search` compares the query against **every** document. Five documents, fine.
Now do the real-world math: a billion documents × 1536 dimensions ≈ **1.5 trillion
multiply-adds per query**. Brute force dies at scale.

The fix is **Approximate Nearest Neighbor (ANN)** search — organize the map *before*
queries arrive so you can find near-neighbors without checking every point. The most
popular structure, **HNSW**, works like navigating a city: take highways to get to
roughly the right district fast, then drop down to main roads, then local streets for
the final answer. You check thousands of points instead of a billion, and find the true
nearest neighbor ~95–99% of the time — a tiny accuracy trade for a ~1000× speedup.

That's a whole topic of its own — a deep dive is linked in [resources](resources.md).
For us, the practical takeaway: **a vector database exists to do exactly this.** In my
course code, Qdrant stores every document's embedding and runs HNSW under the hood —
you call `search`, the map machinery is invisible.

## Vector search in production

Just like you won't hand-roll BM25, you won't hand-roll this loop. Two things change in
production:

- **A stronger model.** Our demo used `all-MiniLM-L6-v2` (384 dims, runs locally, free).
  My course code uses OpenAI's `text-embedding-3-small` (1536 dims, via API) — same
  idea, better map. Swapping models means re-embedding everything: coordinates from
  different models live on **different maps** and can't be compared.
- **A vector database does storage + ANN.** Embed once at index time, store in Qdrant
  (or similar), and each query costs one embedding call plus one ANN lookup.

Last post we called BM25's output a **sparse** vector — thousands of slots, almost all
zero. Today's embedding is its mirror twin: a **dense** vector — only 384 slots, *none*
of them zero. Same database stores both, side by side. Remember that; it's the whole
setup for the next post.

## Where vector search fails — the detective's revenge

Time to be honest, the way we were honest about BM25. Vector search has its own blind
spot, and it's exactly where BM25 shines: **exact identifiers.**

Three documents, two of them near-twins differing only in an error code:

```python
idcorpus = [
    "Error E-4042 refund transaction declined by the payment gateway",
    "Error E-4043 refund transaction succeeded but receipt email failed",
    "How refunds work a general overview of the refund process",
]
```

First, how similar does the model think the two error documents are?

```
doc A (E-4042) vs doc B (E-4043): 0.780
```

**0.780** — to the map, these are nearly the same sentence. The one character that
distinguishes them (`2` vs `3`) barely moves the position. Now search for a specific
code:

```
Query: 'error E-4042'
  0.523  Error E-4043 refund transaction succeeded but receipt email failed
  0.504  Error E-4042 refund transaction declined by the payment gateway
  -0.030  How refunds work a general overview of the refund process
```

**The wrong document wins.** We asked for E-4042 and got E-4043 — vector search found
the right *neighborhood* (error docs, not the overview) but picked the wrong *house*,
because identifiers are noise to a meaning-map. This is the exact failure from the
opening of the last post — now you've watched it happen, with real numbers.

And you already know the tool that gets this right: BM25 gives `e-4042` a huge IDF and
lands on the literal match every time. So:

| Query looks like | Winner |
|---|---|
| `puppy playing outside` (meaning, synonyms) | **Vector search** |
| `error E-4042` (exact identifier) | **BM25** |

Two searchers, opposite blind spots. The obvious question — *why not run both?* — is
the next post.

## The whole post in four lines

The map's rules, now with their math names:

1. **Every text gets coordinates** → an *embedding*: a dense vector (384–1536+ numbers)
   from a trained model.
2. **Similar meaning = nearby** → *cosine similarity*: direction match, +1 same, ~0
   unrelated.
3. **Search = nearest neighbors** → embed docs once, embed the query, rank by cosine.
4. **At scale, take highways** → *ANN* (HNSW) inside a vector database, trading ~1–5%
   recall for ~1000× speed.

Embed everything, measure direction, return the closest. That's vector search.

*(Every snippet and every output block in this post was executed for real — the numbers
come from `sentence-transformers/all-MiniLM-L6-v2` run locally via FastEmbed. A different
model gives different scores, but the behavior — synonyms close, identifiers confused —
is what to expect from any dense embedding model.)*

---

**Next up (2c): Hybrid search** — BM25 caught the identifier vector search missed;
vector search caught the synonyms BM25 missed. Run **both in one query**, fuse the
rankings, and measure whether the union really beats either alone.
