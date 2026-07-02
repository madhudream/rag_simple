# Chunk smart: how you cut your documents decides what your RAG can answer

In the last post we built naive RAG over a handbook of **one-line documents**. That was
a quiet cheat. Real documents are pages — policies, manuals, contracts — and before any
of them can be embedded and searched, someone has to decide how to **cut them into
pieces**. That decision is called **chunking**, it happens before a single query runs,
and this post shows it flipping a RAG system's answer from right to wrong:

```
Q: I booked boarding for a public holiday and want to cancel.
   Will I get my money back?

cut one way  → "Yes. If you cancel boarding, refunds are issued within 5 business days."
cut another  → "No. Bookings made for public holidays are non-refundable, so you
                will not get a refund."
```

Same handbook, same retriever, same LLM, same question. The **only** difference is where
the scissors fell. (The second answer is the correct one.) By the end of this post
you'll have watched both cuts happen, built a chunker from scratch, and — because this
series measures every step — run a small scoreboard that says which cut wins.

We'll cover it in four parts:

1. **Setup** — why chunking exists, and the handbook grows real pages
2. **The two failure modes** — chunks too big (dilution), chunks too small (orphans)
3. **Chunking from scratch** — fixed-size cutting, the mid-sentence problem, overlap
4. **Measure, don't guess** — a scoreboard over golden questions, then production settings

## The one metaphor to hold onto: index cards

Post 00's open-book exam left out one detail: **the book doesn't fit on the desk.**
Embedding models and prompts have size limits, and retrieval needs pieces small enough
to point at. So before the exam, someone copies the book onto **index cards** — and the
whole game is how you cut. Four rules:

1. **The book must become cards.** No way around it — this step exists in every RAG
   system, chosen deliberately or by accident.
2. **A card that's too big blurs.** Cram three topics onto one card and its position on
   the meaning-map (post 2b) becomes a mushy average of all three — plus the student
   gets handed two topics of noise to find one fact.
3. **A card that's too small forgets.** Cut mid-thought and the card loses the sentence
   next door — the rule gets separated from its exception.
4. **Cut at natural seams, and let cards overlap at the edges.** Paragraphs are how the
   author already grouped ideas; overlap insures the sentences that straddle a cut.

## The handbook grows up

Same Sunnyvale Pet Care Center, but now the documents look like real documents — three
multi-paragraph pages (shortened here; full text in the companion notebook):

```python
PAGES = {
"boarding": """Boarding at Sunnyvale Pet Care Center is available seven days a week. The boarding facility closes at 7 pm on weekdays and 5 pm on weekends. All boarding guests must arrive at least one hour before closing time.

Dogs staying longer than three nights receive a complimentary bath before pickup. Blankets and toys from home are welcome and encouraged for comfort.

Refunds for cancelled boarding are issued within 5 business days. Bookings made for public holidays are non-refundable.""",

"grooming": """Grooming appointments must be booked at least 48 hours in advance. ...""",   # + packages, refunds, no-show fee
"daycare":  """Daycare drop off starts at 6:30 am and the last pickup is at 8 pm. ...""",   # + vaccinations, payments
}
```

Note the last boarding paragraph — a **rule** ("refunds within 5 business days")
followed by its **exception** ("public holidays are non-refundable"). Keep an eye on
that pair. Retrieval throughout is the from-scratch vector search of post 2b
(`all-MiniLM-L6-v2`, cosine).

## Failure mode 1 — Chunks too big: dilution

*Index-card rule 2: a card that's too big blurs.*

An embedding is one position on the meaning-map for the *whole* chunk. Watch what
happens to one fact's findability as its document grows around it. Same query, same
fact inside, document getting bigger — real numbers:

```
Q: "What time does boarding close on weekends?"

cosine(query, boarding page alone — 3 paragraphs)        = 0.520
cosine(query, boarding + grooming stapled — 6 paragraphs) = 0.451
cosine(query, whole handbook as ONE document — 9 paras)   = 0.451
```

The fact never moved — the *signal* did. Every added paragraph about grooming refunds
and daycare fees drags the document's position away from where the closing-time question
lands: **0.520 → 0.451**. On three tidy pages that's survivable; on a 50-page policy PDF
embedded whole, the one paragraph you need is a whisper inside a roar.

And dilution has a second cost even when retrieval *succeeds*: the whole giant chunk
goes into the prompt. That number shows up on the scoreboard below.

## Failure mode 2 — Chunks too small: orphans

*Index-card rule 3: a card that's too small forgets.*

Cut the handbook into **one card per sentence** and re-ask the holiday question. Here's
what retrieval returns — top 3 sentence-chunks, real scores:

```
Q: "I booked boarding for a public holiday and want to cancel. Will I get my money back?"

  0.652  Refunds for cancelled boarding are issued within 5 business days.
  0.483  Bookings made for public holidays are non-refundable.
  0.397  A no-show fee of 25 dollars applies if you miss an appointment...
```

Look closely — this is the subtle one. The **rule** outranks the **exception** (the
query says "refund" and "cancel", the rule sentence says both; the exception says
neither). With top-1 retrieval, the LLM gets the rule card and never sees the exception
card. Now the paragraph cut, same query:

```
paragraph-chunk top-1 (0.714):
  Refunds for cancelled boarding are issued within 5 business days.
  Bookings made for public holidays are non-refundable.
```

One card, rule *and* exception. Feed each context to the same LLM:

```
LLM with the sentence chunk : "Yes. If you cancel boarding, refunds are
                               issued within 5 business days."          ← WRONG
LLM with the paragraph chunk: "No. Bookings made for public holidays are
                               non-refundable, so you will not get a refund."  ← RIGHT
```

That's the answer-flip from the top of this post, dissected. Nothing "failed": retrieval
ranked reasonably, the LLM read its context faithfully. The scissors just separated a
rule from its exception, and **a card cut too small lies by omission.** The author put
those two sentences in one paragraph for a reason — the paragraph cut inherits that
reasoning for free.

## Chunking from scratch

The simplest chunker cuts every N characters — here it is, plus what it does to our
boarding page:

```python
def chunk(text, size=150, overlap=0):
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out
```

```
chunk(PAGES["boarding"], size=150, overlap=0):

  [Boarding at Sunnyvale Pet Care Center is available seven days a week. The
   boarding facility closes at 7 pm on weekdays and 5 pm on weekends. All board]
  [ing guests must arrive at least one hour before closing time. Dogs staying
   longer than three nights receive a complimentary bath before pickup. Blanke]
```

Real output, real problem: the cut landed mid-word — `All board / ing guests` — and
every sentence that straddles a boundary gets beheaded like that. The classic first aid
is **overlap**: each chunk re-includes the tail of the previous one, so anything cut at
a boundary survives whole in the *next* card:

```
chunk(PAGES["boarding"], size=150, overlap=50) — chunk 2:

  [t 7 pm on weekdays and 5 pm on weekends. All boarding guests must arrive at
   least one hour before closing time. Dogs staying longer than three nights ]
```

The closing-time sentence, severed by the no-overlap cut, now lives complete inside
chunk 2. Overlap costs storage (the same text embedded twice) and buys robustness at
every boundary — you'll see it earn a point on the scoreboard next.

## Measure, don't guess

*This series' motto: measured every step.* Five chunking strategies, eight golden
questions with known answers, and one metric each way:

- **hit@1** — is the gold answer inside the top-1 retrieved chunk?
- **avg context chars** — how much text that top-1 chunk drags into the prompt.

Real results:

```
strategy                  chunks   hit@1   avg context chars
whole pages                  3      8/8         435
paragraphs                   9      8/8         141
sentences                   19      5/8          66
fixed 150ch, no overlap     10      6/8         141
fixed 150ch, overlap 50     14      7/8         133
```

Read the scoreboard like the post taught you:

- **Whole pages went 8/8 — but at 435 chars a question,** 3× the context of paragraphs
  for the same hits. On this tiny, tidy corpus dilution never got the chance to cause a
  miss (three topically-pure pages are easy to tell apart); the noise cost shows anyway.
  At real scale the misses arrive too — the course-scale numbers below show it.
- **Paragraphs: 8/8 at 141 chars.** Natural seams win — the author already grouped
  rule-with-exception, fact-with-context.
- **Sentences: 5/8.** The orphan failure, now measured — among the misses is exactly the
  holiday-refund question you watched go wrong.
- **Fixed-size, no overlap: 6/8; add overlap: 7/8.** The beheaded-sentence problem is
  real, and overlap buys a point back. Blind cutting with insurance beats blind cutting
  without — but still loses to cutting at seams.

## Chunking in production

In my course code, chunking is LlamaIndex's `SentenceSplitter` — fixed-size in *tokens*,
overlap included, but sentence-aware: it refuses to cut mid-sentence, giving you
seam-respect and size control at once:

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)   # tokens, not chars
nodes = splitter.get_nodes_from_documents(docs)
```

Why 512/128? Because the course *measured it* — a sweep over 1024 / 512 / 256 tokens on
the full 1,421-document corpus, 50 golden questions, scored end to end. Real results
from that run: moving 1024 → 512 improved **every** metric (retrieval recall 0.65 → 0.70,
precision 0.78 → 0.84, faithfulness 0.75 → 0.80) — that's dilution receding at scale,
the thing our tiny corpus was too tidy to show. And 256? Recall held (0.69) but
precision and faithfulness dropped *below* the 1024 baseline — the orphan failure
arriving right on schedule. Same two failure modes, same middle ground, three orders of
magnitude apart in corpus size.

No universal constant though: 512/128 is where *that* corpus's seams live. Legal
contracts, code, chat logs — different seams, different sweet spot. The transferable
part is the method: **sweep, measure, pick.**

## The whole post in four lines

The index cards, now with their engineering names:

1. **The book must become cards** → chunking: every RAG system cuts documents before
   embedding — by design or by default.
2. **Too big blurs** → dilution: one embedding averages the whole chunk; unrelated
   topics drag the score down (0.520 → 0.451) and flood the prompt (435 vs 141 chars).
3. **Too small forgets** → orphans: rule and exception land on different cards, and the
   pipeline answers "Yes" to a non-refundable booking.
4. **Cut at seams, overlap the edges, then measure** → paragraphs beat everything here
   (8/8 at 141 chars); at course scale, a measured sweep picked 512/128.

Cut at the seams, insure the edges, and let a scoreboard pick the size. That's chunking.

*(Every snippet and every output block in this post was executed for real — retrieval is
`all-MiniLM-L6-v2` via FastEmbed run locally, the LLM is `gpt-5.4-mini` at temperature 0,
and both the "Yes" and the "No" answers are unedited outputs. One honest methods note:
the orphan demo uses a plain grounding prompt rather than post 00's strict
refuse-if-absent template — with the strict template the model over-refuses on questions
needing one inference hop, answering "I don't know" even with the exception in view.
Prompts trade failure modes too; that's a later post.)*

---

**Next up: hybrid search** — the retriever itself. This series has already built it
piece by piece: **post 2a** (BM25, the detective), **post 2b** (vector search, the map),
**post 2c** (fusing both, the committee). Then: **reranking** — a second, smarter pass
over whatever the retriever returns.
