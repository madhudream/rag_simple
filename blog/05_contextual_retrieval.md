# Contextual retrieval: stamp every chunk with what it's about

Post 04 ended with a promise: close the vocabulary gap at *index time* — once per
document — instead of paying an LLM call on every query forever. This post keeps it,
with the technique behind the course's biggest single quality jump: **contextual
retrieval** (published by Anthropic in 2024, standard practice by now).

Start with the failure it fixes. Two paragraphs from our handbook, after post 01's
paragraph chunking:

```
chunk 2 [from the boarding page]: "Cancellations must be made 48 hours in advance.
                                   Refunds are issued within 5 business days."
chunk 5 [from the grooming page]: "Cancellations made with less than 24 hours notice
                                   incur a fee. Refunds are issued within 10 business days."
```

Read them as the retriever does — *without* the bracketed labels, because those live in
the page titles the chunker cut away. Now:

```
Q: "How long do grooming refunds take?"

  0.589  "Cancellations must be made 48 hours in advance. Refunds..."   ← boarding! WRONG
  0.541  "Cancellations made with less than 24 hours notice..."         ← the right one, second
```

**Neither chunk contains the word "grooming" or "boarding."** The chunks are twins —
same vocabulary, same shape — and the one thing that told them apart (which page they
came from) died in the chunker. The retriever picks the wrong twin, the LLM faithfully
answers "5 business days," and a grooming customer plans around a refund that will take
10. This is post 01's orphan problem in a new costume: the cut didn't split a rule from
its exception this time — it split every chunk **from its document**.

We'll cover it in four parts:

1. **Setup** — the amnesia problem: chunks that forgot their document
2. **The idea** — stamp each chunk with LLM-written context, at index time
3. **Contextual retrieval from scratch** — real stamps, flipped rankings, a scoreboard
4. **In the real world** — what it costs, what it measured, and what it replaced

## The one metaphor to hold onto: a return address on every card

Post 01 copied the book onto index cards. This post fixes the cards' one remaining flaw:

1. **A card ripped from its book forgets the book.** "Refunds are issued within 5
   business days" — refunds *of what*? The paragraph knew; the card doesn't.
2. **At filing time, you're still holding the book.** So before filing each card, write
   one line across the top: *which book, which chapter, what this card is about.* An
   LLM reads the whole document plus the chunk and writes that line.
3. **The stamp moves the card on the map.** The card embeds *with* its stamp — and
   "boarding cancellation policy: refunds in 5 days" lives in a different neighborhood
   (post 2b) than the bare twin sentences did.
4. **Stamp once, benefit forever.** The work happens at index time. Query time pays
   nothing — the exact mirror image of post 04's per-query bill.

## The corpus: two pages whose chunks are twins

```python
PAGES = {
"boarding": """Sunnyvale Pet Care — Boarding Services

The boarding facility is open seven days a week. It closes at 7 pm on weekdays and 5 pm on weekends.

Guests staying longer than three nights receive a complimentary bath before pickup.

Cancellations must be made 48 hours in advance. Refunds are issued within 5 business days.""",

"grooming": """Sunnyvale Pet Care — Grooming Salon

Appointments must be booked at least 48 hours in advance. Walk-ins are not available.

The full package includes a bath, haircut, nail trim, and ear cleaning.

Cancellations made with less than 24 hours notice incur a fee. Refunds are issued within 10 business days.""",
}
```

Paragraph-chunk them (post 01's winning cut) and the page titles fall away — six chunks,
each an orphan of its page. Three real queries against the plain chunks:

```
Q: "How long do grooming refunds take?"            (right page: grooming)
  0.589  [boarding] "Cancellations must be made 48 hours in advance..."   ← WRONG page
  0.541  [grooming] "Cancellations made with less than 24 hours..."

Q: "When do I get my money back for a cancelled boarding stay?"   (right: boarding)
  0.563  [boarding] "Cancellations must be made 48 hours in advance..."   ← right — by 0.022
  0.541  [grooming] "Cancellations made with less than 24 hours..."

Q: "Does boarding include a bath?"                 (right page: boarding)
  0.523  [boarding] "Guests staying longer than three nights..."
  ...
  0.341  [grooming] "The full package includes a bath, haircut..."        ← lurking third
```

One outright wrong answer, one coin-flip win (0.563 vs 0.541 — that's noise, not
confidence), and one query where the grooming bath chunk hovers near a boarding
question. The pattern behind all three: **the distinguishing fact lives outside the
chunk.** At course scale this exact pattern is the series villain — the corpus there
has tax forms and their instruction booklets, near-twins that differ mostly in *which
document they belong to*.

## The idea: let an LLM write the stamp

At index time we still have everything: the full document *and* the chunk. So for each
chunk, ask an LLM to situate it — this is Anthropic's original prompt, nearly verbatim:

```python
CTX_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

def contextualize(chunk, document):
    ctx = llm(CTX_PROMPT.format(document=document, chunk=chunk))
    return f"{ctx}\n{chunk}"          # stamp on top, original text below
```

The stamped chunk — context prepended — is what gets **embedded and indexed**. The
original text is still in there; it just no longer travels alone. Here are the actual
stamps the LLM wrote for our six chunks:

```
[boarding] "Sunnyvale Pet Care's boarding services include facility hours and
            closing times for weekdays and weekends."
[boarding] "...guest amenities such as a complimentary bath for stays longer
            than three nights."
[boarding] "This chunk covers the boarding service's cancellation and refund
            policy, including the 48-hour advance notice requirement and the
            5-business-day refund timeline."
[grooming] "Booking policy for Sunnyvale Pet Care's grooming salon, including
            advance notice and walk-in availability."
[grooming] "Sunnyvale Pet Care's grooming salon policy and service details..."
[grooming] "This section explains the salon's cancellation policy and refund
            timeline for grooming appointments."
```

Look at the last one next to the third one. The twins aren't twins anymore — each stamp
says *which service*, in exactly the vocabulary a question would use. The LLM did at
index time what post 04's rewrites did at query time: translated between the user's
dialect and the chunk's.

## Same queries, stamped chunks

Re-embed the stamped chunks (changing chunk text means re-embedding — there's no
shortcut), re-run the exact same queries:

```
Q: "How long do grooming refunds take?"
  0.728  [grooming] "This section explains the salon's cancellation policy..."  ← RIGHT
  0.481  [boarding] "This chunk covers the boarding service's cancellation..."

Q: "When do I get my money back for a cancelled boarding stay?"
  0.722  [boarding] "This chunk covers the boarding service's cancellation..."  ← RIGHT
  0.392  [grooming] "This section explains the salon's cancellation policy..."

Q: "Does boarding include a bath?"
  0.467  [boarding]   0.304  [boarding]   0.257  [boarding]    ← all three boarding
```

Read the margins, not just the winners. The grooming-refund query **flipped** (wrong →
right, 0.728 vs 0.481, decisive). The boarding-refund query went from a 0.022 coin-flip
to a 0.33 landslide. The bath query's top-3 is now uniformly the right page — the
lurking grooming chunk got pushed out entirely. Small scoreboard over eight golden
questions:

```
plain chunks          hit@1 = 7/8    miss: "How long do grooming refunds take?"
contextualized chunks hit@1 = 8/8
```

One extra hit understates it — the *margins* are the real product. Thin margins are
where retrieval breaks under paraphrase, under corpus growth, under every future query
you didn't test. The stamps buy distance between right and wrong.

## What it costs — and where the cost lives

Six chunks, six LLM calls, at index time. Generalize that and you get this post's
central trade, the mirror image of post 04:

```
                       query transforms (04)         contextual retrieval (05)
LLM calls              1 per QUERY, forever          1 per CHUNK, once
who pays               every user, every time        the indexing job, one night
query latency          +LLM call (~seconds)          +nothing
when vocabulary        at question time              before any question exists
gets bridged
```

Now the honest ops story from the course, because "1 per chunk, once" can still bite:
the full corpus had **38,000 chunks**. Per-chunk contextualization meant 38,000 LLM
calls — which saturated the API's token-per-minute limit and then **exhausted the
account's quota mid-run**. The fix that shipped: **per-document context** — one
summary per document (~1,400 calls, 27× fewer), prepended to all of that document's
chunks. Slightly blunter stamps, wildly cheaper, and the quality numbers below are
*with* that compromise. Budget the stamping like the batch job it is.

## Contextual retrieval in production

In the course code this is a preprocessing pass before indexing — generate context,
prepend, then embed into a **separate collection** keyed on the contextual flag
(stamped and unstamped chunks must never share an index; their embeddings aren't
comparable). Measured on the full corpus, 50 golden questions, against post 04's
best (hybrid + rerank + multi-query):

- **Recall 0.78 → 0.83** — the biggest retrieval jump in the course.
- **Precision 0.852 → 0.945** — the twins problem, solved at scale: retrieved chunks
  are now overwhelmingly the *right* ones.
- **Faithfulness 0.909 → 0.968** — best in the entire course. Cleaner context in,
  more grounded answers out.
- **Latency 6.50 s → 2.56 s.** Not a typo, and the best part of the story —

— because the course *also dropped multi-query* in the same step. The team tested
contextual + multi-query together, expecting stacked gains: **it was worse than
contextual alone** (same recall, lower precision and faithfulness, double latency).
The stamps had already done multi-query's job — bridging vocabulary — but at index
time, once, so the per-query rewrites added only noise and seconds. The system got
simpler *and* better. That's rare, and worth savoring.

Two notes for your own build: Anthropic's original technique stamps the chunks for the
**BM25 index too** (contextual BM25 — post 2a's detective also benefits from cards
that say what they're about), and reports the combination with reranking (post 03)
compounding to ~67% fewer retrieval failures. The full stack is friends, not rivals:
stamps fix what chunks forgot, hybrid fuses two searchers, the reranker orders the
pool.

## The whole post in four lines

The return-address rules, now with their engineering names:

1. **A card forgets its book** → chunking amnesia: twin chunks ("Refunds are issued
   within N days") that differ only by a page title the chunker deleted — wrong twin
   retrieved at 0.589.
2. **Stamp it while you hold the book** → contextual retrieval: an LLM situates each
   chunk within its document; stamp prepended, then embedded.
3. **The stamp moves the card on the map** → the flip: 0.728 right-twin, coin-flips
   became landslides, hit@1 7/8 → 8/8.
4. **Stamp once, benefit forever** → index-time cost (batch it — 38k calls broke a
   quota; per-document context cut it 27×), zero query-time cost, and it retired
   multi-query: recall 0.83, faithfulness 0.968, latency back to 2.56 s.

Write on every card what the book knew. That's contextual retrieval.

*(Every snippet and every output block in this post was executed for real — stamps
written by `gpt-5.4-mini` at temperature 0, retrieval `all-MiniLM-L6-v2` via FastEmbed,
all local. LLM-written contexts vary between runs; the flip and the widened margins are
what reproduce. Course metrics are one measured run on my corpus with per-document
context; Anthropic's ~67% figure is from their published benchmark, not mine.)*

---

**Next up: semantic cache** — retrieval is now about as good as this course gets it.
The next post attacks a different axis entirely: *cost and speed*. Half your users ask
questions someone already asked, phrased differently. What if the pipeline recognized
"How late is boarding open Saturday?" and "Saturday boarding hours?" as the same
question — and answered the second one in 0.2 seconds for free?
