# What is RAG? Retrieval-Augmented Generation from scratch

Ask an LLM a question about **your** data — your company's handbook, your product docs,
your support policies — and watch what happens. Real question, real model
(`gpt-5.4-mini`), no tricks:

```
Q: How far in advance must grooming appointments be booked at
   Sunnyvale Pet Care Center?

A: I don't have access to Sunnyvale Pet Care Center's current booking
   policy from here. If you want, I can help you find it quickly if you
   share their website link, a screenshot of the grooming page, ...
```

Honest, at least. Now push it — same model, told to commit to a number:

```
Q: (same question) Answer with a specific number of hours only.

A: 24
```

The real answer, from the center's handbook, is **48 hours**. The model didn't know, and
when pressured, it **invented a plausible number with total confidence**. That's a
hallucination, and it's what stands between LLMs and every serious use of them on
private data.

**RAG — Retrieval-Augmented Generation** — is the standard fix, and it's the spine of
this whole blog series. In this post we build the *naive* version from scratch: the
simplest pipeline that works, ~30 lines, every output real. And at the end we watch it
fail in a very specific way — the failure the rest of the series exists to fix.

We'll cover it in four parts:

1. **Setup** — the closed-book problem, and the one idea that fixes it
2. **The three steps** — Retrieve, Augment, Generate
3. **Naive RAG assembled** — the full pipeline from scratch, real answers
4. **RAG in the real world** — the production one-liner, and meeting the villain

## The one metaphor to hold onto: an open-book exam

An LLM answering from memory is a student in a **closed-book exam**. Four rules:

1. **Closed book = memory only.** The student's memory was written during training and
   froze there. Your handbook was never in it — and never will be.
2. **Under pressure, students invent.** A blank is embarrassing; "24 hours" *sounds*
   right. LLMs have the same instinct — that's where hallucinations come from.
3. **So open the book.** Put the actual page in front of the student, and the answer
   comes from the page, not from memory.
4. **You can't hand them the whole library.** Prompts have limits. Someone has to find
   the *right few pages* first — that's a search problem, and you already know how to
   solve it.

RAG is exactly this: an open-book exam where a search engine finds the pages. Hold onto
it — each rule becomes one piece of the pipeline.

## Our corpus: a private handbook

Eight one-line policies from the (fictional) Sunnyvale Pet Care Center. Private data:
no LLM has seen these, which is the entire point:

```python
corpus = [
    "Grooming appointments must be booked at least 48 hours in advance",           # doc 0
    "The boarding facility closes at 7 pm on weekdays and 5 pm on weekends",       # doc 1
    "Dogs staying longer than three nights receive a complimentary bath before pickup",  # doc 2
    "Refunds for cancelled boarding are issued within 5 business days",            # doc 3
    "Refunds for cancelled grooming appointments are issued within 10 business days",    # doc 4
    "All pets must have up to date rabies vaccination records on file",            # doc 5
    "Daycare drop off starts at 6:30 am and the last pickup is at 8 pm",           # doc 6
    "A late pickup fee of 15 dollars applies for every 30 minutes after closing",  # doc 7
]
```

And here's the whole pipeline we're about to build:

```
question: "How far in advance must grooming be booked?"
   │
   ▼
1. RETRIEVE — vector search over the handbook          (find the right pages)
   │    → [doc 0] 0.758   [doc 4] 0.516
   ▼
2. AUGMENT — paste those docs into a strict prompt      (open the book)
   │
   ▼
3. GENERATE — the LLM answers from the pasted context   (answer from the page)
   │
   ▼
"Grooming appointments must be booked at least 48 hours in advance. [doc 0]"
```

Three steps. Every RAG system ever shipped — toy or production — is these three steps
with better and better parts. *Naive* RAG means: each step in its simplest working form.

## Step 1 — Retrieve: find the right pages

*Open-book rule 4: someone has to find the right few pages.*

This step is a search problem, and it's the exact vector search we built from scratch in
post 2b — embed every document once, embed the question, rank by cosine similarity:

```python
import numpy as np
from fastembed import TextEmbedding

emb_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_embs = list(emb_model.embed(corpus))      # embed the handbook ONCE

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(question, top_k=2):
    q = list(emb_model.embed([question]))[0]
    scored = sorted(((cosine(q, e), i) for i, e in enumerate(doc_embs)), reverse=True)
    return scored[:top_k]
```

(If cosine similarity or embeddings are new to you, post 2b builds both by hand — this
post treats them as a solved problem.) Run it on our question:

```
Q: "How far in advance must grooming appointments be booked at Sunnyvale Pet Care Center?"
  0.758  doc 0: Grooming appointments must be booked at least 48 hours in advance
  0.516  doc 4: Refunds for cancelled grooming appointments are issued within 10 business days
```

Right page found. And because this is *semantic* search, wording doesn't have to match —
ask about a "**free** bath after a **long stay**" and it finds the doc that says
"**complimentary** bath" for stays "longer than **three nights**":

```
Q: "Do dogs get a free bath after a long stay?"
  0.779  doc 2: Dogs staying longer than three nights receive a complimentary bath before pickup
  0.252  doc 5: All pets must have up to date rabies vaccination records on file
```

## Step 2 — Augment: open the book inside the prompt

*Open-book rules 3 + 4: the page goes in front of the student — and nothing else.*

"Augment" just means **paste the retrieved docs into the prompt**. But the prompt
wording is the cheapest reliability win in all of RAG, so read it closely — it has three
jobs:

```python
PROMPT = """You are a precise assistant answering questions about the Sunnyvale Pet Care Center handbook.
Use ONLY the context below. Cite the doc id(s) you used in square brackets, e.g. [doc 3].
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."

Context:
---------
{context}
---------
Question: {question}
Answer:"""
```

1. **"Use ONLY the context"** — answer from the page, not from memory. This line is what
   makes it an open-book exam instead of a student peeking at both.
2. **"Cite the doc id(s)"** — every claim traceable to a page. When the answer is wrong,
   you can see *which retrieved doc* misled it.
3. **The "I don't know" escape hatch** — this one matters more than it looks. Without
   explicit permission to refuse, an LLM handed irrelevant context will *still* try to
   answer — helpfully, confidently, wrongly. Rule 2 of the metaphor: under pressure,
   students invent. This line removes the pressure.

## Step 3 — Generate: the LLM answers from the page

Wire the three steps together and the whole pipeline is one short function:

```python
from openai import OpenAI

client = OpenAI()

def ask_llm(prompt):
    r = client.chat.completions.create(
        model="gpt-5.4-mini", temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()

def rag_answer(question, top_k=2):
    hits = retrieve(question, top_k)                                    # 1. RETRIEVE
    context = "\n".join(f"[doc {i}] {corpus[i]}" for score, i in hits)  # 2. AUGMENT
    return ask_llm(PROMPT.format(context=context, question=question))   # 3. GENERATE
```

That's naive RAG, complete. Run it — three questions, real output:

```
Q: How far in advance must grooming appointments be booked at Sunnyvale Pet Care Center?
retrieved: [(0.758, doc 0), (0.516, doc 4)]
A: Grooming appointments must be booked at least 48 hours in advance. [doc 0]

Q: Do dogs get a free bath after a long stay?
retrieved: [(0.779, doc 2), (0.252, doc 5)]
A: Yes. Dogs staying longer than three nights receive a complimentary bath before pickup [doc 2].

Q: Does Sunnyvale Pet Care offer cat grooming?
retrieved: [(0.454, doc 0), (0.355, doc 4)]
A: I don't know based on the provided documents.
```

## What the results teach us

- **The hallucination is gone.** The model that invented "24" now answers **48 hours,
  with a citation** — because the true number was on the page in front of it.
- **Synonyms worked for free.** "Free bath after a long stay" never appears in the
  handbook; semantic retrieval bridged it, and the answer cites the right doc.
- **The escape hatch fired.** For cat grooming, retrieval returned the closest docs it
  could find (scores 0.454, 0.355 — low), the context didn't contain the answer, and the
  model said the exact refusal string instead of improvising. All three prompt jobs,
  visible in three answers.

## Meeting the villain

Time for the honesty section — and this one sets up the entire rest of the series. Ask
something a real customer would ask:

```
Q: "Until what time can I pick up my dog on a Saturday, and what happens if I am late?"
```

The handbook has both answers: boarding closes at **5 pm on weekends** [doc 1], and late
pickup costs **15 dollars per 30 minutes** [doc 7]. Now look at what retrieval actually
returned:

```
  0.562  doc 2: Dogs staying longer than three nights receive a complimentary bath...
  0.496  doc 6: Daycare drop off starts at 6:30 am and the last pickup is at 8 pm
  0.426  doc 7: A late pickup fee of 15 dollars applies for every 30 minutes after closing
```

The document we need most — doc 1, the weekend closing time — is **not even in the top
3**. The question says "Saturday", the doc says "weekends"; the question says "pick up
my dog", which smells like the *pickup*-flavored docs about baths and daycare. The
retriever drifted to the wrong pages. And here's the answer the full pipeline produced
from them (top-2 context):

```
A: The last pickup time is 8 pm on Saturday [doc 6]. If your dog is staying
   longer than three nights, they receive a complimentary bath before pickup [doc 2].
```

**Confident. Cited. Wrong.** The true Saturday cutoff is 5 pm — this answer would get a
customer locked out with a $15-per-half-hour fee ticking. And notice the scary part:
every safety feature *worked*. The model used only the context. It cited its sources.
It didn't hallucinate a word. **Grounding can't save you from retrieving the wrong
pages** — it just makes the wrong answer better-dressed.

This is the villain of the whole series: **retrieval is the bottleneck.** The LLM at the
end is only as good as the pages we hand it. Everything ahead — measuring, chunking,
hybrid search, reranking — is a different weapon against this exact failure.

## Naive RAG in production: the one-liner

In my course code the whole pipeline above collapses into a few LlamaIndex lines — same
three steps, industrial parts:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-5.4-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

docs = [Document(text=t, metadata={"doc_id": f"doc {i}"}) for i, t in enumerate(corpus)]

index = VectorStoreIndex.from_documents(docs)       # chunk + embed + index — Step 1's index
engine = index.as_query_engine(similarity_top_k=2)  # retrieve + augment + generate

print(engine.query("How far in advance must grooming appointments be booked?"))
```

Real output:

```
Q: How far in advance must grooming appointments be booked?
A: Grooming appointments must be booked at least 48 hours in advance.
retrieved: ['doc 0', 'doc 4']

Q: Does Sunnyvale Pet Care offer cat grooming?
A: The available information does not say whether cat grooming is offered. It only
   states that grooming appointments must be booked at least 48 hours in advance...
retrieved: ['doc 0', 'doc 4']
```

Map it to our from-scratch code: `VectorStoreIndex.from_documents` is our
"embed everything once" (plus automatic chunking for long documents — our one-line docs
didn't need it, real PDFs do). `as_query_engine(similarity_top_k=2)` is `retrieve` +
`PROMPT` + `ask_llm` in one object.

One difference worth seeing: LlamaIndex ships a *default* prompt, not our strict one —
and it shows on the cat-grooming question. On one run it hedged politely ("the available
information does not say..."); on another it answered **"Yes, Sunnyvale Pet Care offers
grooming appointments"** — inferring past the page, exactly the failure our escape-hatch
line exists to block. The lesson transfers: in production, pass a custom template (like
our strict one) — the default gets you a RAG system whose refusal behavior you never
chose.

## The whole post in four lines

The open-book exam, now with its engineering names:

1. **Closed book = memory only** → an LLM can't know your private data, and its memory
   froze at training time.
2. **Under pressure, students invent** → hallucination; our model said "24" when the
   truth was 48.
3. **Open the book** → *Augment*: paste retrieved docs into a strict prompt — only the
   context, cite sources, allowed to say "I don't know".
4. **Someone must find the right pages** → *Retrieve*: vector search (post 2b), and it
   is the bottleneck — when it drifts, you get confident, cited, wrong answers.

Retrieve the pages, open the book, answer from the page. That's RAG.

*(Every snippet and every output block in this post was executed for real — the LLM is
`gpt-5.4-mini` at temperature 0, retrieval is `all-MiniLM-L6-v2` via FastEmbed run
locally, and the wrong "8 pm on Saturday" answer is a genuine, unedited pipeline output.
LLM outputs vary slightly between runs; the failure pattern is what reproduces.)*

---

**Next up: chunk smart** — this post cheated with one-line documents. Real documents
are pages, and someone has to decide how to *cut* them before anything gets embedded.
That decision alone can flip an answer from right to wrong — and from here on, every
step in this series comes with a scoreboard: golden questions and hit rates, so each
fix has to *prove* it helps.

**Also in this series:** posts 2a (BM25), 3b (vector search), and 3c (hybrid) crack open
the `retrieve` step this post treated as a black box.
