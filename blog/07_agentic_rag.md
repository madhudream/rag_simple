# Agentic RAG: the pipeline becomes a loop with a brain

Every post in this series built the same shape: a **fixed pipeline**. Question goes in,
retrieve (maybe rerank), stuff a prompt, generate, done — the same moves in the same
order for every question, forever. This final post breaks that shape, because 2026-style
RAG isn't a pipeline anymore. Watch a real trace from the system we're about to build:

```
Q: "I need to cancel booking B-1002. Will I get my money back?"

  → CALL get_booking("B-1002")
    RESULT: {'customer': 'Tom', 'service': 'boarding', 'nights': 2,
             'public_holiday': True}
  → CALL search_handbook("cancellation refund policy for boarding booking public holiday")
    RESULT: (0.70) Refunds for cancelled boarding are issued within 5 business days
            (0.67) Bookings made for public holidays are non-refundable

  ANSWER: B-1002 is a boarding booking on a public holiday, so it's non-refundable.
```

Read the second tool call carefully. Nobody's question said "public holiday" — the
**agent** learned that from the booking lookup, *wrote it into its own search query*,
retrieved the exception alongside the rule, and applied the right one. That's the
series-long rule-vs-exception villain (posts 01, 03, 05), handled by a system that
**thinks between retrievals.**

*(This post builds the concept for real, with the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python), on the handbook
corpus this whole series has used — and reads real traces. A conceptual overview and
related reading are linked in [resources](resources.md).)*

We'll cover it in four parts:

1. **The big picture** — the librarian gets promoted to researcher
2. **Why fixed pipelines fall short** — four failures, all of them old friends
3. **Agentic RAG, built** — the brain, the hands, the loop; real code, real traces
4. **Patterns, trade-offs, and the whole series in one belt of tools**

## The one metaphor to hold onto: the researcher

Post 03 introduced the senior librarian — she *reads* candidates and reorders them, but
she still works one fixed shift in one fixed pipeline. Standard RAG, end to end, is **a
librarian who fetches one book**: take the question, grab the best match, hand it over,
done — whether or not it answered anything.

Agentic RAG is **a researcher**. Four rules:

1. **A researcher thinks before fetching.** Does this question even need a search?
   Which source? What should the search *say*?
2. **A researcher judges what came back.** Good enough to answer? Or search again,
   differently?
3. **A researcher uses many sources.** The handbook, the booking system, the error-code
   index — whatever the question demands, in whatever order.
4. **A researcher costs more than a librarian.** More time, more money. You hire one
   when the question deserves it.

One sentence to rule the post: **agentic RAG turns one-shot retrieval into a loop,
driven by an LLM that keeps working until it can answer well.**

## Quick recap: what we're upgrading

Standard RAG — the shape of every previous post:

```
Question ──► [ Embed ] ──► [ Vector search, top-k ] ──► [ LLM ] ──► Answer
              (post 2b)      (posts 2a–2c, 03, 05)      (post 00)
```

One pass, left to right, no second chances. And an **AI agent**, in one paragraph: an
LLM given a set of **tools** it may call. The LLM itself only ever outputs text —
"call `get_booking` with `B-1002`" — and a **runtime** actually executes the tool and
feeds the result back into the conversation. The LLM reads the result, thinks, and
either calls another tool or writes the final answer. Think → act → observe, in a loop,
with the conversation history serving as the researcher's **scratchpad** of everything
learned so far.

## Why the fixed pipeline falls short

Four failures — every one of them something this series already met:

1. **Multi-hop questions.** "Does *my* booking get a free bath?" — step one, look up
   the booking (5 nights, boarding); step two, search the bath policy. One retrieval
   can't do two dependent steps: the second query *depends on the first answer*.
2. **The wrong dialect.** Post 04's villain: the user says "Saturday", the handbook
   says "weekends". The fixed pipeline searched with the user's words and confidently
   answered wrong. We fixed it with query transforms — bolted on, for *every* query,
   whether needed or not.
3. **Multiple sources.** Bookings live in a database, policies in a vector index,
   error codes match by keyword (post 2a vs 2b — opposite strengths). A fixed pipeline
   hits one source. Real questions don't care about your architecture.
4. **Nobody checks the retrieval.** Post 00's ur-failure: wrong pages in, confident
   cited garbage out. The pipeline *cannot notice* that its own retrieval failed —
   there is no step where anything judges the chunks.

The core problem in one line: **standard RAG is rigid — it cannot adapt.** Every fix
in posts 01–06 made the pipeline *better*; none made it *adaptive*.

## What is agentic RAG?

Put the LLM in charge of retrieval itself. At every step it decides:

- Does this question need retrieval at all?
- Which tool — meaning search, keyword search, database, web?
- What should the query *say* (not necessarily the user's words)?
- Are the results good enough, or retrieve again differently?
- Combine sources? Stop and answer?

```
User Question
     │
     ▼
┌──────────────────┐
│    AI Agent      │◄────────────────────────────┐
│   (the LLM)      │                             │
└──────────────────┘                             │
     │  thinks: "need retrieval? which tool?     │
     │           what should the query say?"     │
     ▼                                           │
┌──────────────────┐                             │
│      Tool        │  vector / keyword / DB      │
└──────────────────┘                             │
     │  returns results                          │
     ▼                                           │
   agent judges: good enough? ──── No ───────────┘
     │                            (new query, new tool)
    Yes
     │
     ▼
  Final Answer
```

## The three building blocks, built

The classic anatomy — **the brain, the hands, the loop** — here's each one as real code
with the OpenAI Agents SDK (`pip install openai-agents`).

### The hands: three tools

A tool is just a Python function with a good docstring — and the docstring is not
documentation, it's **the routing table**: it is literally what the agent reads when
deciding which tool fits the question.

```python
from agents import Agent, Runner, function_tool

@function_tool
def search_handbook(query: str) -> str:
    """Search the pet care handbook BY MEANING. Best for policy questions
    (hours, refunds, fees, requirements). Handbooks use general policy language:
    prefer category words like weekday/weekend, boarding/daycare over specific
    days or pet names."""
    q = list(emb_model.embed([query]))[0]                      # post 2b, verbatim
    top = sorted(((cosine(q, e), i) for i, e in enumerate(doc_embs)), reverse=True)[:3]
    return "\n".join(f"(score {s:.2f}) {corpus[i]}" for s, i in top)

@function_tool
def keyword_search(query: str) -> str:
    """Search the handbook by EXACT keyword match. Best for identifiers:
    error codes, form numbers, product names — anything where the literal
    token must match."""
    scored = sorted(((bm25_score(query, d), i) for i, d in enumerate(docs_tok)),
                    reverse=True)                              # post 2a, verbatim
    hits = [(s, i) for s, i in scored if s > 0][:3]
    return "\n".join(f"(score {s:.2f}) {corpus[i]}" for s, i in hits) or "No keyword matches."

@function_tool
def get_booking(booking_id: str) -> str:
    """Look up a customer booking by its id (e.g. 'B-1001'). Returns the
    service type, number of nights, and whether it falls on a public holiday."""
    b = BOOKINGS.get(booking_id.upper())
    return str(b) if b else f"No booking found with id {booking_id}."
```

Notice what the tools *are*: `search_handbook` is post 2b's vector search, unchanged.
`keyword_search` is post 2a's BM25, unchanged. `get_booking` stands in for the classic
"SQL database" source, minimized to a dict. **The series didn't get replaced — it got put on a
tool belt.** And one sly detail: the meaning-search docstring teaches the dialect
lesson from post 04 ("prefer weekday/weekend over specific days") — prompt engineering
moved into the tool description, where the agent reads it exactly when relevant.

### The brain: the agent

```python
agent = Agent(
    name="Sunnyvale Support Agent",
    model="gpt-5.4-mini",
    instructions=(
        "You are a support agent for Sunnyvale Pet Care Center. Answer customer "
        "questions using the tools.\n"
        "- Decide whether you need to search at all; greetings need no tools.\n"
        "- Pick the right tool: meaning search for policies, keyword search for "
        "identifiers like error codes, booking lookup for a customer's own booking.\n"
        "- After a search, JUDGE the results: do they actually answer the question? "
        "If not, search again with different, more general wording (handbooks say "
        "'weekends', not 'Saturday').\n"
        "- Multi-part questions may need several tools. Answer only from tool "
        "results; if nothing answers it, say you don't know."
    ),
    tools=[search_handbook, keyword_search, get_booking],
)
```

Read the instructions against the researcher's rules: decide-before-fetching (rule 1),
judge-and-retry (rule 2), many sources (rule 3). The grounding discipline from post 00
survives, one line, at the end.

### The loop: the runtime

```python
result = Runner.run_sync(agent, question, max_turns=8)
print(result.final_output)
```

`Runner` is the loop: it sends the conversation to the LLM, executes whatever tool the
LLM names, appends the result to the scratchpad, and repeats — until the LLM answers in
plain text or `max_turns` fires (more on that guardrail below).

## Watch it work — four traces

All real, unedited. **One: knowing when *not* to retrieve.**

```
Q: "Thanks, that's all I needed!"
  (no tool calls)
  ANSWER: You're welcome!
```

Zero retrievals, zero wasted tokens. The fixed pipeline would have dutifully embedded
"thanks" and searched the handbook with it.

**Two: routing — the detective summoned on demand.**

```
Q: "What does error E-4042 mean?"
  → CALL keyword_search("E-4042")
    RESULT: (2.34) Error E-4042 refund transaction declined by the payment gateway
  ANSWER: E-4042 means: refund transaction declined by the payment gateway.
```

The agent read the docstrings and picked the keyword tool for an identifier — post 2b
proved meaning-search ranks the wrong twin first here; the agent never gave it the
chance. Post 2c solved this with always-on hybrid fusion; the agent solves it with a
*choice*.

**Three: the Saturday question — post 04, self-serve.**

```
Q: "Until what time can I pick up my dog on a Saturday? He is boarding."
  → CALL search_handbook("boarding pickup time on weekends")
    RESULT: (0.65) The boarding facility closes at 7 pm on weekdays and 5 pm on weekends
  ANSWER: On Saturdays, boarding pickup is until 5 pm.
```

Look at the query the agent wrote: the user said **"Saturday"**; the agent searched
**"on weekends."** That is post 04's query transform — performed by the agent, on the
fly, only because this question needed it. No standing multi-query tax on every query;
translation happens when a researcher decides it should.

**Four: multi-hop — the finale from the top of the post.** The B-1002 trace: booking
lookup first, then a search query *composed from the lookup's result* ("…public
holiday"), then the exception correctly beats the rule. And its friendlier sibling:

```
Q: "My booking is B-1001. Does my dog get a free bath, and if I cancel,
    do I get a refund?"
  → CALL get_booking("B-1001")          → boarding, 5 nights, no holiday
  → CALL search_handbook("free bath included with booking or service")
  → CALL search_handbook("cancellation refund policy")
  ANSWER: Yes — B-1001 is a 5-night boarding stay, so your dog gets a
          complimentary bath. If you cancel, refunds are issued within
          5 business days.
```

Three tools, two sources, one compound answer. A fixed pipeline cannot express this
question at all — the bath answer *depends on* data that isn't in the handbook.

## The named patterns

What you just watched has names in the literature:

- **ReAct-style RAG** — think, act, observe, repeat. That's the loop itself; every
  trace above is ReAct.
- **Self-RAG** — the agent grades its own retrievals (relevant? sufficient?) before
  using them. That's the "JUDGE the results" line in our instructions.
- **Corrective RAG (CRAG)** — when the primary source fails, fall back to another.
  That's the routing between meaning/keyword/booking tools — and at bigger scale,
  a web-search fallback when the index has nothing.

Same core in all three: **an agent making decisions between retrievals.** The labels
just mark which decision gets emphasized.

## Standard vs agentic — and when to use which

| Aspect | Standard RAG (posts 00–06) | Agentic RAG |
|---|---|---|
| Flow | Fixed pipeline | Dynamic loop |
| Retrievals per question | One | As many as needed |
| Tools | One (usually vector search) | Many (vector, keyword, DB, web…) |
| Query rewriting | Bolted on (post 04), always-on | Agent decides, per question |
| Judges its own retrieval | No | Yes |
| Multi-hop questions | Can't | Can |
| Latency / cost | Low | Higher — LLM call per step |
| Debuggability | Deterministic | Same question, different paths |

Use the researcher when questions are multi-hop, sources are plural, phrasing is messy,
or being right matters more than being fast. Keep the librarian when questions are
simple and volume is high — and remember post 06: **a semantic cache in front of an
agent is the best of both** (repeat questions get millisecond answers; novel questions
get the researcher).

Honesty section, per house rules — the researcher's bill:

- **Latency and cost.** Every loop step is an LLM call, and tool results ride along in
  every subsequent prompt. Our B-1001 trace = 4 LLM turns where post 00 spent 1.
- **Non-determinism.** Run the Saturday question twice and the agent may phrase its
  search differently (ours did across runs — same right answer, different query
  wording). Debugging means reading traces, not diffing outputs.
- **Termination.** An agent that keeps judging its results insufficient will loop
  forever — that's what `max_turns=8` is for. Ship a stopping rule or the agent ships
  one for you, in production, at 3 a.m.
- **Complexity.** Tool design, instructions, evaluation, stopping rules. The simple
  system from post 00 is now a small society. Don't reach for it before a fixed
  pipeline has actually failed you.

## The whole series on one tool belt

The researcher's rules, and where every previous post ended up:

1. **Think before fetching** → the agent skips retrieval for small talk, routes error
   codes to the detective (**2a**) and policies to the map (**2b**), reads tool
   docstrings as its routing table.
2. **Judge what came back** → self-grading retrievals (Self-RAG), retrying with the
   phrasebook's translation (**04**) only when needed.
3. **Many sources** → handbook index (chunked per **01**, stamped per **05**), booking
   database, keyword index — fused not by a fixed formula (**2c**) but by a decision.
4. **Researchers cost more** → so the receptionist (**06**) answers the repeats in a
   millisecond, the librarian (**03**) still orders any wide pool, and the researcher
   handles what's left: the questions that were never answerable by one retrieval.

Retrieval stopped being a pipeline and became a decision. That's agentic RAG — and
that's the course.

*(Every trace in this post is a real, unedited run — OpenAI Agents SDK with
`gpt-5.4-mini`, tools running post 2a's BM25 and post 2b's vector search locally via
FastEmbed, `max_turns=8`. Agent behavior varies between runs: tool-call wording and
occasionally tool order will differ; the decisions — routing, self-translation,
multi-hop composition — are what reproduce.)*

---

**Next up: the whole story** — every post in this series compressed into one
end-to-end article: eight metaphors, one corpus, and the measured numbers from naive
RAG's 0.50 hit-rate to the full 2026 stack. One of a kind.
