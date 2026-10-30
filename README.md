# halshield

Stack anti-hallucination layers around any LangGraph node. Pick a protection level, or list the layers
yourself; `Shield(...).wrap(node)` returns a node of the same shape (`state -> dict`), so you add it to a graph
like any other node.

```python
shield = Shield(Protection.MEDIUM, sources=docs)
graph.add_node("writer", shield.wrap(writer))
```

## Install

```bash
pip install -e .                 # core: numpy, openai, python-dotenv
pip install -e ".[langchain]"    # reuse LangChain retrievers / vector stores
pip install -e ".[examples]"     # langgraph + pandas, to run examples/
```

Put `OPENAI_API_KEY` in the environment or in `.env` (see `.env.example`). The default model is `gpt-4o-mini`.

## Example

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from halshield import Protection, Shield, get_llm

docs = {
    "k2": "The Kestrel K2 flies for up to 38 minutes on a single charge.",
    "k3": "The Kestrel K3, released in 2021, flies for 52 minutes per charge.",
}
llm = get_llm()


class QAState(TypedDict, total=False):
    question: str
    context: list[str]
    answer: str
    sources: list[str]
    confidence: float


def writer(state: dict) -> dict:
    """Your node: answer from `context`, honour `instructions` and `temperature` set by the layers."""
    rules = "\n".join(f"- {rule}" for rule in state.get("instructions", []))
    reply = llm.json(
        "Context:\n" + "\n".join(state.get("context", [])) + f"\n\nQuestion: {state['question']}\n{rules}\n"
        'Return JSON {"answer": str, "sources": [ids], "confidence": number 0-1}.',
        "writer",
        temperature=state.get("temperature", 0.0),
    )
    return {
        "answer": reply.get("answer", ""),
        "sources": reply.get("sources", []),
        "confidence": float(reply.get("confidence") or 0.0),
    }


shield = Shield(Protection.MEDIUM, sources=docs)

graph = StateGraph(QAState)
graph.add_node("writer", shield.wrap(writer))
graph.add_edge(START, "writer")
graph.add_edge("writer", END)
app = graph.compile()

print(app.invoke({"question": "How long does the K3 fly?"})["answer"])  # "The K3 flies for 52 minutes [k3]."
print(llm.usage.totals())  # tokens and calls, per layer in .tokens / .calls
```

A complete graph (retriever node → shielded writer) is in [examples/demo.py](examples/demo.py).

## Protection levels

`Protection` is an enum. Give a level to `Shield` and it builds and connects that level's layers for you:

```python
from halshield import Protection, Shield

Shield(Protection.MINIMAL, sources=docs)  # cheapest
Shield(Protection.MEDIUM, sources=docs)  # best accuracy for the cost
Shield(Protection.MAXIMUM, sources=docs)  # fewest hallucinations
Shield("medium", sources=docs)  # the level's name works too
Shield(Protection.MAXIMUM)  # no sources: checks your node's own RAG
```

| level | layers (closest to the node first) | LLM calls per question |
|---|---|---|
| `MINIMAL` | `RAGGrounding(k=4)` | 1 |
| `MEDIUM` | `SelfConsistency`, `RAGGrounding(k=4, rewrite_queries=3)` | 3 when the first two samples agree, else 5 |
| `MAXIMUM` | `MEDIUM` + `CoVe`, `Critic`, `Abstain(0.7)` | about 7 to 9 |

Options, passed after the level:

| option | default | meaning |
|---|---|---|
| `sources` | none | documents to search: `{id: text}` or a list of texts |
| `retriever` | none | instead of `sources`: a halshield `Retriever` or any LangChain retriever / vector store |
| `k` | 4 | chunks to retrieve |
| `llm` | `get_llm()` | the client every layer uses |
| `threshold` | 0.7 | `MAXIMUM` only: the Abstain threshold. Lower answers more questions but lets more hallucinations through: at 0.5 the bench had about twice as many |

With neither `sources` nor `retriever`, nothing is retrieved: the level checks the context your node reports
(see [Wrap a node that already does RAG](#wrap-a-node-that-already-does-rag)).

To change a level, take its layers as a list and edit it:

```python
from halshield import Abstain

layers = Protection.MEDIUM.layers(docs)  # [SelfConsistency, RAGGrounding]
shield = Shield([*layers, Abstain(threshold=0.6)])
```

Measured on the bench (28 hard questions × 3 runs, `gpt-4o-mini` writer, `gpt-4o` judge):

| | no protection | `MINIMAL` | `MEDIUM` | `MAXIMUM` |
|---|---|---|---|---|
| correct and clean answers | 48% | 57% (+10) | 60% (+12) | 60% (+12) |
| answers with a hallucination | 32% | 26% (−6) | 23% (−10) | 8% (−24) |
| declined to answer | 21% | 17% | 11% | 27% |
| seconds per question | 1.4 | 1.6 (×1.1) | 4.8 (×3.5) | 9.8 (×7.1) |
| tokens per question | 325 | 735 (×2.3) | 2,604 (×8.0) | 3,816 (×11.7) |

Tokens per question by layer:

| | writer | retrieval | query rewriting | sample pick | CoVe | critic |
|---|---|---|---|---|---|---|
| `MINIMAL` | 718 | 17 | | | | |
| `MEDIUM` | 2,052 | 65 | 340 | 148 | | |
| `MAXIMUM` | 1,997 | 66 | 341 | 132 | 911 | 371 |

The run-to-run spread is about ±3 to 4.5 points. `MAXIMUM` is the level to pick when a wrong answer costs more
than no answer: it hallucinates a quarter as often as no protection but declines more questions. Re-measure
with `examples/run_eval.py --levels` after you change a layer (see [Evaluate](#evaluate)).

## Your own layers

Pass a list instead of a level. Layers run in list order: `layers[0]` is closest to your node, the last layer
has the final say. Any subset works.

```python
from halshield import Abstain, CoVe, Critic, RAGGrounding, SelfConsistency, Shield

shield = Shield(
    [
        SelfConsistency(n=3),  # inside RAGGrounding: retrieval runs once, every sample sees the same context
        RAGGrounding(sources=docs, require_citations=True),
        CoVe(),
        Critic(model="gpt-4o-mini"),
        Abstain(threshold=0.7),
    ]
)
```

## Supported methods

| layer | method | extra LLM calls |
|---|---|---|
| `RAGGrounding(sources=None, retriever=None, require_citations=True, k=3, rewrite_queries=0, check_subjects=True)` | Retrieval-augmented grounding: finds the top-k chunks (embeddings + keyword overlap), injects them as `[id] text`, requires inline citations and drops uncited claims. With `check_subjects` it also drops claims about a model variant ("K3 Lite") whose cited chunks only describe the base model ("K3"). With `rewrite_queries=n` it also searches for n LLM-rewritten queries phrased like the documents ("CTO" for "head of technology", one query per step of a multi-part question) plus one hypothetical document-style answer sentence (HyDE), and merges the results with reciprocal rank fusion | 0, or 1 with rewriting (+ embeddings) |
| `SelfConsistency(n=3, temperature=0.8, early_stop=True)` | Self-consistency: samples n answers in parallel and keeps the one most consistent with the others; confidence is capped by agreement. With `early_stop`, if the first two samples state the same numbers from the same chunks, the rest is skipped | 1 run, or n − 1 runs + 1 |
| `CoVe(max_questions=3)` | Chain-of-Verification: one check question per claim, all answered in one call from the context only (never seeing the draft); refuted claims are removed | 3 |
| `Critic(model=None)` | LLM critic: a reviewer model (can differ from the writer) removes claims the context does not support | 1 |
| `Abstain(threshold=0.7)` | Abstention: replaces the answer with `"Not enough information to answer."` when confidence is low or no sourced answer is left | 0 |

CoVe and Critic check the answer only against the chunks it cites (all chunks if it cites none): a claim must
hold in its own source, and their prompts are several times shorter. Set `cited_context_only = False` on
the layer to check against the whole context. Citations they drop from sentences they keep word for word
are put back, and a citation such as `[doc06]` that clearly means the context's `[d06]` is corrected.

### Use your own retriever

Pass any LangChain retriever, vector store or Runnable that returns `Document`s. You don't need to maintain a
vector DB:

```python
RAGGrounding(retriever=chroma_store.as_retriever(search_kwargs={"k": 4}))
```

### Wrap a node that already does RAG

Your RAG node can still hallucinate. Wrap it as it is: `RAGGrounding()` with no `sources` or `retriever`
retrieves nothing and only enforces citations against the chunks your node used.

```python
def my_rag_node(state: dict) -> dict:
    documents = retriever.invoke(state["question"])  # your existing retrieval
    reply = my_chain.invoke(
        {"question": state["question"], "documents": documents, "rules": state.get("instructions", [])}
    )  # show the layers' rules to the model
    return {
        "answer": reply.answer,
        "sources": reply.sources,
        "confidence": reply.confidence,
        "context": documents,
    }  # what the answer was written from: "[id] text" strings or Documents


shield = Shield(Protection.MAXIMUM)  # or Shield([RAGGrounding(), CoVe(), Critic(), Abstain(threshold=0.7)])
graph.add_node("rag", shield.wrap(my_rag_node))
```

- **`context`:** return the chunks the answer was written from. CoVe and Critic check claims against them too.
- **Documents:** they get ids from `metadata["id"]`, else `"<file name>#p<page>"`.
- **Citations:** your prompt should show the chunks with those ids and include `state["instructions"]`, which
  holds the rule to cite them inline. Sentences that cite none of the chunks are dropped.
- **No `context` returned:** the answer is not filtered, and a flag says so.

### Node contract

- **in:** `question`, `context` (`"[id] text"` entries), plus `instructions` and `temperature` set by the layers
- **out:** `answer`, `sources`, `confidence`, optional `flags` (what each layer changed; declare it in
  your state schema to keep it)

### Errors

Errors raised at run time derive from `HalshieldError`: `InvalidStateError` (no question) and `LLMRequestError`
(the API failed). Bad constructor arguments raise `ValueError` / `TypeError`.

### Extending

Subclass an interface from `halshield.interfaces`:

| interface | implement | for |
|---|---|---|
| `Layer` | `run(state, node)` | a new layer |
| `RevisionLayer` | `revise(question, answer, context)` | a layer that rewrites the answer |
| `Retriever` | `retrieve(question)` | another search backend |
| `LLMClient` | `chat`, `embed`, `with_model` | another model provider (`set_llm(...)`) |

## Evaluate

```bash
pytest                                                        # offline, no API key
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py      # 28 hard cases x 10 configs
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py --levels --repeats 3  # no protection vs the 3 levels
python examples/run_eval.py --configs baseline,maximum --repeats 3 --limit 5
python examples/run_eval.py --out results/after.jsonl --compare results/before.jsonl  # what a fix changed
python examples/show_results.py results/levels.jsonl          # print a saved run again, tokens by layer
```

28 hard cases about a fictional company. Each answer is graded by an LLM judge, a fact rubric and regex
checks for known traps. See [docs/evaluation.md](docs/evaluation.md) for the bench, the scoring, how to
read the report and how to add cases.
