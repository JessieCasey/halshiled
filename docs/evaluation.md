# Evaluation

The evaluation measures how often each layer stack hallucinates, how often it still answers correctly, and
what that costs. It runs every configuration over the same hard bench and grades each answer three ways.

## Quick start

```bash
pip install -e ".[examples]"
export OPENAI_API_KEY=...                 # or put it in .env
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py --levels --repeats 3   # before/after the library
```

| flag | default | meaning |
|---|---|---|
| `--configs baseline,maximum` | all 10 | which layer stacks to run |
| `--levels` | off | run only `baseline` and the `minimal`, `medium` and `maximum` levels |
| `--categories false_premise,scope` | all | only cases in these categories |
| `--limit N` | all | at most N cases (quick checks) |
| `--repeats N` | 1 | answer every case N times; adds run-to-run spread to the summary |
| `--workers N` | 8 | answers evaluated in parallel (threads); `1` runs one at a time |
| `--threshold 0.7` | 0.7 | `Abstain` confidence threshold |
| `--writer helpful\|strict` | `helpful` | writer prompt (see below) |
| `--bench path.json` | `examples/bench.json` | bench file |
| `--out results.jsonl` | `results.jsonl` | per-answer details (folders are created) |
| `--compare before.jsonl` | none | after the run, diff it against an earlier results file |

Environment: `HALSHIELD_JUDGE_MODEL` picks the judge. Use a stronger model than the writer, e.g. `gpt-4o`; by
default the judge uses the library's default model. The exit code is 2 for a malformed bench, bad flags or
an API error.

**Cost:** one run is cases × configs × (writer + layer calls + 1 judge call). The `maximum` level makes about 8 LLM
calls per question, so a full run with `--repeats 3` makes a few thousand calls. Try `--limit 3` first.

**Speed:** `--workers 8` runs 8 answers at once. If you hit OpenAI rate limits (HTTP 429), lower it. The client
retries a 429 twice, then the run stops with `LLMRequestError`.

## What is compared

The pipeline is a LangGraph graph with two nodes: a keyword retriever that puts the 2 best-matching documents
into `context`, then the writer wrapped by a `Shield`. Every configuration uses the same writer and differs
only in its layers:

| config | layers (processing order) |
|---|---|
| `baseline` | none: the pipeline's own keyword RAG only (the "before") |
| `+grounding(rewrite)` | RAGGrounding (k=4) with `rewrite_queries=3` |
| `+citations(own RAG)` | RAGGrounding with no retriever: it only enforces citations against the pipeline's own keyword retrieval (wrapping an existing RAG node) |
| `+grounding+cove` | RAGGrounding, CoVe |
| `+grounding+cove+abstain` | RAGGrounding, CoVe, Abstain |
| `+grounding+sc` | SelfConsistency, RAGGrounding |
| `+grounding+critic` | RAGGrounding, Critic |
| `minimal` | `Protection.MINIMAL`: RAGGrounding (k=4) |
| `medium` | `Protection.MEDIUM`: SelfConsistency, RAGGrounding (k=4) with `rewrite_queries=3` |
| `maximum` | `Protection.MAXIMUM`: `medium`, then CoVe, Critic, Abstain (threshold 0.7) |

`minimal` was called `+grounding` and `maximum` was called `full` in earlier results files; `--compare` and
`compare_results.py` match the old names to the new ones. SelfConsistency sits inside RAGGrounding, so retrieval
and query rewriting run once per question and every sample sees the same context.

The default `helpful` writer prompt is a typical product-assistant prompt. It asks for complete, confident
answers and says customers dislike "information is unavailable", which is where models start to invent. Use
`--writer strict` to see the effect of a neutral prompt.

## The bench

`examples/bench.json` describes a fictional drone company (Kestrel Robotics), so the model cannot answer
from memory. The 22 documents deliberately contain traps:

- an outdated 2019 press release and an old 2022 price list
- a look-alike product (K3 Lite vs K3)
- an unrelated company with the same name
- a stale case study and a spec sheet in kilometres
- a competitor's longer-flying drone
- long release notes with the key fact buried in the middle

The questions are worded differently from the documents ("the budget version of the K3", "led
technology"), so keyword matching alone doesn't find the answer.

| id | category | question tests |
|---|---|---|
| c01 | conflicting_sources | prefer the release spec (52 min) over the 2019 announcement (45 min) |
| c02 | entity_confusion | the K3 Lite isn't cold-rated, though the K3 is |
| c03 | attribute_transfer | K3 Lite payload: unanswerable, only the K3's is given |
| c04 | multi_hop_arithmetic | minimum contract cost: identify the model, use current prices, apply the 3-year term (84,000 €) |
| c05 | unit_conversion | 4.5 km in miles (2.8) |
| c06 | temporal_reasoning | CTO at the K3 launch (Saar) and today (Kivi) |
| c07 | stale_data | Baltic Freight's current fleet: unanswerable, 12 is a retired pilot |
| c08 | distractor | no Kestrel drone flies an hour; the 60-min F9 is a competitor's |
| c09 | false_premise | the K3 was never withdrawn (the K2's sales ended) |
| c10 | false_premise | Anneli Saar was never CEO |
| c11 | partial | K3 Lite flight time (40 min) yes, weight unknown |
| c12 | elaboration | Atlas's internal model: unanswerable, only its behaviour is documented |
| c13 | buried_fact | lowest alert threshold (0.5%), not the 2% default |
| c14 | multi_hop_aggregation | Verdemarkt's total fleet: 210 + 4 × 15 = 270 |
| c15 | multi_hop_temporal | founding to CE marking: 8 years |
| c16 | entity_confusion | headcount 340, not the namesake's 1,200 |
| c17 | scope | only the K3 is documented as CE-marked |
| c18 | hypothetical | a legitimate inference (330) that layers must not delete |
| c19 | temporal_reasoning | CTO in 2022 (Saar), between Leht (until 2020) and Kivi (from 2024) |
| c20 | entity_confusion | budget K3 vs K2 flight time: 2 minutes, not the full K3's 14 |
| c21 | multi_hop_arithmetic | ten lidar-less drones per year: 38,000 € at 2024 prices (not 53,000 or 46,000) |
| c22 | attribute_transfer | K3 Lite temperature range: unanswerable, -20 to 40 °C is the K3's |
| c23 | false_premise | the move was Tallinn → Rotterdam (2019), not the reverse |
| c24 | entity_confusion | Kestrel's trucks: unanswerable, the 90 trucks are the namesake's |
| c25 | buried_fact | the Atlas release that added the K3 Lite (4.0) |
| c26 | multi_hop_temporal | Series B (2020) to second factory production (2022): 2 years |
| c27 | false_premise | replacement in 48 h; drones can't be bought, only leased |
| c28 | multi_hop_aggregation | models a new customer can lease: two (K2 discontinued, K4 a prototype) |

Five cases (c03, c07, c12, c22, c24) are unanswerable: the correct behaviour is to decline. c18 checks the
other direction, that layers don't remove a correct derived answer. c19–c28 were added later;
results files from before then have only c01–c18, and `--compare` diffs only the cases both runs share.

## Scoring

Each answer is graded by `HallucinationJudge` (`examples/evaluation/hallucination_judge.py`):

1. **LLM judge.** It splits the answer into atomic claims and checks each against all the documents. These
   count as unsupported:
   - outdated figures presented as current
   - facts moved to the wrong product or company
   - accepted false premises

   These are not claims at all:
   - "X is not mentioned"
   - a correction of the premise ("Rotterdam, not Berlin")

   Arithmetic and unit conversions from the documents count as supported if they are right.
2. **Rubric (`must_include`).** These are facts the answer must state, checked in code: whole-word,
   case-insensitive, citations ignored. Each group is a list of alternatives, and every group must be matched.
3. **Forbidden assertions (`forbidden`).** These are regexes for the specific wrong answer the case baits,
   checked per sentence. An `unless` regex exempts sentences that only mention the trap to correct it
   ("the 2019 press release promised 45 minutes").

These combine into the reported metrics:

| metric | an answer counts when |
|---|---|
| hallucination | the judge found an unsupported claim, **or** a forbidden assertion matched |
| trap hit | the judge says it fell for a listed trap, **or** a forbidden assertion matched |
| correct | answerable: rubric met (or judge says correct when there is no rubric), not abstained, no forbidden assertion. Unanswerable: abstained with no unsupported claims |
| clean & correct | correct **and** no hallucination: the headline number |
| abstain | the answer declines (`ABSTAIN_MSG`, or the judge says it declines) |

Steps 2 and 3 are deterministic, so a lenient judge cannot inflate the score.

## Before and after the library

The first table of the report, **Before (baseline) and after (each config)**, compares every config with
`baseline`, the same pipeline without halshield:

- `c&c`, `correct`, `halluc`: the rates, and `Δ …, pp`: the change in percentage points
- `sec/q`, `tokens/q`: cost per question, and `x time`, `x tokens`: how many times the baseline's

To print the tables of a saved run again, without the API:

```bash
python examples/show_results.py results/levels.jsonl
```

Each record keeps `tokens_by_label` and `calls_by_label`, so **Tokens per question by label** shows where each
level spends its tokens (writer, grounding, self_consistency, cove, critic). Runs saved before this field
existed show only a `total` column.

For the three protection levels only:

```bash
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py --levels --repeats 3 --out results/levels.jsonl
```

## Before and after a fix

To show that a change fixed something, save one run before the change and one after, with the same flags:

```bash
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py --out results/before.jsonl
# ... change a layer, a prompt or the retriever ...
HALSHIELD_JUDGE_MODEL=gpt-4o python examples/run_eval.py --out results/after.jsonl --compare results/before.jsonl
```

Or diff two saved runs at any time without calling the API:

```bash
python examples/compare_results.py results/before.jsonl results/after.jsonl
```

The comparison covers the configs and cases both runs share:

- **Before -> after by config:** clean & correct, hallucination, sec/q and tokens/q, before, after and the Δ.
- **Changed cases:** every case whose clean & correct rate changed, regressions first. For a regression the
  reason is the new failure; for a fix it is the failure that went away.

Use the same `--repeats` for both runs, and treat changes smaller than `c&c sd` as noise.

## Reading the report

Answers run in parallel (`--workers`, 8 by default), so they finish out of order. While it runs,
`run_eval.py` prints one row per finished answer, with a `done` counter:

```
done     config                   run  case  status  time    tokens  calls  answer                     reason
-------  -----------------------  ---  ----  ------  ------  ------  -----  -------------------------  -----------------
14/378   baseline                 0    c01   HALLUC  1.4s    298     1      It flies 45 minutes [d13]. trap: 45 minutes…
```

`time`, `tokens` and `calls` cover the pipeline for that answer; judging is not included. Under parallel load
`time` includes waiting for the API, so compare it between runs that used the same `--workers`.

When it finishes, it prints:
- the wall time
- **All answers** as one table, sorted by config (in the order above), then case, then run
- these tables:

- **Before (baseline) and after (each config):** see [Before and after the library](#before-and-after-the-library).
- **Summary:** one row per config.
  - the rates above
  - `c&c sd` / `halluc sd`: standard deviation across runs, only with `--repeats` > 1
  - `Δ c&c vs baseline`
  - `tokens/q` and `LLM calls/q`, which exclude judge tokens and the one-time index embedding
  - `evidence recall`: share of each case's `evidence` documents that retrieval put in the context
  - `sec/q` and `p95 sec`: mean and 95th-percentile pipeline time per answer
- **Clean & correct by category** and **Hallucination by category:** where each stack wins or loses.
- **Clean & correct by case (hardest first):** a fraction per config, averaged over runs.
- **Failures:** each failed answer with its reason and the number of runs it happened in. The reason is the
  first that applies:
  - `trap: …` (a forbidden assertion)
  - `unsupported: …` (judge claims)
  - `abstained…`
  - `missing: …` (rubric facts)
  - `judged incorrect`
- **Tokens per question by label:** cost split by writer, grounding, self_consistency, cove, critic.

`results.jsonl` has one JSON line per answer:
- config, run, case id, category, question
- answer, sources, flags
- `seconds`, `tokens`, `calls` for that answer
- `context_ids` (the chunks the answer was written from) and `missing_evidence` (the evidence docs none of them came from)
- with query rewriting, the searched queries appear in `flags` as `grounding: also searched for '…'`
- every verdict field: `unsupported`, `missing_facts`, `forbidden_hits`, etc.

```bash
jq -c 'select(.config == "full" and .clean_and_correct == false) | {id, answer, forbidden_hits, unsupported}' results.jsonl
```

**How much to trust the numbers:**
- With 28 cases, one case moves a rate by about 3.6 points.
- Use `--repeats 3` and compare differences against `c&c sd` before you conclude that one stack beats another.
- The writer runs at temperature 0, so repeats mainly capture API nondeterminism and SelfConsistency sampling.

## Adding a case

Add an object to `cases` in `bench.json`:

```json
{
  "id": "c19",
  "category": "conflicting_sources",
  "question": "Worded differently from the documents",
  "answerable": true,
  "gold": "The reference answer, including corrections of any false premise.",
  "must_include": [["52"], ["no", "none"]],
  "forbidden": [
    {"label": "presents the outdated figure", "pattern": "\\b45\\b", "unless": "announc|2019"}
  ],
  "traps": ["the wrong answers, in words, for the judge"],
  "evidence": ["d03", "d13"],
  "why_hard": "Why a model gets this wrong."
}
```

The loader rejects:
- missing fields, duplicate ids, or evidence that names an unknown document
- invalid regexes
- a `must_include` rubric on an unanswerable case

`pytest` also checks that every gold answer passes its own rubric and forbidden checks. Add the wrong answers
you expect to `test_forbidden_assertions` in `tests/test_bench.py`, so the patterns are known to catch them.

Tips:
- Write forbidden patterns around the wrong assertion itself, not around the trap's number alone. Use `unless`
  for the ways a correct answer would mention it.
- Prefer several alternatives in `must_include` groups over a single phrasing.
- Unanswerable cases have no `must_include`. Use `forbidden` to catch invented answers.
