# Eval results

Committed transcripts from real eval runs, so the numbers are visible without
needing the hardware. Both evals hit a live local model (they are **not** mocked
— that's the point; the unit tests mock the model, these measure its judgement).

**Run:** 2026-07-07 · **model:** `qwen3.6-35b-a3b` (Qwen3.6-35B-A3B MoE, served
by LM Studio on an RTX 3090) · **embed:** `text-embedding-qwen3-embedding-0.6b`

| Eval | Result | Detail |
|---|---|---|
| Classifier ([`run_eval.py`](../run_eval.py)) | **23/25** | [`classifier-eval-2026-07-07.txt`](classifier-eval-2026-07-07.txt) |
| Optimism ([`run_optimism_eval.py`](../run_optimism_eval.py)) | **19/20** within ±0.1 | [`optimism-eval-2026-07-07.txt`](optimism-eval-2026-07-07.txt) |

## Classifier — 23/25

Per-field accuracy across the 25 hand-labeled fixtures:

| Field | Accuracy |
|---|---|
| `is_good_news` | 25/25 (100%) |
| `is_corporate_pr` | 3/3 (100%) |
| `is_pure_luck` | 2/2 (100%) |
| `category` | 14/16 (88%) |

Both misses (`pol_1`, `pol_2`) are **category-only** — the model correctly keeps
the story and gets every boolean right, but files a tenant-protection / voter-ID
ruling under a neighbouring category. The judgement that matters (keep vs. drop,
PR vs. genuine, luck vs. kindness) is perfect on this set.

## Optimism — 19/20 within ±0.1

A reference-graded eval: each fixture carries a reference optimism score (0–1) —
LLM-drafted against `CRITERIA`, then reviewed and adjusted by hand — and a case
passes when the model lands within ±0.1 of it.

- **Within tolerance:** 19/20
- **Mean absolute error:** 0.049
- **Mean signed error:** −0.007 (essentially unbiased; a hair pessimistic)

The near-zero signed error is the headline: an earlier prompt scored the
0.25–0.55 "announced / pledged / pilot" band systematically **high** on two
different model families. Ordering the JSON schema so the model writes its
`reason` before the number, and turning the soft "lean lower" guidance into hard
caps, removed that one-directional inflation — see
[`../../LEARNINGS.md`](../../LEARNINGS.md) for the before/after. The single miss
(`opt_microgrant_pilot`, 0.30 vs. 0.40 reference) is off by exactly 0.10.

## Reproduce

Requires LM Studio reachable at `BASE_URL` (see `src/good_news/config.py`) with
the chat + embedding models loaded:

```bash
pip install -e ".[test]"
python evals/run_eval.py --verbose
python evals/run_optimism_eval.py --verbose
```
