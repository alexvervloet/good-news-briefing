#!/usr/bin/env python
"""
Optimism eval: scores the model's `optimism` against agent-labeled reference scores.

This is an "agentic" eval — the reference optimism scores in optimism_fixtures.json
were assigned by an agent reading each article against CRITERIA, not derived from any
ground-truth label. It measures whether the model's *sense of how uplifting and
substantive* a story is tracks a considered human-ish judgement, which the pass/fail
classifier eval (run_eval.py) deliberately doesn't probe.

Both scores live on the model's native 0.0-1.0 optimism scale. A case PASSES when the
model's optimism is within TOLERANCE of the reference score.

Requires:
  - LM Studio running (BASE_URL in src/good_news/config.py)
  - package installed: pip install -e .  (from project root)

Usage:
  python evals/run_optimism_eval.py             # summary — failures only
  python evals/run_optimism_eval.py --verbose   # show all cases including passes
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from good_news.llm import classify
from good_news.models import Article, Verdict

FIXTURES_PATH = Path(__file__).parent / "optimism_fixtures.json"

# A case passes when |model.optimism - expected| <= TOLERANCE, on the 0.0-1.0 scale.
TOLERANCE = 0.1


@dataclass
class Result:
    id: str
    description: str
    expected: float
    verdict: Verdict | None

    @property
    def actual(self) -> float | None:
        return None if self.verdict is None else self.verdict.optimism

    @property
    def errored(self) -> bool:
        return self.verdict is None

    @property
    def delta(self) -> float | None:
        return None if self.actual is None else abs(self.actual - self.expected)

    @property
    def passed(self) -> bool:
        return self.delta is not None and self.delta <= TOLERANCE


def run() -> list[Result]:
    fixtures = json.loads(FIXTURES_PATH.read_text())
    results: list[Result] = []
    total = len(fixtures)

    print(f"Running {total} optimism eval cases against the live model")
    print(f"(pass = model optimism within ±{TOLERANCE:g} of the reference score)\n")

    for i, fx in enumerate(fixtures, 1):
        print(f"  [{i:2d}/{total}] {fx['id']:<30}", end="", flush=True)
        article = Article(**fx["article"])
        verdict = classify(article)
        r = Result(fx["id"], fx.get("description", ""), float(fx["optimism"]), verdict)
        results.append(r)

        if r.errored:
            print("ERROR (classify returned None)")
        else:
            mark = "pass" if r.passed else "FAIL"
            print(f"{mark}   expected={r.expected:.2f}  got={r.actual:.2f}  Δ={r.delta:.2f}")

    return results


def report(results: list[Result], verbose: bool) -> None:
    scored = [r for r in results if not r.errored]
    passed = sum(1 for r in scored if r.passed)
    errors = sum(1 for r in results if r.errored)
    total = len(results)

    suffix = f"  ({errors} error{'s' if errors != 1 else ''})" if errors else ""
    print(f"\n{'=' * 62}")
    print(f"  OPTIMISM EVAL   {passed}/{total} within ±{TOLERANCE:g}{suffix}")
    print(f"{'=' * 62}")

    if scored:
        mae = sum(r.delta for r in scored) / len(scored)  # type: ignore[misc]
        bias = sum(r.actual - r.expected for r in scored) / len(scored)  # type: ignore[operator]
        print(f"  mean abs error: {mae:.3f}     mean signed error: {bias:+.3f}")
        print(f"  ({'model runs optimistic' if bias > 0 else 'model runs pessimistic'} "
              f"relative to the reference)")

    for r in results:
        if r.errored:
            print(f"\n  ERROR  [{r.id}]  {r.description}")
            print("           classify() returned None — model unreachable or bad output")
        elif not r.passed:
            print(f"\n  FAIL   [{r.id}]  {r.description}")
            print(f"           expected={r.expected:.2f}  got={r.actual:.2f}  Δ={r.delta:.2f}")
            if verbose and r.verdict:
                print(f"           reason: \"{r.verdict.reason}\"")
        elif verbose:
            print(f"\n  pass   [{r.id}]  {r.description}")
            print(f"           expected={r.expected:.2f}  got={r.actual:.2f}  Δ={r.delta:.2f}")
            if r.verdict:
                print(f"           reason: \"{r.verdict.reason}\"")

    print()


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    results = run()
    report(results, verbose=verbose)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
