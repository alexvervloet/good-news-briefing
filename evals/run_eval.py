#!/usr/bin/env python
"""
Classifier eval: runs classify() against labeled fixtures and reports accuracy.

Unlike unit tests (which swap out the LLM), this calls the real model to measure
whether the classifier is making the *right judgements* — the thing unit tests
deliberately don't check.

Requires:
  - LM Studio running (BASE_URL in src/good_news/config.py)
  - package installed: pip install -e .  (from project root)

Usage:
  python evals/run_eval.py             # summary — failures only
  python evals/run_eval.py --verbose   # show all cases including passes
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from good_news.llm import classify
from good_news.models import Article, Verdict

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


@dataclass
class Result:
    id: str
    description: str
    expect: dict[str, Any]
    verdict: Verdict | None
    failures: list[tuple[str, Any, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict is not None and not self.failures

    @property
    def errored(self) -> bool:
        return self.verdict is None


def _check(verdict: Verdict, expect: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return (field, expected, actual) tuples for every field that doesn't match."""
    return [
        (field, exp, getattr(verdict, field))
        for field, exp in expect.items()
        if getattr(verdict, field) != exp
    ]


def run() -> list[Result]:
    fixtures = json.loads(FIXTURES_PATH.read_text())
    results: list[Result] = []
    total = len(fixtures)

    print(f"Running {total} eval cases against the live model...\n")

    for i, fx in enumerate(fixtures, 1):
        print(f"  [{i:2d}/{total}] {fx['id']:<32}", end="", flush=True)
        article = Article(**fx["article"])
        verdict = classify(article)

        if verdict is None:
            print("ERROR (classify returned None)")
            results.append(Result(fx["id"], fx.get("description", ""), fx["expect"], None))
            continue

        failures = _check(verdict, fx["expect"])
        status = "pass" if not failures else f"FAIL ({', '.join(f[0] for f in failures)})"
        print(status)
        results.append(Result(fx["id"], fx.get("description", ""), fx["expect"], verdict, failures))

    return results


def report(results: list[Result], verbose: bool) -> None:
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.errored)
    total = len(results)

    suffix = f"  ({errors} error{'s' if errors != 1 else ''})" if errors else ""
    print(f"\n{'=' * 62}")
    print(f"  CLASSIFIER EVAL   {passed}/{total} passed{suffix}")
    print(f"{'=' * 62}")

    for r in results:
        if r.errored:
            print(f"\n  ERROR  [{r.id}]  {r.description}")
            print("           classify() returned None — model unreachable or bad output")
        elif r.failures:
            print(f"\n  FAIL   [{r.id}]  {r.description}")
            for f, exp, got in r.failures:
                print(f"           {f}: expected={exp!r}  got={got!r}")
            if verbose and r.verdict:
                print(f"           reason: \"{r.verdict.reason}\"")
        elif verbose:
            print(f"\n  pass   [{r.id}]  {r.description}")
            if r.verdict:
                print(f"           reason: \"{r.verdict.reason}\"")

    # Per-field accuracy
    field_stats: dict[str, list[int]] = {}  # field -> [pass_count, total_count]
    for r in results:
        if r.errored:
            continue
        failed = {f[0] for f in r.failures}
        for f in r.expect:
            ps = field_stats.setdefault(f, [0, 0])
            ps[1] += 1
            if f not in failed:
                ps[0] += 1

    print(f"\n{'─' * 40}")
    print("  Per-field accuracy:\n")
    for f, (p, n) in sorted(field_stats.items()):
        pct = 100 * p / n
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"    {f:<22}  {p:2d}/{n}  {bar}  {pct:.0f}%")

    print()


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    results = run()
    report(results, verbose=verbose)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
