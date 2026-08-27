"""The editorial filter and dedup math -- the decisions that shape the briefing.

`keep` is a pure function, so it's the easiest, highest-value thing to test.
`dedupe` calls the model's embeddings, so we mock that one function and let the
real cosine/sorting logic run.
"""

from __future__ import annotations
from dataclasses import replace

import pytest

from good_news import config, pipeline
from good_news.models import Article


# --- keep(): which verdicts clear the bar ----------------------------------

def test_keep_passes_good_news(good_verdict):
    assert pipeline.keep(good_verdict) is True


def test_keep_rejects_none():
    assert pipeline.keep(None) is False


def test_keep_rejects_not_good_news(good_verdict):
    assert pipeline.keep(replace(good_verdict, is_good_news=False)) is False


def test_keep_rejects_corporate_pr(good_verdict):
    assert pipeline.keep(replace(good_verdict, is_corporate_pr=True)) is False


def test_keep_rejects_pure_luck_unless_community(good_verdict):
    lucky = replace(good_verdict, is_pure_luck=True, category="environment")
    assert pipeline.keep(lucky) is False
    # ...but luck is allowed when it's people helping people.
    community = replace(lucky, category="community_helping")
    assert pipeline.keep(community) is True


def test_keep_respects_optimism_threshold(good_verdict):
    below = replace(good_verdict, optimism=config.OPTIMISM_THRESHOLD - 0.01)
    above = replace(good_verdict, optimism=config.OPTIMISM_THRESHOLD)
    assert pipeline.keep(below) is False
    assert pipeline.keep(above) is True


# --- cosine(): pure vector math --------------------------------------------

def test_cosine_identical_vectors_is_one():
    assert pipeline.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert pipeline.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_safe():
    # Guards the na/nb == 0 branch instead of dividing by zero.
    assert pipeline.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- dedupe_links(): the same URL fetched twice ----------------------------

def test_dedupe_links_keeps_first_of_a_repeated_url():
    # Two reddit submissions pointing at one article arrive as two Articles
    # sharing a link. Only the first should survive to be classified.
    a = Article("first", "", "https://example.com/x", "reddit")
    b = Article("second", "", "https://example.com/x", "reddit")
    c = Article("other", "", "https://example.com/y", "guardian")
    assert pipeline.dedupe_links([a, b, c]) == [a, c]


def test_dedupe_links_preserves_order_and_passes_through_unique():
    items = [
        Article("a", "", "https://example.com/a", "s"),
        Article("b", "", "https://example.com/b", "s"),
    ]
    assert pipeline.dedupe_links(items) == items


# --- dedupe(): collapse near-identical coverage ----------------------------

def _art(title, optimism):
    a = Article(title, "", f"https://example.com/{title}", "src")
    a.optimism = optimism
    return a


def test_dedupe_keeps_highest_optimism_of_a_cluster(monkeypatch):
    items = [_art("dup-low", 0.6), _art("dup-high", 0.9), _art("other", 0.7)]

    # Fake embeddings: first two are identical vectors (a duplicate cluster),
    # the third is orthogonal. We map each title to a vector so the real
    # cosine/threshold logic does the actual deduping.
    vecs = {
        "dup-low": [1.0, 0.0],
        "dup-high": [1.0, 0.0],
        "other": [0.0, 1.0],
    }
    monkeypatch.setattr(pipeline, "embed", lambda titles: [vecs[t] for t in titles])

    kept = pipeline.dedupe(items)
    titles = {a.title for a in kept}
    assert titles == {"dup-high", "other"}  # the lower-optimism dup is dropped


def test_dedupe_skips_when_embeddings_fail(monkeypatch):
    items = [_art("a", 0.6), _art("b", 0.9)]

    def boom(_titles):
        raise RuntimeError("embeddings server down")

    monkeypatch.setattr(pipeline, "embed", boom)
    # Failure must degrade gracefully: return items unchanged, not crash.
    assert pipeline.dedupe(items) == items


def test_dedupe_noop_for_single_item():
    items = [_art("solo", 0.5)]
    assert pipeline.dedupe(items) == items


def test_dedupe_relaxes_threshold_when_below_min_keep(monkeypatch):
    # near-dup-high and near-dup-low have cosine similarity of 0.89 -- between
    # the strict threshold (0.86) and the relaxed one (0.92).
    # Strict pass: 0.89 > 0.86 → they collapse → 2 items survive.
    # min_keep=3: 2 < 3 → retry with relaxed pass.
    # Relaxed pass: 0.89 < 0.92 → they are NOT duplicates → 3 items survive.
    import math
    sim = 0.89
    vecs = {
        "near-dup-high": [1.0, 0.0],
        "near-dup-low":  [sim, math.sqrt(1 - sim ** 2)],  # cosine([1,0], this) == 0.89
        "other":         [0.0, 1.0],
    }
    items = [_art("near-dup-high", 0.9), _art("near-dup-low", 0.6), _art("other", 0.7)]
    monkeypatch.setattr(pipeline, "embed", lambda titles: [vecs[t] for t in titles])

    kept = pipeline.dedupe(items, min_keep=3)
    assert {a.title for a in kept} == {"near-dup-high", "near-dup-low", "other"}


def test_dedupe_stays_strict_when_min_keep_is_met(monkeypatch):
    # Same vectors as above, but min_keep=2. Strict pass gives 2 items (≥ 2),
    # so the relaxed threshold must NOT be used and the lower-optimism dup is dropped.
    import math
    sim = 0.89
    vecs = {
        "near-dup-high": [1.0, 0.0],
        "near-dup-low":  [sim, math.sqrt(1 - sim ** 2)],
        "other":         [0.0, 1.0],
    }
    items = [_art("near-dup-high", 0.9), _art("near-dup-low", 0.6), _art("other", 0.7)]
    monkeypatch.setattr(pipeline, "embed", lambda titles: [vecs[t] for t in titles])

    kept = pipeline.dedupe(items, min_keep=2)
    assert {a.title for a in kept} == {"near-dup-high", "other"}
