"""End-to-end orchestration tests for pipeline.run().

run() wires together fetch -> classify -> filter -> dedupe -> digest -> deliver.
We mock every external seam (the feeds, the model, the store, the file/email
output) and let the REAL orchestration run, then assert the wiring: that the
filter is applied, that only kept articles reach the digest, that already-seen
links are skipped, and that the briefing is delivered.
"""

from __future__ import annotations

import pytest

from good_news import config, pipeline
from good_news.models import Article, Verdict


def _article(title, link, reddit=False):
    return Article(title, f"summary of {title}", link, "src", is_reddit_article=reddit)


def _verdict(good, optimism=0.9, category="environment"):
    """A verdict that passes the filter when good=True, fails when good=False."""
    return Verdict(
        is_good_news=good,
        category=category,
        optimism=optimism,
        is_corporate_pr=False,
        is_pure_luck=False,
        reason="reason",
    )


class FakeStore:
    """Stand-in for SeenStore: in-memory, records what it was asked to do."""

    def __init__(self, already_seen=()):
        self.seen = set(already_seen)
        self.marked = []
        self.committed = False

    @classmethod
    def open(cls):
        return cls()

    def is_seen(self, link):
        return link in self.seen

    def mark_seen(self, link):
        self.marked.append(link)
        self.seen.add(link)

    def commit(self):
        self.committed = True


@pytest.fixture
def wire(monkeypatch):
    """Mock every seam of run() and return a dict to inspect/override.

    By default: two articles, one passes the filter and one doesn't; the model
    and feeds are faked; the digest and delivery are recorded, not performed.
    """
    state = {
        "articles": [
            _article("Good story", "https://example.com/good"),
            _article("Bad story", "https://example.com/bad"),
        ],
        # title -> verdict the fake model returns for that article
        "verdicts": {
            "Good story": _verdict(True),
            "Bad story": _verdict(False),
        },
        "classified": [],     # articles the model was asked to judge
        "digest_items": None,  # articles that reached write_digest
        "written": [],         # (document, today) passed to write_briefing
        "emailed": [],         # (subject, body) passed to send_email
        "store": None,
    }

    def fake_classify(article):
        state["classified"].append(article)
        return state["verdicts"].get(article.title)

    def fake_write_digest(items):
        state["digest_items"] = items
        return "DIGEST BODY"

    def fake_store_open():
        store = FakeStore()
        state["store"] = store
        return store

    monkeypatch.setattr(pipeline, "fetch", lambda per_feed: state["articles"])
    monkeypatch.setattr(pipeline, "classify", fake_classify)
    monkeypatch.setattr(pipeline, "embed", lambda titles: [[1.0] for _ in titles])
    monkeypatch.setattr(pipeline, "write_digest", fake_write_digest)
    monkeypatch.setattr(pipeline.SeenStore, "open", staticmethod(fake_store_open))
    monkeypatch.setattr(
        pipeline, "write_briefing",
        lambda doc, today: state["written"].append((doc, today)),
    )
    monkeypatch.setattr(
        pipeline, "send_email",
        lambda subject, body: state["emailed"].append((subject, body)),
    )
    return state


# --- the filter is applied: only good news reaches the digest --------------

def test_run_only_digests_articles_that_pass_the_filter(wire):
    pipeline.run(dry_run=True)
    titles = [a.title for a in wire["digest_items"]]
    assert titles == ["Good story"]  # the bad story was filtered out


def test_run_copies_verdict_fields_onto_kept_article(wire):
    pipeline.run(dry_run=True)
    kept = wire["digest_items"][0]
    assert kept.category == "environment"
    assert kept.optimism == 0.9
    assert kept.reason == "reason"


def test_run_dry_run_prints_digest_to_stdout(wire, capsys):
    pipeline.run(dry_run=True)
    out = capsys.readouterr().out
    assert "DIGEST BODY" in out
    assert out.lstrip().startswith("# Good News")


def test_run_dry_run_does_not_touch_store_or_disk(wire):
    pipeline.run(dry_run=True)
    assert wire["store"] is None     # SeenStore.open() never called
    assert wire["written"] == []     # nothing written to disk


# --- short-circuit when nothing clears the bar -----------------------------

def test_run_writes_nothing_when_no_news_passes(wire):
    wire["verdicts"]["Good story"] = _verdict(False)  # now everything fails
    pipeline.run(dry_run=True)
    assert wire["digest_items"] is None
    assert wire["written"] == []


# --- a real run: store + delivery ------------------------------------------

def test_run_marks_seen_and_writes_briefing(wire):
    pipeline.run(dry_run=False)
    store = wire["store"]
    assert store is not None
    assert set(store.marked) == {
        "https://example.com/good",
        "https://example.com/bad",
    }
    assert store.committed is True
    assert len(wire["written"]) == 1
    document, _today = wire["written"][0]
    assert "DIGEST BODY" in document


def test_run_skips_already_seen_links(wire, monkeypatch):
    # Pre-seed the store so the good story is already seen -> never re-judged.
    seeded = FakeStore(already_seen={"https://example.com/good"})
    wire["store_seeded"] = seeded
    monkeypatch.setattr(
        pipeline.SeenStore, "open", staticmethod(lambda: seeded)
    )
    pipeline.run(dry_run=False)
    judged = [a.title for a in wire["classified"]]
    assert judged == ["Bad story"]  # the seen one was skipped before classify


def test_run_emails_only_when_send_mail(wire):
    pipeline.run(dry_run=False, send_mail=True)
    assert len(wire["emailed"]) == 1
    subject, body = wire["emailed"][0]
    assert subject.startswith("Good News")
    assert "DIGEST BODY" in body


# --- reddit articles get their body crawled before judging -----------------

def test_run_crawls_reddit_article_body(wire, monkeypatch):
    reddit = _article("Reddit story", "https://news.example.com/x", reddit=True)
    wire["articles"] = [reddit]
    wire["verdicts"]["Reddit story"] = _verdict(True)
    monkeypatch.setattr(pipeline, "fetch_article_text", lambda url: "CRAWLED BODY")

    pipeline.run(dry_run=True)
    # The crawled body replaced the RSS summary before the model saw it.
    assert wire["classified"][0].summary == "CRAWLED BODY"


# --- the whole briefing is capped, not just each category ------------------

def test_run_caps_the_briefing_at_max_items(wire, monkeypatch):
    # MAX_PER_CATEGORY bounds the internal categories, which the reader never
    # sees: the model merges them into its own headings at write time, so a
    # briefing can be seven categories long. MAX_ITEMS is what bounds the thing
    # that actually arrives.
    cats = [
        "politics_social", "anti_corporate", "technology", "community_helping",
        "science_health", "environment", "other",
    ]
    n = config.MAX_ITEMS * 2
    wire["articles"] = [_article(f"Story {i}", f"https://example.com/{i}") for i in range(n)]
    wire["verdicts"] = {
        f"Story {i}": _verdict(True, optimism=0.9 - i / 100, category=cats[i % len(cats)])
        for i in range(n)
    }
    # Spread thinly enough that MAX_PER_CATEGORY can't be what trims them.
    assert n / len(cats) <= config.MAX_PER_CATEGORY
    # Orthogonal vectors: every story is distinct, so dedupe collapses nothing.
    monkeypatch.setattr(
        pipeline, "embed",
        lambda texts: [[1.0 if j == i else 0.0 for j in range(len(texts))]
                       for i in range(len(texts))],
    )

    pipeline.run(dry_run=True)

    kept = wire["digest_items"]
    assert len(kept) == config.MAX_ITEMS
    # And it keeps the best ones, not the first ones it happened to see.
    assert [a.title for a in kept] == [f"Story {i}" for i in range(config.MAX_ITEMS)]


def test_run_leaves_a_short_briefing_alone(wire):
    # The cap must only ever trim; a normal evening is well under it.
    pipeline.run(dry_run=True)
    assert len(wire["digest_items"]) == 1
