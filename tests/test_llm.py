"""Testing the model-calling functions WITHOUT a model.

This is the template for the whole "AI pipeline" testing approach: we replace
`llm.client` (the OpenAI client) with a fake whose `.create` returns a canned,
OpenAI-shaped response. Then classify()/write_digest() run for real against
known input -- so we test OUR parsing, guardrails, and error handling, never the
model's judgement (that's an eval, not a unit test).
"""

from __future__ import annotations
import json
from types import SimpleNamespace

import pytest

from good_news import llm
from good_news.models import Article
from conftest import fake_chat, topic_embed


def install_fake_embed(monkeypatch, embed_fn=None):
    """Point llm.embed at a fake. The default embeds everything to zero, so the
    digest link check (guardrails.alignments) finds no evidence either way and
    stays out of tests that aren't about it; pass a topic_embed() to give the
    sentences meaning."""
    monkeypatch.setattr(
        llm, "embed", embed_fn or (lambda texts: [[0.0] for _ in texts])
    )


def install_fake_client(monkeypatch, response):
    """Point llm.client at a fake that returns `response` from both the chat
    and embeddings endpoints. Returns a list that records the call kwargs."""
    install_fake_embed(monkeypatch)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        embeddings=SimpleNamespace(create=create),
    )
    monkeypatch.setattr(llm, "client", fake)
    return calls


# --- classify(): RSS item -> structured Verdict ----------------------------

def test_classify_parses_verdict(monkeypatch, article):
    payload = {
        "is_good_news": True,
        "category": "environment",
        "optimism": 0.8,
        "is_corporate_pr": False,
        "is_pure_luck": False,
        "reason": "real reforestation",
    }
    install_fake_client(monkeypatch, fake_chat(content=json.dumps(payload)))

    v = llm.classify(article)
    assert v is not None
    assert v.category == "environment" and v.optimism == 0.8


def test_classify_returns_none_on_bad_json(monkeypatch, article):
    # Model emitted prose instead of JSON -- classify must swallow it and skip
    # the article rather than crash the whole run.
    install_fake_client(monkeypatch, fake_chat(content="Sorry, I can't do that."))
    assert llm.classify(article) is None


def test_classify_returns_none_when_client_raises(monkeypatch, article):
    def boom(**kwargs):
        raise ConnectionError("LM Studio offline")

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
    )
    monkeypatch.setattr(llm, "client", fake)
    assert llm.classify(article) is None


# --- write_digest(): items -> markdown, with link splicing -----------------

def test_write_digest_restores_links(monkeypatch, article):
    article.category, article.reason = "environment", "great stuff"
    # The "model" echoes the @@1@@ marker; restore_links swaps in the real URL.
    install_fake_client(
        monkeypatch, fake_chat(content="Wonderful news! @@1@@")
    )

    out = llm.write_digest([article])
    assert article.link in out
    assert "@@1@@" not in out


def install_fake_sequence(monkeypatch, responses):
    """Point llm.client.chat at a fake that returns `responses` in order (the
    last is reused once exhausted). Returns the list recording call kwargs."""
    calls = []
    seq = list(responses)

    def create(**kwargs):
        calls.append(kwargs)
        return seq[min(len(calls) - 1, len(seq) - 1)]

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(llm, "client", fake)
    install_fake_embed(monkeypatch)
    return calls


def test_write_digest_retries_when_markers_inline(monkeypatch, article):
    # First draft staples both markers inline (the wrong-links sloppiness); the
    # retry comes back clean, so write_digest adopts it and links resolve.
    other = type(article)("Other win", "good", "https://example.com/b", "src")
    article.category, article.reason = "environment", "great stuff"
    other.category, other.reason = "community_helping", "kind stuff"
    calls = install_fake_sequence(
        monkeypatch,
        [
            fake_chat(content="One. @@1@@ Two. @@2@@"),
            fake_chat(content="One.\n@@1@@\n\nTwo.\n@@2@@"),
        ],
    )

    out = llm.write_digest([article, other])
    assert len(calls) == 2  # it regenerated once
    assert article.link in out and other.link in out
    assert "@@1@@" not in out and "@@2@@" not in out


def test_write_digest_no_retry_when_markers_clean(monkeypatch, article):
    # A well-formatted first draft must not trigger a wasteful second call.
    article.category, article.reason = "environment", "great stuff"
    calls = install_fake_client(
        monkeypatch, fake_chat(content="Wonderful news!\n@@1@@")
    )
    llm.write_digest([article])
    assert len(calls) == 1


def test_write_digest_keeps_first_when_retry_still_bad(monkeypatch, article):
    # If the re-roll is no better, keep the usable first draft rather than a
    # worse one. Links still resolve from the inline first draft.
    article.category, article.reason = "environment", "great stuff"
    calls = install_fake_sequence(
        monkeypatch,
        [
            fake_chat(content="Wonderful. @@1@@"),
            fake_chat(content="Still inline. @@1@@"),
        ],
    )
    out = llm.write_digest([article])
    assert len(calls) == 2  # it tried once more
    assert article.link in out and "@@1@@" not in out


def test_write_digest_raises_on_truncation(monkeypatch, article):
    # finish_reason == "length" is how chain-of-thought leaked before; fail loud.
    install_fake_client(
        monkeypatch, fake_chat(content="half a dige", finish_reason="length")
    )
    with pytest.raises(RuntimeError, match="token cap"):
        llm.write_digest([article])


def test_write_digest_raises_on_empty_answer(monkeypatch, article):
    # Model emitted only reasoning -> answer_text is blank -> no digest.
    install_fake_client(monkeypatch, fake_chat(content="<think>hmm</think>"))
    with pytest.raises(RuntimeError, match="no digest text"):
        llm.write_digest([article])


# --- write_digest(): the right link on the right story ---------------------
#
# The bug these cover: the model copies every marker exactly once but permutes
# them across sentences, so each story is mailed its neighbour's link. Presence
# checks can't see it; only the semantic check can.

def _two_stories():
    trees = Article("Town plants 10,000 trees", "", "https://example.com/trees",
                        "src", category="environment", reason="a hillside reforested")
    chair = Article("Park gets all-terrain wheelchair", "",
                        "https://example.com/chair", "src",
                        category="community_helping", reason="barriers removed")
    return [trees, chair]


_ROTATED = "A hillside is dense with trees again.\n@@2@@\n\nThe park lends a wheelchair.\n@@1@@"
_CORRECT = "A hillside is dense with trees again.\n@@1@@\n\nThe park lends a wheelchair.\n@@2@@"
_EXPECTED = (
    "A hillside is dense with trees again.\n"
    "https://example.com/trees\n"
    "\n"
    "The park lends a wheelchair.\n"
    "https://example.com/chair"
)


def test_write_digest_regenerates_when_links_are_misassigned(monkeypatch):
    # The markers are perfectly formatted -- one per line, each used once -- so
    # every older guardrail passes. Only the meaning of the sentences reveals
    # that they have been swapped, and that has to be enough to force a re-roll.
    calls = install_fake_sequence(
        monkeypatch, [fake_chat(content=_ROTATED), fake_chat(content=_CORRECT)]
    )
    install_fake_embed(monkeypatch, topic_embed("trees", "wheelchair"))

    out = llm.write_digest(_two_stories())
    assert len(calls) == 2  # it regenerated
    assert out == _EXPECTED


def test_write_digest_repairs_when_the_retry_is_also_misassigned(monkeypatch):
    # Both drafts swap the markers. Rather than mail a story its neighbour's
    # link, put each marker back on the sentence it was written for.
    calls = install_fake_sequence(
        monkeypatch, [fake_chat(content=_ROTATED), fake_chat(content=_ROTATED)]
    )
    install_fake_embed(monkeypatch, topic_embed("trees", "wheelchair"))

    out = llm.write_digest(_two_stories())
    assert len(calls) == 2
    assert out == _EXPECTED


def test_write_digest_does_not_reroll_a_correct_digest(monkeypatch):
    # The check must not fire on a good draft: a needless re-roll costs a whole
    # generation and risks a worse briefing.
    calls = install_fake_sequence(monkeypatch, [fake_chat(content=_CORRECT)])
    install_fake_embed(monkeypatch, topic_embed("trees", "wheelchair"))

    assert llm.write_digest(_two_stories()) == _EXPECTED
    assert len(calls) == 1


def test_write_digest_survives_a_dead_embedding_model(monkeypatch, capsys):
    # Fail open: an unreachable embedding model must not cost us the briefing,
    # and must not trigger a pointless re-roll either.
    def down(_texts):
        raise RuntimeError("connection refused")

    calls = install_fake_sequence(monkeypatch, [fake_chat(content=_ROTATED)])
    install_fake_embed(monkeypatch, down)

    out = llm.write_digest(_two_stories())
    assert len(calls) == 1
    assert "https://example.com/trees" in out and "https://example.com/chair" in out
    assert "skipping the digest link check" in capsys.readouterr().err
