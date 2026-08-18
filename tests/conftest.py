"""Shared fixtures and helpers for the test suite.

The single most important idea in here is `fake_chat` / `fake_message`: the
model speaks to us through OpenAI-shaped response objects, so to test our code
WITHOUT a running LM Studio we just hand-build those objects with whatever the
"model" should have said. No network, no GPU, fully deterministic.
"""

from __future__ import annotations
from types import SimpleNamespace

import pytest

from good_news.models import Article, Verdict


# --- builders for fake OpenAI responses -----------------------------------
# The real client returns: resp.choices[0].message.content (+ .reasoning_content
# and, for completions, choice.finish_reason). SimpleNamespace lets us mimic
# exactly that shape with plain Python -- no openai import, no mocks library.


def fake_message(content="", reasoning_content=None):
    """A stand-in for resp.choices[0].message."""
    return SimpleNamespace(content=content, reasoning_content=reasoning_content)


def fake_chat(content="", reasoning_content=None, finish_reason="stop"):
    """A stand-in for a whole chat-completion response."""
    msg = fake_message(content, reasoning_content)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


# --- a fake embedding model -----------------------------------------------
# The digest link check (guardrails.alignments) asks an embedding model whether
# each sentence really belongs to the item whose marker it carries. To test that
# logic without a GPU we embed by *topic word*: a text mentioning topics[i]
# becomes the i-th unit vector, so cosine is 1.0 for a matching sentence/item
# pair and 0.0 for a mismatched one -- meaning made deterministic.


def topic_embed(*topics):
    """An embed_fn keyed on topic words: a text mentioning topics[i] embeds to
    the i-th unit vector. A text mentioning none embeds to zero, which cosine
    scores 0 against everything."""

    def embed(texts):
        vecs = []
        for text in texts:
            v = [0.0] * len(topics)
            for i, word in enumerate(topics):
                if word in text.lower():
                    v[i] = 1.0
            vecs.append(v)
        return vecs

    return embed


@pytest.fixture
def article():
    """A minimal Article to flow through the pipeline."""
    return Article(
        title="Town plants 10,000 trees",
        summary="Volunteers reforested a hillside.",
        link="https://example.com/trees",
        source="Positive News",
    )


@pytest.fixture
def good_verdict():
    """A verdict that clears the editorial filter (see pipeline.keep)."""
    return Verdict(
        is_good_news=True,
        category="environment",
        optimism=0.9,
        is_corporate_pr=False,
        is_pure_luck=False,
        reason="Community-led reforestation with lasting impact.",
    )
