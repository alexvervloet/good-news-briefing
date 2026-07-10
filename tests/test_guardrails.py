"""Guardrails are the safety net over raw model output. Two of these tests are
regression tests for real bugs (see git log): chain-of-thought leaking into the
digest, and the model writing its own (broken) URLs instead of using markers.
"""

from __future__ import annotations

from good_news.guardrails import (
    answer_text,
    markers_each_on_own_line,
    message_text,
    restore_links,
)
from good_news.models import Article
from conftest import fake_message


# --- message_text: pull the answer out, falling back to reasoning ----------

def test_message_text_prefers_content():
    msg = fake_message(content="the answer", reasoning_content="some thinking")
    assert message_text(msg) == "the answer"


def test_message_text_falls_back_to_reasoning_when_content_blank():
    # Some Qwen3 builds route the whole reply into reasoning_content.
    msg = fake_message(content="   ", reasoning_content="the answer")
    assert message_text(msg) == "the answer"


def test_message_text_handles_none_content():
    assert message_text(fake_message(content=None, reasoning_content=None)) == ""


# --- answer_text: FINAL answer only, never reasoning -----------------------

def test_answer_text_strips_think_block():
    # Regression: <think> reasoning must never reach the reader.
    msg = fake_message(content="<think>let me reason...</think>Real digest text")
    assert answer_text(msg) == "Real digest text"


def test_answer_text_strips_multiline_think():
    msg = fake_message(content="<think>\nline1\nline2\n</think>\nHello")
    assert answer_text(msg) == "Hello"


def test_answer_text_does_not_fall_back_to_reasoning():
    # Unlike message_text, a blank content here means "no answer" -- it must
    # NOT surface reasoning_content (which is raw chain-of-thought).
    msg = fake_message(content="", reasoning_content="secret reasoning")
    assert answer_text(msg) == ""


# --- restore_links: swap @@N@@ markers for real URLs -----------------------

def _items():
    return [
        Article("A", "", "https://example.com/a", "src"),
        Article("B", "", "https://example.com/b", "src"),
    ]


def test_restore_links_substitutes_markers():
    # Each marker ends an item, so a blank line lands after each restored URL —
    # whatever follows a marker is the next item, never a continuation.
    out = restore_links("First @@1@@\n\nSecond @@2@@", _items())
    assert out == (
        "First https://example.com/a\n\nSecond https://example.com/b"
    )


def test_restore_links_leaves_out_of_range_marker_untouched():
    out = restore_links("ref @@9@@", _items())
    assert "@@9@@" in out


def test_restore_links_warns_on_model_written_url(capsys):
    # Regression: model paraphrases URLs into dead links; it must copy markers.
    restore_links("see https://made-up.example", _items())
    assert "model-written URL" in capsys.readouterr().err


def test_restore_links_warns_on_unresolved_marker(capsys):
    restore_links("ref @@9@@", _items())
    assert "unresolved link marker" in capsys.readouterr().err


def test_restore_links_warns_on_duplicate_markers(capsys):
    # Regression: model repeated @@1@@ for every item, making all links point
    # to the same article. The mismatch check must catch this.
    restore_links("First @@1@@\n\nSecond @@1@@", _items())
    assert "marker mismatch" in capsys.readouterr().err


def test_restore_links_warns_on_missing_marker(capsys):
    # Model dropped @@2@@ entirely — one item gets no link.
    restore_links("Only @@1@@ here", _items())
    assert "marker mismatch" in capsys.readouterr().err


def test_restore_links_separates_items_with_blank_line():
    # Regression: without a blank line between items a category collapses into
    # one run-on markdown paragraph. Each link on its own line gets one after it.
    digest = "Sentence one.\n@@1@@\nSentence two.\n@@2@@"
    out = restore_links(digest, _items())
    assert out == (
        "Sentence one.\n"
        "https://example.com/a\n"
        "\n"
        "Sentence two.\n"
        "https://example.com/b"
    )


def test_markers_each_on_own_line_true_when_isolated():
    assert markers_each_on_own_line("Sentence one.\n@@1@@\n\nSentence two.\n@@2@@")


def test_markers_each_on_own_line_false_when_inline():
    # The failure mode from the wrong-links run: markers stapled to the sentence.
    assert not markers_each_on_own_line("Sentence one. @@1@@ Sentence two. @@2@@")


def test_markers_each_on_own_line_false_when_no_markers():
    assert not markers_each_on_own_line("No markers at all here.")


def test_restore_links_separates_items_with_inline_markers():
    # Regression: the model sometimes writes the marker inline at the end of the
    # sentence instead of on its own line, leaving the restored URL mid-line. The
    # blank line still has to land after each link so the items don't run on.
    digest = "Sentence one. @@1@@ Sentence two. @@2@@"
    out = restore_links(digest, _items())
    assert out == (
        "Sentence one. https://example.com/a\n"
        "\n"
        "Sentence two. https://example.com/b"
    )
