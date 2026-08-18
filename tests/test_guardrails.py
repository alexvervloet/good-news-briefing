"""Guardrails are the safety net over raw model output. Two of these tests are
regression tests for real bugs (see git log): chain-of-thought leaking into the
digest, and the model writing its own (broken) URLs instead of using markers.
"""

from __future__ import annotations

from good_news.guardrails import (
    alignments,
    answer_text,
    markers_each_on_own_line,
    message_text,
    misaligned,
    realign,
    restore_links,
)
from good_news.models import Article
from conftest import fake_message, topic_embed


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


# --- alignments / misaligned / realign: is each marker on the RIGHT sentence?
#
# Regression tests for the failure the marker audit above cannot see: the model
# copies every marker exactly once, but *permuted* across the sentences, so each
# story is mailed with its neighbour's link.

def _stories():
    return [
        Article("Town plants 10,000 trees", "", "https://example.com/trees", "src",
                reason="volunteers reforested a hillside"),
        Article("Park gets all-terrain wheelchair", "", "https://example.com/chair",
                "src", reason="removing barriers for disabled visitors"),
        Article("Neighbours build micro-shelters", "", "https://example.com/shelter",
                "src", reason="housing before winter arrives"),
    ]


_EMBED = topic_embed("trees", "wheelchair", "shelter")


def _digest(markers):
    """A three-item digest whose sentences carry `markers` in output order."""
    sentences = [
        "A hillside is dense with trees again.",
        "The park now lends an all-terrain wheelchair.",
        "A shelter goes up before the cold does.",
    ]
    return "\n\n".join(
        f"{s}\n@@{m}@@" for s, m in zip(sentences, markers)
    )


def test_alignments_sees_nothing_wrong_in_a_correct_digest():
    aligns = alignments(_digest([1, 2, 3]), _stories(), _EMBED)
    assert [a.written for a in aligns] == [1, 2, 3]
    assert misaligned(aligns) == []


def test_alignments_catches_a_rotation():
    # The reported bug: every sentence carries the NEXT story's marker. Each
    # number still appears exactly once, so restore_links' audit stays silent.
    aligns = alignments(_digest([2, 3, 1]), _stories(), _EMBED)
    assert [a.best for a in aligns] == [1, 2, 3]
    assert len(misaligned(aligns)) == 3


def test_alignments_skips_headers_and_the_opening_line():
    # The sentence a marker belongs to is never the thematic header above it,
    # nor the tone-setting line that opens the briefing.
    digest = (
        "Here are a few quiet reminders that care still moves through the world.\n\n"
        "**Green things**\n\n"
        "A hillside is dense with trees again.\n@@1@@\n\n"
        "## Community & Access\n\n"
        "The park now lends an all-terrain wheelchair.\n@@2@@\n\n"
        "A shelter goes up before the cold does.\n@@3@@"
    )
    assert misaligned(alignments(digest, _stories(), _EMBED)) == []


def test_alignments_reads_inline_markers():
    # The sloppy format the model sometimes falls into must still be scored --
    # that is exactly the run most likely to have permuted its markers.
    digest = (
        "A hillside is dense with trees again. @@2@@ "
        "The park now lends an all-terrain wheelchair. @@1@@"
    )
    assert len(misaligned(alignments(digest, _stories()[:2], _EMBED))) == 2


def test_alignments_is_inert_for_a_single_item():
    # One item cannot be mixed up with anything, so don't spend an embed call.
    def explode(_texts):
        raise AssertionError("should not embed")

    assert alignments("Only one.\n@@1@@", _stories()[:1], explode) == []


def test_alignments_fails_open_when_embeddings_are_down(capsys):
    # A link check must never be the reason a briefing fails to go out.
    def down(_texts):
        raise RuntimeError("connection refused")

    assert alignments(_digest([2, 3, 1]), _stories(), down) == []
    assert "skipping the digest link check" in capsys.readouterr().err


def test_misaligned_respects_the_margin():
    # Two stories in the same category can sit close enough that the rival
    # scores a hair higher. A near-tie is not evidence of a swap, so the margin
    # has to swallow it -- otherwise a clean digest gets re-rolled for nothing.
    def near_tie(_texts):
        return [
            [1.00, 0.0000],  # sentence 1
            [0.00, 1.0000],  # sentence 2
            [0.70, 0.7141],  # item 1 (unit): 0.70 with sentence 1
            [0.72, 0.6939],  # item 2 (unit): 0.72 -- 0.02 better, and the same
        ]                    # 0.02 the other way round for sentence 2

    text = "First.\n@@1@@\n\nSecond.\n@@2@@"
    aligns = alignments(text, _stories()[:2], near_tie)
    assert misaligned(aligns, margin=0.05) == []
    assert len(misaligned(aligns, margin=0.01)) == 2  # the gap is there, ignored
    assert realign(text, aligns, margin=0.05) == text


def test_realign_repoints_a_rotated_digest():
    aligns = alignments(_digest([2, 3, 1]), _stories(), _EMBED)
    fixed = realign(_digest([2, 3, 1]), aligns)
    assert fixed == _digest([1, 2, 3])


def test_realign_leaves_a_correct_digest_untouched():
    text = _digest([1, 2, 3])
    assert realign(text, alignments(text, _stories(), _EMBED)) is text


def test_realign_refuses_a_repair_that_reuses_a_link(capsys):
    # Both sentences are about trees, so both want @@1@@. Repairing would link
    # one story twice and leave another unlinked -- worse than the model's guess.
    text = "Trees are back.\n@@1@@\n\nMore trees are back.\n@@2@@"
    aligns = alignments(text, _stories(), _EMBED)
    assert realign(text, aligns) == text
    assert "would reuse a link" in capsys.readouterr().err


def test_realign_rescues_an_out_of_range_marker():
    # restore_links can only warn about @@9@@; the semantic check knows which
    # story the sentence was written for and can put the right marker back.
    text = _digest([1, 9, 3])
    fixed = realign(text, alignments(text, _stories(), _EMBED))
    assert fixed == _digest([1, 2, 3])
