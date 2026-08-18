"""Guardrails over what the model returns.

Everything here acts on a model's raw output -- pulling out the answer,
stripping leaked chain-of-thought, and splicing real links back in -- so that
nothing the reader should never see (reasoning, hallucinated URLs) reaches the
digest. The request-side knobs (think_suffix, think_extra_body) live in llm.py.
"""

from __future__ import annotations
import math
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .models import Article


def message_text(msg: Any) -> str:
    """Pull the answer text out of a chat message.

    Some LM Studio builds of Qwen3.6 route the whole reply into
    `reasoning_content` and leave `content` empty (even with /no_think), so
    fall back to reasoning_content when content is blank.
    """
    content = (msg.content or "").strip()
    if content:
        return content
    return (getattr(msg, "reasoning_content", None) or "").strip()


# Qwen3 wraps reasoning in <think>...</think>. When a build inlines that into
# `content` instead of a separate reasoning_content field, strip it out so the
# chain-of-thought never reaches the reader.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# The outermost {...} in a string. Greedy + DOTALL so it spans a multi-line,
# pretty-printed object from the first brace to the last.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def verdict_json(msg: Any) -> str:
    """Pull a complete JSON verdict out of a reply, ignoring leaked reasoning.

    A reasoning build may prepend a <think>...</think> block (or stray prose)
    before the JSON; we strip that and grab the outermost {...}. Returns "" when
    there is no *complete* object -- e.g. the token cap fired mid-JSON, leaving
    an unterminated string -- so the caller can report truncation instead of
    handing json.loads a fragment and getting a cryptic parse error.
    """
    raw = _THINK_RE.sub("", message_text(msg)).strip()
    m = _JSON_OBJ_RE.search(raw)
    return m.group(0) if m else ""


def answer_text(msg: Any) -> str:
    """The model's FINAL answer only -- never its reasoning.

    Unlike message_text(), this deliberately does NOT fall back to
    reasoning_content: on builds that emit a chain-of-thought, that field holds
    raw thinking, which must never be surfaced to the reader. Blank content here
    means the model was still reasoning when it stopped, so there is no answer.
    """
    content = _THINK_RE.sub("", msg.content or "").strip()
    # If the token cap fired mid-reasoning, content starts with an unclosed
    # <think> tag (the regex above requires a closing tag to match). Return ""
    # so the caller sees "emitted only reasoning" rather than a confusing error.
    if content.startswith("<think>"):
        return ""
    return content


# Each item is tagged with an @@N@@ marker the model echoes onto the link line;
# we swap markers for real links after generation. The model paraphrases URLs
# (turning reasonstobecheerful.world into a dead reasonsbecheerful.world), so it
# never sees or writes a URL -- it only copies the opaque marker.
_LINK_MARK_RE = re.compile(r"@@(\d+)@@")


def markers_each_on_own_line(text: str) -> bool:
    """True when every @@N@@ marker sits alone on its own line, as instructed.

    The digest prompt tells the model to copy each item's marker onto its own
    line. When it instead writes the marker inline at the end of a sentence, the
    run tends to be a sloppy one -- the same runs that staple links onto the
    wrong stories. So a marker that shares its line with other text is a cheap
    signal to regenerate before the mismatch reaches the reader.
    """
    markers = _LINK_MARK_RE.findall(text)
    if not markers:
        return False
    alone = len(re.findall(r"(?m)^[ \t]*@@\d+@@[ \t]*$", text))
    return alone == len(markers)


# --- assignment: does each sentence carry ITS OWN item's marker? -----------
#
# The audit inside restore_links() is a *presence* check: it proves every number
# appears exactly once, which a permutation satisfies too. So the dangerous
# failure -- markers rotated across sentences, every link pointing at the next
# story -- is invisible to it, and markers_each_on_own_line() only catches the
# sloppy runs that happen to correlate with it. The one thing that can prove an
# *assignment* is meaning: embed each generated sentence and each source item,
# and check that a sentence really is closest to the item whose marker sits next
# to it. The embedding model is already loaded for dedupe, so this is one extra
# call per digest.

# How much closer another item must be before we believe the model misassigned a
# marker. Rival items here are stories dedupe already judged distinct, so a real
# mismatch shows a large gap; the margin only guards against two same-category
# items scoring near-identically. See DIGEST_ALIGNMENT_MARGIN in config.
ALIGNMENT_MARGIN = 0.05

# A thematic grouping header, in either shape the model writes it (`## Theme` or
# a bold-only line). Headers belong to no single item, so they are never the
# sentence a marker was written for.
_HEADER_RE = re.compile(r"^(#{1,6}\s|\*\*[^\n*]+\*\*[ \t]*$)")

EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when either has no magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Alignment:
    """One digest sentence, the marker it got, and the marker it deserves."""

    start: int  # span of the written marker in the digest text, for repair
    end: int
    sentence: str
    written: int  # marker number the model put on this sentence
    best: int  # marker of the item the sentence actually matches best
    written_score: float
    best_score: float

    @property
    def gap(self) -> float:
        """How much better the best-matching item fits than the written one."""
        return self.best_score - self.written_score


def _marker_sentences(text: str) -> list[tuple[re.Match[str], str]]:
    """Pair every @@N@@ marker with the sentence it was written for.

    An item is a sentence followed by its marker, so the sentence is the last
    line of the block between the previous marker and this one. Dropping headers
    and blank lines on the way there also drops the opening tone-setting line,
    which belongs to no item. When the model writes the marker inline
    (`... home. @@1@@`) that last line is the sentence up to the marker, which is
    exactly what we want -- so this reads the sloppy format correctly too.
    """
    pairs: list[tuple[re.Match[str], str]] = []
    prev = 0
    for m in _LINK_MARK_RE.finditer(text):
        lines = [ln.strip() for ln in text[prev : m.start()].splitlines()]
        body = [ln for ln in lines if ln and not _HEADER_RE.match(ln)]
        pairs.append((m, body[-1] if body else ""))
        prev = m.end()
    return pairs


def alignments(
    text: str, items: list[Article], embed_fn: EmbedFn
) -> list[Alignment]:
    """Score every digest sentence against every item it could belong to.

    Returns one Alignment per marker, in output order. Returns [] -- meaning "no
    evidence of a problem" -- when there is nothing to mix up (a single item) or
    when the embedding model is unreachable: a link check must never be the
    reason a briefing fails to go out.
    """
    pairs = _marker_sentences(text)
    if len(pairs) < 2 or len(items) < 2:
        return []
    sentences = [s for _, s in pairs]
    item_texts = [f"{it.title} {it.reason or ''}".strip() for it in items]
    try:
        vecs = embed_fn(sentences + item_texts)
    except Exception as e:
        print(
            f"  ! embeddings unavailable, skipping the digest link check: {e}",
            file=sys.stderr,
        )
        return []
    if len(vecs) != len(sentences) + len(item_texts):
        print(
            "  ! embedding count mismatch, skipping the digest link check",
            file=sys.stderr,
        )
        return []
    sent_vecs, item_vecs = vecs[: len(sentences)], vecs[len(sentences) :]

    out: list[Alignment] = []
    for (m, sentence), sv in zip(pairs, sent_vecs):
        scores = [cosine(sv, iv) for iv in item_vecs]
        best = max(range(len(scores)), key=lambda i: scores[i])
        written = int(m.group(1))
        # An out-of-range marker scores 0, so any real match beats it and the
        # repair below can rescue it -- restore_links would only warn.
        written_score = scores[written - 1] if 1 <= written <= len(scores) else 0.0
        out.append(
            Alignment(
                start=m.start(),
                end=m.end(),
                sentence=sentence,
                written=written,
                best=best + 1,
                written_score=written_score,
                best_score=scores[best],
            )
        )
    return out


def misaligned(
    aligns: list[Alignment], margin: float = ALIGNMENT_MARGIN
) -> list[Alignment]:
    """The sentences whose marker is confidently the wrong one."""
    return [a for a in aligns if a.best != a.written and a.gap > margin]


def realign(
    text: str, aligns: list[Alignment], margin: float = ALIGNMENT_MARGIN
) -> str:
    """Move each confidently-misplaced marker onto the sentence it belongs to.

    A last resort, not the first line of defence: a cleanly regenerated draft is
    always better than a repaired one, so this runs only once a re-roll has come
    back misaligned too. It rewrites nothing unless the corrected mapping is
    still one-to-one -- a repair that would link one item twice and leave another
    unlinked is a sign the similarity evidence is muddled, and then the model's
    own mapping (plus the loud warning) is the safer thing to ship.
    """
    if not misaligned(aligns, margin):
        return text
    corrected = [
        a.best if (a.best != a.written and a.gap > margin) else a.written
        for a in aligns
    ]
    if len(set(corrected)) != len(corrected):
        print(
            "  ! digest links look misassigned, but the repair would reuse a "
            "link; leaving the model's markers alone",
            file=sys.stderr,
        )
        return text

    out: list[str] = []
    prev = 0
    for a, new in zip(aligns, corrected):
        out.append(text[prev : a.start])
        out.append(f"@@{new}@@")
        if new != a.written:
            print(
                f'  ! digest: "{a.sentence[:60]}" matches @@{new}@@ '
                f"({a.best_score:.2f}) far better than the @@{a.written}@@ the "
                f"model wrote ({a.written_score:.2f}); repointing its link",
                file=sys.stderr,
            )
        prev = a.end
    out.append(text[prev:])
    return "".join(out)


def restore_links(text: str, items: list[Article]) -> str:
    """Replace @@N@@ markers with the real link for items[N-1]."""

    def sub(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return items[idx - 1].link if 1 <= idx <= len(items) else m.group(0)

    # Check the raw output (not the substituted result, which is all real links):
    # a URL here means the model wrote one despite being told to copy markers.
    if "http" in text:
        print("  ! digest contains a model-written URL (markers expected)", file=sys.stderr)
    # Catch the model reusing the same marker number for every item, which would
    # make all links resolve to the same article.
    found = sorted(int(m) for m in _LINK_MARK_RE.findall(text))
    expected = list(range(1, len(items) + 1))
    if found != expected:
        print(
            f"  ! marker mismatch: expected @@1@@–@@{len(items)}@@ each once, "
            f"got {found}",
            file=sys.stderr,
        )
    restored = _LINK_MARK_RE.sub(sub, text)
    # Markers the model dropped or mangled instead of echoing them verbatim.
    if leftover := _LINK_MARK_RE.findall(restored):
        print(
            f"  ! digest left {len(leftover)} unresolved link marker(s)",
            file=sys.stderr,
        )
    return _space_items(restored)


def _space_items(text: str) -> str:
    """Guarantee a blank line after every link and bold-only header line.

    Each item is a sentence followed by its link; without a blank line between
    items markdown collapses a whole category into one run-on paragraph. The
    model's spacing is unreliable, so enforce it on the links we control rather
    than asking the model to get it right.

    Every URL here came from a restored @@N@@ marker, and every marker sits at an
    item boundary, so a blank line belongs after each URL wherever it landed --
    including when the model wrote the marker inline at the end of the sentence
    (`... home. @@1@@ A patient ...`) instead of on its own line. Anchoring to
    line start missed that case and ran the items together, so match every URL.

    When the model writes **Category** (bold-only) instead of ## Category, the
    Python markdown library places the header and the following text in the same
    <p>, so email clients render them on the same visual line. Inserting \n\n
    after such lines splits them into separate blocks.
    """
    text = re.sub(r"(https?://\S+)[ \t]*\n*", r"\1\n\n", text)
    text = re.sub(r"(?m)^(\*\*[^\n*]+\*\*)[ \t]*\n(?!\n)", r"\1\n\n", text)
    return text.rstrip("\n")
