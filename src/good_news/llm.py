"""The LM Studio / OpenAI-client wrapper: classify(), write_digest(), embed().

LM Studio speaks the OpenAI API, so we talk to it through the openai client
pointed at the LAN host. The model is local, so the api_key can be anything.
"""

from __future__ import annotations
import json
import re
import sys
from typing import Any, cast
from openai import OpenAI

from . import config
from .guardrails import (
    Alignment,
    alignments,
    answer_text,
    markers_each_on_own_line,
    misaligned,
    realign,
    restore_links,
    verdict_json,
)
from .models import Article, Verdict
from .prompts import CRITERIA, DIGEST_PROMPT, VERDICT_SCHEMA

client = OpenAI(base_url=config.BASE_URL, api_key="lm-studio")  # key can be anything


def _model_family() -> str:
    """Which reasoning-control dialect config.CHAT_MODEL speaks.

    Switching models is done by (un)commenting a CHAT_MODEL line in config, so
    we detect the family from that id at call time -- no separate setting to
    keep in sync. The '/no_think' suffix and `enable_thinking` flag below are
    Qwen3-specific; Gemma and others ignore both, so we must not send them or a
    Gemma run reasons freely (~8000 tokens/article instead of ~100-200).
    """
    model = config.CHAT_MODEL.lower()
    if "qwen" in model:
        return "qwen"
    if "gemma" in model:
        return "gemma"
    return "other"


def think_suffix(thinking: bool) -> str:
    """Prompt-level reasoning toggle. Only Qwen3 reads '/no_think'."""
    if thinking or _model_family() != "qwen":
        return ""
    return " /no_think"


def think_extra_body(thinking: bool) -> dict[str, Any]:
    """Server-side reasoning toggle over the OpenAI API, per model family.

    Each family disables reasoning differently (verified with
    scripts/check_no_think.py):

    - Qwen3 reads the chat template's `enable_thinking` flag. (The '/no_think'
      suffix from think_suffix() is the unreliable belt to this suspenders.)
    - Gemma-4 ignores `enable_thinking` entirely and instead honours the OpenAI
      `reasoning_effort` param; "none" suppresses its <|channel>thought block.
      This matters beyond cost: with reasoning on, the json_schema grammar
      forbids the thought block, forcing the model cold into JSON, where it
      derails into garbage in the free-text `reason` field. Reasoning off, it
      answers directly and the grammar behaves.

    Other families have no known toggle, so send nothing.
    """
    family = _model_family()
    if family == "qwen":
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    if family == "gemma":
        return {} if thinking else {"reasoning_effort": "none"}
    return {}


def classify(article: Article) -> Verdict | None:
    """Judge one article against CRITERIA, returning None on any failure."""
    user = (
        f"SOURCE: {article.source}\nTITLE: {article.title}\n"
        f"SUMMARY: {article.summary}{think_suffix(config.THINKING)}"
    )
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            temperature=0,
            # Breaks the greedy-decoding repetition loop that makes some quants
            # (e.g. Gemma Q4_K_M) spam one token until the cap truncates the JSON.
            frequency_penalty=config.CLASSIFY_FREQUENCY_PENALTY,
            # Runaway guard: a verdict is a few hundred tokens, but a degenerating
            # model can emit thousands. Bound it so one bad article can't stall
            # the whole run.
            max_tokens=config.CLASSIFY_MAX_TOKENS,
            messages=[
                {"role": "system", "content": CRITERIA},
                {"role": "user", "content": user},
            ],
            # If your LM Studio build rejects json_schema, swap for
            # response_format={"type": "json_object"} and the parse still works.
            response_format=cast(
                Any, {"type": "json_schema", "json_schema": VERDICT_SCHEMA}
            ),
            extra_body=think_extra_body(config.THINKING),
        )
        choice = resp.choices[0]
        text = verdict_json(choice.message)
        if not text:
            why = (
                f"hit the {config.CLASSIFY_MAX_TOKENS}-token cap mid-output "
                "(the model likely reasoned first; disable thinking or raise "
                "CLASSIFY_MAX_TOKENS)"
                if choice.finish_reason == "length"
                else "no JSON verdict in the reply"
            )
            print(f"  ! classify failed: {why}", file=sys.stderr)
            return None
        return Verdict.from_json(json.loads(text))
    except Exception as e:
        print(f"  ! classify failed: {e}", file=sys.stderr)
        return None


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _digest_faults(text: str, aligns: list[Alignment]) -> str:
    """Why this draft is worth re-rolling, as a phrase to log -- "" when sound.

    Two checks, in increasing order of what they prove: the format proxy (a
    marker written inline is the visible symptom of a sloppy run) and the
    semantic one (a sentence that matches a different item than its marker says).
    """
    why = []
    if not markers_each_on_own_line(text):
        why.append("markers not each on their own line")
    if bad := misaligned(aligns, config.DIGEST_ALIGNMENT_MARGIN):
        why.append(f"{len(bad)} sentence(s) carrying another story's marker")
    return "; ".join(why)


def write_digest(items: list[Article]) -> str:
    payload = "\n\n".join(
        f"[{it.category}] {it.title}\n{it.reason}\n@@{i}@@"
        for i, it in enumerate(items, 1)
    ) + think_suffix(config.DIGEST_THINKING)

    def _generate(temperature: float):
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            temperature=temperature,
            max_tokens=config.DIGEST_MAX_TOKENS,
            messages=[
                {"role": "system", "content": DIGEST_PROMPT},
                {"role": "user", "content": payload},
            ],
            extra_body=think_extra_body(config.DIGEST_THINKING),
        )
        return resp.choices[0]

    choice = _generate(config.DIGEST_TEMPERATURE)
    text = answer_text(choice.message)
    aligns: list[Alignment] = []
    # One retry when the draft is unsound: markers written off-format, or -- the
    # fault that actually reaches the reader -- a sentence carrying another
    # story's marker. Skip the retry on a truncated reply (that's a token-cap
    # problem, not a formatting one) and only adopt the retry if it comes back
    # clean, so a worse re-roll never replaces a usable first draft.
    if text and choice.finish_reason != "length":
        aligns = alignments(text, items, embed)
        if faults := _digest_faults(text, aligns):
            print(f"  ! digest {faults}; regenerating once", file=sys.stderr)
            retry = _generate(0.0)
            retry_text = answer_text(retry.message)
            if retry_text and retry.finish_reason != "length":
                retry_aligns = alignments(retry_text, items, embed)
                if not _digest_faults(retry_text, retry_aligns):
                    choice, text, aligns = retry, retry_text, retry_aligns
    if not text:
        raise RuntimeError(
            "model returned no digest text (it likely emitted only reasoning); "
            "disable thinking for the digest call."
        )
    if choice.finish_reason == "length":
        # Trim to the last complete @@N@@ marker so we don't return a half-written item.
        last = list(re.finditer(r"@@\d+@@", text))
        if last:
            text = text[: last[-1].end()]
            print(
                f"  ! digest hit the {config.DIGEST_MAX_TOKENS}-token cap; "
                f"returning {len(last)} of {len(items)} items",
                file=sys.stderr,
            )
            # Trimming moved every marker span, so score the text we will ship.
            aligns = alignments(text, items, embed)
        else:
            hint = (
                " (token cap hit mid-reasoning — try setting DIGEST_THINKING = True"
                " so reasoning tokens count against max_tokens properly, or raise"
                " DIGEST_MAX_TOKENS further)"
                if "<think>" in text
                else ""
            )
            raise RuntimeError(
                f"digest hit the {config.DIGEST_MAX_TOKENS}-token cap with no complete items{hint}"
            )
    # Last resort: the retry (if any) came back misaligned too, so repair the
    # mapping we have rather than mailing a story its neighbour's link.
    return restore_links(realign(text, aligns, config.DIGEST_ALIGNMENT_MARGIN), items)
