"""Orchestration: wire fetch -> classify -> filter -> dedupe -> digest -> deliver."""

from __future__ import annotations
import sys
import datetime

from . import config
# cosine lives in guardrails now that the digest link check uses it too; dedupe
# is still its other consumer, so it stays reachable as pipeline.cosine.
from .guardrails import cosine
from .models import Article, Verdict
from .sources import fetch, fetch_article_text
from .store import SeenStore
from .llm import classify, embed, write_digest
from .deliver import write_briefing, send_email


def keep(v: Verdict | None) -> bool:
    """The editorial filter: which verdicts clear the bar for the briefing."""
    if not v or not v.is_good_news:
        return False
    if v.is_corporate_pr:
        return False
    if v.is_pure_luck and v.category != "community_helping":
        return False
    return v.optimism >= config.OPTIMISM_THRESHOLD


def dedupe(items: list[Article], min_keep: int = 1) -> list[Article]:
    """Collapse near-identical coverage, keeping the highest-optimism version.

    If the strict threshold would leave fewer than min_keep items, retries with
    DEDUPE_SIMILARITY_RELAXED so sparse categories aren't over-collapsed.
    Embeddings are computed once and reused across both passes.
    """
    if len(items) < 2:
        return items
    try:
        vecs = embed([f"{it.title} {it.reason or ''}".strip() for it in items])
    except Exception as e:
        print(f"  ! embeddings unavailable, skipping dedupe: {e}", file=sys.stderr)
        return items

    def _run(threshold: float) -> list[Article]:
        kept: list[Article] = []
        kept_vecs: list[list[float]] = []
        for it, v in sorted(zip(items, vecs), key=lambda p: -(p[0].optimism or 0)):
            if all(cosine(v, kv) < threshold for kv in kept_vecs):
                kept.append(it)
                kept_vecs.append(v)
        return kept

    result = _run(config.DEDUPE_SIMILARITY)
    if len(result) < min_keep:
        result = _run(config.DEDUPE_SIMILARITY_RELAXED)
    return result


def _print_verdict(article: Article, v: Verdict, passed: bool) -> None:
    mark = "✓" if passed else "✗"
    flags = []
    if v.is_corporate_pr:
        flags.append("PR")
    if v.is_pure_luck:
        flags.append("luck")
    tag = f"  [{','.join(flags)}]" if flags else ""
    print(
        f"  {mark} {v.optimism:.2f} {v.category:17} {article.title[:70]}{tag}",
        file=sys.stderr,
    )
    print(f"        {v.reason}", file=sys.stderr)


def run(
    dry_run: bool = False,
    limit: int | None = None,
    show_verdicts: bool = False,
    send_mail: bool = False,
) -> None:
    store = None if dry_run else SeenStore.open()
    per_feed = limit if limit is not None else (5 if dry_run else config.MAX_ENTRIES_PER_FEED)

    articles = fetch(per_feed=per_feed)
    print(f"fetched {len(articles)} articles", file=sys.stderr)

    if dry_run:
        fresh = [a for a in articles if a.link]
        print(
            f"{len(fresh)} to judge (dry run, ignoring seen-history)", file=sys.stderr
        )
    else:
        assert store is not None
        fresh = [a for a in articles if a.link and not store.is_seen(a.link)]
        print(f"{len(fresh)} new since last run", file=sys.stderr)

    kept: list[Article] = []
    for a in fresh:
        # Reddit's RSS summary is just the submission blurb, so crawl the real
        # article and judge the model on that instead of the reddit post.
        if config.FETCH_REDDIT_ARTICLES and a.is_reddit_article:
            body = fetch_article_text(a.link)
            if body:
                a.summary = body
        v = classify(a)
        if store is not None:
            store.mark_seen(a.link)
        passed = keep(v)
        if show_verdicts and v is not None:
            _print_verdict(a, v, passed)
        if passed and v is not None:
            a.category = v.category
            a.optimism = v.optimism
            a.reason = v.reason
            kept.append(a)
    if store is not None:
        store.commit()
    print(f"{len(kept)} passed the filter", file=sys.stderr)

    by_cat: dict[str, list[Article]] = {}
    for a in kept:
        by_cat.setdefault(a.category or "other", []).append(a)

    all_cats = [
        "politics_social", "anti_corporate", "technology",
        "community_helping", "science_health", "environment", "other",
    ]
    cat_counts = {c: len(by_cat.get(c, [])) for c in all_cats}
    print(
        "  category breakdown: "
        + "  ".join(f"{c}={n}" for c, n in cat_counts.items()),
        file=sys.stderr,
    )

    selected: list[Article] = []
    for cat, group in by_cat.items():
        group = dedupe(group, min_keep=config.MIN_PER_CATEGORY)
        group.sort(key=lambda x: -(x.optimism or 0))
        selected.extend(group[:config.MAX_PER_CATEGORY])

    if not selected:
        print(
            "No good news cleared the bar. Try lowering OPTIMISM_THRESHOLD.",
            file=sys.stderr,
        )
        return

    selected.sort(key=lambda x: -(x.optimism or 0))
    md = write_digest(selected)
    today = datetime.date.today().isoformat()
    document = f"# Good News — {today}\n\n{md}\n"

    if dry_run:
        # Progress goes to stderr above, so stdout is just the clean digest.
        print("\n" + "=" * 60 + "\n", file=sys.stderr)
        print(document)
        if send_mail:  # opt-in during dry runs, for testing the email path
            send_email(f"Good News — {today}", document)
        return

    write_briefing(document, today)
    if send_mail:
        send_email(f"Good News — {today}", document)
