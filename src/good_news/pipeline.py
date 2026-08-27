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


def dedupe_links(items: list[Article]) -> list[Article]:
    """Drop repeat appearances of the same URL, keeping the first.

    Two reddit submissions can link the same article, and a story can sit in two
    feeds at once, so fetch() genuinely returns the same link more than once.
    SeenStore only filters against *previous* runs, so without this the copies
    are classified independently, get different `reason` text, and can land in
    different categories -- which is how the 2026-08-26 briefing shipped one
    404media URL twice.
    """
    seen: set[str] = set()
    out: list[Article] = []
    for a in items:
        if a.link not in seen:
            seen.add(a.link)
            out.append(a)
    return out


def dedupe_key(item: Article) -> str:
    """What dedupe compares: the story as published, never the model's verdict.

    This used to be title + `reason`, on the theory that the verdict captures
    what a story is about better than a headline does. Measured on the
    2026-08-27 briefing (7 same-event pairs against 164 unrelated pairs), it
    does the opposite -- `reason` is generated text that varies run to run, so
    it adds noise to the very number the threshold is drawn against:

        title only      dupes 0.546-0.809   others <=0.624   overlap
        title+reason    dupes 0.511-0.894   others <=0.561   overlap
        title+summary   dupes 0.645-0.760   others <=0.539   0.107 gap

    Only the feed's own summary separates the two populations cleanly, so the
    threshold below has somewhere stable to sit.
    """
    return f"{item.title} {item.summary[:600]}".strip()


def dedupe(items: list[Article], min_keep: int = 1) -> list[Article]:
    """Collapse near-identical coverage, keeping the highest-optimism version.

    If the strict threshold would leave fewer than min_keep items, retries with
    DEDUPE_SIMILARITY_RELAXED so sparse categories aren't over-collapsed.
    Embeddings are computed once and reused across both passes.
    """
    if len(items) < 2:
        return items
    try:
        vecs = embed([dedupe_key(it) for it in items])
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
        fresh = dedupe_links([a for a in articles if a.link])
        print(
            f"{len(fresh)} to judge (dry run, ignoring seen-history)", file=sys.stderr
        )
    else:
        assert store is not None
        fresh = dedupe_links(
            [a for a in articles if a.link and not store.is_seen(a.link)]
        )
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

    # Dedupe across the whole batch, before the category split. One event is
    # rarely filed under one category -- the 2026-08-27 Meta settlement landed
    # in both anti_corporate and technology -- so a per-category pass never
    # compares the copies that matter. Running it here also means the
    # MIN_PER_CATEGORY relaxation below can no longer re-admit a pair the
    # strict pass merged, which is how two write-ups of one FDA approval
    # survived into the same section.
    kept = dedupe(kept)
    print(f"{len(kept)} after collapsing duplicate coverage", file=sys.stderr)

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
        # No dedupe pass here: the global one above already collapsed every
        # pair at or above DEDUPE_SIMILARITY, so a second per-category pass
        # can only spend embedding calls to find nothing.
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
