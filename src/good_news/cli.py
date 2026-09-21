"""Command-line entry point: parse flags and kick off the pipeline."""

from __future__ import annotations
import argparse
import sys

from .llm import ServerUnavailable
from .pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Good news evening briefing")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="test run: ignore the seen-history, print the digest to the terminal, and "
        "don't save or open a file (safe to run over and over)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="entries to pull per feed (default: 5 in --dry-run, else 25)",
    )
    p.add_argument(
        "--verdicts",
        action="store_true",
        help="print every article's keep/drop decision and reason; great for tuning CRITERIA",
    )
    p.add_argument(
        "--email",
        action="store_true",
        help="also send the briefing during a --dry-run (for testing the email path)",
    )
    p.add_argument(
        "--no-email",
        action="store_true",
        help="suppress the email on a real run (it otherwise sends when EMAIL_TO is set)",
    )
    args = p.parse_args(argv)
    # Real runs email by default; dry runs only when --email is passed.
    send_mail = args.email if args.dry_run else not args.no_email
    try:
        run(
            dry_run=args.dry_run,
            limit=args.limit,
            show_verdicts=args.verdicts,
            send_mail=send_mail,
        )
    except ServerUnavailable as e:
        # Exit non-zero so cron, and anything reading $?, sees a failed run
        # instead of a successful one that happened to produce no briefing.
        print(f"\n! run aborted: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
