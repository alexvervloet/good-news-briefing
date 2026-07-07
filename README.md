# Good News Briefing

[![CI](https://github.com/alexvervloet/good-news-briefing/actions/workflows/ci.yml/badge.svg)](https://github.com/alexvervloet/good-news-briefing/actions/workflows/ci.yml)

A small, self-hosted pipeline that pulls a set of RSS feeds, uses a **local LLM**
(served by [LM Studio](https://lmstudio.ai/)) to judge each story against a
tunable editorial point of view, collapses duplicate coverage, and composes a
warm Markdown "good news" briefing — optionally emailing it to you and a friend.

Nothing leaves your network except the RSS fetches and the outgoing email: the
classification and writing all happen on your own GPU.

> **The engineering, in three findings.** This project is mostly an exercise in
> making a small local model behave. The non-obvious lessons, each with the
> change and the measured effect, live in **[`LEARNINGS.md`](LEARNINGS.md)**:
> - **The model hallucinates plausible-but-dead URLs**, so it never sees one —
>   each item carries an opaque `@@N@@` marker it echoes, and real links are
>   spliced back in by code after generation.
> - **Structured-output field order is generation order.** Putting the free-text
>   `reason` *before* the `optimism` number in the JSON schema makes the model
>   justify before it scores — a free calibration win that erased a
>   one-directional scoring bias across two model families.
> - **A rubric the model under-applies needs hard caps, not adjectives.** "Lean
>   lower" lost to central tendency; "cap at 0.45 if only pledged" held.
>
> Both claims are checked by evals against a live model — latest real transcripts
> (classifier 23/25, optimism 19/20 within ±0.1) are committed under
> [`evals/results/`](evals/results/).

## Example output

A real briefing, exactly as generated on 2026-07-06 — grouped by theme, each
story a calm one-liner over a real (code-spliced, never model-written) link.
This is the unedited output of one scheduled run:

```markdown
# Good News — 2026-07-06

The day has softened into evening; here are five quiet reminders that care still moves through the world.

**Preserving memory**

A grassroots network has safeguarded half a million photographs and documents in distributed servers, keeping Palestinian cultural memory alive despite ongoing pressures to erase it.
https://www.wired.com/story/how-palestinians-are-building-a-digital-archive-that-cant-be-erased/

**Neighbors stepping in**

When a boy with cerebral palsy was injured at a local park, five passing teenagers immediately coordinated rescue efforts and stayed with the family until professional help arrived.
https://www.theguardian.com/lifeandstyle/2026/jul/06/the-kindness-of-strangers-my-son-was-unconscious-and-i-frantically-called-out-for-help-then-five-teenagers-came-running

An eleven-year-old jumped into a residential pool without hesitation and pulled a struggling adult to safety, proving that quick thinking can change an entire family’s trajectory.
https://www.actionnews5.com/2026/07/05/11-year-old-rescues-man-nearly-drowning-apartment-complex-pool/

**Protecting habitat and nurturing talent**

Targeted anti-poaching patrols have gradually increased leopard density in Benin’s Pendjari National Park, offering a measured victory for one of the region’s most vulnerable wild cats.
https://news.mongabay.com/2026/07/endangered-west-african-leopards-show-signs-of-recovery-despite-odds-its-a-win/

A national performing arts initiative is now funding stage time for emerging dancers between sixteen and twenty-four, giving young creators reliable access to professional development and public performance.
https://www.positive.news/lifestyle/arts/the-national-dance-company-opening-doors-for-young-performers/
```

The pipeline has run nightly via `cron` since 2026-06-19 (see [Scheduling](#scheduling-optional)).

## How it works

1. **Fetch** — pull a list of RSS feeds (`feedparser`).
2. **Dedupe history** — skip anything seen on a previous run (SQLite).
3. **Classify** — ask your local chat model to judge each item against the
   editorial criteria, returning structured JSON.
4. **Filter** — drop corporate PR and pure-luck fluff; keep genuine good news
   above an optimism threshold.
5. **Collapse duplicates** — use a local embedding model to merge near-identical
   coverage of the same story (degrades gracefully if no embed model is loaded).
6. **Compose** — have the model write a calm, grouped Markdown briefing.
7. **Deliver** — save it to `~/good-news/`, open it (macOS), and optionally email it.

## Requirements

- A machine running **LM Studio** with its server enabled and **bound to
  `0.0.0.0`**, serving a chat model and (optionally) an embedding model.
  Defaults are tuned for an RTX 3090: `qwen3.6-35b-a3b` (IQ4_XS) +
  `qwen3-embedding-0.6b`.
- Python 3.10+ on the machine that runs this script (it talks to LM Studio over
  the LAN, so it can be a different computer).

## Setup

```bash
git clone https://github.com/alexvervloet/good-news-briefing.git
cd good-news-briefing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env (see below)
```

Edit `.env` with your own values:

| Variable             | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `PC_HOST`            | LAN IP of the machine running LM Studio (e.g. `192.168.1.106`) |
| `EMAIL_FROM`         | Sender Gmail address (leave blank to disable email)            |
| `EMAIL_TO`           | Comma-separated recipients                                     |
| `GMAIL_APP_PASSWORD` | Gmail **app password** (not your normal password)              |
| `SMTP_HOST`/`SMTP_PORT` | Optional overrides (default Gmail SSL)                      |

In LM Studio: load your chat and embedding models, start the server, and bind it
to `0.0.0.0` so other machines on the LAN can reach it. Copy the exact model ids
from the server panel into `CHAT_MODEL` / `EMBED_MODEL` in `src/good_news/config.py`
if they differ from the defaults.

## Usage

```bash
# Safe test: ignores seen-history, prints the digest, saves/sends nothing
python3 good_news_briefing.py --dry-run

# Show every keep/drop decision and the model's reasoning (great for tuning)
python3 good_news_briefing.py --dry-run --verdicts

# Real run: saves ~/good-news/briefing-YYYY-MM-DD.md, opens it, and emails it
python3 good_news_briefing.py
```

### Flags

| Flag         | Effect                                                                 |
| ------------ | ---------------------------------------------------------------------- |
| `--dry-run`  | Ignore seen-history, print to terminal, save/open/email nothing        |
| `--limit N`  | Entries to pull per feed (default: 5 in `--dry-run`, else 25)          |
| `--verdicts` | Print each article's keep/drop decision and reason                     |
| `--email`    | Also send the email during a `--dry-run` (for testing the email path)  |
| `--no-email` | Suppress the email on a real run                                       |

## Email setup (Gmail)

Gmail's SMTP needs an **app password**, not your normal password:

1. Enable 2-Step Verification on the account.
2. Create one at <https://myaccount.google.com/apppasswords>.
3. Put the 16 characters in `.env` as `GMAIL_APP_PASSWORD` (or export it in your
   shell). If it's unset, the script skips emailing with a warning instead of
   failing.

## Configuration

The code lives in the `src/good_news/` package; the knobs worth tuning live in
two files:

- `src/good_news/prompts.py`
  - **`CRITERIA`** — the editorial point of view: what counts as "good news."
    This is the one knob worth rewriting to taste.
- `src/good_news/config.py`
  - **`FEEDS`** — the list of RSS sources.
  - `CHAT_MODEL`, `EMBED_MODEL`, `THINKING` — model ids and the Qwen reasoning toggle.
  - `OPTIMISM_THRESHOLD`, `DEDUPE_SIMILARITY`, `MAX_PER_CATEGORY`, `MAX_ENTRIES_PER_FEED`.

Run it with `python3 good_news_briefing.py ...` from a source checkout, or
`pip install -e .` and then use `python3 -m good_news ...` (or the `good-news`
console script).

## Testing & evals

**Unit tests** (fast, no model required):

```bash
pip install -e ".[test]"
pytest
```

Tests mock the LLM at its boundary and run in under a second. See
`tests/README.md` for details.

**Classifier eval** (requires LM Studio running):

```bash
python evals/run_eval.py             # summary — failures only
python evals/run_eval.py --verbose   # show all 25 cases including passes
```

Runs `classify()` against 25 hand-labeled articles and reports per-field
accuracy. Use this after editing `CRITERIA` to check for regressions. Exit
code is non-zero if any case fails.

**Optimism eval** (requires LM Studio running):

```bash
python evals/run_optimism_eval.py             # summary — failures only
python evals/run_optimism_eval.py --verbose   # show all cases including passes
```

An agentic eval: each article in `evals/optimism_fixtures.json` carries a
reference `optimism` score (0.0–1.0) assigned by an agent reading it against
`CRITERIA`, not a hard ground-truth label. The runner scores the model's
`optimism` against that reference and passes a case when they agree within
±0.1, also reporting mean absolute error and whether the model skews
optimistic or pessimistic. Exit code is non-zero if any case falls outside the
tolerance.

See [`LEARNINGS.md`](LEARNINGS.md) for a write-up of what this eval surfaced.

## Engineering learnings

[`LEARNINGS.md`](LEARNINGS.md) is a running log of the non-obvious lessons this
project has taught me — measured findings about model behavior, evaluation, and
prompt design, each with the change I made and what the numbers did. For example:
the model paraphrases URLs into plausible-but-dead links, so it never sees one —
each item carries an opaque `@@N@@` marker it echoes, and real links are spliced
back in by code after generation; or, further down the log, the model silently
compressing the optimism scale onto a single value until the prompt gave it
calibration anchors.

## Scripts

`scripts/check_no_think.py` — verifies that `/no_think` and `enable_thinking:false`
actually suppress reasoning tokens on your model build before you run the
full pipeline. Useful when setting up a new model or after an LM Studio update:

```bash
python scripts/check_no_think.py
```

## Output

- `~/good-news/briefing-YYYY-MM-DD.md` — the briefing for each real run.
- `~/good-news/seen.sqlite3` — dedupe history so future runs only surface new items.

## Scheduling (optional)

Run it automatically in the evening with `cron` or `launchd` on macOS, e.g. a
crontab line for 6pm daily:

```cron
0 18 * * * cd /path/to/good-news-briefing && .venv/bin/python good_news_briefing.py >> ~/good-news/cron.log 2>&1
```

## Privacy

`.env` (your IP, email addresses, and app password) is gitignored and never
committed. Only safe placeholder defaults live in the source.
