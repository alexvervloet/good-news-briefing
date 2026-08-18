"""Configuration knobs and environment loading.

Importing this module also performs the two bits of environment setup the rest
of the package depends on: pointing urllib at certifi's CA bundle (so HTTPS
feeds don't silently fail) and loading the .env file. Both run once, at import.
"""

from __future__ import annotations
import os
import sys
import pathlib

# macOS Python often ships without root certificates, so feedparser's urllib
# fails every HTTPS feed with CERTIFICATE_VERIFY_FAILED and silently returns
# zero entries. Point urllib at certifi's CA bundle before any feed is parsed.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    print(
        "  ! certifi not installed; HTTPS feeds may fail (pip install certifi)",
        file=sys.stderr,
    )

# Load private/local settings from a .env file at the project root (gitignored).
# Real environment variables take precedence, so nothing here is required.
try:
    from dotenv import load_dotenv

    # src/good_news/config.py -> parents[2] is the repo root, where .env lives.
    load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    print(
        "  ! python-dotenv not installed; relying on real env vars (pip install python-dotenv)",
        file=sys.stderr,
    )


def _env_list(name: str, default: str = "") -> list[str]:
    """Read a comma-separated env var into a clean list of strings."""
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


# ----------------------------------------------------------------------
# MODEL / SERVER
# ----------------------------------------------------------------------
PC_HOST = os.environ.get("PC_HOST", "127.0.0.1")  # set in .env to your PC's LAN IP
BASE_URL = f"http://{PC_HOST}:1234/v1"
# Tuned for an RTX 3090 (24GB). Qwen3.6-35B-A3B is a 3B-active MoE: fast and
# high-quality. Use the IQ4_XS quant (~18GB) so it stays fully on the card and
# leaves room for context + the embedding model. Copy the EXACT ids from the
# LM Studio server panel after loading each model.
CHAT_MODEL = "unsloth/qwen3.6-35b-a3b"  # load the IQ4_XS (or MTP) GGUF in LM Studio
# CHAT_MODEL = "google/gemma-4-26b-a4b"  # load the Q4_K_M (or MTP) GGUF in LM Studio
EMBED_MODEL = (
    "text-embedding-qwen3-embedding-0.6b"  # tiny; co-loads with the chat model fine
)

# Reasoning toggle. False makes the model skip reasoning tokens -- much faster,
# which is what you want for high-volume classification. Set True only if you
# ever want it to deliberate (slower).
#
# How "off" is enforced is model-specific (see llm.think_extra_body): Qwen3
# reads enable_thinking, Gemma-4 ignores that and honours reasoning_effort
# instead. CLASSIFY_MAX_TOKENS below bounds the cost if a model reasons anyway.
THINKING = False

# Hard cap on a single classify() completion. A verdict is a few hundred tokens,
# so this is a runaway guard, not a normal limit -- it bounds the damage when a
# model degenerates (see below) instead of letting it burn thousands of tokens.
CLASSIFY_MAX_TOKENS = 2000

# Repetition penalty for classify(). At temperature=0 (greedy decoding) some
# models/quants fall into a degenerate loop, repeating a token until the cap
# fires and truncates the JSON (Gemma Q4_K_M does this; Qwen doesn't). A small
# frequency_penalty nudges the logits off the repeated token and breaks the
# loop. It adjusts logits *before* the argmax, so scoring stays deterministic.
# 0.0 disables it; raise toward 1.0 if a model still loops.
CLASSIFY_FREQUENCY_PENALTY = 0
# Thinking is always disabled for the digest: it's creative writing, not analysis,
# so reasoning tokens waste budget without improving output.
DIGEST_THINKING = False

# Sampling temperature for the digest. The digest is a format-faithful task as
# much as a creative one -- the model must copy each item's @@N@@ marker onto the
# right sentence -- and higher temperatures made it wander from that format and,
# in the same runs, staple links onto the wrong stories. Kept warm enough for
# varied prose but low enough to hold the marker discipline.
DIGEST_TEMPERATURE = 0.3

# How much better a rival item must match a digest sentence before we conclude
# the model stapled the wrong @@N@@ marker onto it (guardrails.alignments). The
# items reaching the digest are stories dedupe already judged distinct, so a real
# mismatch shows a gap of several tenths; this only stops two near-neighbours in
# the same category from triggering a pointless re-roll. Raise it if a clean
# digest gets regenerated; lower it if a subtle swap slips through.
DIGEST_ALIGNMENT_MARGIN = 0.05

# Upper bound on the digest completion. Without an explicit cap LM Studio applies
# its own default limit, which truncates the briefing mid-sentence once enough
# items pass the filter. Sized to comfortably fit a full multi-category digest
# with up to 25 items (thinking is disabled, so all tokens go to output).
DIGEST_MAX_TOKENS = 50000

# ----------------------------------------------------------------------
# PIPELINE THRESHOLDS
# ----------------------------------------------------------------------
MAX_PER_CATEGORY = 5
MIN_PER_CATEGORY = 3  # relax dedupe if a category would have fewer than this
OPTIMISM_THRESHOLD = 0.55  # 0..1; raise to be pickier
DEDUPE_SIMILARITY = 0.86  # cosine above this = treat as the same story
DEDUPE_SIMILARITY_RELAXED = (
    0.92  # fallback threshold used when a category is below MIN_PER_CATEGORY
)
MAX_ENTRIES_PER_FEED = 25
# Reddit's RSS links to the comments page, and its "summary" is just the
# submission blurb. When True, swap in the real article URL and crawl the
# article body so the model judges the story, not the reddit post. Adds a
# network fetch per reddit item; failures fall back to the RSS summary.
FETCH_REDDIT_ARTICLES = True
ARTICLE_FETCH_TIMEOUT = 10  # seconds per article crawl
ARTICLE_MAX_CHARS = 4000  # trim crawled body before sending to the model

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------
OUT_DIR = pathlib.Path.home() / "good-news"
DB_PATH = OUT_DIR / "seen.sqlite3"
OPEN_WHEN_DONE = True  # `open` the file on macOS when finished

# ---- Email (optional) -----------------------------------------------------
# After a real run, mail the briefing to EMAIL_TO. Leave EMAIL_TO empty to
# disable. Auth uses a Gmail APP PASSWORD (not your normal password), read from
# the GMAIL_APP_PASSWORD env var so no secret lives in this file:
#   1. Turn on 2-Step Verification for the Gmail account.
#   2. Visit https://myaccount.google.com/apppasswords, create one, copy 16 chars.
#   3. Add to ~/.zshrc:  export GMAIL_APP_PASSWORD="xxxxxxxxxxxxxxxx"
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")  # set in .env
EMAIL_TO = _env_list("EMAIL_TO")  # set in .env, comma-separated
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))  # SSL

# ----------------------------------------------------------------------
# FEEDS  --  verify each of these resolves in a browser; RSS paths drift.
# ----------------------------------------------------------------------
FEEDS = [
    # progressive / social-progress
    "https://www.theguardian.com/inequality/rss",
    "https://www.propublica.org/feeds/propublica/main",
    "https://theintercept.com/feed/?lang=en",
    "https://www.motherjones.com/feed/",
    "https://www.commondreams.org/rss.xml",
    # anti-corporate / tech
    "https://www.404media.co/rss/",
    "https://www.eff.org/rss/updates.xml",
    "https://www.wired.com/feed/rss",
    "https://feeds.arstechnica.com/arstechnica/index",
    # solutions / good news
    "https://reasonstobecheerful.world/feed/",
    "https://www.positive.news/feed/",
    "https://www.yesmagazine.org/feed",
    # community / humans helping humans
    "https://www.reddit.com/r/UpliftingNews/.rss",
    # major outlets — good/uplifting filtered sections
    "https://feeds.bbci.co.uk/news/topics/cx2pk70323et/rss.xml",  # BBC Uplifting Stories
    "https://www.theguardian.com/world/series/the-upside/rss",  # Guardian: The Upside
]
