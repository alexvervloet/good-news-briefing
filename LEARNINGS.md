# Engineering Learnings

A running log of the non-obvious lessons this project has taught me — things I
worked out by building and measuring, not by reading about them. Each entry
records what I observed, what I changed, what the numbers did, and the
general principle I'm taking forward. In chronological order, oldest first.

Each entry follows the same shape:

- **Context** — what I was doing and why.
- **Finding** — what I observed, with evidence.
- **Action** — what I changed in response.
- **Result** — what happened, in numbers where possible.
- **Takeaway** — the transferable principle.

---

## 2026-06-19 — The most dangerous failure mode is the one that returns zero instead of raising

**Context.** Early on the pipeline would occasionally just produce an empty
briefing with no error — every feed quietly yielding nothing.

**Finding.** Two independent "succeeds with zero results" traps, both upstream of
any code I'd written. (1) macOS Python often ships without root certificates, so
feedparser's urllib fails *every* HTTPS feed with `CERTIFICATE_VERIFY_FAILED` and
returns zero entries — no exception. (2) feedparser itself never raises on a
broken or unreachable feed: it swallows network, SSL, and parse errors into a
`d.bozo` flag and hands back an empty `entries` list. Both look identical to "the
feed legitimately had no new items."

**Action.** Point urllib at certifi's CA bundle at import time, before any feed is
parsed ([`config.py`](src/good_news/config.py)), and warn loudly if certifi is
missing. In the fetch loop, check `d.bozo` and surface `d.bozo_exception` when a
feed comes back empty rather than counting it as zero
([`sources.py`](src/good_news/sources.py)).

**Takeaway.**

- A library that returns an empty result on failure is more dangerous than one
  that throws, because the empty result flows downstream looking like valid data.
  When integrating one, hunt for its silent-failure flag (`d.bozo` here) and
  convert it into a visible warning yourself.
- Environment assumptions (CA certs present) belong in one-time setup that runs
  *before* the first network call and announces when its precondition is missing —
  not in a comment hoping the next machine is configured the same way.

---

## 2026-06-19 — Keep data the model can mangle out of the model's hands entirely

**Context.** The digest is one model-written paragraph per story, each ending with
the article's link. The obvious design is to hand the model each item's URL and
ask it to place the link under its sentence.

**Finding.** The model paraphrases URLs. It would turn
`reasonstobecheerful.world` into a plausible-looking but dead
`reasonsbecheerful.world` — a silently broken link in something a reader is meant
to click. Asking nicely ("copy the URL exactly") doesn't fix a model that treats
text as something to rewrite.

**Action.** The model never sees a URL. Each item is tagged with an opaque
`@@N@@` marker; the model only copies that marker onto the link line, and
[`restore_links()`](src/good_news/guardrails.py) swaps each marker for the real
URL *after* generation, in code. The same step audits the raw output for the
failure modes that remain — a literal `http` (the model wrote a URL anyway), a
reused or missing marker number, leftover unresolved markers — and warns on each.

**Takeaway.**

- If a value must be exact and you can supply it yourself, don't route it through
  the generative step at all. Give the model an opaque token to echo and splice the
  real value back deterministically. This generalizes to IDs, citations, prices,
  any verbatim string.
- A generative pipeline needs a *verification* stage that inspects the raw output
  for the specific ways this model misbehaves, not just a hope that the prompt held.

---

## 2026-06-19 — A reasoning model leaks its chain-of-thought two different ways, and the same output needs two different extraction rules

**Context.** The local model is Qwen3, which emits reasoning tokens before its
answer. Two of my calls consume that output: `classify()` needs the JSON verdict,
and `write_digest()` needs the prose briefing. I assumed "turn thinking off" with
the documented `/no_think` prompt suffix would be the end of it.

**Finding.** Two surprises. First, *disabling* thinking is unreliable: some LM
Studio builds of Qwen3 ignore the `/no_think` suffix, and some route the entire
reply into a separate `reasoning_content` field leaving `content` empty. Second,
when reasoning *does* leak it arrives by two different paths — either inlined into
`content` wrapped in `<think>...</think>` tags, or in the separate
`reasoning_content` field — and the two call sites need *opposite* handling of it.
For `classify()`, the JSON is the answer and sometimes lands in
`reasoning_content`, so I must fall back to that field to find it. For the digest,
`reasoning_content` is raw thinking that must *never* reach the reader, so falling
back to it would be exactly wrong.

**Action.** Belt-and-suspenders on the request side: send both the `/no_think`
suffix *and* the server-side `chat_template_kwargs: {enable_thinking: False}` flag
([`llm.py`](src/good_news/llm.py)). On the response side, two functions with
deliberately different rules ([`guardrails.py`](src/good_news/guardrails.py)):
`message_text()` falls back to `reasoning_content` (used where that field may hold
the real answer); `answer_text()` strips `<think>` blocks and refuses to fall back
(used for reader-facing text — blank means "still reasoning," not "look in the
reasoning field").

**Takeaway.**

- "The answer" and "the final answer the reader sees" are two different
  extractions of the same model output, and conflating them is how raw
  chain-of-thought ends up in front of a user. Name and separate them.
- Treat a vendor's thinking toggle as advisory, not load-bearing. Set it by every
  mechanism available (prompt suffix *and* server flag) and still validate the
  output, because builds disagree about which mechanism they honor.

---

## 2026-06-19 — `max_tokens` budgets the reasoning tokens too, so cap failures need a known cut-point

**Context.** Long digests were truncating mid-sentence. I set an explicit
`DIGEST_MAX_TOKENS` because, without one, LM Studio applies its own short default
limit. Then individual runs started *erroring* instead of truncating cleanly.

**Finding.** Two coupled problems. (1) When the completion hits the cap
(`finish_reason == "length"`), naively returning the text yields a half-written
final item. (2) More subtly: if thinking is left on, the reasoning tokens count
against the *same* `max_tokens` budget — so the model can burn the entire cap
*before writing any answer*, leaving output that is nothing but an unclosed
`<think>` block. Raising the token limit doesn't help; the reasoning just expands
to fill it.

**Action.** On a length finish, trim back to the last complete `@@N@@` marker so
the digest ends on a whole item, and log how many of N items survived
([`llm.py`](src/good_news/llm.py)). If *no* complete item exists and the text is
an unclosed `<think>` block, raise with a pointed hint — disable digest thinking
so output tokens aren't eaten by reasoning, rather than blindly raising the cap.
`DIGEST_THINKING` is forced off for the digest for exactly this reason: it's
creative writing, not analysis, so reasoning spends budget without improving the
result.

**Takeaway.**

- `max_tokens` is a budget over *everything the model emits*, reasoning included.
  On a thinking model an output-length failure can mean "spent it all thinking,"
  which a bigger budget won't fix — diagnose *what* filled the budget before
  raising it.
- When a length cap is possible, design the output so there's a safe place to cut.
  The `@@N@@` markers gave truncation a clean item boundary to fall back to instead
  of a ragged half-sentence.

---

## 2026-06-20 — Independent calls can't leak order, but the fixture file still can

**Context.** While building the optimism eval, I'd written the 20 reference
cases into [`evals/optimism_fixtures.json`](evals/optimism_fixtures.json) in
neatly descending score order. I asked myself whether the model could pick up on
that ordering and "cheat."

**Finding.** In this harness it can't: `classify()` is called once per article,
each call a fresh, stateless request at `temperature=0` with only that one
article in the prompt. The model never sees the fixtures as a sequence, so file
order cannot influence its scores. But the ordering still mattered for two other
reasons — it can anchor *me* while labeling (each score drifting off the last
instead of being judged on its own), and it becomes a tell if the file is ever
fed as a batch or read by an LLM-as-judge later.

**Action.** Shuffled the fixtures so the file is non-monotonic, while leaving the
runner to report results in file order.

**Takeaway.** Separate "can this leak into the system *as built*" from "is this a
latent hazard." The first was a no — the architecture made it impossible. The
second was a yes, and cost nothing to fix. Reasoning about data-leakage means
reasoning about the exact call boundary, not a vague feeling that "the model
sees the file."

---

## 2026-06-20 — An LLM will quietly compress a rating scale unless you anchor it

**Context.** The classifier ([`src/good_news/llm.py`](src/good_news/llm.py))
asks a local model to score each story's `optimism` from 0.0 to 1.0. Unit tests
mock the model, so they verify plumbing but never whether the *judgement* is any
good. To probe that, I built a reference-graded eval
([`evals/run_optimism_eval.py`](evals/run_optimism_eval.py)): 20 articles with
reference optimism scores spread across the full range — drafted by an LLM
reading each against `CRITERIA`, then reviewed and corrected by hand — with the
eval passing a case only when the model landed within ±0.1 of the reference.

**Finding.** The model scored **6/20**, with a mean absolute error of **0.242**
and a mean *signed* error of **+0.222** — consistently optimistic. The real
problem wasn't the bias, it was the shape: the model collapsed almost every
mid-to-high story onto a single value, **0.85**. A regional clean-water project,
a one-off river cleanup, a single coffee shop unionizing, and an honorary
"volunteer day" all came back 0.85 — the same score as genuine national wins. It
could tell *great* from *barely-news* at the extremes but was effectively blind
across the entire middle, which is where most real stories live. The original
prompt only said "a vague positive-sounding headline scores low" — no anchors,
so the model had no idea what a 0.4 versus a 0.6 was supposed to look like.

**Action.** I rewrote only the `SCORING` block in
[`src/good_news/prompts.py`](src/good_news/prompts.py): five labeled bands
(0.10–0.25 up to 0.90–1.00), each with concrete examples, plus two explicit
instructions — *reserve the top of the range* and *when between bands, lean
lower if the good is mostly announced, symbolic, or tiny*. No code changes.

**Result.**

| Metric | Before | After |
|---|---|---|
| Within ±0.1 of reference | 6/20 | **13/20** |
| Mean absolute error | 0.242 | **0.095** |
| Mean signed error | +0.222 | **+0.044** |

The plateau broke: scores now spread across 0.15–0.95 and the systematic
optimism nearly vanished. Remaining misses clustered on "announced but not yet
delivered" stories (pledges, grants, pilots) the model still floors around 0.55
— and a few of those are genuine judgement disagreement rather than model error.

**Takeaway.**

- A model handed an unanchored numeric scale will use a fraction of it and bunch
  scores together. The fix is calibration anchors with concrete examples per
  band, not adjectives like "high" and "low."
- **Test the judgement, not just the plumbing.** Mocked unit tests would never
  have surfaced this; it only showed up because the eval measured the model
  against considered reference labels.
- **Spread your reference labels deliberately.** Clustered ground truth can't
  reveal whether a model discriminates — the failure was only visible because
  the references covered the whole 0.2–0.95 range.
- **Watch signed error, not just absolute error.** Mean *absolute* error says
  "how wrong"; mean *signed* error revealed the wrongness had a direction (a
  fixable bias) rather than being random noise.
- **Know when to stop.** At 13/20 I stopped tuning the prompt: chasing the last
  few cases would have meant overfitting the prompt to my 20 specific labels
  rather than improving real calibration.

---

## 2026-06-20 — A JSON-schema grammar and a model's reasoning format can fight each other, and the structured fields hide it

**Context.** I run the optimism eval against a local model swapped by
(un)commenting a `CHAT_MODEL` line in [`config.py`](src/good_news/config.py).
Under Qwen it cost ~100–200 tokens per article; switching to a Gemma-4 build it
ballooned to ~8000, then started *failing* with `json.loads` errors like
`Unterminated string`. `classify()` constrains output with a `json_schema`
grammar (`response_format`).

**Finding.** A four-layer onion, each layer masking the next:

1. The reasoning-disable knobs were Qwen-only. `/no_think` and
   `chat_template_kwargs.enable_thinking` are Qwen3 conventions; Gemma silently
   ignores both, so it reasoned freely.
2. The failures weren't reasoning leakage — `reasoning_content` was empty and the
   garbage was *inside* the JSON's `reason` string, repeating one token
   (`progress-progress-…`) until the cap truncated it. A `frequency_penalty`
   didn't fix it; it just diversified the loop into multilingual word-salad
   (`solidifyingify`, a stray `고`), which ruled out "simple repetition."
3. The tell was *where* it broke: `category`, `optimism`, the booleans all came
   out perfect — only the one free-text field degenerated. The grammar was
   holding up the constrained fields and exposing the model only where it had
   freedom.
4. Root cause, found by probing the raw API (`scripts/diagnose_gemma.py`,
   `scripts/check_no_think.py`): this Gemma build is a reasoning model that emits
   a `<|channel>thought` block *before* its answer. The `json_schema` grammar
   only permits JSON tokens, so it *forbids* that thought block, forcing the
   model cold into `{` off-distribution — where the low-bit quant derails in the
   free-text field. Without the grammar the model reasoned coherently. So the
   grammar wasn't *constraining* the model, it was *fighting its output format*.

**Action.** Made the reasoning toggle model-family-aware in
[`llm.py`](src/good_news/llm.py): `_model_family()` reads `CHAT_MODEL`, and
`think_extra_body()` sends `enable_thinking` for Qwen but `reasoning_effort:
"none"` for Gemma — the one flag a strategy-matrix probe found that actually
silences its `<|channel>thought` (verified `clean`/`finish=stop`). With reasoning
off the model answers directly, stays on-distribution, and the existing grammar
behaves. Hardened the parse as a backstop: `verdict_json()` strips any leaked
reasoning, extracts the outermost `{...}`, and returns `""` on a truncated reply
so a cap hit reports plainly instead of throwing a cryptic `json.loads` error.

**Result.** One config-driven branch fixed cost, garbage, and the parse errors
together, because they were all the same root cause. The Qwen path is unchanged;
the Gemma path goes straight to a clean verdict. The `frequency_penalty` I added
mid-investigation turned out to be treating a symptom and could be reverted once
the real fix landed.

**Takeaway.**

- **Grammar-constrained decoding and a model's native output format can be
  incompatible.** A channel/reasoning model expects to emit a thought block
  first; a schema grammar that forbids it forces a cold start that low-bit quants
  can't survive. When structured output degenerates, test the *same prompt
  without the grammar* — if free-form is coherent, the grammar is the suspect.
- **Constrained fields hide model failure; the free-text field is your canary.**
  Perfect enums/numbers next to one garbage string field means the grammar is
  masking incoherence, not preventing it. Don't read "valid JSON" as "the model
  is fine."
- **Per-vendor capability flags don't transfer.** `enable_thinking` (Qwen) vs
  `reasoning_effort` (Gemma) do the same job under different names; key the
  request off a detected model family rather than assuming one knob works
  everywhere. A short strategy-matrix probe beats guessing.
- **Peel symptoms in order and resist fixing the visible one.** Cap → repetition
  → word-salad → grammar/reasoning conflict were nested; each "fix" (raise cap,
  add penalty) addressed a symptom and revealed the next layer. The real cause
  was four steps below the first error message.

---

## 2026-06-20 — Make the model justify before it scores, and write rubric caps as hard rules, not nudges

**Context.** The band anchors from the earlier calibration pass got both local
models to **13/20** on the optimism eval, but the residual failures were
lopsided in a telling way. Across *two* model families (Qwen and Gemma) almost
every miss was the model scoring *too high*, and the misses all clustered in the
0.25–0.55 "announced / pledged / pilot / modest local win" band — grants not yet
spent, counselors pledged but not hired, a tiny guaranteed-income pilot. The
same fixtures failing on both models pointed at the prompt, not either model.

**Finding.** Two structural causes. (1) `classify()` ran with thinking off *and*
`reason` was the last field in `VERDICT_SCHEMA`, so under grammar-constrained
decoding the model emitted the number **cold**, before writing any justification —
the worst setup for calibration. (2) The downward-pressure instruction was a soft
closing line ("lean lower if mostly announced, symbolic, or tiny"); the model's
central-tendency pull toward ~0.55 overrode the nudge. The clustering even landed
on `OPTIMISM_THRESHOLD` (0.55), so borderline items piled up exactly on the
include/exclude line.

**Action.** No code changes — only [`prompts.py`](src/good_news/prompts.py).
(1) Reordered `VERDICT_SCHEMA` so the booleans and `reason` generate *before*
`optimism`; property order is generation order under the grammar, so the model now
commits to a justification and conditions the number on it. (2) Replaced the soft
nudge with explicit hard caps (announced/pledged → 0.45, purely symbolic → 0.25,
fewer than a few hundred helped → 0.65) plus a "0.50–0.55 is not a safe hedge"
line. (3) Relabeled one fixture (`kindness_proclamation` 0.30 → 0.25) once the
model *and* the new symbolic-cap rule agreed it sat in the symbolic band — the
label was the outlier, not the model.

**Result.**

| Metric | Qwen before | Qwen after | Gemma before | Gemma after |
|---|---|---|---|---|
| Within ±0.1 | 13/20 | **18/20** | 13/20 | **16/20** |
| Mean absolute error | — | **0.058** | — | **0.068** |
| Mean signed error | +0.044 | **+0.003** | (optimistic) | **+0.050** |

The one-directional inflation is gone on both models, which confirms it was a
prompt problem. The remaining misses are genuine judgement edges (is a planted-but-
not-yet-fruiting orchard "delivered"? is a funded 60k-student meals program a 0.70
or a 0.85?), not systematic bias.

**Takeaway.**

- **Order the structured output so reasoning precedes the score.** A model forced
  to emit a number before its justification is scoring on vibes; put the free-text
  rationale field *first* in the schema and the grammar makes it think before it
  commits. This is a free calibration win — no extra tokens, no thinking mode.
- **A rubric the model under-applies needs hard caps, not adjectives.** "Lean
  lower" loses to central tendency; "cap at 0.45 if only pledged" holds, and you
  can see the model cite the rule back in its `reason`.
- **The prompt is one global context — edits have non-local effects.** A carve-out
  I added to rescue two cases (orchard, kidney) silently pushed two *unrelated*
  cases up by 0.15, because the MoE re-decodes the whole context. Always re-run the
  full set after any CRITERIA edit, and watch for the cobra effect of an over-broad
  exception ("concrete work done" leaked into "funding pledged").
- **When the model and a strengthened rule agree against a label, suspect the
  label.** The lone fixture both the model and the new cap pushed below its
  reference was mis-anchored; fixing reference data is part of calibration, not
  cheating — as long as you change it to match the *rubric*, not to match the model.
- **Cross-model agreement localizes the bug.** The same fixtures failing the same
  direction on two different model families is strong evidence the fault is in the
  shared prompt, not the model — and a fix that lifts both confirms it.

---

## 2026-07-10 — A presence check can't catch a permutation: the marker guardrail passed while every link pointed at the wrong story

**Context.** The digest tags each item with an opaque `@@N@@` marker that the
model copies onto the link line, and [`restore_links()`](src/good_news/guardrails.py)
swaps each marker for the real URL afterward (see the earlier "keep data out of
the model's hands" entry). That step already audits the raw output — literal
`http`, reused or missing marker numbers, leftover markers. I trusted it to catch
a bad mapping.

**Finding.** One `temperature=0.7` run produced a briefing where six of nine
stories carried the *wrong* link: the cities-cooling story linked to a John Deere
right-to-repair article, the John Deere story to an urban-climate article, and so
on — the markers had been **permuted** across sentences. The guardrail waved it
through, and the reason is structural: it only checks that the multiset of marker
numbers equals `{1..N}`. A permutation uses each number exactly once, so
`found == expected` holds — presence is intact, *assignment* is scrambled, and a
presence check is blind to assignment. The same run also wrote its markers inline
at the end of each sentence instead of on their own line, which (a) broke
`_space_items`, whose blank-line regex was anchored to URLs at line start, so the
items ran together, and (b) was the visible symptom that a run had gone sloppy —
the format drift and the mis-mapping travelled together.

**Action.** Three changes. (1) `_space_items` now inserts a blank line after
*every* restored URL wherever it landed — each URL is an item boundary — instead
of only line-leading ones, so inline markers no longer produce run-ons. (2) Added
`markers_each_on_own_line()` and made [`write_digest()`](src/good_news/llm.py)
**regenerate once at `temperature=0`** when the first draft fails it, adopting the
retry only if it comes back clean and complete (a worse re-roll never replaces a
usable first draft; a truncated reply skips the retry, since that's a token-cap
problem, not a formatting one). (3) Pulled the hardcoded `0.7` into
`DIGEST_TEMPERATURE = 0.3` — the digest is a format-faithful task as much as a
creative one, and the high temperature is what let the model wander off-format.

**Result.** The spacing bug is fixed and covered by a regression test; six new
tests exercise the fidelity check and all three retry paths (retries-and-adopts,
no-retry-when-clean, keeps-first-when-retry-still-bad). The retry keys off the
*format* deviation, which strongly co-occurred with the mis-mapping here — an
honest limitation remains: a run that formats markers perfectly yet still permutes
them would slip through. Fully closing that would need structured (non-free-form)
digest output or a semantic sentence↔item check; the temperature drop is the
bigger lever against it.

**Takeaway.**

- **A presence check is not an assignment check.** Verifying that every ID appears
  once says nothing about whether each ID is attached to the *right* thing. When a
  model carries opaque tokens across a reordering, the dangerous failure is the
  permutation, and it's invisible to any check that only counts tokens. Know which
  property your guardrail actually proves.
- **Watch for a cheap correlated signal when the real fault is undetectable.** I
  can't detect a permutation from the markers alone, but the model that permuted
  them also ignored the one-marker-per-line format — an observable deviation I
  *can* check and retry on. A proxy you can measure beats a fault you can't.
- **A format-faithful task wants a low temperature even when its output is prose.**
  The digest reads as creative writing, so `0.7` felt right — but it also has to
  copy IDs verbatim onto the correct sentence, and that discipline is what the heat
  eroded. Separate "should the wording vary" from "must the structure hold."
- **Post-processing that repairs the model's slop must not assume the model's
  format.** The blank-line fix broke because it assumed markers sat at line start;
  the model's whole point of failure is *not* following the format, so the cleanup
  has to handle the marker wherever it actually landed.
