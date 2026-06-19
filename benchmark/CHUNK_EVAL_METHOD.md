# Order-invariant chunk-aligned OCR scoring — method

Status: working draft (full pages). Code: `benchmark/chunk_eval.py` (core),
`benchmark/run_chunk_eval.py` (runner). Output: `results/fullpage_chunk_eval.json`.

## Why this exists

Corpus CER/WER score the OCR output as one **linear character sequence** against
the gold's linear sequence (`benchmark/ocr_metrics.py`). On multi-column
newspaper pages that conflates three independent things:

1. **Character recognition** — did the tool read the glyphs correctly?
2. **Reading-order serialization** — did it emit the columns/articles in the
   same order the transcriber wrote them down?
3. **Page scope** — did it transcribe the same regions (ads/mastheads in or out)?

A tool that reads every character perfectly but serializes the six columns in a
different order scores as a near-total failure under linear CER. Measured this
way, much of Chandra's and Infinity's "error" on full pages was reading-order,
not recognition (Chandra 23.6%→12.7%, Infinity 14.5%→6.9% once order is removed).

For the downstream goal — text mining / information extraction — what matters is
that each **coherent chunk (article, or paragraph)** comes out accurately and
intact, *regardless of the order the tool emitted the chunks in*. This method
measures that.

## Design in one sentence

Segment only the **gold** into chunks; leave each tool's output whole; for every
gold chunk find its best-matching contiguous span anywhere in the tool output
(order-invariant by construction); score character/word accuracy **within** that
span; track separately how much gold content was recovered and how much tool
text was never claimed by any gold chunk.

This deliberately does **not** require segmenting the tool output into articles
(which is unreliable for plain-text tools). It is the generalization of the
manuscript segment-alignment already in `locate_article.best_match_span`.

## Step 1 — Gold chunk parsing (`segment_gold`)

Input: a full-page `*_review.md` gold. Only the body after the `## Page Content`
heading is used. The body interleaves transcribed text with bracketed layout
annotations, e.g. `[Masthead]`, `[left column]`, `[ad]`,
`[line for separation but no header]`, `[cartoon image ...]`.

Parsing:
1. Split the body into raw blocks on blank lines.
2. Within each block, drop `#` heading lines; classify every **bracket-only**
   line as a *marker*, not content:
   - ad / masthead marker (`_AD`) — also counted in `ad_blocks`;
   - column marker (`_COL`), separator marker (`_SEP`), image marker (`_IMG`).
   Any of these sets a **boundary flag** for the block.
3. Strip inline bracket spans from remaining lines (`_clean`); the surviving
   text is the block's content paragraph. If the first content line passes the
   headline heuristic (`_looks_like_header`), the block also starts a boundary.

Two granularities are produced from the same parse:

- **paragraph** — every content block is its own chunk. Reliable on all 8 golds
  (depends only on blank lines, not on layout annotations). This is the trusted
  number.
- **article** — content blocks are merged left-to-right until the next boundary
  (column / separator / ad / detected header). Approximate: the golds are
  inconsistently annotated (only one of 8 pages has separator markers; 3 have no
  column markers), so on unannotated pages article boundaries fall back entirely
  to the header heuristic. Use as corroboration, not as the headline.

`_looks_like_header`: a line is treated as a headline if, after stripping a
trailing period, it has ≤8 alphabetic words, ≤64 chars, and is either all-caps
or ≥80% of its words are capitalized. (Matches the golds' described headers:
"one to four words in all caps or title case".)

## Step 2 — Order-invariant matching (`_best_span`)

For one gold chunk `g` against the whole tool output `hyp`:

1. Whitespace-normalize and lowercase both.
2. `rapidfuzz.fuzz.partial_ratio_alignment(g, hyp)` finds the best-matching
   window of `hyp` and a similarity score in [0,1]. Because the search is over
   the entire `hyp`, the chunk's position/order in the tool output is irrelevant.
3. Map the normalized window back to a raw substring of `hyp` (proportional
   offset by length ratio, padded by `max(200, len(g)//4)` chars each side).
4. `locate_article._best_window` refines that padded window to the substring of
   gold length with the lowest edit distance (coarse slide + fine step).

Returns `(score, span)` where `span` is the raw tool substring aligned to `g`.

## Step 3 — Scoring (`score_chunks`)

Per gold chunk:
- **located** iff `score >= LOCATE_THRESHOLD` (0.55) and a span was found.
- If located: compute CER and WER of the span vs the chunk, both on
  semantic-normalized text (`ocr_metrics.normalize_text(..., semantic=True)`:
  lowercase, strip punctuation, collapse whitespace). Also record `coverage`
  (span length / chunk length, capped at 1) and BLEU.
- If not located: the chunk's full character/word count is added to the
  denominators as a complete miss (dropped or garbled content).

Per page/tool aggregate:
- **coverage_chunks** = located chunks / total chunks.
- **coverage_chars** = gold (semantic) chars in located chunks / total gold
  chars. Catches dropped content (olmOCR ~47% on full pages; total miss = the
  chunk simply isn't anywhere in the output).
- **within_cer / within_wer** = char/word-weighted error over located chunks
  only — the clean recognition signal, with ordering removed. (`run_chunk_eval`
  reports within_cer; within_wer is available per file.)
- **extra_ratio** = tool (semantic) chars not matched by any gold chunk, over
  total tool chars. A rough gauge of ads/mastheads/noise the tool transcribed
  that the gold omits. Reported, **not** penalized: per the gold's scope a tool
  should not be punished for reading more than the transcriber recorded.

## Interpreting the outputs

- Low `within_cer` + high `coverage` = accurate and complete (Infinity).
- Low `within_cer` + lower `coverage` = accurate but drops content (Chandra's
  truncation on Davidson p02).
- Higher `within_cer` + high `coverage` = reads everything but genuinely misreads
  (Gemini — its errors are real, not ordering).
- `coverage` near 0 with a non-zero `within_cer` printed = a *total miss*: almost
  nothing matched, so the within-chunk number is over a tiny fragment and is not
  meaningful (olmOCR, Davidson p02).

## Known limitations

- **Article granularity is approximate** where the gold lacks layout markers;
  paragraph-level is the reliable headline. Both agreed on tool ranking in the
  full-page run.
- Independent per-chunk search can let two gold chunks claim overlapping `hyp`
  regions; `extra_ratio` is therefore an estimate, not an exact disjoint cover.
- Coherence/fragmentation is currently captured only indirectly (a scattered
  article yields a short best span → low `coverage` and higher CER). An explicit
  fragmentation metric (one gold article → how many tool fragments) is a possible
  extension.
- Method is wired for the **full-page** dataset. Tables need a 2D (cell-grid)
  analogue — see notes in the tables discussion, not yet built.
