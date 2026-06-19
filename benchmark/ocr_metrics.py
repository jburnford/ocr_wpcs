#!/usr/bin/env python3
"""CER/WER metrics core for the OCR benchmark.

Normalization (strict + semantic) matches the prior project
(archive-olm-pipeline/evaluation/evaluate_chandra_ocr.py); the from-scratch DP
Levenshtein there is replaced with rapidfuzz's C implementation (identical
results, far faster — needed for 600+ BLN600 pages).
"""
from __future__ import annotations
import re

from rapidfuzz.distance import Levenshtein

import failure_signals as _fs


# Typographic punctuation variants are cosmetic, not OCR errors: a transcriber
# writes a curly apostrophe ’ where an OCR tool emits a straight '. Canonicalize
# the whole class (single/double quotes, dashes) to ASCII before scoring so even
# strict CER/WER does not penalize them.
_SINGLE = "‘’‚‛ʼ`´"
_DOUBLE = "“”„‟"
_DASH = "–—‐‑‒―"
_PUNCT_CANON = str.maketrans(
    _SINGLE + _DOUBLE + _DASH,
    "'" * len(_SINGLE) + '"' * len(_DOUBLE) + "-" * len(_DASH),
)


def normalize_text(text: str, semantic: bool = False) -> str:
    """strict: canonicalize typographic punctuation + collapse whitespace.
    semantic: also lowercase + drop punctuation."""
    text = text.translate(_PUNCT_CANON)
    if semantic:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Character error rate. Returns (rate, edit_distance, reference_length)."""
    edits = Levenshtein.distance(reference, hypothesis)
    n = len(reference)
    return (edits / n if n else 0.0), edits, n


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Word error rate (Levenshtein over word sequences)."""
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    edits = Levenshtein.distance(ref_w, hyp_w)
    n = len(ref_w)
    return (edits / n if n else 0.0), edits, n


# --- evaluation-chapter metrics ------------------------------------------------
# Mirrors the co-author's Chapter 1 evaluation: WER, significant-word accuracy
# (WER over content words only), BLEU-4, and hallucination rate (real-word
# errors absent from gold, dictionary = NLTK words corpus, ~236k entries).
from nltk.corpus import stopwords as _nltk_stopwords  # noqa: E402
from nltk.corpus import words as _nltk_words  # noqa: E402
from nltk.translate.bleu_score import (  # noqa: E402
    SmoothingFunction,
    sentence_bleu,
)

_DICT = frozenset(w.lower() for w in _nltk_words.words())
_STOP = frozenset(_nltk_stopwords.words("english"))
_SMOOTH = SmoothingFunction().method1


def is_dictionary_word(word: str) -> bool:
    """True if `word` (stripped of punctuation, lowercased) is a real English
    word per the NLTK words corpus — used to tell a hallucination (a wrong but
    real, plausible word) from a visibly-garbled misread."""
    w = re.sub(r"[^\w]", "", word.lower())
    return bool(w) and w in _DICT


def significant_word_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """WER over content words only (English function words removed)."""
    ref_w = [w for w in reference.split() if w not in _STOP]
    hyp_w = [w for w in hypothesis.split() if w not in _STOP]
    edits = Levenshtein.distance(ref_w, hyp_w)
    n = len(ref_w)
    return (edits / n if n else 0.0), edits, n


def bleu_score(reference: str, hypothesis: str) -> float:
    """Smoothed sentence BLEU-4 of hypothesis against reference."""
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w or not hyp_w:
        return 0.0
    return float(sentence_bleu([ref_w], hyp_w, smoothing_function=_SMOOTH))


def hallucination_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Fraction of OCR words that are real dictionary words absent from gold.

    Per the evaluation chapter: an error word (not in gold) that is nonetheless
    a real English word is a 'hallucination' — plausible but wrong, the kind of
    error NER would silently extract. Returns (rate, count, ocr_word_total).
    """
    gold_set = set(reference.split())
    hyp_w = hypothesis.split()
    if not hyp_w:
        return 0.0, 0, 0
    halluc = sum(1 for w in hyp_w if w not in gold_set and w in _DICT)
    return halluc / len(hyp_w), halluc, len(hyp_w)


def hallucination_split(reference: str, hypothesis: str,
                        max_edit: int = 2) -> tuple[int, int]:
    """Split the hallucinations counted by hallucination_rate into two kinds.

    A 'hallucination' is a real dictionary word in the OCR that is absent from
    the gold. On historical text these are not all the same error:

      - modernization: the OCR word is within `max_edit` characters of a word
        that IS on the page (gold). It is a misread or a silently normalized
        spelling variant of real page content — e.g. early-modern 'bloud'
        rendered as 'blood', 'armes' as 'arms'. A fidelity problem (the model
        editorialises the orthography), not an invention.
      - fabrication: no gold word is within `max_edit`. Text the model added
        that is nowhere on the page — the kind a downstream NER would wrongly
        extract.

    Returns (modernization, fabrication); their sum equals the count from
    hallucination_rate. Pass semantic-normalized text (as evaluate_pair does).
    """
    gold_set = set(reference.split())
    by_len: dict[int, list[str]] = {}
    for g in gold_set:
        by_len.setdefault(len(g), []).append(g)
    modern = fabricate = 0
    for w in hypothesis.split():
        if w in gold_set or w not in _DICT:
            continue
        lw = len(w)
        near = False
        for length in range(lw - max_edit, lw + max_edit + 1):
            for g in by_len.get(length, ()):
                if Levenshtein.distance(w, g, score_cutoff=max_edit) <= max_edit:
                    near = True
                    break
            if near:
                break
        if near:
            modern += 1
        else:
            fabricate += 1
    return modern, fabricate


def evaluate_pair(filename: str, gold_text: str, ocr_text: str) -> dict:
    """Score one gold/OCR pair, strict + semantic. Schema matches the prior
    project's per-file result dict so the comparison/analysis scripts apply."""
    g_strict = normalize_text(gold_text, semantic=False)
    o_strict = normalize_text(ocr_text, semantic=False)
    g_sem = normalize_text(gold_text, semantic=True)
    o_sem = normalize_text(ocr_text, semantic=True)

    cer_s, cer_e_s, _ = cer(g_strict, o_strict)
    wer_s, wer_e_s, _ = wer(g_strict, o_strict)
    cer_m, cer_e_m, _ = cer(g_sem, o_sem)
    wer_m, wer_e_m, _ = wer(g_sem, o_sem)

    # evaluation-chapter metrics, computed on semantic-normalized text
    sig_wer, sig_e, sig_n = significant_word_wer(g_sem, o_sem)
    bleu = bleu_score(g_sem, o_sem)
    hr, hr_count, hr_total = hallucination_rate(g_sem, o_sem)
    hm, hf = hallucination_split(g_sem, o_sem)
    # gold-free failure signals on the (cleaned) OCR text — flag silent failures
    # (loops/runaway/empty) even where gold exists. expected_words=None here, so
    # the length-blowup trigger is off for variable-length corpus documents.
    fsig = _fs.failure_signals(ocr_text)

    return {
        "filename": filename,
        "gold_chars": len(g_strict),
        "ocr_chars": len(o_strict),
        "gold_words": len(g_strict.split()),
        "ocr_words": len(o_strict.split()),
        "cer": cer_s,
        "cer_edits": cer_e_s,
        "wer": wer_s,
        "wer_edits": wer_e_s,
        "gold_chars_semantic": len(g_sem),
        "ocr_chars_semantic": len(o_sem),
        "gold_words_semantic": len(g_sem.split()),
        "ocr_words_semantic": len(o_sem.split()),
        "cer_semantic": cer_m,
        "cer_edits_semantic": cer_e_m,
        "wer_semantic": wer_m,
        "wer_edits_semantic": wer_e_m,
        # chapter metrics
        "sig_wer": sig_wer,
        "sig_wer_edits": sig_e,
        "sig_gold_words": sig_n,
        "sig_word_accuracy": max(0.0, 1.0 - sig_wer),
        "bleu": bleu,
        "hallucination_rate": hr,
        "hallucination_count": hr_count,
        "hallucination_modernization": hm,
        "hallucination_fabrication": hf,
        "ocr_word_total": hr_total,
        # gold-free failure signals (see failure_signals.py)
        "gzip_ratio": fsig["gzip_ratio"],
        "failure_label": fsig["failure_label"],
        "gold_text_preview": g_strict[:120],
        "ocr_text_preview": o_strict[:120],
    }


def corpus_summary(results: list[dict]) -> dict:
    """Aggregate per-file results into corpus + average metrics."""
    n = len(results)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / n

    def overall(edits_key: str, len_key: str) -> float:
        tot_len = sum(r[len_key] for r in results)
        tot_edits = sum(r[edits_key] for r in results)
        return tot_edits / tot_len if tot_len else 0.0

    tot_ocr_words = sum(r["ocr_word_total"] for r in results)
    tot_halluc = sum(r["hallucination_count"] for r in results)
    tot_modern = sum(r.get("hallucination_modernization", 0) for r in results)
    tot_fabricate = sum(r.get("hallucination_fabrication", 0) for r in results)
    return {
        "total_files": n,
        "strict": {
            "total_gold_chars": sum(r["gold_chars"] for r in results),
            "total_gold_words": sum(r["gold_words"] for r in results),
            "average_cer": avg("cer"),
            "overall_cer": overall("cer_edits", "gold_chars"),
            "average_wer": avg("wer"),
            "overall_wer": overall("wer_edits", "gold_words"),
        },
        "semantic": {
            "total_gold_chars": sum(r["gold_chars_semantic"] for r in results),
            "total_gold_words": sum(r["gold_words_semantic"] for r in results),
            "average_cer": avg("cer_semantic"),
            "overall_cer": overall("cer_edits_semantic", "gold_chars_semantic"),
            "average_wer": avg("wer_semantic"),
            "overall_wer": overall("wer_edits_semantic", "gold_words_semantic"),
        },
        "chapter": {
            "average_wer": avg("wer_semantic"),
            "average_sig_word_accuracy": avg("sig_word_accuracy"),
            "overall_sig_wer": overall("sig_wer_edits", "sig_gold_words"),
            "average_bleu": avg("bleu"),
            "overall_hallucination_rate": (
                tot_halluc / tot_ocr_words if tot_ocr_words else 0.0
            ),
            "average_hallucination_rate": avg("hallucination_rate"),
            # split of the same hallucinations: misread/normalized real page
            # content vs text invented from nowhere (see hallucination_split)
            "overall_modernization_rate": (
                tot_modern / tot_ocr_words if tot_ocr_words else 0.0
            ),
            "overall_fabrication_rate": (
                tot_fabricate / tot_ocr_words if tot_ocr_words else 0.0
            ),
            "hallucination_modernization_count": tot_modern,
            "hallucination_fabrication_count": tot_fabricate,
        },
        # distribution of gold-free failure labels across the corpus
        "failure": _label_counts(results),
    }


def _label_counts(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        lbl = r.get("failure_label", "clean")
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts
