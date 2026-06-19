#!/usr/bin/env python3
"""Run Chandra 2's CLI with an added 'do not modernize spelling' instruction.

Prompt experiment for the WPCS paper: Chandra's default `ocr_layout` prompt says
nothing about orthography, and on the early-modern (Jacob) corpus it silently
normalizes some archaic spellings (counted as 'modernization' hallucinations).
We test whether an explicit diplomatic-transcription instruction reduces that.

No package edits: we mutate the in-memory PROMPT_MAPPING (chandra.model.vllm
resolves PROMPT_MAPPING[item.prompt_type] at call time, so the mutation is live),
then hand off to the normal `chandra` CLI entrypoint. argv passes straight
through, e.g.:  python this_wrapper.py <pdf> <outdir> --method vllm
"""
import chandra.prompts as P

FIDELITY = (
    "\n\nCRITICAL — DIPLOMATIC TRANSCRIPTION: Transcribe every word exactly as "
    "printed, preserving the original early-modern and archaic spelling, "
    "capitalization, and punctuation. Do NOT modernize, normalize, regularize, "
    "or correct spelling. Keep period forms exactly as written (e.g. 'bloud' "
    "not 'blood', 'armes' not 'arms', 'widdow' not 'widow', 'goodnesse' not "
    "'goodness'). Render the long-s as a normal 's' but change nothing else."
)

for _k in ("ocr_layout", "ocr"):
    if _k in P.PROMPT_MAPPING and isinstance(P.PROMPT_MAPPING[_k], str):
        P.PROMPT_MAPPING[_k] = P.PROMPT_MAPPING[_k] + FIDELITY

from chandra.scripts.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
