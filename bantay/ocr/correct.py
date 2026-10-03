"""Post-OCR correction: domain lexicon replacement.

Design note, because it is the part a panel will ask about:

The corrector is NOT a seq2seq model. Nothing generates free text, so it can
never invent a sentence the scanner did not see. It only ever replaces one
out-of-lexicon token with an in-lexicon token, and only when the string
similarity clears a deliberately high bar. Worst case it leaves the OCR output
alone; it cannot fabricate a legal claim into a blotter record. That property
is worth more here than the extra point of CER a generative corrector might buy.

One channel decides every replacement: a difflib ratio against the barangay's
own vocabulary (models/ocr_lexicon.json, built from the corpus, so it already
contains IPA-BLOTTER, NAGSADYA, PANINILIP and the purok names).

Pure stdlib - no torch, no transformers, no checkpoint. This is the offline
fallback for when the Gemini repair pass in gemini.py is unavailable, and it
must run on a barangay hall machine with no GPU and no ML extras installed.
"""
import difflib
import json
import os
import re

# Digits belong INSIDE the word pattern: OCR turns S into 5 and O into 0, so
# "NAG5ADY4" is one corrupted token, not three fragments. _suspicious() below
# is what keeps genuine numbers ("2023", "P3M") out of the candidate search.
_WORD_RE = re.compile(r"[A-Za-zÑñ][A-Za-z0-9Ññ'\-]*")


class OCRCorrector:
    def __init__(self, lexicon=None, max_candidates=5,
                 sim_floor=0.70, accept_floor=0.86):
        self.lexicon = lexicon or {}
        self.vocab = list(self.lexicon)
        self.max_candidates = max_candidates
        self.sim_floor = sim_floor        # cheap prefilter for get_close_matches
        self.accept_floor = accept_floor  # the bar an edit must actually clear

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_dir(cls, model_dir, **kw):
        """Load model_dir/ocr_lexicon.json  ({token: frequency}).

        Missing file is not an error: an empty lexicon makes correct() a no-op,
        which is the right degradation for a corrector that can only ever
        replace tokens with ones it already knows.
        """
        lex_path = os.path.join(model_dir, "ocr_lexicon.json")
        lexicon = {}
        if os.path.exists(lex_path):
            with open(lex_path, encoding="utf-8") as fh:
                lexicon = json.load(fh)
        return cls(lexicon=lexicon, **kw)

    # ---- correction -------------------------------------------------------
    def _suspicious(self, word):
        w = word.upper()
        if len(w) < 3 or w in self.lexicon:
            return False
        digits = sum(c.isdigit() for c in w)
        return digits / len(w) < 0.5      # mostly-numeric token is a real number

    def _candidates(self, word):
        return difflib.get_close_matches(word.upper(), self.vocab, n=self.max_candidates,
                                         cutoff=self.sim_floor)

    def correct(self, text):
        """Return {"text", "edits": [{before, after, score, start, end}], "n_edits"}."""
        if not text or not self.vocab:
            return {"text": text, "edits": [], "n_edits": 0}

        edits = []
        for m in _WORD_RE.finditer(text):
            word = m.group()
            if not self._suspicious(word):
                continue
            best, best_score = None, 0.0
            # accept_floor is deliberately far above sim_floor: get_close_matches
            # is only a prefilter, and an edit that barely clears 0.70 similarity
            # is a coin flip on a legal record. The looser bar is what put 11,084
            # edits at 0.58 precision into models/ocr_correction_report.json.
            for cand in self._candidates(word):
                sim = difflib.SequenceMatcher(a=word.upper(), b=cand).ratio()
                if sim >= self.accept_floor and sim > best_score:
                    best, best_score = cand, sim
            if best and best.upper() != word.upper():
                edits.append({"start": m.start(), "end": m.end(), "before": word,
                              "after": _match_case(word, best), "score": round(best_score, 4)})
        if not edits:
            return {"text": text, "edits": [], "n_edits": 0}

        out, cursor = [], 0
        for e in edits:
            out.append(text[cursor:e["start"]])
            out.append(e["after"])
            cursor = e["end"]
        out.append(text[cursor:])
        return {"text": "".join(out), "edits": edits, "n_edits": len(edits)}


def _match_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.title()
    return replacement.lower()


def build_lexicon(texts, min_count=2, extra=()):
    """Frequency lexicon from clean narratives. Writes models/ocr_lexicon.json."""
    counts = {}
    for t in texts:
        for m in _WORD_RE.finditer(t or ""):
            w = m.group().upper()
            if len(w) >= 2:
                counts[w] = counts.get(w, 0) + 1
    for w in extra:
        counts[w.upper()] = max(counts.get(w.upper(), 0), min_count)
    return {w: c for w, c in counts.items() if c >= min_count}
