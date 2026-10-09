"""Post-OCR correction: domain lexicon replacement.

Design note, because it is the part a panel will ask about:

The corrector is NOT a seq2seq model. Nothing generates free text, so it can
never invent a sentence the scanner did not see. It only ever replaces one
out-of-lexicon token with an in-lexicon token, and only when the string
similarity clears a deliberately high bar. Worst case it leaves the OCR output
alone; it cannot fabricate a legal claim into a blotter record. That property
is worth more here than the extra point of CER a generative corrector might buy.

One channel decides every replacement: similarity against the barangay's own
vocabulary (models/ocr_lexicon.json, built from the corpus, so it already
contains IPA-BLOTTER, NAGSADYA, PANINILIP and the purok names). Similarity is
1 - edit distance / length of the longer word, where swapping two glyphs the
scanner is known to confuse (_CONFUSIONS) costs half an edit.

It used to be difflib's ratio, which scores a dropped or added affix as a near
match (KASAGUTAN -> SAGUTAN 0.875, IYONG -> IYON 0.889). On the real Vision
pages in data/ocr_calibration.csv that rewrote correctly read Tagalog words the
lexicon lacks more often than it repaired garbled ones - precision 0.46 on the
holdout pages, token accuracy down, not up (tools/eval_corrector.py). Now an
affix change costs a full edit and a misread glyph half of one, so PUNOK
becomes PUROK (R read as N) rather than PUNO (a letter dropped).

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

# Glyph pairs Google Vision confuses on this barangay's handwritten capitals, as
# (read, written): "SG" is a written G that came back as S. Directional, because
# the scanner's mistakes are: G read as S is common, S read as G is not. Fitted
# on the 151 dev pages only, never the 37 holdout pages; `python
# tools/eval_corrector.py --fit` re-derives it.
_CONFUSIONS = frozenset(
    "SG AG MN NR OA PD CL EG CG IA DA UN HN WN UO LK TA DG RK TI OE HA RA RN EO "
    "GA LG UR GE KA MA BD AT RB IO JT EA OG YW".split())
CONFUSION_COST = 0.5

# Tagalog marks verb aspect on the first letter - MAKAKITA (will see) against
# NAKAKITA (saw), MAGBAYAD / NAGBAYAD / PAGBAYAD - and M read for N is one of
# the scanner's commonest misreads. One token cannot tell the two apart, and a
# silent tense change is a changed legal fact, so that swap is never made.
_ASPECT = frozenset("MNP")


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
            upper = word.upper()
            best, best_key = None, None
            # accept_floor is deliberately far above sim_floor: get_close_matches
            # is only a prefilter, and an edit that barely clears 0.70 similarity
            # is a coin flip on a legal record. The looser bar is what put 11,084
            # edits at 0.58 precision into models/ocr_correction_report.json.
            for cand in self._candidates(word):
                if upper[0] != cand[0] and upper[0] in _ASPECT and cand[0] in _ASPECT:
                    continue
                # Ties go to the commoner word: deterministic, and the likelier one.
                key = (_similarity(upper, cand), self.lexicon.get(cand, 0))
                if key[0] >= self.accept_floor and (best_key is None or key > best_key):
                    best, best_key = cand, key
            if best and best != upper:
                edits.append({"start": m.start(), "end": m.end(), "before": word,
                              "after": _match_case(word, best), "score": round(best_key[0], 4)})
        if not edits:
            return {"text": text, "edits": [], "n_edits": 0}

        out, cursor = [], 0
        for e in edits:
            out.append(text[cursor:e["start"]])
            out.append(e["after"])
            cursor = e["end"]
        out.append(text[cursor:])
        return {"text": "".join(out), "edits": edits, "n_edits": len(edits)}


def _similarity(word, cand):
    """1 - weighted Levenshtein distance / longer length; both upper-case.

    Insertions and deletions cost 1, so an affix gained or lost is a full edit;
    a substitution in _CONFUSIONS costs CONFUSION_COST.
    """
    prev = list(range(len(cand) + 1))
    for i, a in enumerate(word, 1):
        cur = [i]
        for j, b in enumerate(cand, 1):
            sub = 0 if a == b else CONFUSION_COST if a + b in _CONFUSIONS else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return 1 - prev[-1] / max(len(word), len(cand))


def _match_case(source, replacement):
    # Majority case, not str.isupper(): a stray lowercase glyph in an all-caps
    # page ("PANINlLIP") must not turn the fix into lowercase mid-sentence.
    letters = [c for c in source if c.isalpha()]
    if sum(c.isupper() for c in letters) * 2 > len(letters):
        return replacement.upper()
    if source[:1].isupper():
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
