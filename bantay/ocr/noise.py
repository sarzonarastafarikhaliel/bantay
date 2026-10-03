"""OCR noise model — makes (noisy, clean) training pairs from the clean corpus.

Why synthesise at all: the corrector needs thousands of supervised pairs and
the barangay has 73 scanned pages. So corrupt the clean narratives with a
confusion model instead, and calibrate that model on whatever real pairs exist.

DEFAULT_CONFUSIONS is a literature-standard uppercase-handwriting set (the
blotter is written in all caps). It is a starting point, NOT a measurement.
Run `fit_confusions()` on even 10 real (ocr_output, gold_transcription) pairs
and the generator matches the real scanner instead of a textbook — that one
call is the difference between a corrector that works on the pilot pages and
one that only works on paper.
"""
import difflib
import random

# char -> what the scanner tends to emit instead
DEFAULT_CONFUSIONS = {
    "O": ["0", "Q", "D"], "0": ["O", "D"], "I": ["1", "L", "T"], "1": ["I", "L"],
    "L": ["I", "1"], "S": ["5", "8"], "5": ["S"], "B": ["8", "R", "P"], "8": ["B", "S"],
    "G": ["6", "C"], "6": ["G"], "Z": ["2", "7"], "2": ["Z"], "U": ["V", "Y"],
    "V": ["U", "Y"], "C": ["G", "O", "("], "E": ["F", "B"], "F": ["E", "P"],
    "R": ["P", "B", "K"], "P": ["R", "F"], "N": ["M", "H"], "M": ["N", "H"],
    "H": ["N", "M", "K"], "A": ["4", "R"], "4": ["A"], "D": ["O", "0", "P"],
    "T": ["7", "I", "F"], "7": ["T", "Z"], "Y": ["V", "U"], "K": ["X", "R"],
    "W": ["V", "M"], "J": ["I", "T"], "Q": ["O", "G"], "X": ["K", "Y"],
    ",": [".", ""], ".": [",", ""], "-": ["_", ""],
}

# multi-char merges/splits the single-char table cannot express
DEFAULT_LIGATURES = {"RN": "M", "M": "RN", "CL": "D", "VV": "W", "II": "U", "NG": "MG"}


def corrupt(text, rate=0.08, seed=None, confusions=None, ligatures=None,
            p_space=0.15, p_drop=0.10):
    """Return an OCR-degraded copy of `text`.

    rate     fraction of characters touched (0.08 ~ CER 8%, typical for a good
             engine on clean handwriting; push to 0.15-0.25 for hard pages).
    p_space  share of edits that split/merge a word instead of swapping a char.
    p_drop   share of edits that delete the character outright.
    """
    rng = random.Random(seed)
    conf = confusions or DEFAULT_CONFUSIONS
    ligs = ligatures if ligatures is not None else DEFAULT_LIGATURES
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.strip() and rng.random() < rate:
            roll = rng.random()
            pair = text[i:i + 2].upper()
            if ligs and pair in ligs and roll < 0.10:
                out.append(ligs[pair])
                i += 2
                continue
            if roll < p_space:
                out.append(" " if ch != " " else "")      # spurious split
            elif roll < p_space + p_drop:
                pass                                       # dropped glyph
            elif ch.upper() in conf:
                sub = rng.choice(conf[ch.upper()])
                out.append(sub if ch.isupper() else sub.lower())
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def fit_confusions(pairs, min_count=2):
    """Learn a confusion table from real (noisy, clean) pairs.

    pairs: iterable of (ocr_text, gold_text). Character-aligned with difflib,
    so it tolerates insertions and deletions. Returns a dict in the same shape
    as DEFAULT_CONFUSIONS, with substitutions repeated by frequency so
    random.choice() samples them proportionally.
    """
    counts = {}
    for noisy, clean in pairs:
        sm = difflib.SequenceMatcher(a=clean, b=noisy, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace":
                continue
            src, dst = clean[i1:i2], noisy[j1:j2]
            if len(src) == len(dst):
                for a, b in zip(src, dst):
                    counts.setdefault(a.upper(), {}).setdefault(b, 0)
                    counts[a.upper()][b] += 1
    table = {}
    for src, dsts in counts.items():
        weighted = [d for d, n in dsts.items() if n >= min_count for _ in range(n)]
        if weighted:
            table[src] = weighted
    return table


def cer(hyp, ref):
    """Character error rate: edit distance / len(ref). The headline OCR metric."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for j, rc in enumerate(ref, 1):
        cur = [j]
        for i, hc in enumerate(hyp, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hc != rc)))
        prev = cur
    return prev[-1] / len(ref)


def wer(hyp, ref):
    """Word error rate. Same distance, token granularity."""
    h, r = hyp.split(), ref.split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for j, rw in enumerate(r, 1):
        cur = [j]
        for i, hw in enumerate(h, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hw != rw)))
        prev = cur
    return prev[-1] / len(r)
