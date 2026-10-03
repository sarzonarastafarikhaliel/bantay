"""Narrative-preservation guard for blotter transcription. Standard library only.

This is the portable port of the guards in bantay/ocr/gemini.py, for use where
that package is not installed (Claude Desktop / claude.ai, where the skill runs
in a sandbox with no project checkout). Same thresholds and the same per-edit
rules, so a page this accepts is a page the app would accept.

It is deliberately STRICTER than the app in two places, because it can only
report where the app can act. The app reverts a rejected edit silently before
storing the page; here a rejected edit fails the whole check, since nothing else
will revert it. And edit detection counts numbers as tokens (see TOKEN_RE),
which the app does not need to do - its splice protects the digits structurally.

The point is that "I only made minor corrections" is not something a model can
be trusted to assert about its own output. This measures it.

    python guard.py check verbatim.txt corrected.txt         # verdict + edit list
    python guard.py check verbatim.txt corrected.txt --json  # machine-readable
    python guard.py types                                    # the 35 valid types
    python guard.py lookup "Theft"                           # group / PNP / KP

Exit code 0 = ACCEPT, 1 = REJECT (keep the verbatim text instead).
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

# Identical to bantay/ocr/correct.py's tokeniser - N-tilde included because the
# corpus is Tagalog/English, and a tokeniser that split on it would report
# phantom edits on every name containing one.
WORD_RE = re.compile(r"[A-Za-zÑñ][A-Za-z0-9Ññ'\-]*")

# Edit detection needs a WIDER tokeniser than the fidelity rates do. In the app,
# a bare number is invisible to WORD_RE and is therefore never spliced back into
# the page - the original digits survive structurally, so the guard never has to
# see them. This script has no splice: it only reports, so if it cannot SEE a
# 2023 -> 2028 change it cannot stop one. Numbers are tokens here for that reason.
TOKEN_RE = re.compile(r"[A-Za-z0-9Ññ][A-Za-z0-9Ññ'\-]*")

# Thresholds, matching gemini.py's defaults.
MAX_PAGE_CHANGE = 0.35        # BANTAY_RESTORE_MAX_CHANGE
MAX_TOKEN_GAIN = 0.10         # BANTAY_MAX_TOKEN_GAIN
MAX_TOKEN_LOSS = 0.10         # a repair may not quietly drop the page
MIN_SIMILARITY = 0.5          # BANTAY_MIN_RESTORE_SIMILARITY

TAXONOMY = Path(__file__).with_name("taxonomy.json")


def tokens(text, pattern=WORD_RE):
    """Content tokens, upper-cased. WORD_RE for the rates (parity with the app),
    TOKEN_RE for edit detection (see that constant)."""
    return [m.group().upper() for m in pattern.finditer(text or "")]


def mostly_numeric(token):
    """A date, time, amount or case number. Never eligible for correction: in a
    blotter record 2023 -> 2028 is the error that surfaces in a hearing."""
    return sum(c.isdigit() for c in token) / max(len(token), 1) >= 0.5


def fidelity(source, candidate):
    """Token-level loss / gain / distortion, as shares of the SOURCE length.

    loss       source tokens with no output  - the page being silently dropped
    gain       output tokens with no source  - the model inventing text
    distortion aligned but changed           - the actual repair, the wanted part
    """
    a, b = tokens(source), tokens(candidate)
    loss = gain = distortion = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            continue
        nb, na = i2 - i1, j2 - j1
        distortion += min(nb, na)
        loss += max(nb - na, 0)
        gain += max(na - nb, 0)
    n = max(len(a), 1)
    return {"n_tokens": len(a), "loss": loss, "gain": gain, "distortion": distortion,
            "loss_rate": round(loss / n, 4), "gain_rate": round(gain / n, 4),
            "distortion_rate": round(distortion / n, 4),
            "change_rate": round((loss + gain + distortion) / n, 4)}


def _keep(before, after):
    """Per-edit guards. Returns (keep, reason) so a rejection can be explained
    rather than merely counted."""
    if not before and not after:
        return False, "empty"
    if mostly_numeric(before) or mostly_numeric(after):
        return False, "numeric - dates, times, amounts and case numbers are never altered"
    if not after.strip():
        return False, "deletion - repairing an unreadable word is in scope, erasing it is not"
    if before and difflib.SequenceMatcher(
            a=before.upper(), b=after.upper()).ratio() < MIN_SIMILARITY:
        return False, "not a plausible misread - reads as a paraphrase, not a correction"
    return True, "plausible misread"


def edits(source, candidate):
    """Every changed token pair, each with its verdict and reason."""
    a, b = tokens(source, TOKEN_RE), tokens(candidate, TOKEN_RE)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            continue
        before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        keep, why = _keep(before, after)
        out.append({"before": before, "after": after, "keep": keep, "reason": why})
    return out


def check(source, candidate):
    """ACCEPT or REJECT the whole corrected page, with reasons."""
    fid = fidelity(source, candidate)
    ed = edits(source, candidate)
    reasons = []
    if fid["change_rate"] > MAX_PAGE_CHANGE:
        reasons.append("page changed {:.1%} > {:.0%} limit".format(
            fid["change_rate"], MAX_PAGE_CHANGE))
    if fid["gain_rate"] > MAX_TOKEN_GAIN:
        reasons.append("added {:.1%} new tokens > {:.0%} limit - text not on the page".format(
            fid["gain_rate"], MAX_TOKEN_GAIN))
    if fid["loss_rate"] > MAX_TOKEN_LOSS:
        reasons.append("dropped {:.1%} of tokens > {:.0%} limit - narrative not preserved".format(
            fid["loss_rate"], MAX_TOKEN_LOSS))
    rejected = [e for e in ed if not e["keep"]]
    # Any failed per-edit guard is fatal here. The app can afford a softer line
    # because it silently reverts a rejected edit before the page is stored;
    # this script only reports, so the only way a rejected edit gets reverted is
    # if the verdict stops the caller and makes them do it.
    if rejected:
        reasons.append("{} edit(s) fail the per-edit guards - revert them to the "
                       "verbatim reading and re-run".format(len(rejected)))
    return {"verdict": "REJECT" if reasons else "ACCEPT",
            "reasons": reasons, "fidelity": fid, "edits": ed,
            "n_edits": len(ed), "n_rejected": len(rejected)}


def _load_taxonomy():
    if not TAXONOMY.exists():
        sys.exit("taxonomy.json not found beside this script")
    return json.loads(TAXONOMY.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="guard a corrected page against the verbatim one")
    c.add_argument("verbatim")
    c.add_argument("corrected")
    c.add_argument("--json", action="store_true")

    sub.add_parser("types", help="list the 35 valid incident types")

    lk = sub.add_parser("lookup", help="group / PNP tier / KP status for a type")
    lk.add_argument("name")

    args = ap.parse_args()

    if args.cmd == "types":
        for cat in _load_taxonomy()["categories"]:
            print("{:<38} {}".format(cat["name"], cat["group"]))
        return

    if args.cmd == "lookup":
        tax = _load_taxonomy()
        hit = next((c for c in tax["categories"]
                    if c["name"].lower() == args.name.lower()), None)
        if not hit:
            print("NOT A VALID TYPE: {!r}".format(args.name))
            print("Run `python guard.py types` - never invent a type outside that list.")
            sys.exit(1)
        for k in ("name", "group", "pnp_tier", "pnp_basis", "kp_status", "kp_basis"):
            if hit.get(k):
                print("{:<11} {}".format(k + ":", hit[k]))
        return

    src = Path(args.verbatim).read_text(encoding="utf-8")
    cand = Path(args.corrected).read_text(encoding="utf-8")
    res = check(src, cand)

    if args.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        f = res["fidelity"]
        print("VERDICT: {}".format(res["verdict"]))
        for r in res["reasons"]:
            print("  ! " + r)
        print("\n{} source tokens | {} repaired | {} added | {} lost".format(
            f["n_tokens"], f["distortion"], f["gain"], f["loss"]))
        print("change {:.1%} (limit {:.0%}) | gain {:.1%} (limit {:.0%}) | "
              "loss {:.1%} (limit {:.0%})".format(
                  f["change_rate"], MAX_PAGE_CHANGE, f["gain_rate"], MAX_TOKEN_GAIN,
                  f["loss_rate"], MAX_TOKEN_LOSS))
        if res["n_rejected"]:
            print("\n{} of {} edits fail the per-edit guards and must be reverted "
                  "to the verbatim reading:".format(res["n_rejected"], res["n_edits"]))
            for e in res["edits"]:
                if not e["keep"]:
                    print("  {!r} -> {!r}\n      {}".format(
                        e["before"], e["after"], e["reason"]))
    sys.exit(1 if res["verdict"] == "REJECT" else 0)


if __name__ == "__main__":
    main()
