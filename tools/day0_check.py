"""Day 0 pre-flight for the OCR accuracy work. Run this before planning anything.

Answers four questions, in the order that decides your schedule:

  1. Is a Gemini backend configured at all, and which one bills?
  2. Does the model ID in bantay/ocr/gemini.py actually exist on that backend?
  3. Was the 21/30 error run in models/ocr_correction_report.json a TIMEOUT?
     -> if yes, the fix is one line and Day 1 starts on time
     -> if no, credentials or the model ID are wrong and the schedule shifts a day
  4. Does the model accept IMAGE input here? Day 2's restore_from_image() is
     built on this, so failing now is much cheaper than failing on Day 2.

Why this exists rather than `python -m bantay.ocr.gemini`: that prints config and
lists models, which answers 1 and 2. It does not time anything, so it cannot
confirm or kill the timeout hypothesis, and it never sends an image. Those are
the two facts that actually change what you do next.

Cost: one Vision call and up to ~8 Gemini calls. Cents. It reuses the Vision
output already in data/ocr_calibration.csv for the Gemini probes rather than
re-OCRing pages, so Vision quota is untouched beyond the single liveness check.

Usage (from the project root, in the venv):
    python tools/day0_check.py
    python tools/day0_check.py --skip-image      # skip probe 5
    python tools/day0_check.py --timeouts 20 90  # override the probe deadlines
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD, INFO = "  [ok]  ", " [warn] ", " [FAIL] ", "        "


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def mask(value):
    """Show that a secret is set without printing it."""
    if not value:
        return "(not set)"
    return f"set, {len(value)} chars, ends {value[-4:]!r}"


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
def check_config():
    rule("1. CONFIGURATION")
    env = os.environ
    print(f"{INFO}BANTAY_GEMINI                 = {env.get('BANTAY_GEMINI', '(unset -> on)')}")
    print(f"{INFO}GOOGLE_GENAI_USE_VERTEXAI     = {env.get('GOOGLE_GENAI_USE_VERTEXAI', '(unset)')}")
    print(f"{INFO}GOOGLE_CLOUD_PROJECT          = {env.get('GOOGLE_CLOUD_PROJECT', '(unset)')}")
    print(f"{INFO}GOOGLE_CLOUD_LOCATION         = {env.get('GOOGLE_CLOUD_LOCATION', '(unset -> global)')}")
    print(f"{INFO}GEMINI_API_KEY                = {mask(env.get('GEMINI_API_KEY'))}")
    print(f"{INFO}BANTAY_GEMINI_MODEL           = {env.get('BANTAY_GEMINI_MODEL', '(unset -> file default)')}")
    print(f"{INFO}BANTAY_OCR_BACKEND            = {env.get('BANTAY_OCR_BACKEND', '(unset -> gvision in scan.py)')}")

    cred = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred:
        exists = Path(cred).exists()
        print(f"{OK if exists else BAD}GOOGLE_APPLICATION_CREDENTIALS = {cred}"
              f"{'' if exists else '   <-- FILE DOES NOT EXIST'}")
    else:
        print(f"{INFO}GOOGLE_APPLICATION_CREDENTIALS = (unset — relying on gcloud ADC)")

    try:
        from bantay.ocr import gemini
    except Exception as exc:
        print(f"{BAD}cannot import bantay.ocr.gemini: {exc}")
        return None, None

    mode = gemini._mode()
    model = env.get("BANTAY_GEMINI_MODEL", gemini._DEFAULT_MODEL)
    if not mode:
        print(f"\n{BAD}backend resolves to OFF. verify() and restore() are no-ops.")
        print(f"{INFO}Set either GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT,")
        print(f"{INFO}or GEMINI_API_KEY. Nothing else in this check can run.")
    else:
        print(f"\n{OK}backend = {mode}   model = {model}")
    return mode, model


# ---------------------------------------------------------------------------
# 2. Does the model exist on the backend that bills?
# ---------------------------------------------------------------------------
def check_model(mode, model):
    rule("2. MODEL AVAILABILITY")
    if not mode:
        print(f"{INFO}skipped — no backend configured")
        return False
    from bantay.ocr import gemini
    try:
        names = sorted(m.name.split("/")[-1] for m in gemini._client().models.list())
    except Exception as exc:
        print(f"{BAD}could not list models: {type(exc).__name__}: {exc}")
        print(f"{INFO}This is a credential or project problem, not a model problem.")
        print(f"{INFO}-> Day 1 becomes 'fix auth'. Everything downstream shifts one day.")
        return False

    print(f"{OK}backend served {len(names)} models")
    if model in names:
        print(f"{OK}'{model}' is available")
    else:
        print(f"{BAD}'{model}' is NOT in the list — every scan will fail per-page")
        flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n
                 and "live" not in n and "audio" not in n]
        if flash:
            print(f"{INFO}flash-tier models you could use instead:")
            for n in flash[:12]:
                print(f"{INFO}    {n}")
            print(f"{INFO}-> set BANTAY_GEMINI_MODEL, or change _DEFAULT_MODEL in gemini.py")
        return False
    return True


# ---------------------------------------------------------------------------
# 3. Vision liveness — one call, so a Vision problem is not misread as a Gemini one
# ---------------------------------------------------------------------------
def check_vision(images_dir):
    rule("3. GOOGLE VISION LIVENESS (1 call)")
    folder = Path(images_dir)
    if not folder.is_dir():
        print(f"{WARN}{folder} not found — skipping")
        return
    pages = sorted(p for p in folder.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
    if not pages:
        print(f"{WARN}no images in {folder} — skipping")
        return
    page = pages[0]
    try:
        from bantay.ocr.engine import QuotaExceeded, extract
    except Exception as exc:
        print(f"{BAD}cannot import engine: {exc}")
        return
    t0 = time.perf_counter()
    try:
        out = extract(page, backend="gvision")
    except QuotaExceeded as exc:
        print(f"{BAD}quota exhausted: {exc}")
        return
    except Exception as exc:
        print(f"{BAD}{type(exc).__name__}: {exc}")
        print(f"{INFO}Vision is the intake path — fix this before anything else.")
        return
    dt = time.perf_counter() - t0
    print(f"{OK}{page.name}: {len(out['text'])} chars, {len(out['lines'])} lines, "
          f"mean_conf {out['mean_conf']}, {dt:.1f}s")
    if out["text"]:
        print(f"{INFO}first line: {out['text'].splitlines()[0][:60]!r}")


# ---------------------------------------------------------------------------
# 4. THE TIMEOUT PROBE — the decisive one
# ---------------------------------------------------------------------------
def probe_timeouts(mode, model, calibration, timeouts):
    rule("4. TIMEOUT PROBE  (is the 21/30 error run a deadline problem?)")
    if not mode:
        print(f"{INFO}skipped — no backend configured")
        return None
    import pandas as pd
    from bantay.ocr import gemini

    path = Path(calibration)
    if not path.exists():
        print(f"{WARN}{path} not found — skipping")
        return None
    df = pd.read_csv(path)
    df = df.assign(n=df["ocr_text"].astype(str).str.len()).sort_values("n")
    # shortest / median / longest: latency scales with input, and the longest page
    # is the one that decides what the deadline has to be.
    picks = [df.iloc[0], df.iloc[len(df) // 2], df.iloc[-1]]

    print(f"{INFO}3 real Vision transcriptions, {len(timeouts)} deadlines, "
          f"{3 * len(timeouts)} calls total")
    print(f"{INFO}the report ran verify() at its 20s default and got 21/30 errors\n")
    print(f"{'page':<24}{'chars':>7}{'timeout':>9}{'wall':>8}   status")
    print("-" * 78)

    results = []
    for timeout in timeouts:
        for row in picks:
            t0 = time.perf_counter()
            try:
                out = gemini.verify(str(row.ocr_text), model=model, timeout=float(timeout))
                status = out["status"]
            except Exception as exc:                      # verify() should never raise
                status = f"RAISED {type(exc).__name__}: {exc}"
            dt = time.perf_counter() - t0
            ok = not status.startswith("error")
            results.append({"timeout": timeout, "chars": int(row.n), "wall": dt,
                            "ok": ok, "status": status})
            print(f"{str(row.source_file)[:23]:<24}{int(row.n):>7}{timeout:>9}"
                  f"{dt:>7.1f}s   {'OK  ' if ok else 'ERR '}{status[:44]}")

    print()
    by_timeout = {}
    for r in results:
        by_timeout.setdefault(r["timeout"], []).append(r)
    for timeout, group in sorted(by_timeout.items()):
        n_ok = sum(g["ok"] for g in group)
        print(f"{INFO}timeout={timeout:>4}s   {n_ok}/{len(group)} succeeded   "
              f"mean wall {sum(g['wall'] for g in group) / len(group):.1f}s")

    low, high = min(by_timeout), max(by_timeout)
    ok_low = sum(g["ok"] for g in by_timeout[low])
    ok_high = sum(g["ok"] for g in by_timeout[high])

    print()
    if ok_low < ok_high:
        succeeded = [r["wall"] for r in results if r["ok"]]
        worst = max(succeeded) if succeeded else float(high)
        print(f"{OK}HYPOTHESIS CONFIRMED — the failures are deadline, not credentials.")
        print(f"{INFO}{low}s got {ok_low}/{len(by_timeout[low])}; {high}s got "
              f"{ok_high}/{len(by_timeout[high])}.")
        print(f"{INFO}FIX: raise verify()'s default timeout in bantay/ocr/gemini.py")
        print(f"{INFO}     (slowest successful call here: {worst:.0f}s — leave real headroom)")
        print(f"{INFO}Day 1 starts on schedule.")
    elif ok_high == 0:
        print(f"{BAD}FAILS AT EVERY DEADLINE — not a timeout.")
        print(f"{INFO}Read the status strings above: they carry the real exception.")
        print(f"{INFO}-> Day 1 becomes 'fix the backend'. Shift the schedule one day.")
    elif ok_low == len(by_timeout[low]):
        print(f"{WARN}Everything succeeded, including at {low}s.")
        print(f"{INFO}So the 21/30 came from something transient — quota that day, or a")
        print(f"{INFO}model ID that has since changed. Re-run the arm and check `errors`")
        print(f"{INFO}before trusting any number from it. Raise the timeout anyway:")
        print(f"{INFO}these are your 3 pages, not the 30 the arm runs.")
    else:
        print(f"{WARN}Mixed. Sample of 3 is small — re-run with more pages before concluding.")
    return results


# ---------------------------------------------------------------------------
# 5. Image input — Day 2 depends entirely on this
# ---------------------------------------------------------------------------
def probe_image(mode, model, images_dir):
    rule("5. IMAGE INPUT  (Day 2's restore_from_image depends on this)")
    if not mode:
        print(f"{INFO}skipped — no backend configured")
        return False
    folder = Path(images_dir)
    pages = sorted(p for p in folder.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png"}) if folder.is_dir() else []
    if not pages:
        print(f"{WARN}no images in {folder} — skipping")
        return False
    page = pages[0]

    from google.genai import types

    from bantay.ocr import gemini

    mime = "image/png" if page.suffix.lower() == ".png" else "image/jpeg"
    prompt = ("This is a photo of a page from a handwritten Philippine barangay blotter "
              "logbook. Return JSON only: "
              '{"readable": true|false, "first_line": "<the first line of handwriting '
              'you can read, verbatim>", "language": "<what language(s) it appears to be>"}')
    t0 = time.perf_counter()
    try:
        resp = gemini._client().models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=page.read_bytes(), mime_type=mime), prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                http_options=gemini._http_options(120.0),
            ),
        )
    except Exception as exc:
        print(f"{BAD}{type(exc).__name__}: {exc}")
        print(f"{INFO}Day 2's whole plan rests on image input. If this cannot be made to")
        print(f"{INFO}work, fall back to Gemini-on-text and say so in Limitations —")
        print(f"{INFO}but note that caps you at the 85.7% ceiling (plan section 1.1).")
        return False
    dt = time.perf_counter() - t0
    print(f"{OK}image accepted — {page.name} ({page.stat().st_size / 1024:.0f} KB) in {dt:.1f}s")
    print(f"{INFO}reply: {(resp.text or '').strip()[:400]}")
    print(f"\n{INFO}Budget ~{dt:.0f}s/page for the image path. 73 pages "
          f"~= {dt * 73 / 60:.0f} min for a full run.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw/Blotter Pics")
    ap.add_argument("--calibration", default="data/ocr_calibration.csv")
    ap.add_argument("--timeouts", nargs="+", type=int, default=[20, 90])
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-image", action="store_true")
    args = ap.parse_args()

    print("BANTAY — Day 0 pre-flight")
    print(f"cwd: {Path.cwd()}")

    mode, model = check_config()
    model_ok = check_model(mode, model)
    if not args.skip_vision:
        try:
            check_vision(args.images)
        except Exception:
            print(f"{BAD}unexpected error in the Vision check:")
            traceback.print_exc()

    probe = None
    if model_ok:
        try:
            probe = probe_timeouts(mode, model, args.calibration, args.timeouts)
        except Exception:
            print(f"{BAD}unexpected error in the timeout probe:")
            traceback.print_exc()

    image_ok = False
    if model_ok and not args.skip_image:
        try:
            image_ok = probe_image(mode, model, args.images)
        except Exception:
            print(f"{BAD}unexpected error in the image probe:")
            traceback.print_exc()

    rule("VERDICT")
    if not mode:
        print(f"{BAD}NO-GO — no Gemini backend configured.")
        print(f"{INFO}Next: set GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT")
        print(f"{INFO}      (or GEMINI_API_KEY), then re-run this.")
    elif not model_ok:
        print(f"{BAD}NO-GO — backend reachable but the model ID is wrong or auth failed.")
        print(f"{INFO}Next: fix that first. Day 1 shifts by one day.")
    elif probe and any(r["ok"] for r in probe) and image_ok:
        print(f"{OK}GO — backend live, model serves, image input works.")
        print(f"{INFO}Next: Day 1 AM — human-verify 20 gold records against their photos.")
        print(f"{INFO}      Day 1 PM — write tools/eval_fields.py and get the baseline.")
    elif probe and any(r["ok"] for r in probe):
        print(f"{WARN}PARTIAL GO — text path works, image input did not.")
        print(f"{INFO}Day 1 proceeds unchanged. Re-test image input before committing to Day 2.")
    else:
        print(f"{BAD}NO-GO — no Gemini call succeeded at any deadline.")
        print(f"{INFO}Next: read the status strings in section 4; they carry the exception.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
