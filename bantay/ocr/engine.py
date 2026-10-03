"""Pluggable OCR backends.

Handwriting engines differ wildly in install cost, so the backend is a config
value, not a hard dependency:

  easyocr    pip-only, torch-backed. Weak on this handwriting (see paddle).
  trocr      microsoft/trocr-base-handwritten via transformers. Best on
             handwriting, English-trained, ~1.3 GB VRAM. Detects lines with
             easyocr's detector (recognition unused) and runs TrOCR per line -
             needs easyocr installed too, even to use only the trocr backend.
  paddle     PaddleOCR PP-OCRv5/v6 via paddlepaddle. Own detector and
             recognizer, no torch. Reads the printed blotter form far better
             than the other local engines; still a print-first model on
             cursive. CPU-only wheel by default (paddlepaddle-gpu is a
             separate install), so `gpu` only takes effect if that wheel is
             the one installed.
  gvision    Google Cloud Vision DOCUMENT_TEXT_DETECTION. Strongest on the
             handwritten narrative, no local model. The page leaves the
             machine, unlike the local engines - a data-privacy call the
             barangay makes, not this module (RA 10173).

Every backend returns the same shape so downstream code never branches:

    {"text": str, "lines": [{"text": str, "conf": float}], "mean_conf": float,
     "engine": str}

`conf` is 0..1. Engines that do not report confidence return 0.0 and a
mean_conf of 0.0 — treat that as "unknown", not "bad" (see review gate in
routes/scan.py).
"""
import io
import os

_READERS = {}  # backend -> loaded reader, cached; model load is seconds, not ms


class QuotaExceeded(RuntimeError):
    """Cloud backend gave up after its retry budget - a run-level failure, not
    a bad page. Subclasses RuntimeError so existing `except RuntimeError`
    handlers keep working; catch it first to stop a bulk loop instead of
    skipping every remaining page and writing a silently truncated output.
    """


def available_engines():
    """Backends importable right now. Cheap — import check only."""
    found = []
    # Ordered best-first for this project's pages, because extract() falls back
    # to the first entry when no backend is named: paddle recovers 43.0% of gold
    # tokens, trocr 24.6%, easyocr 8.1%, on the 188 audited scans (see tools/audit_ocr_gold.py). gvision
    # beats them all but stays last - sending the page off-machine is the
    # barangay's call, never an automatic fallback.
    for name, mod in (("paddle", "paddleocr"), ("trocr", "transformers"),
                      ("easyocr", "easyocr"),
                      ("gvision", "google.cloud.vision")):
        try:
            __import__(mod)
            found.append(name)
        except ImportError:
            pass
    return found


def _as_bytes(image):
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "read"):
        return image.read()
    with open(image, "rb") as fh:
        return fh.read()


def _as_image(data, mode="RGB"):
    """Decode page bytes to a PIL image with EXIF orientation applied.

    Phone captures of a blotter page carry an EXIF orientation tag (10 of the
    104 collected scans do). PIL hands back the sensor's raw pixel grid and
    leaves the tag unapplied, so without exif_transpose those pages reach the
    recognizer rotated 90 degrees and score as near-total garbage - an artifact
    of the loader, not of the engine. Every local backend decodes through here
    for that reason. gvision is the exception: it gets the original bytes and
    Google applies the tag itself.
    """
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert(mode) if mode else img


def _easyocr_reader(langs, gpu):
    import easyocr

    key = ("easyocr", tuple(langs), gpu)
    if key not in _READERS:
        # 'tl' rides the latin_g2 recognizer; ['en'] alone loses Tagalog-specific
        # frequency priors but is a valid fallback if the tl model is missing.
        _READERS[key] = easyocr.Reader(list(langs), gpu=gpu)
    return _READERS[key]


def _easyocr(data, langs, gpu):
    import numpy as np

    reader = _easyocr_reader(langs, gpu)
    img = np.array(_as_image(data))
    out = reader.readtext(img, detail=1, paragraph=False)
    lines = [{"text": t, "conf": float(c)} for _box, t, c in out if t.strip()]
    return lines


def _cluster_lines(horizontal_list):
    """Group word-level detection boxes into line-level bounding boxes.

    easyocr's detect() returns boxes as [x_min, x_max, y_min, y_max] but does
    NOT merge them into lines the way readtext()'s internal recognition pass
    does - group_text_box's default thresholds are tuned for scene text, not
    a page of widely word-spaced cursive. A word crop gives a handwriting
    model less context than a line crop (TrOCR was trained on IAM lines, not
    isolated words), so this recovers that grouping: cluster by y-center
    within one box-height, then sort each row left to right.
    """
    if not horizontal_list:
        return []
    boxes = sorted(horizontal_list, key=lambda b: (b[2] + b[3]) / 2)
    rows = []
    for x_min, x_max, y_min, y_max in boxes:
        yc, h = (y_min + y_max) / 2, max(y_max - y_min, 1)
        row = next((r for r in rows if abs(yc - r["yc"]) < r["h"] * 0.6), None)
        if row is None:
            rows.append({"yc": yc, "h": h, "n": 1, "boxes": [(x_min, x_max, y_min, y_max)]})
        else:
            row["boxes"].append((x_min, x_max, y_min, y_max))
            row["yc"] = (row["yc"] * row["n"] + yc) / (row["n"] + 1)
            row["h"] = max(row["h"], h)
            row["n"] += 1
    lines = []
    for row in sorted(rows, key=lambda r: r["yc"]):
        bs = row["boxes"]
        lines.append((min(b[0] for b in bs), max(b[1] for b in bs),
                      min(b[2] for b in bs), max(b[3] for b in bs)))
    return lines


def _trocr(data, langs, gpu, batch_size=8, margin=6):
    """Detect text lines with easyocr's detector (recognition unused), then
    run TrOCR per line. TrOCR is a line-level model - one whole-page pass
    truncates past max_new_tokens on anything longer than a few words, so line
    segmentation is not optional here, it is the difference between usable
    output and a cut-off fragment.
    """
    import numpy as np
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    name = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    if name not in _READERS:
        proc = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
        model.to("cuda" if gpu and torch.cuda.is_available() else "cpu").eval()
        _READERS[name] = (proc, model)
    proc, model = _READERS[name]

    img = _as_image(data)
    img_np = np.array(img)
    horizontal, _free = _easyocr_reader(langs, gpu).detect(img_np)
    line_boxes = _cluster_lines(horizontal[0] if horizontal else [])

    w, h = img.size
    if not line_boxes:
        line_boxes = [(0, w, 0, h)]  # detector found nothing - fall back to the whole page

    crops = []
    for x0, x1, y0, y1 in line_boxes:
        box = (max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin))
        crops.append(img.crop(box))

    lines = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        px = proc(images=batch, return_tensors="pt").pixel_values.to(model.device)
        with torch.no_grad():
            ids = model.generate(px, max_new_tokens=64)
        for text in proc.batch_decode(ids, skip_special_tokens=True):
            if text.strip():
                lines.append({"text": text.strip(), "conf": 0.0})
    return lines


def _paddle(data, langs, gpu):
    """PaddleOCR, detector + recognizer in one pipeline.

    langs is mapped to a single paddle lang code (it takes one, not a list):
    'tl' if asked for, else the first code paddle recognises. 'tl' loads the
    latin recognizer, which covers the English on the form too, so the Taglish
    pages do not need two passes.

    Three sub-models are switched off on purpose: document orientation
    classification, unwarping, and per-textline orientation. The blotter scans
    are upright flatbed/phone captures of a flat form, so all three only add
    seconds per page and a chance to rotate a correctly-oriented crop.

    oneDNN is disabled: paddle 3.3 on Windows raises
    ConvertPirAttribute2RuntimeAttribute on the v6 detector with it enabled.
    """
    import numpy as np

    lang = "tl" if "tl" in langs else (list(langs) or ["en"])[0]
    key = ("paddle", lang)
    if key not in _READERS:
        from paddleocr import PaddleOCR

        _READERS[key] = PaddleOCR(
            lang=lang, enable_mkldnn=False,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    reader = _READERS[key]

    img = np.array(_as_image(data))[:, :, ::-1]  # paddle takes arrays as BGR
    result = reader.predict(img)
    lines = []
    for page in result:
        for text, score in zip(page["rec_texts"], page["rec_scores"]):
            if text.strip():
                lines.append({"text": text.strip(), "conf": float(score)})
    return lines


def _gvision(data, langs, gpu):
    """Google Cloud Vision, document/handwriting mode.

    Auth is standard ADC - point GOOGLE_APPLICATION_CREDENTIALS at a service
    account JSON. `gpu` is ignored; the inference is Google's.

    Vision has no line object, only words plus per-symbol break hints, so the
    lines the rest of the pipeline expects are rebuilt from those hints. A
    paragraph end also closes a line: the last word of a paragraph often
    carries no break hint at all, and without that flush the final line of
    every block would be swallowed.
    """
    from google.cloud import vision

    if "gvision" not in _READERS:
        _READERS["gvision"] = vision.ImageAnnotatorClient()
    client = _READERS["gvision"]

    # Quota (429) is the failure that actually bites on a bulk run, and
    # api-core's default retry does not cover it. Retry with exponential
    # backoff, then surface it as QuotaExceeded so a caller can stop rather
    # than skip. 5 minutes of backoff is generous for a per-minute quota and
    # still bounded.
    from google.api_core import exceptions as gexc
    from google.api_core import retry as gretry

    retry = gretry.Retry(
        predicate=gretry.if_exception_type(
            gexc.ResourceExhausted, gexc.ServiceUnavailable,
            gexc.DeadlineExceeded, gexc.InternalServerError, gexc.TooManyRequests),
        initial=2.0, maximum=60.0, multiplier=2.0, timeout=300.0,
    )
    try:
        resp = client.document_text_detection(
            image=vision.Image(content=data),
            image_context=vision.ImageContext(language_hints=list(langs)),
            retry=retry, timeout=120.0,
        )
    except (gexc.ResourceExhausted, gexc.TooManyRequests, gexc.RetryError) as exc:
        raise QuotaExceeded(
            "Google Vision quota exhausted after 5 minutes of retries. Check the "
            "project's Vision API quota in the Cloud console, or fall back with "
            "BANTAY_OCR_BACKEND=easyocr."
        ) from exc
    except gexc.GoogleAPICallError as exc:
        raise RuntimeError(f"Google Vision: {exc.message or exc}") from exc
    if resp.error.message:
        raise RuntimeError(f"Google Vision: {resp.error.message}")

    brk = vision.TextAnnotation.DetectedBreak.BreakType
    lines, buf, confs = [], [], []

    def flush():
        text = "".join(buf).strip()
        if text:
            lines.append({"text": text,
                          "conf": round(sum(confs) / len(confs), 4) if confs else 0.0})
        buf.clear()
        confs.clear()

    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    buf.append("".join(sym.text for sym in word.symbols))
                    confs.append(float(word.confidence))
                    tail = word.symbols[-1].property.detected_break.type_
                    if tail in (brk.LINE_BREAK, brk.EOL_SURE_SPACE):
                        flush()
                    elif tail in (brk.SPACE, brk.SURE_SPACE):
                        buf.append(" ")
                flush()
    return lines


_BACKENDS = {"easyocr": _easyocr, "trocr": _trocr,
             "paddle": _paddle, "gvision": _gvision}


def extract(image, backend=None, langs=("tl", "en"), gpu=None):
    """Run OCR on one page image.

    image   path, bytes, or a file-like object (Flask FileStorage works).
    backend one of _BACKENDS; defaults to $BANTAY_OCR_BACKEND, then the first
            installed engine.
    """
    backend = backend or os.environ.get("BANTAY_OCR_BACKEND") or (available_engines() or [None])[0]
    if backend not in _BACKENDS:
        raise RuntimeError(
            f"No OCR backend available (asked for {backend!r}). "
            f"Install one of: pip install easyocr | transformers | "
            f"paddleocr | google-cloud-vision"
        )
    if gpu is None:
        gpu = os.environ.get("BANTAY_OCR_GPU", "1") == "1"
    lines = _BACKENDS[backend](_as_bytes(image), langs, gpu)
    scored = [l["conf"] for l in lines if l["conf"] > 0]
    return {
        "text": "\n".join(l["text"] for l in lines),
        "lines": lines,
        "mean_conf": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "engine": backend,
    }
