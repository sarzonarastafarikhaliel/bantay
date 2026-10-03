"""OCR intake for scanned blotter pages.

Flow:  image -> engine.extract() [Google Vision] -> gemini.restore()
       -> narrative template -> ml.sealion_classify.

gemini.restore() does the repair AND fills the record template in one call, and
is what routes/scan.py uses. It is a no-op unless a backend (Vertex AI or an API
key) is configured; with none, routes/scan.py falls back to correct.OCRCorrector
so the app still works offline.

Two other entry points are still live and are NOT part of the scan flow:
  correct.OCRCorrector  the lexicon corrector. The offline fallback, and the
                        baseline arm tools/eval_fields.py measures against.
  gemini.verify()       the guarded edit-list pass. Kept because
                        tools/run_gemini_arms.py measures it as a separate arm
                        against the corrector; see gemini.py for why it returns
                        edits rather than text.

Nothing here imports torch or the Google SDKs at module level; the Flask app must
start on a machine with no GPU, no ML extras and no network (requirements-ml.txt).
"""
from .engine import extract, available_engines
from .correct import OCRCorrector

__all__ = ["extract", "available_engines", "OCRCorrector", "gemini"]
