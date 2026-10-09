# BANTAY Session Context: 2026-10-09 (legal wording, province, confidence label, no TF-IDF fallback)

Uncommitted on `main`. Line endings were kept per file. CRLF: `__init__.py`, `normalize.py`, `export_docx.py`, `routes/scan.py`, `dashboard.html`, `tests/test_records_route.py`, `tests/test_scan_route.py`. All other edited files are LF.

## Round 1 (done)

| Task | Change | Files |
|---|---|---|
| 1 Sec. 412 wording | "Lupon Chairman's determination/call" replaced with: certification to file action is issued by the Lupon or Pangkat secretary and attested by the Lupon or Pangkat chairman (RA 7160 Sec. 412); "this tag does not replace it". No other occurrences in .py/.html/.js. | `templates/records_view.html:123`, `templates/dashboard.html:73-75`, `normalize.py:233` (comment) |
| 2 Province | `PROVINCE = "Pampanga"` (was "Bulacan"). The remaining "Bulacan" hits are in data CSVs only (complainant addresses), left as-is. | `export_docx.py:25` |
| 3 Confidence label | Jinja filter `confidence_label`: >= 0.6 "Models agree", >= 0.4 "Unverified (check model unavailable)", else "Models disagree". Stored values (0.8/0.5/0.3), the 0.6 threshold and the hidden `model_confidence` inputs are unchanged. The Readability (Vision OCR) meter stays as a %. | `__init__.py`, `dashboard.html`, `records_list.html`, `review_resolve.html`, `records_new.html` |
| 4 Re-tier "Marital Relation" | NOT done. Waiting for an explicit "do task 4" (report affected tests and dashboard KP counts first). | `normalize.py:~248` |

## Round 2 (done)

| Task | Change | Files |
|---|---|---|
| A Last % displays | The record-view bar, the review-queue `%.2f` badges and the review-resolve JS re-run text now show the label ("Model check: <label> — <review>"). The JS `confidenceLabel()` moved to `base.html` (one shared copy, removed from `records_new.html`). The "Classification Confidence Score" heading was renamed "Classification Model Check". | `records_view.html`, `review_queue.html`, `review_resolve.html`, `records_new.html`, `base.html` |
| B Colours match labels | Amber branch `>= 0.3` changed to `>= 0.4`: 0.8 is green, 0.5 is amber, 0.3 is red. | `dashboard.html`, `records_list.html`, `records_view.html`, `review_queue.html`, `review_resolve.html` |
| C No TF-IDF fallback (matches the thesis) | When SEA-LION/Ollama gives no answer, the result is `incident_type/confidence/category_group = None` with `source="none"`, and the status text is kept. No `app.classifier.classify` call remains in the request path. `/records/predict` also returns `status`. Comments now say zero-shot (no examples in the prompt). | `routes/records.py`, `routes/scan.py`, `records_new.html`, `review_resolve.html` |

Task C details:
- `scan.py`: `pred = {"source": "none"}`. The model string is "sealion zero-shot" or "none (encoder classifies)". The unused `get_category_group` import was removed.
- UI wording:
  - The banner reads "Classification off: ... no incident type will be suggested".
  - With no type, the preview reads "No suggestion - classifier <status>. Pick the type yourself."
  - All TF-IDF wording is removed.
- Guard added in `new_record()`: a record saved with no incident type is always `needs_review`. Without it, a scan draft (which posts `review_status=accepted`) would save as Accepted.
- Kept: `ml/infer.py`, `ml/train.py`, `app.classifier` and the admin "reload model" action, for research tooling.

Round 1 Task 3 details (still true):
- The table bars use `width:100%`, so colour shows the category and the bar no longer shows a percentage.
- `records_new.html`: the classification meter is a static `conf-meter` block that shows the label. The `conf_meter` macro is used only by Readability.

## Tests

- `tests/test_scan_route.py`:
  - Expects "Models agree", "Classification Model Check" and "sealion zero-shot", and that "80%" is absent.
  - The Ollama-off scan test checks for "No suggestion - classifier off (set BANTAY_OLLAMA=1)" and "none (encoder classifies)", and that "TF-IDF" is absent.
- `tests/test_records_route.py`:
  - `test_classify_narrative_falls_back_to_tfidf_when_sealion_off` was renamed `test_classify_narrative_leaves_type_blank_when_sealion_off`. It now uses a real narrative and checks for a blank type, `source="none"` and the "off" status.
  - New: `test_save_without_a_confirmed_type_goes_to_review`.
- Full suite: **259 passed** (about 9 min). The shell has `BANTAY_OLLAMA=1`, so tests that don't stub Ollama make real calls, which is why the suite is slow.

## Verified live (default admin from `seed.py`)

- **Ollama off (port 5056):**
  - New Blotter Record shows the "Classification off" banner.
  - `/records/predict` returns label None, `source=none` and status "off (set BANTAY_OLLAMA=1)".
  - Scanning `data/raw/Blotter Pics/Blotter_Pic (1).jpg` shows the same status, the "No suggestion" preview, no type selected and no TF-IDF text.
- **Ollama on (port 5057):**
  - The dashboard, `/records/`, `/records/1`, `/review/` and `/review/10` render with no errors and no percentages, and every label has the right colour.
  - `/records/predict` returns Theft, 0.8, from gemma4:e4b.
- **Round 1:** `/records/21/export.docx` contains "Province of Pampanga".

## Not verified

- No record is stored at 0.5 in the live DB, so "Unverified" (amber) has not been seen on a page.
- The JS preview text was checked through the endpoint response, not in a browser (the chrome-devtools MCP was down).

## Open items

- Task 4 (Marital Relation re-tier) is waiting for the user.
- Commit not made yet.
