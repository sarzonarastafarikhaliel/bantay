"""Regenerate bantay/static/templates/blotter_import_template.pdf.

The blocks are rendered by narrative.render() itself, so the PDF always shows
the exact layout narrative.parse() reads back - re-run this after any edit to
narrative._TEMPLATE. Printed with a headless Edge/Chrome (already on every
Windows machine this runs on), the same way the previous template was made.

    python tools/make_pdf_template.py
"""
import html
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay import narrative  # noqa: E402

OUT = ROOT / "bantay" / "static" / "templates" / "blotter_import_template.pdf"

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
)

# Synthetic example values - never real records.
EXAMPLE = {
    "blotter_no": "2026-001", "date_reported": "2026-01-03",
    "reporting_party": "Juan Dela Cruz", "complainant_address": "Purok 3, Brgy. Anunas",
    "complainant_contact": "09170000000", "complainant_age": "45",
    "respondent": "Pedro Santos", "respondent_address": "Purok 5, Brgy. Anunas",
    "respondent_age": "38",
    "complaint": "Pagkawala ng manok",
    "date": "2026-01-02", "time": "21:00", "location_purok": "Purok 3",
    "incident_summary": "Nagsadya si Juan upang ireklamo ang pagkawala ng kanyang manok.",
    "statement_writer": "Juan Dela Cruz", "statement_date": "2026-01-03", "statement_time": "14:30",
    "desk_officer": "Kag. Maria Reyes",
    "status": "For Hearing", "action_taken": "Both parties summoned for mediation.",
}

INSTRUCTIONS = """\
<h1>BANTAY - Blotter Record Import Template (PDF)</h1>
<p><b>How to use:</b> copy one BLOTTER FORM block per blotter record into Word or Google
Docs, replace the values after each colon, and write the account under E. SALAYSAY and
AKSYONG GINAWA. The labels follow the barangay's paper Blotter Form - keep them exactly as
shown, in capitals. Leave a field as "(not stated on page)" if the page does not state it.
Export as PDF, then upload on Import Blotter Records.</p>
<p>Every block needs an E. SALAYSAY; blocks without one are skipped. The first block is a
filled-in example: replace or delete it before importing. Do not type anything between
blocks. KATAYUAN (status) and AKSYONG GINAWA (action taken) are not on the paper form and
may be left as they are.</p>
<p>Incident type, PNP classification, and Katarungang Pambarangay referability are filled in
automatically, the same as on New Blotter Record. Batch Number and Encoded By can be entered
on the import page.</p>
"""


def build_html():
    blocks = [narrative.render(EXAMPLE), narrative.render({}), narrative.render({})]
    pres = "\n".join(f'<pre>{html.escape(b)}</pre>' for b in blocks)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Blotter Import Template</title>
<style>
  @page {{ size: A4; margin: 16mm 14mm; }}
  body {{ font-family: Arial, sans-serif; font-size: 10pt; color: #000; }}
  h1 {{ font-size: 13pt; margin: 0 0 8pt; }}
  p {{ margin: 0 0 6pt; line-height: 1.35; }}
  pre {{ font-family: "Courier New", monospace; font-size: 9.5pt; line-height: 1.3;
         white-space: pre; break-before: page; margin: 0; }}
</style></head><body>
{INSTRUCTIONS}
{pres}
</body></html>"""


def main():
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("No Edge/Chrome found - set one in BROWSERS.")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "pdf_template.html"
        src.write_text(build_html(), encoding="utf-8")
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", src.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
