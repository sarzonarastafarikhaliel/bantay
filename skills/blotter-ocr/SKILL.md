---
name: blotter-ocr
description: Transcribe scanned or photographed barangay blotter pages and extract the BANTAY encoding fields (date, time, purok, parties, narrative, action taken, status, incident type), preserving the whole narrative verbatim with only guarded minor corrections. Use whenever the user uploads blotter page images or photos of logbook records, asks to digitise or OCR blotter pages, wants scanned records turned into a CSV for encoding, or mentions blotter intake, page transcription, or adding records to the BANTAY corpus.
---

# Blotter page transcription

Turn photographed or scanned barangay blotter pages into reviewable encoding
rows, **without rewriting what the page says**.

You are the OCR engine here. There is no Vision API and no local pipeline in this
environment — you read the page image directly. That makes the preservation rule
below the whole job, because nothing else is enforcing it for you.

> Working inside the BANTAY project checkout instead, with the venv and a Vision
> or Gemini credential available? Use the `blotter-scan-pipeline` skill — it runs
> the measured Google Vision pipeline over a whole folder, which is more accurate
> than reading the image here and is the path the thesis figures were produced
> with.

## The rule

**The narrative is transcribed, not summarised, paraphrased, translated, tidied
or shortened.** A blotter is a legal record. An uncorrected error in it is a far
smaller failure than an invented one.

Never do any of these, even when the result would read better:

- Do not summarise or condense. Every sentence on the page appears in the output.
- Do not translate. Taglish stays Taglish. ALL CAPS stays ALL CAPS.
- Do not fix grammar, or rewrite a clumsy sentence into a clear one.
- Do not add connective words, headings, or punctuation that is not on the page.
- Do not delete a word you cannot read — transcribe your best reading of it, or
  mark it `[?]`. Erasing evidence is not transcription.
- **Never change a number.** Dates, times, ages, amounts, case numbers, blotter
  numbers. `2023` is never `2028`, `12:50` is never `12:15`. This is the single
  most damaging error possible here, because it is the one that surfaces in a
  hearing.

What you *may* correct: an obvious character-level misread of a word that is
plainly a real word — `MANQK` → `MANOK`, `IREKLAM0` → `IREKLAMO`,
`PANAB0NG` → `PANABONG`. A correction must look like fixing a misread letter,
never like choosing a better word.

## Procedure

Work one page at a time. Do not batch pages into a single pass.

**1. Transcribe verbatim.** Read the image and write down exactly what is there,
misreads and all, line breaks preserved. Save it as `verbatim_<page>.txt`. Correct
nothing in this pass — this file is the evidence everything else is measured
against.

**2. Correct, minimally.** Copy the verbatim text and fix only clear
character-level misreads, per the rule above. Save as `corrected_<page>.txt`.

**3. Verify — do not skip this.** Run the guard:

```bash
python scripts/guard.py check verbatim_<page>.txt corrected_<page>.txt
```

It measures what actually changed and prints ACCEPT or REJECT. **If it says
REJECT, you did too much.** Revert the edits it names back to the verbatim
reading and run it again until it accepts. Do not argue with it, do not edit the
thresholds, and do not proceed on a REJECT.

You cannot verify this by reading your own output — that is exactly the judgement
this step exists to replace. Run it every time, on every page.

**4. Extract the fields** from the accepted text (see below).

**5. Write the row** into the output CSV.

## Fields to extract

Take values from the page. Leave a field **blank** if the page does not say — a
blank slot an encoder fills is a much smaller problem than a confident wrong
value they never think to check.

| Field | Notes |
|---|---|
| `Source File/Page` | the image filename |
| `Date` | ISO `YYYY-MM-DD` |
| `Date (as written)` | exactly as on the page, e.g. `Nov. 19, 2023` |
| `Time` | 24-hour `HH:MM` |
| `Location/Purok` | e.g. `Purok I` |
| `Reporting Party` | complainant / nagrereklamo |
| `Respondent` | the other party |
| `Narrative/Summary` | **the full accepted text** — never a summary |
| `Action Taken` | |
| `Status` | e.g. Filed, Settled, Referred |
| `Incident Type` | one of the 35 — see below |
| `Incident Type (Secondary)` | only if the page genuinely describes two |
| `Readability` | `readable` / `partial` / `unreadable` |

### Incident Type must come from the taxonomy

```bash
python scripts/guard.py types            # the 35 valid types
python scripts/guard.py lookup "Theft"   # group, PNP tier, KP status, legal basis
```

**Never invent a type outside that list.** If nothing fits, use
`Others / Miscellaneous`. An off-taxonomy label breaks every downstream lookup.

Never hand-fill `Category Group`, `PNP Classification` or `KP Referability` —
`lookup` derives them from the type. A typed copy drifts from the table.

Your suggested type is a **suggestion**. On this corpus the project's own
classifier reaches 69% exact-type accuracy, and the two boundaries that account
for most errors are Physical Injury vs. Slight Physical Injury (which turns on a
medical certificate the narrative often does not state) and Property Claim /
Damage against Malicious Mischief and Boundary Dispute. When a record sits on one
of those boundaries, say so in `Remarks` rather than picking silently.

## Privacy

Blotter pages carry real names, and the Data Privacy Act of 2012 governs them.

- `Narrative/Summary` keeps real names — it is the operational record.
- `Narrative (masked)` is the same text with the reporting party as `[PERSON_1]`
  and the respondent as `[PERSON_2]`. Mask **every** occurrence, including
  surname-only and first-name-only later references.
- If the page names people you could not assign to either party slot, mask them
  as `[PERSON_3]`, `[PERSON_4]`… and set `_pii_spotcheck_needed` to `YES`.
- Never paste a full narrative with real names into your visible reply. Refer to
  the row and the file instead.

## Output

Append one row per page to `blotter_rows.csv` with these columns:

```
Record ID, Batch Number, Source File/Page, Date, Date (as written), Time,
Location/Purok, Reporting Party, Respondent, Narrative/Summary,
Narrative (masked), Action Taken, Status, Incident Type,
Incident Type (Secondary), Readability, Include in ML, Fold, Encoded By,
Date Encoded, Remarks, Label Confidence, Label Source, Second Encoder Label,
_qa_verdict, _qa_change_rate, _qa_loss_rate, _qa_gain_rate, _qa_n_edits,
_qa_blank_slots, _qa_needs_review, _pii_spotcheck_needed
```

Fill the `_qa_*` columns from the guard's `--json` output. Set
`_qa_needs_review = YES` whenever anything downstream would be guessing: a REJECT
you had to work back from, a blank required slot, an unreadable passage, or a type
you were not confident in. Leave `Fold`, `Encoded By` and `Second Encoder Label`
blank — a person fills those.

Tell the user plainly at the end: how many pages you did, how many are flagged,
and which fields came back blank. Every row still needs a human pass before it
enters the corpus.

## If the page is unreadable

Say so. Set `Readability` to `unreadable`, `Include in ML` to `FALSE`, and leave
the fields blank. A row that admits it failed is useful; a row invented from an
illegible page is worse than no row, because nothing downstream can tell.
