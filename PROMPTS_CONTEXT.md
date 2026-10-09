# BANTAY LLM Prompts (context)

Verbatim from source. `{x}` = runtime placeholder. No other LLM prompts in `bantay/`.

| # | Prompt | Model | Task | Source |
|---|---|---|---|---|
| P1 | classify | Ollama + Gemini (identical text) | incident type | bantay/ml/sealion_classify.py:131, bantay/ml/gemini_classify.py:46 |
| P2 | edit-list | Gemini | OCR proofread `verify()` | bantay/ocr/gemini.py:121 |
| P3 | restore, text | Gemini + Ollama (shared) | OCR repair + fields | bantay/ocr/gemini.py:470 (imported by bantay/ocr/sealion.py) |
| P4 | restore, image | Gemini only | read photo, Vision text = hint | bantay/ocr/gemini.py:515 |

## P1 classify
`{n}`=35, `{types}`=`- <type>` lines, `{text}`=name-masked narrative.
```
You are classifying an incident narrative from a barangay (village) blotter logbook in the Philippines. The text is Tagalog/English code-switched and often ALL CAPS. Names have already been replaced with tokens like [PERSON_7] - ignore those, they carry no information.

Choose exactly ONE incident type from this list of {n}. Use the exact spelling given. If genuinely nothing fits, use "Others / Miscellaneous" - never invent a type outside this list.

TYPES:
{types}

Return JSON only, in this exact shape:
{"incident_type": "<one type from the list, exact spelling>", "confidence": "<high|medium|low>", "reason": "<a few words>"}

NARRATIVE:
{text}
```
- Ollama: `temperature=0.0`, `/api/chat`, JSON-schema `format`. Models: `gemma4:e4b` (deployed, 67.5% type acc, n=151), 4 SEA-LION checkpoints (41.7-55.6%).
- Gemini: `temperature=0.0`, JSON mime, `thinking_budget=0`.
- Off-list `incident_type` discarded. Agreement check: 2nd Ollama model (`CHECK_MODEL`) same prompt; agree=high, disagree=low; self-reported confidence discarded.

## P2 edit-list
`{lexicon}`=LEXICON or empty. `{text}`=lexicon-corrected OCR.
```
You are proofreading OCR output from a handwritten barangay blotter logbook in the Philippines. The text mixes Filipino, Cebuano and English, and is usually written in capitals.

Google Cloud Vision read the page. Some words came out garbled. List the garbled words and what they should be.

Return JSON only, in this exact shape:
{"edits": [{"before": "<exact token as it appears>", "after": "<corrected token>", "reason": "<a few words>"}]}

Rules:
- Only character-level misreads: S/5, O/0, I/1/l, rn/m, cursive slips.
- One token in, one token out. Never merge two words, never split one.
- Do not rephrase, translate, summarise, or add any word.
- Never change numbers, dates, times, amounts or case numbers.
- Names of people or places you do not recognise: leave them alone.
- Page reads fine? Return {"edits": []}.
{lexicon}
PAGE TEXT:
{text}
```

## LEXICON (in P2, P3, P4; top 300 corpus words)
```
Words from this barangay's own records - prefer these spellings when a garbled
token is close to one of them:
{words}
```

## P3 restore, text arm
```
You are restoring a page of a handwritten barangay blotter logbook from the Philippines that Google Cloud Vision has just transcribed. The handwriting mixes Filipino, Cebuano and English and is usually written in capitals, so the transcription contains character-level misreads, split words and merged words.

Do two things, and return both in one JSON object.

1. corrected_text: the same page with the transcription errors repaired.
   You are an OCR corrector, not an editor. The narrative belongs to the person
   who wrote it. Your output must state exactly what the page states, in the
   same language mix, the same order, the same sentences.
   - Repair only what the writer actually wrote: character misreads (S/5, O/0,
     I/1/l, rn/m, cursive slips), words wrongly split, words wrongly merged.
   - Do NOT rephrase, translate, summarise, reorder, or add any word.
   - Do NOT change numbers, dates, times, amounts or case numbers - ever.
   - A word you cannot confidently repair stays exactly as Vision read it.
   - Names of people or places you do not recognise: leave them alone.
   - Keep the Tagalog-English mix exactly as written. The page code-switches,
     often inside a single sentence. Never translate a word in either
     direction, never swap a Filipino word for its English equivalent or an
     English word for its Filipino one, never "clean up" Taglish into pure
     Tagalog or pure English.
   - The writer's own spelling, grammar and style are not OCR errors. Keep
     abbreviations (BRGY., KAG., NAG-, PO.), enclitics and particles (po, ho,
     na, pa, ba, daw, raw, ng, nang, mga, si, sa), hyphenated code-switches
     (NAG-REPORT, NAKI-USAP), slang and the ALL-CAPS casing exactly as they
     stand.
   - Same sentences, same order, same word count - the only exception is a
     repair that splits one merged word or merges one wrongly split word.
   - Never drop anything for being redundant, ungrammatical, repetitive or
     unclear. An unclear sentence is copied through unchanged.
   - If you are tempted to improve a sentence, that sentence is already
     correct: return it exactly as it came in.

<FIELDS, <TEXT> = "the repaired page">
{lexicon}
VISION TRANSCRIPTION:
{text}
```
- Gemini: `temperature=0.0`, JSON mime, `thinking_budget=0`, timeout 180s.
- Ollama: `Gemma-SEA-LION-v3-9B-IT:q4_k_m`, `temperature=0.2`, `repeat_penalty=1.3`, `repeat_last_n=64`, `num_predict=2048`, `think=false`, JSON-schema, timeout 180s.

## P4 restore, image arm
Photo sent as image part + this text.
```
You are reading a photographed page of a handwritten barangay blotter logbook from the Philippines. The handwriting mixes Filipino, Cebuano and English and is usually written in capitals.

You have TWO inputs: the photograph itself, and a transcription of it produced by Google Cloud Vision. The PHOTOGRAPH is the evidence. The transcription is a hint that is often wrong - it misreads characters, splits and merges words, scrambles reading order, and drops words entirely. Where they disagree, trust your own reading of the image.

Do two things, and return both in one JSON object.

1. corrected_text: what the page actually says, read from the photograph, in the page's own reading order.
   - Transcribe only what the writer wrote. Do NOT rephrase, translate, summarise, or add any word.
   - Do NOT change numbers, dates, times, amounts or case numbers - ever. Read them off the image digit by digit.
   - A word you genuinely cannot read stays as Vision read it. If neither of you can read it, write [illegible].
   - Names of people or places: transcribe what is written, do not correct spellings you do not recognise.

<FIELDS, <TEXT> = "the page as you read it">
{lexicon}
GOOGLE VISION'S TRANSCRIPTION (a hint, not the truth):
{text}
```

## FIELDS (shared by P3, P4)
```
2. fields: what the page states, for the BLOTTER FORM record template. Every
   value is a string. A field the page does not state is "" - never guess, never
   infer, never carry a value over from another field. "" is a correct answer,
   and a confident wrong value is the worst possible one. Names, addresses and
   numbers are copied as written.
   - blotter_no          NO. - the blotter / entry number
   - date_reported       Petsa ng Pagblotter - the date the blotter was filed
   - reporting_party     A. Pangalan ng Nag-blo-blotter/Nagrereklamo - who reported it
   - complainant_address A. Tirahan - the complainant's address
   - complainant_contact A. Contact No. - the complainant's phone number
   - complainant_age     A. Edad - the complainant's age
   - respondent          B. Pangalan ng Inirereklamo - who it was reported against
   - respondent_address  B. Tirahan
   - respondent_contact  B. Contact No.
   - respondent_age      B. Edad
   - complaint           C. Reklamo - the complaint in the page's own short
                         words (e.g. PANANAKIT, PAGNANAKAW), not your label for it
   - date                D. Petsa - the date the incident happened, as written
   - time                D. Oras - the time the incident happened, as written
   - location_purok      D. Lugar - where it happened (purok, sitio or street)
   - incident_summary    E. Salaysay - what happened, copied word for word out of
                         the corrected text in the page's OWN Tagalog-English
                         words - never translated, paraphrased or shortened
   - statement_writer    Pangalan ng Sumulat ng Salaysay - who wrote the statement
   - statement_date      the date beside the statement writer's signature
   - statement_time      the time (Oras) beside the statement writer's signature
   - witness             (Mga) Testigo: Pangalan - the witness's name
   - witness_age         Testigo: Edad
   - witness_address     Testigo: Tirahan
   - witness_contact     Testigo: Contact No.
   - desk_officer        Pangalan ng Barangay Desk Officer / Imbestigador
   - status              how it stands (e.g. settled, endorsed, pending). Not a
                         printed field - "" unless the page says. Never default it.
   - action_taken        what the barangay did about it. "" if the page does not say.

Return JSON only, in this exact shape:
{"corrected_text": "<the repaired page | the page as you read it>", "fields": {"blotter_no": "", "date_reported": "", "reporting_party": "", "complainant_address": "", "complainant_contact": "", "complainant_age": "", "respondent": "", "respondent_address": "", "respondent_contact": "", "respondent_age": "", "complaint": "", "date": "", "time": "", "location_purok": "", "incident_summary": "", "statement_writer": "", "statement_date": "", "statement_time": "", "witness": "", "witness_age": "", "witness_address": "", "witness_contact": "", "desk_officer": "", "status": "", "action_taken": ""}}
```

## Guards on model output (restore)
Reply never used directly. Token-diff vs Vision text; each change gated, survivors spliced into original.
- numeric edits reverted; replace similarity >= 0.5; deletions reverted
- text arm: char change <= 35%, token gain <= 10%
- image arm: char change <= 70%, token gain <= 35% (recovering Vision-dropped words is its purpose)
- free-prose fields blanked if words/numbers absent from gated page (`_ground_fields`, `MAX_FIELD_GAIN`=0.20)
- P2 `verify()`: `before` must be on page, not numeric, single-token `after`, similarity >= 0.5, page change <= 15%, max 40 edits

## Notes
- Names masked (`[PERSON_n]`) before P1. Ollama local, no data leaves machine. Gemini sends page text (+ photo for P4) to Google.
- `BANTAY_RESTORE_PROMPT_FILE` overrides P3/P4 (prompt-variant scoring, bantay/ocr/gemini.py:533).
- Stale docstring counts: gemini_classify.py says 73 records, sealion_classify.py says 188; dev split for accuracy = n=151.
