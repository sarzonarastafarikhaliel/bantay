"""Optional LLM verification pass over the corrected OCR text.

Why this exists: the lexicon corrector in correct.py can only propose tokens the
barangay corpus already contains. A word the corpus never saw - a new purok, a
surname, a Cebuano verb form outside the 2023-2025 logbooks - stays garbled no
matter how good the masked LM is. Gemini has seen Filipino and Cebuano text this
corpus never will.

Why it is shaped like this: Gemini is a generator, and correct.py's whole safety
claim is that nothing in the pipeline can invent a sentence the scanner did not
see. Handing a generator the page and taking back prose would throw that away, so
it is never asked for text. It is asked for an EDIT LIST - {before, after} token
pairs - and this module applies the edits itself, dropping every pair that fails
a guard below. The model's output can only ever become a token-for-token swap of
a word that is actually on the page. Worst case it proposes nothing usable and
the lexicon corrector's output passes through untouched.

Guards, in the order they fire (see _keep and verify):
  - `before` must exist on the page as a whole token
  - a mostly-numeric token is never touched: dates, times, amounts and case
    numbers are legal facts, and "fixing" 2023 to 2028 is the one failure mode
    that would matter in a hearing
  - the swap must look like a misread, not a rewrite (similarity >= 0.5)
  - `after` must be a single token - no merging, no splitting, no phrases
  - and if the accepted edits together change more than 15% of the page's
    characters, ALL of them are dropped: that is a model rewriting the narrative,
    not proofreading it, and there is no safe way to keep half of a rewrite

restore() and restore_from_image() are the other shape, and they DO take back
whole text - the page is too garbled for an edit list and the record template
has to be filled from something. They are safe for a different reason: the reply
is never used directly. It is diffed against Vision's transcription, each changed
token is gated (numeric revert, no deletions, and a similarity floor so a swap
has to look like a misread and not a paraphrase), and only survivors are spliced
back into the ORIGINAL string. Then two page-level ceilings:

  MAX_RESTORE_CHANGE   share of CHARACTERS changed. A model rewriting rather
                       than repairing trips this.
  MAX_TOKEN_GAIN       share of TOKENS the model ADDED. Its own ceiling because
                       characters cannot see this failure: three invented
                       sentences in a 500-character page is 4% of it, and in a
                       blotter it is invented evidence. The image arm gets a
                       much larger allowance - recovering words Vision never
                       emitted is the entire reason it reads the pixels.

And the guard that matters most, because it protects what actually becomes the
record: _ground_fields. The template slots come back from the model in the
model's own words, and until they were gated a page whose TEXT was rejected as
a rewrite still handed that rewrite to the encoder through incident_summary. A
free-prose slot whose words are not on the gated page is now blanked, and
routes/scan.py falls back to the gated text or the regex reader - both grounded
in the page by construction.

fidelity() reports the three error types (loss / gain / distortion) on every
result, after Google Research's minimally-lossy simplification work. It is the
narrative-preservation number the field metrics cannot see, and it is what
tools/run_gemini_arms.py logs per page when scoring a prompt variant.

Two ways to reach the model, both through the same SDK:

  vertex    Vertex AI on your own GCP project. Bills the project, so Google
            Cloud credits apply, and it authenticates with the same ADC service
            account the gvision backend already uses - one credential for both
            Google hops instead of a second loose secret. Set
            GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT.
  api-key   AI Studio key in GEMINI_API_KEY. Simplest to start with, but it
            bills separately from GCP - credits do NOT cover it.

Off by default. Neither configured, or BANTAY_GEMINI=0, and verify() is a no-op
that returns the text it was given. The Flask app must run with no key, no SDK
and no network - same rule as the ML extras.

Model IDs move faster than this file does. BANTAY_GEMINI_MODEL sets it, and
`python -m bantay.ocr.gemini` prints the current config plus every model the
configured backend will actually serve you - check the ID there before putting
it in a deployment, because a wrong one fails per-page at scan time.
"""
import difflib
import json
import os
import re

from .. import narrative
from .correct import _WORD_RE, _match_case

_CLIENT = None
# The deployment target. Model IDs move faster than this file, and a default that
# no longer exists fails per-page at scan time rather than at startup - so verify
# it against the backend you actually bill with `python -m bantay.ocr.gemini`
# before trusting it, and override with BANTAY_GEMINI_MODEL for anything else.
_DEFAULT_MODEL = "gemini-3.7-flash"

# Above this share of the page's CHARACTERS changed, the model stopped
# proofreading. Dropping everything is the only safe response - a partial accept
# keeps whichever half of a rewrite happened to sort first.
#
# Measured as characters actually changed, not the length of every edited token:
# repairing NAG5ADYA -> NAGSADYA changes one character, and counting all eight
# made a badly garbled page - the case this layer exists for, see
# docs/REIMPLEMENTATION.md 6.1 - trip a guard aimed at paraphrasing. Found by
# `python -m bantay.ocr.gemini --smoke`, which scored five one-character repairs
# as a 32% rewrite.
# Extended thinking is OFF for every call this module makes, and it is the single
# biggest latency lever here - not a micro-optimisation. Measured on
# Blotter_Pic (10).jpg, an 838-char page, same prompt, same model:
#
#     thinking default   240s, and it still 504'd without finishing
#     thinking_budget=0    5.0s, 753 output tokens
#
# Proofreading OCR at temperature 0 is a mechanical, local task: decide whether
# NAG5ADYA is NAGSADYA. There is nothing here for a reasoning budget to buy, and
# what it costs is a scan the encoder watches time out. Raise it only if you have
# a page where the model demonstrably needs to reason, and measure before and
# after - BANTAY_GEMINI_THINKING is the knob.
THINKING_BUDGET = int(os.environ.get("BANTAY_GEMINI_THINKING", "0"))

MAX_PAGE_CHANGE = 0.15
MAX_EDITS = 40
MIN_SIMILARITY = 0.5

_PROMPT = """You are proofreading OCR output from a handwritten barangay blotter \
logbook in the Philippines. The text mixes Filipino, Cebuano and English, and is \
usually written in capitals.

Google Cloud Vision read the page. Some words came out garbled. List the garbled \
words and what they should be.

Return JSON only, in this exact shape:
{{"edits": [{{"before": "<exact token as it appears>", "after": "<corrected token>", "reason": "<a few words>"}}]}}

Rules:
- Only character-level misreads: S/5, O/0, I/1/l, rn/m, cursive slips.
- One token in, one token out. Never merge two words, never split one.
- Do not rephrase, translate, summarise, or add any word.
- Never change numbers, dates, times, amounts or case numbers.
- Names of people or places you do not recognise: leave them alone.
- Page reads fine? Return {{"edits": []}}.
{lexicon}
PAGE TEXT:
{text}
"""

_LEXICON_BLOCK = """
Words from this barangay's own records - prefer these spellings when a garbled
token is close to one of them:
{words}
"""


def _mode():
    """Which backend is configured: 'vertex', 'api-key', or '' for off.

    Vertex wins when both are set - if someone went to the trouble of pointing
    this at a GCP project, that is the billing account they meant.
    """
    if os.environ.get("BANTAY_GEMINI", "1") == "0":
        return ""
    if (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
            and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        return "vertex"
    if os.environ.get("GEMINI_API_KEY"):
        return "api-key"
    return ""


def available():
    """True if a verification pass would actually call out. Cheap - no import."""
    return bool(_mode())


def _client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        if _mode() == "vertex":
            # Explicit project/location rather than letting the SDK read them:
            # a missing location otherwise surfaces as a confusing 404 on the
            # model instead of a plain "you did not configure a region".
            _CLIENT = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            _CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _CLIENT


def _http_options(timeout):
    """Request deadline plus retry, shared by both passes.

    Two separate failures hide behind a 504 DEADLINE_EXCEEDED, and they want
    opposite fixes:

      the deadline was too short   the model was still generating when the
                                   budget ran out. Retrying an identical request
                                   against the same deadline just fails again,
                                   three times, for three times the wall clock.
                                   The fix is a bigger timeout - see
                                   RESTORE_TIMEOUT for why restore() needs a much
                                   bigger one than verify().
      the backend hiccupped        a genuinely transient 502/503/429. Here a
                                   retry is the whole fix.

    So: a deadline generous enough that the first attempt can actually finish,
    and only two attempts, because the retry is there for the blip, not to paper
    over an under-budgeted call. Worst case stays bounded - this runs inside a
    synchronous Flask request with a human watching the page.

    Retry comes from the SDK rather than a loop here: google-genai ships
    HttpRetryOptions with the backoff and jitter already written.
    """
    from google.genai import types

    return types.HttpOptions(
        timeout=int(timeout * 1000),          # milliseconds
        retry_options=types.HttpRetryOptions(
            attempts=2, initial_delay=2.0, max_delay=20.0, exp_base=2.0,
            # 504 is deliberately NOT here, and the docstring above is why: a
            # deadline retried against the same deadline fails again, so the only
            # thing the retry buys is double the wall clock before the same
            # error. Measured on Blotter_Pic (10) - a page that times out cost
            # 247s to fail instead of 124s. Keep the transient codes, which are
            # what the retry is actually for.
            http_status_codes=[429, 500, 502, 503],
        ),
    )


def _ask(text, lexicon, model, timeout):
    """One call. Returns the parsed edit list."""
    from google.genai import types

    words = ""
    if lexicon:
        top = sorted(lexicon, key=lexicon.get, reverse=True)[:300]
        words = _LEXICON_BLOCK.format(words=", ".join(top))

    resp = _client().models.generate_content(
        model=model,
        contents=_PROMPT.format(lexicon=words, text=text),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            # No tools are passed, so AFC is inert - but leaving it on makes the
            # SDK log a recommendation on every page the app scans.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=_http_options(timeout),
        ),
    )
    return _parse(resp.text)


def _parse(raw):
    """Defensive parse: a malformed reply is zero edits, never an exception."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    edits = data.get("edits") if isinstance(data, dict) else data
    if not isinstance(edits, list):
        return []
    out = []
    for e in edits[:MAX_EDITS]:
        if isinstance(e, dict) and isinstance(e.get("before"), str) and isinstance(e.get("after"), str):
            out.append((e["before"].strip(), e["after"].strip(), str(e.get("reason", ""))[:60]))
    return out


def _changed_chars(before, after):
    """Characters this swap actually alters, ignoring the parts left identical."""
    ops = difflib.SequenceMatcher(a=before, b=after).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def _mostly_numeric(token):
    """Same rule correct.py uses to leave real numbers alone."""
    return sum(c.isdigit() for c in token) / max(len(token), 1) >= 0.5


def _keep(before, after, present):
    """Per-edit guards. `present` is the set of upper-cased tokens on the page."""
    if not before or not after or before.upper() == after.upper():
        return False
    if before.upper() not in present:            # not a token on this page
        return False
    if _mostly_numeric(before) or _mostly_numeric(after):
        return False                             # dates, times, amounts, case numbers
    if _WORD_RE.fullmatch(after) is None:
        return False                             # phrase, split, or punctuation
    return difflib.SequenceMatcher(a=before.upper(), b=after.upper()).ratio() >= MIN_SIMILARITY


def _apply(text, pairs):
    """Rewrite every whole-token occurrence of each accepted pair.

    Walks the same token spans correct.py walks, so an edit from here has the
    same {start, end, before, after, score} shape the review table already
    renders - plus `source`, because the encoder must be able to see which
    corrections came from the model that can hallucinate.
    """
    edits, out, cursor = [], [], 0
    for m in _WORD_RE.finditer(text):
        hit = pairs.get(m.group().upper())
        if hit is None:
            continue
        after, reason, score = hit
        edits.append({"start": m.start(), "end": m.end(), "before": m.group(),
                      "after": _match_case(m.group(), after), "score": round(score, 4),
                      "reason": reason, "source": "gemini"})
    for e in edits:
        out.append(text[cursor:e["start"]])
        out.append(e["after"])
        cursor = e["end"]
    out.append(text[cursor:])
    return "".join(out), edits


# The old default was 20s, and it is what produced the 21/30 error run in
# models/ocr_correction_report.json. tools/day0_check.py timed three real pages at
# both deadlines: 20s failed the 1043-char page with 504 DEADLINE_EXCEEDED, 90s
# passed all three, and the slowest successful call took 28s. Not network speed -
# generation time scales with page length, and a logbook page is long.
VERIFY_TIMEOUT = 90.0


def verify(text, lexicon=None, model=None, timeout=VERIFY_TIMEOUT):
    """Second-opinion pass over already-corrected OCR text.

    Returns {"text", "edits", "status"} and never raises: a dead key, a dead
    network or a model having a bad day must degrade the scan, not break it. The
    status string is surfaced in the UI, so a no-op is never silent.
    """
    mode = _mode()
    if not mode:
        return {"text": text, "edits": [], "status": "off"}
    if not text or not text.strip():
        return {"text": text, "edits": [], "status": "skipped: empty page"}

    model = model or os.environ.get("BANTAY_GEMINI_MODEL", _DEFAULT_MODEL)
    try:
        proposed = _ask(text, lexicon, model, timeout)
    except Exception as exc:                      # noqa: BLE001 - any SDK/network failure
        return {"text": text, "edits": [], "status": f"error: {type(exc).__name__}: {exc}"[:200]}

    present = {m.group().upper() for m in _WORD_RE.finditer(text)}
    pairs = {}
    for before, after, reason in proposed:
        if _keep(before, after, present):
            score = difflib.SequenceMatcher(a=before.upper(), b=after.upper()).ratio()
            pairs[before.upper()] = (after, reason, score)
    if not pairs:
        n = len(proposed)
        return {"text": text, "edits": [],
                "status": "ok: no edits" if not n
                          else f"ok: {n} proposed, none passed the guards"}

    new_text, edits = _apply(text, pairs)

    touched = sum(_changed_chars(e["before"], e["after"]) for e in edits) / max(len(text), 1)
    if touched > MAX_PAGE_CHANGE:
        return {"text": text, "edits": [],
                "status": f"rejected: model rewrote {touched:.0%} of the page "
                          f"(limit {MAX_PAGE_CHANGE:.0%})"}
    return {"text": new_text, "edits": edits,
            "status": f"ok: {len(edits)} edits ({model} via {mode})"}


# ---------------------------------------------------------------------------
# Full-page restore: one call that both repairs the Vision transcription and
# fills the record template. See restore() for why it is one call and not two.
# ---------------------------------------------------------------------------

# The restore pass may change more of the page than verify() may, because it is
# doing a different job: verify() proofreads text the lexicon corrector already
# cleaned, while restore() takes raw Vision output off handwritten cursive,
# where a third of the page genuinely coming back wrong is a normal Tuesday and
# not a rewrite. Still bounded, still a hard reject. Tune it against your own
# pages - this is the knob, and the default is a starting guess, not a
# measurement.
MAX_RESTORE_CHANGE = float(os.environ.get("BANTAY_RESTORE_MAX_CHANGE", "0.35"))

# A replace must look like a misread, not a paraphrase. verify() has had this
# floor since day one and restore() never did, which is how SINASBOZ came back
# as "pinuntahan" on Blotter_Pic (10) - a word bearing no relation to the one on
# the page, accepted because the only ceiling was a page-wide average.
# INSERTIONS are exempt: `before` is empty there because Vision dropped the word
# entirely, and recovering those is the whole reason restore_from_image() exists.
MIN_RESTORE_SIMILARITY = float(os.environ.get("BANTAY_MIN_RESTORE_SIMILARITY", "0.5"))

# Share of the page's tokens the model may ADD. Fabrication on the text arm,
# recovery on the image arm, hence two ceilings - see MAX_IMAGE_TOKEN_GAIN.
#
# This is the axis nothing else bounded. MAX_RESTORE_CHANGE counts characters,
# and three invented words in a 500-character page is 4% of it - nowhere near
# the 35% ceiling, and still three sentences of evidence that no scanner saw.
#
# There is deliberately no matching LOSS budget: _keep_restore already reverts
# every deletion one edit at a time, before the splice, so page-level loss can
# only ever be the residue of a merge repair - which is a repair the prompt asks
# for. fidelity() still REPORTS loss, because the eval and the prompt loop need
# the number even where the guard has nothing left to do.
MAX_TOKEN_GAIN = float(os.environ.get("BANTAY_MAX_TOKEN_GAIN", "0.10"))

# Seconds before the request deadline expires. Much larger than verify()'s,
# and the reason is output length, not network speed: verify() asks for a short
# edit list, while restore() asks the model to write the WHOLE corrected page
# back out. Generation time scales with output tokens, so a full logbook page
# takes tens of seconds to emit - and a deadline sized for the edit list returns
# 504 DEADLINE_EXCEEDED on a page that was generating perfectly well.
#
# If you still see 504s, this is the first knob, not the retry count: a retry
# against a too-small deadline just fails again. Second knob is the model -
# BANTAY_GEMINI_MODEL, a flash tier emits faster than a pro tier. Raise this
# past a couple of minutes only if nobody is watching the page render.
RESTORE_TIMEOUT = float(os.environ.get("BANTAY_GEMINI_TIMEOUT", "180"))

# The field list both restore prompts share. Keys and order are
# narrative.SLOTS, which follows the paper BLOTTER FORM; each is described by
# the form label it sits under, because a page is either that form filled in by
# hand or a free logbook entry stating the same facts in prose. Built once here
# so the two prompts cannot drift apart. Braces are doubled: the prompts are
# str.format templates.
_FIELD_KEYS = narrative.SLOTS

_FIELDS_BLOCK = """2. fields: what the page states, for the BLOTTER FORM record template. Every
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
{{"corrected_text": "<<<TEXT>>>", "fields": {{""" + ", ".join(
    '"%s": ""' % k for k in _FIELD_KEYS) + """}}}}
"""

_RESTORE_PROMPT = """You are restoring a page of a handwritten barangay blotter \
logbook from the Philippines that Google Cloud Vision has just transcribed. The \
handwriting mixes Filipino, Cebuano and English and is usually written in \
capitals, so the transcription contains character-level misreads, split words \
and merged words.

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

""" + _FIELDS_BLOCK.replace("<<TEXT>>", "the repaired page") + """{lexicon}
VISION TRANSCRIPTION:
{text}
"""

# The image arm's prompt. Differs from the text-only one in exactly one respect,
# and it is the whole point: the PHOTOGRAPH is the evidence and Vision's
# transcription is only a hint. The text-only prompt has to treat the
# transcription as ground truth because it is the only input there is - which is
# also why that arm can never recover a word Vision dropped.
_RESTORE_IMAGE_PROMPT = """You are reading a photographed page of a handwritten barangay blotter logbook from the Philippines. The handwriting mixes Filipino, Cebuano and English and is usually written in capitals.

You have TWO inputs: the photograph itself, and a transcription of it produced by Google Cloud Vision. The PHOTOGRAPH is the evidence. The transcription is a hint that is often wrong - it misreads characters, splits and merges words, scrambles reading order, and drops words entirely. Where they disagree, trust your own reading of the image.

Do two things, and return both in one JSON object.

1. corrected_text: what the page actually says, read from the photograph, in the    page's own reading order.
   - Transcribe only what the writer wrote. Do NOT rephrase, translate,      summarise, or add any word.
   - Do NOT change numbers, dates, times, amounts or case numbers - ever. Read      them off the image digit by digit.
   - A word you genuinely cannot read stays as Vision read it. If neither of you      can read it, write [illegible].
   - Names of people or places: transcribe what is written, do not correct      spellings you do not recognise.

""" + _FIELDS_BLOCK.replace("<<TEXT>>", "the page as you read it") + """{lexicon}
GOOGLE VISION'S TRANSCRIPTION (a hint, not the truth):
{text}
"""


def _restore_template(image):
    """The restore prompt, overridable so prompt variants can be scored.

    The Google minimally-lossy work did not fine-tune anything. It ran a loop:
    generate, auto-rate the output for readability and fidelity, hand the
    ratings to a second model and ask it to rewrite THE PROMPT, repeat - 824
    iterations, until the scores plateaued. On 73 gold pages that loop is the
    affordable half of that method and supervised tuning is not (see the module
    docstring), so this hook is what makes it runnable here:

        BANTAY_RESTORE_PROMPT_FILE=prompts/v3.txt python tools/run_gemini_arms.py \\
            --arm text --tag v3
        python tools/eval_fields.py models/preds_gemini_text_v3.csv

    A variant file must still carry the {lexicon} and {text} placeholders and
    still ask for the same JSON shape, or _parse_restore returns nothing and
    every page degrades to raw OCR - which the status column will say plainly.
    Read per call rather than cached: a scan is one file read against one
    multi-second API call, and a cache is a variant you edited but did not test.
    """
    path = os.environ.get("BANTAY_RESTORE_PROMPT_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return _RESTORE_IMAGE_PROMPT if image else _RESTORE_PROMPT


def _ask_restore(text, lexicon, model, timeout, image=None, mime=None):
    """One call, one timeout, no retry. Returns (corrected_text, fields).

    With `image`, the page photo is sent alongside Vision's transcription and the
    prompt swaps to _RESTORE_IMAGE_PROMPT. One call, not two: the model reads the
    pixels itself and gets Vision's reading as a prior, which is the only way to
    recover the 14.3% of gold tokens Vision never emitted at all (see
    OCR_ACCURACY_PLAN.md 1.1 - that share is the ceiling on every text-only
    corrector, this project's included).
    """
    from google.genai import types

    words = ""
    if lexicon:
        top = sorted(lexicon, key=lexicon.get, reverse=True)[:300]
        words = _LEXICON_BLOCK.format(words=", ".join(top))

    prompt = _restore_template(image).format(lexicon=words, text=text)
    contents = ([types.Part.from_bytes(data=image, mime_type=mime or "image/jpeg"), prompt]
                if image else prompt)

    resp = _client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=_http_options(timeout),
        ),
    )
    return _parse_restore(resp.text)


def _parse_restore(raw):
    """Defensive parse: a malformed reply is (None, {}), never an exception."""
    if not raw:
        return None, {}
    try:
        data = json.loads(raw)
    except ValueError:
        return None, {}
    if not isinstance(data, dict):
        return None, {}
    corrected = data.get("corrected_text")
    corrected = corrected if isinstance(corrected, str) and corrected.strip() else None
    raw_fields = data.get("fields")
    fields = {}
    if isinstance(raw_fields, dict):
        for key in _FIELD_KEYS:
            value = raw_fields.get(key)
            # A model that answers a field with a list or a nested object is not
            # answering the question. An empty slot the encoder fills beats a
            # stringified dict landing in a legal record.
            if isinstance(value, str):
                fields[key] = value.strip()
    return corrected, fields


def _diff_edits(before, after):
    """Token-level diff between the two texts, in the edit table's shape.

    Whole-token alignment rather than a character diff: the review table shows
    the encoder one row per word that changed, and a character diff would split
    NAG5ADY4 -> NAGSADYA into three unreadable rows.
    """
    spans = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(before)]
    new_tokens = [m.group() for m in _WORD_RE.finditer(after)]
    matcher = difflib.SequenceMatcher(
        a=[t[2].upper() for t in spans], b=[t.upper() for t in new_tokens])

    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src = " ".join(t[2] for t in spans[i1:i2])
        dst = " ".join(new_tokens[j1:j2])
        # A pure insertion has no span in `before` to anchor to; anchor it at the
        # following token's start so the splice in restore() stays ordered.
        start = spans[i1][0] if i1 < len(spans) else len(before)
        end = spans[i2 - 1][1] if i2 > i1 else start
        edits.append({
            "start": start, "end": end,
            "before": src, "after": dst,
            "score": round(difflib.SequenceMatcher(a=src.upper(), b=dst.upper()).ratio(), 4),
            "reason": tag, "source": "gemini",
        })
    return edits


# ---------------------------------------------------------------------------
# Minimally-lossy accounting, after Google Research's "minimally lossy text
# simplification with Gemini" (research.google, 2025).
#
# That work does not score a rewrite with one blunt "how different is it"
# number. It MAPS CLAIMS from the source onto the output and labels each
# mismatch as information LOSS, GAIN or DISTORTION, then weights by severity.
# The decomposition is the right instrument here for the same reason it was
# there: the three failures want opposite responses, and one ceiling cannot
# tell them apart. Claims are more than this module can compute locally without
# a second billed call, so the unit here is the content token - the same unit
# tools/eval_fields.py already scores narratives in.
#
#   distortion  an aligned token came back different. This IS the repair, and
#               it is the entire reason the pass runs. Wanted.
#   gain        a token appears that the source never had. On the image arm
#               that is the point - Vision never emits 14.3% of gold tokens and
#               only the pixels can recover them. On the text arm there are no
#               pixels, so the same event is fabrication.
#   loss        a token the source had is gone. Never wanted on either arm: a
#               blotter is a legal record, and the narrative has to come out of
#               this pass whole.
#
# The old single ceiling scored a faithfully repaired page and a paraphrased
# page identically, because both change the same share of characters. These
# three separate them.
# ---------------------------------------------------------------------------

def _tokens(text):
    """Content tokens, upper-cased. Same tokeniser the edit table walks."""
    return [m.group().upper() for m in _WORD_RE.finditer(text or "")]


def fidelity(source, candidate):
    """Token-level loss / gain / distortion between two texts.

    Rates are shares of the SOURCE token count, so they compare across pages of
    different lengths. Returned on every restore() result: a caller enforcing
    the guard reads `loss_rate` and `gain_rate`, a caller writing an eval row
    logs the whole dict, and the encoder sees it summarised in the status line.
    """
    a, b = _tokens(source), _tokens(candidate)
    loss = gain = distortion = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            continue
        nb, na = i2 - i1, j2 - j1
        distortion += min(nb, na)                 # aligned but changed - the repair
        loss += max(nb - na, 0)                   # source tokens with no output
        gain += max(na - nb, 0)                   # output tokens with no source
    n = max(len(a), 1)
    return {"loss": loss, "gain": gain, "distortion": distortion,
            "n_tokens": len(a),
            "loss_rate": round(loss / n, 4), "gain_rate": round(gain / n, 4),
            "distortion_rate": round(distortion / n, 4)}


# Slots the model writes in its own hand and NO second reader checks:
# ocr/reconcile.py only cross-checks the five closed-vocabulary slots, and the
# names, addresses, contacts and prose have no regex counterpart by design.
# date/time-like slots, location_purok and status are deliberately absent - they
# are reformatted on purpose ("NOV. 19, 2023" -> "2023-11-19"), so token
# grounding would reject every correct answer. Contact numbers, ages and the
# blotter number carry no words, so for them it is the number guard below that
# bites: a digit string the page never showed drops the slot.
_GROUNDED_SLOTS = tuple(s for s in narrative.SLOTS if s not in (
    "date", "time", "date_reported", "statement_date", "statement_time",
    "location_purok", "status"))

# Share of a field's own tokens that may be absent from the page. Not zero: the
# model legitimately joins names with a comma, drops line-break hyphenation and
# expands "BRGY" - and a two-word field would otherwise fail on one such token.
MAX_FIELD_GAIN = float(os.environ.get("BANTAY_MAX_FIELD_GAIN", "0.20"))

# _WORD_RE only matches tokens that START with a letter, so a field's NUMBERS
# are invisible to the word-grounding above - and a number is the one thing in a
# blotter that must never drift. Blotter_Pic (100) on the dev split: Vision read
# "P10,000", the text guard correctly reverted the model's "110,000", and
# "110,000 PESOS" was filed in the summary anyway because no guard looked at the
# fields. Compared on digits alone, so a page's "P10,000" still supports a
# summary's "10,000 pesos" - it is the VALUE that has to be on the page, not the
# formatting. Unlike the word budget this one is absolute: one unsupported
# number drops the field.
_NUM_RE = re.compile(r"\d[\d,.:]*")


# Addresses are exempt from the NUMBER check only (their words are still
# grounded). Vision routinely drops the purok digit ("ADD. PUROK BRGY ANUNAS")
# that the photo shows, and the restore guard reverts every numeric edit, so the
# digit can never reach the gated text - the check dropped complainant_address
# on 36-42 of 151 dev pages (gemini_*_form runs), almost all for a missing
# purok number. location_purok, which carried the same purok before the
# address slots existed, was never number-grounded either; this keeps the
# addresses at that same exposure rather than below it.
_NUMBER_EXEMPT = ("complainant_address", "respondent_address", "witness_address")


def _numbers(text):
    """Digit strings, separators stripped: 'P10,000' and '10.000' both -> '10000'."""
    return {re.sub(r"[^0-9]", "", m.group()) for m in _NUM_RE.finditer(text or "")} - {""}


def _ground_fields(fields, page):
    """Drop free-prose fields the page does not support. Returns (fields, dropped).

    This is the guard this module was missing, and it is the one that matters
    most, because `fields` - not `corrected_text` - is what becomes the record.
    Every other guard here protects the text; `fields` came back from the model
    completely ungated, so a page whose text was REJECTED as a rewrite handed
    that same rewrite to the encoder anyway through incident_summary. Both
    failures are visible in models/preds_gemini_text.csv on the dev split:

      Blotter_Pic (10)   text rejected at 69%, "raw OCR kept" - and the
                         paraphrase ("KUMUHA NG MANOK NA PANABONG" ->
                         "PAGKUHA ng manok na pinalalaki") filed as the summary
      Blotter_Pic (100)  numeric guard correctly reverted P10,000 in the text,
                         and "110,000 PESOS" was filed in the summary regardless

    A dropped slot costs nothing: routes/scan.py falls back to the gated text
    for incident_summary and to the regex reader for the rest, and both of those
    are grounded in the page by construction.
    """
    page_tokens = set(_tokens(page))
    page_numbers = _numbers(page)
    kept, dropped = dict(fields or {}), []
    for slot in _GROUNDED_SLOTS:
        value = kept.get(slot)
        invented = [] if slot in _NUMBER_EXEMPT else sorted(_numbers(value) - page_numbers)
        if invented:
            kept[slot] = ""
            dropped.append(f"{slot} (number {invented[0]} is not on the page)")
            continue
        toks = _tokens(value)
        if not toks:
            continue
        outside = sum(1 for t in toks if t not in page_tokens)
        if outside / len(toks) > MAX_FIELD_GAIN:
            kept[slot] = ""
            dropped.append(f"{slot} ({outside}/{len(toks)} words not on the page)")
    return kept, dropped


def _result(source, page, edits, fields, status):
    """Every restore() exit, with the field guard and the fidelity row attached.

    One helper rather than five hand-written dicts because the grounding has to
    run on EVERY path - including the rejection paths, which are exactly where
    an ungated paraphrase does the most damage, since the text it was read out
    of has already been thrown away.
    """
    fields, dropped = _ground_fields(fields, page)
    if dropped:
        status += " | dropped ungrounded " + ", ".join(dropped)
    return {"text": page, "edits": edits, "fields": fields,
            "fidelity": fidelity(source, page), "status": status}


def _keep_restore(edit):
    """Per-edit guards for the full-text path.

    restore() lets the model return whole text, so unlike verify() there is no
    "is this token on the page" check to lean on - the guards have to run on the
    diff instead. Numbers are the one that matters: a date, a time, an amount or
    a case number is a legal fact, and 2023 -> 2028 is the failure that would
    actually surface in a hearing. Anything numeric is reverted, whatever the
    model thought it saw.
    """
    if not edit["before"] and not edit["after"]:
        return False
    if _mostly_numeric(edit["before"]) or _mostly_numeric(edit["after"]):
        return False
    # A replace must look like a misread, not a paraphrase. Insertions are
    # exempt - see MIN_RESTORE_SIMILARITY. sealion.py imports this function, so
    # the floor lands on that arm too, which is correct: it is the same model
    # failure, not a Gemini-specific one.
    if edit["before"] and difflib.SequenceMatcher(
            a=edit["before"].upper(), b=edit["after"].upper()
    ).ratio() < MIN_RESTORE_SIMILARITY:
        return False
    # A deletion is the model dropping a word it could not read. Repairing is in
    # scope; erasing evidence is not.
    return bool(edit["after"].strip())


def _spaced(edit, out, text):
    """An accepted edit's text, with the whitespace an insertion does not carry.

    A pure insertion has a zero-width span (see _diff_edits), so appending it
    verbatim glues it to whichever neighbour the splice lands against: recovering
    "ANUNAS" next to "SALITA39" produced the single token "SALITA39ANUNAS",
    corrupting a word the scanner DID read in the course of restoring one it
    missed. Replaces need none of this - they consume the span they sit in.

    Only reachable from the image arm in practice, which is the arm insertions
    exist for, and the arm the accuracy argument rests on.
    """
    piece = edit["after"]
    if edit["before"]:
        return piece
    prev = out[-1][-1] if out and out[-1] else " "
    nxt = text[edit["end"]:edit["end"] + 1] or " "
    return ("" if prev.isspace() else " ") + piece + ("" if nxt.isspace() else " ")


def restore(text, lexicon=None, model=None, timeout=None, image=None, mime=None,
            max_change=None, max_gain=None):
    """Repair a Vision transcription and fill the record template, in one call.

    One call rather than two because the extraction needs the repaired text to
    read fields out of, and a second call would either re-send the page (double
    the quota, double the latency) or read fields off text the first call had
    already changed underneath it.

    The model returns whole text, but it does not get to hand that text straight
    through: the reply is diffed against the Vision output, every changed token
    is gated by _keep_restore, and only the surviving edits are spliced back into
    the ORIGINAL string. So the stored page is still built from what the scanner
    saw, the encoder still gets one review row per change, and a model that
    decides to rewrite the narrative produces a rejection rather than a record.

    The FIELDS get the same treatment, via _ground_fields: a free-prose slot
    whose words are not on the gated page is blanked rather than filed. Without
    that, every guard above was decorative - the record is built from `fields`,
    and they used to come back from the model untouched.

    Returns {"text", "edits", "fields", "fidelity", "status"} and never raises -
    a dead key, a dead network, or a bad day for the model must degrade the
    scan, not break it. `fields` is {} when nothing usable came back; callers
    fall back to the regex reader in routes/scan.py.
    """
    mode = _mode()
    if not mode:
        return _result(text, text, [], {}, "off")
    if not text or not text.strip():
        return _result(text, text, [], {}, "skipped: empty page")

    model = model or os.environ.get("BANTAY_GEMINI_MODEL", _DEFAULT_MODEL)
    timeout = RESTORE_TIMEOUT if timeout is None else timeout
    max_change = MAX_RESTORE_CHANGE if max_change is None else max_change
    max_gain = MAX_TOKEN_GAIN if max_gain is None else max_gain
    try:
        corrected, fields = _ask_restore(text, lexicon, model, timeout, image, mime)
    except Exception as exc:                      # noqa: BLE001 - any SDK/network failure
        # A deadline is the one failure here with an obvious operator action, and
        # "504 DEADLINE_EXCEEDED" does not suggest it. Name the knob in the
        # status line, which is the only place the encoder will ever see this.
        detail = f"{type(exc).__name__}: {exc}"
        if "DEADLINE_EXCEEDED" in detail or "504" in detail:
            detail = (f"timed out after {timeout:.0f}s generating the corrected page. "
                      f"Raise BANTAY_GEMINI_TIMEOUT, or use a faster model via "
                      f"BANTAY_GEMINI_MODEL (currently {model}).")
        return _result(text, text, [], {}, f"error: {detail}"[:300])

    if corrected is None:
        return _result(text, text, [], fields,
                       "ok: no corrected text returned, fields only")

    proposed = _diff_edits(text, corrected)
    kept = [e for e in proposed if _keep_restore(e)]
    dropped = len(proposed) - len(kept)

    # Nothing survived, but the model did propose changes. "ok: 0 edits" is the
    # same string a page that genuinely read fine produces, and the encoder has
    # to be able to tell those apart: one is a clean page, the other is a model
    # that tried to rewrite the narrative and was stopped. Say which.
    if proposed and not kept:
        return _result(text, text, [], fields,
                       f"rejected: {len(proposed)} proposed edits, none passed the "
                       f"guards - raw OCR kept")

    # Splice the surviving edits into the original, so nothing the model wrote
    # outside an accepted edit can reach the page.
    out, cursor = [], 0
    for e in sorted(kept, key=lambda e: e["start"]):
        if e["start"] < cursor:                   # overlapping spans - keep the first
            continue
        out.append(text[cursor:e["start"]])
        out.append(_spaced(e, out, text))
        cursor = e["end"]
    out.append(text[cursor:])
    new_text = "".join(out)

    touched = sum(_changed_chars(e["before"], e["after"]) for e in kept) / max(len(text), 1)
    if touched > max_change:
        return _result(text, text, [], fields,
                       f"rejected: model rewrote {touched:.0%} of the page "
                       f"(limit {max_change:.0%}) - raw OCR kept")

    # The minimally-lossy gate. Separate from `touched` above, which averages
    # all three error types into one number and so cannot distinguish a page
    # repaired character by character from a page reworded wholesale.
    fid = fidelity(text, new_text)
    if fid["gain_rate"] > max_gain:
        return _result(text, text, [], fields,
                       f"rejected: not minimally lossy - added {fid['gain']} words the "
                       f"scanner never read ({fid['gain_rate']:.0%} of the page, limit "
                       f"{max_gain:.0%}) - raw OCR kept")

    note = f" ({dropped} rejected by guards)" if dropped else ""
    return _result(text, new_text, kept, fields,
                   f"ok: {len(kept)} edits{note}, {fid['distortion']} words repaired / "
                   f"{fid['gain']} added / {fid['loss']} lost ({model} via {mode})")


# The image arm may diverge from Vision far more than the text arm may, and the
# divergence is the SIGNAL rather than the alarm: this pass reads the photograph,
# so a word Vision never emitted is exactly what it is here to recover, and
# restore()'s 35% ceiling would reject the pages it helps most. Still bounded and
# still a hard reject - a model that returns a page bearing no relation to what
# Vision saw has stopped transcribing. Tune against your own pages.
MAX_IMAGE_CHANGE = float(os.environ.get("BANTAY_IMAGE_MAX_CHANGE", "0.70"))

# GAIN is the one error type this arm is allowed far more of than the text arm,
# and the asymmetry is the architecture: a word Vision never emitted can only
# arrive here as an insertion, and 14.3% of gold tokens are in that category
# (OCR_ACCURACY_PLAN.md 1.1). LOSS gets no such allowance - MAX_TOKEN_LOSS is
# shared by both arms, because there is no reading of the pixels under which
# losing a word the scanner already found is an improvement.
MAX_IMAGE_TOKEN_GAIN = float(os.environ.get("BANTAY_IMAGE_MAX_TOKEN_GAIN", "0.35"))


def restore_from_image(image, vision_text, lexicon=None, model=None, timeout=None,
                       mime=None, max_change=None, max_gain=None):
    """Read the page photo directly, with Vision's transcription as a prior.

    This is the only path in the pipeline not capped by what Vision transcribed.
    Every text corrector - the lexicon, Gemini's own verify() -
    can at best repair a garbled token that is already on the page, which on this
    corpus tops out at 54.3% gold-token recall because 45.7% of gold tokens never
    appear in the transcription at all (OCR_ACCURACY_PLAN.md 1.1). Pixels are the
    only way past that number.

    `image` is raw bytes. Everything else - the numeric revert, the deletion
    guard, the token-level diff the review table renders, the splice back into
    the original - is restore()'s, unchanged, because those guards do not care
    where the proposed text came from. Only the change ceiling differs; see
    MAX_IMAGE_CHANGE.

    Returns restore()'s dict: {"text", "edits", "fields", "status"}.
    """
    return restore(vision_text, lexicon=lexicon, model=model, timeout=timeout,
                   image=image, mime=mime,
                   max_change=MAX_IMAGE_CHANGE if max_change is None else max_change,
                   max_gain=MAX_IMAGE_TOKEN_GAIN if max_gain is None else max_gain)


if __name__ == "__main__":       # python -m bantay.ocr.gemini [--smoke]
    # Config check. Model IDs change; this is how you confirm the one you want
    # exists on the backend you are actually billing, before a deployment finds
    # out for you one scanned page at a time. --smoke spends one call to prove
    # the whole path works on a deliberately garbled line.
    import sys

    mode = _mode()
    print(f"backend        : {mode or 'off (no Vertex project, no API key, or BANTAY_GEMINI=0)'}")
    print(f"model          : {os.environ.get('BANTAY_GEMINI_MODEL', _DEFAULT_MODEL)}"
          f"{'' if os.environ.get('BANTAY_GEMINI_MODEL') else '   (default - set BANTAY_GEMINI_MODEL to change)'}")
    if mode == "vertex":
        print(f"project/region : {os.environ['GOOGLE_CLOUD_PROJECT']} / "
              f"{os.environ.get('GOOGLE_CLOUD_LOCATION', 'global')}")

    if mode and "--smoke" in sys.argv:
        # Garbled on purpose: 0-for-O and 5-for-S, the two misreads Vision makes
        # most on this logbook. The date is in there to watch it survive - if
        # "JAN. 3, 2026" comes back changed, the numeric guard has a hole.
        sample = ("NAG5ADYA DIT0 SA BRGY. HALL NG ANUNAS UPANG IPA-BL0TTER ANG "
                  "NANGYARING PANL0LOOB SA KANILANG BAHAY N00NG JAN. 3, 2026")
        print()
        print("smoke test - one API call")
        print("  in     :", sample)
        out = verify(sample)
        print("  out    :", out["text"])
        print("  status :", out["status"])
        for e in out["edits"]:
            print(f"    {e['before']!r} -> {e['after']!r}   {e['reason']}")
        if "JAN. 3, 2026" not in out["text"]:
            print("  WARNING: the date changed. That must not happen - stop and check _keep().")
        elif out["text"] == sample:
            print("  no change made. Read the status line above before assuming it is broken:")
            print("  'off' is config, 'error:' is the backend, 'none passed the guards' is the wall.")
    elif mode:
        try:
            names = sorted(m.name.split("/")[-1] for m in _client().models.list())
            print()
            print(f"{len(names)} models available here:")
            for n in names:
                print("   ", n)
        except Exception as exc:                  # noqa: BLE001 - this IS the diagnostic
            print(f"could not list models: {type(exc).__name__}: {exc}")
