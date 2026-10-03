# BANTAY Semi-Synthetic Data Pipeline

**Input:** 74 valid blotter reports (73 rows after the header, all `Include in ML = Yes`)
**Output:** 2,100-row ML corpus — 73 real + 2,027 semi-synthetic — with canonical
`Incident Type`, `Category Group`, and `PNP Tier` on every row, split into
train / validation / test.

---

## 0. Why the current model scores 0.26

`models/comparison_results.json` reports best F1 = 0.2578 (Decision Tree). Four
things cause that, and all four are data problems, not model problems:

| Cause | Evidence |
|---|---|
| Two label vocabularies in one dataset | The 73 real records use 23 free-form Incident Type strings; only 12 of them are in `IncidentTypeCategoryGroup.csv`. `Molestation`, `Vehicular Accident`, `Consumer / Fare Dispute` etc. have no canonical home. |
| `merge_rare_categories(min_count=3)` | With 73 rows over 23 labels, most classes have 1–3 members and collapse into `Others / Miscellaneous`. That is why `Unclassified` and `Catch-all` are the only tiers with non-zero F1 in the Naive Bayes and Decision Tree rows. |
| 15-row test set | `train_test_split(test_size=0.2)` on 73 rows. One flipped prediction moves accuracy by 6.7 points. The metric cannot distinguish a good model from a lucky one. |
| Synthetic seed covers the wrong classes | `bantay_synthetic_seed_2023_1.csv` has 15 rows for each of 23 types — but only the types that **do not** appear in the real data. The 12 real classes stay at their original 1–12 rows, so the imbalance is untouched. |

The pipeline below fixes all four.

---

## 1. Stage map

```
data_in/bantay_ml_ready_2023.csv   (73 real records, 23 free-form labels)
data_in/IncidentTypeCategoryGroup.csv  (34 canonical types + Category Group)
        │
        ▼
  ① LABEL BRIDGE                              semisynth/label_bridge.py
     23 encoded strings ─► 35 canonical types
     + Category Group  + PNP Tier
     + confidence (HIGH / MED / LOW) + written rationale per rewrite
        │
        ▼
  ② PHRASE BANK                               semisynth/phrasebank.py
     6 openers · 6 closers · 6 detail clauses · 1 header
     × 6 core event clauses per incident type (210 total)
     + slot vocabularies (items, amounts, vehicles, places)
        │
        ▼
  ③ GENERATION                                semisynth/generate.py
     seeded template + slot recombination
     top-up to 60 rows per type  (60 − real_count, floored at 0)
     structured fields sampled from the REAL empirical distribution
        │
        ▼
  ④ CORPUS + SPLITS                           semisynth/build_corpus.py
     synthetic split by core-clause variant  0-3 train │ 4 val │ 5 test
     real split stratified 60/20/20 by Category Group
        │
        ▼
  ⑤ MODEL                                     semisynth/model.py
     dual-axis: global type model + dedicated group model
        │
        ▼
  ⑥ VERIFICATION                              semisynth/verify.py
     7 integrity gates + leave-one-out on the 73 real records
```

Run it:

```bash
python run_pipeline.py --real data_in/bantay_ml_ready_2023.csv --out out/
python run_pipeline.py --loocv            # adds the real-only headline metric (~9 min)
```

---

## 2. Stage ① — Label bridge

The taxonomy CSV is the authority. Eleven encoded strings need a rewrite, and
each rewrite is declared in `BRIDGE_RULES` with a confidence and a rationale
that a panel can read. Full table ships as `out/label_bridge_audit.csv`.

| Encoded | Canonical | Conf | Why |
|---|---|---|---|
| Molestation (×2) | Peeping/Voyeurism | HIGH | Both narratives are `PANINILIP` while bathing. No contact alleged. |
| Grave Threats (×2) | Grave Threats / Light Threats | HIGH | Naming only. |
| Vehicular Accident (×2) | Vehicular Incident | HIGH | Naming only. |
| Trespassing / Suspicious Persons | Trespassing | HIGH | Entry onto property by named persons. |
| Neighbor / Nuisance Complaint | Neighbor Dispute | HIGH | Garbage nuisance escalating to argument. |
| Contract Dispute / Estafa | Estafa (Swindling) | HIGH | P3M taken, only excavation done — deceit plus damage. |
| Threats / Coercion (Financial) | Coercion | HIGH | Forced to sign a blank sheet (RPC Art. 286). |
| Other / Unclear | Others / Miscellaneous | HIGH | No offence stated. |
| Verbal Altercation / Disturbance | Public Disturbance | MED | Shouting match; Alarms and Scandals needs tumultuous public disorder. |
| Rental / Lease Dispute | Breach of Contract | MED | Lease dispute, not eviction. Runner-up: Collection of Sum of Money. |
| Consumer / Fare Dispute | Breach of Contract | **LOW** | CSV has no consumer type. Flagged for adviser. |
| Property Deposit / Safekeeping | Property Claim / Damage | **LOW** | Administrative custody entry, not a complaint. Flagged. |

**One taxonomy extension.** `Peeping/Voyeurism` is added to the 34 CSV types
(→ 35). Folding the two peeping records into `Unjust Vexation` would understate
a penal-code offence. This is the only addition, and it is flagged as an
extension in `taxonomy.py` so the adviser can veto it in one line.

**Nothing is dropped.** The 2 LOW rows and 2 records with out-of-window dates
(`1-3-25` transcribed as 2025, and a 2026 date) go to
`out/bantay_needs_review.csv` *and stay in the corpus*. Flag, don't delete.

---

## 3. Stage ② — Phrase bank (why this method)

Every skeleton is lifted from the 73 real narratives or written in their exact
register. Openers, closers and detail clauses are verbatim; the 210 core event
clauses are new sentences in the same register.

This is what makes the corpus **semi**-synthetic rather than synthetic: the
lexical distribution stays inside the barangay's own Tagalog, only the
recombination is new. Three properties that matter at defense:

1. **Deterministic.** Same seed + same CSV = byte-identical output. A panelist
   can re-run it.
2. **No hallucinated content.** No LLM writes a narrative at generation time,
   so no fabricated legal claim can enter the training data.
3. **Auditable provenance.** Every synthetic row carries `_template_id`
   recording which opener, core variant, details and closer produced it.

Cost: lower lexical diversity than LLM-authored text. That is stated in
§8 Limitations, not hidden.

---

## 4. Stage ③ — Generation rules

| Field | Rule |
|---|---|
| `Location/Purok` | Sampled from the real empirical distribution (19 observed values, weighted). |
| `Time` | Sampled from the real time distribution. |
| `Status` | Sampled from real: Filed 60 / Settled 8 / For Hearing 5. |
| `Action Taken` | Conditioned on Status — `Settled` → confrontation held; `For Hearing` → a scheduled date 3–15 days out; `Filed` → blank, matching the 51 blank real rows. |
| `Date` | **Uniform across the calendar year, deliberately.** |
| Persons | `[PERSON_SYN_n]` tokens only. Never a real or invented name. |
| Volume | `60 − real_count` per type, floored at 0. Classes with real data get topped up, not duplicated. |

**The date decision is a design choice, not an oversight.** The real records
only span Nov 2023 – Jan 2024. If synthetic dates were sampled to match, every
synthetic row would land in that window and the model would learn
"November ⇒ blotter". BANTAY does trend analysis; a fabricated seasonal signal
is worse than no signal. Uniform dates keep the month feature deliberately
uninformative, so any temporal trend the deployed system reports comes from
real data only.

---

## 5. Stage ④ — Splitting, and the leakage trap

Random row-level splitting of template-generated text is the easiest way to
publish a fake 0.95 F1: the same sentence skeleton appears in train and test,
and the model gets graded on memorisation.

**Control:** synthetic rows are split by *core clause variant*.

| Variants | Split |
|---|---|
| 0, 1, 2, 3 | train |
| 4 | validation |
| 5 | test |

A test row is therefore always built from a sentence skeleton the model has
never seen. Real rows are stratified 60/20/20 by Category Group so the test
fold always contains genuine logbook language (15 real records).

### Three numbers, three questions

| Metric | Question it answers | Belongs in |
|---|---|---|
| synthetic-test F1 | Can the model generalise across sentence skeletons? | Chapter 4 diagnostics |
| real-test F1 (n=15) | Does it transfer to real narratives? | Chapter 4, with the n=15 caveat stated |
| **LOOCV on all 73 real** | Same question, 73 test points instead of 15 | **Abstract and conclusion** |

LOOCV: for each real record, train on all synthetic-train rows + the other 72
real records, predict the held-out one. It is the highest-power real-data
estimate available at this sample size, and it is what should be reported.

---

## 6. Stage ⑤ — Model

Features are word 1–2 grams **plus character 3–5 grams**. The character channel
is not optional: the source is hand-transcribed Tagalog with inconsistent
spelling (`IPA-BLOTTER` / `IPA BLOTTER` / `PINA BLOTTER`), which word-only
TF-IDF treats as three unrelated tokens.

Three architectures were measured. The result is not the one the plan assumed:

**LOOCV over all 73 real records**

| Model | Type acc | Type macro F1 | Group acc | Group macro F1 |
|---|---|---|---|---|
| Flat (35-way) | **0.644** | **0.609** | 0.795 | 0.705 |
| Hierarchical (group → type) | 0.575 | 0.416 | **0.808** | **0.790** |
| Constrained (group model gates global type model) | 0.589 | 0.446 | **0.808** | **0.790** |

The hierarchy loses on the type axis because stage-1 group errors cascade and
cannot be recovered, and because seven small per-group models each see a slice
of the data instead of all 2,000 rows. But the dedicated group model beats a
group derived from a predicted type (0.790 vs 0.705 macro F1).

**So neither axis should obey the other.** `DualAxisClassifier` runs both and
reports each from its own model:

| | LOOCV, n=73 |
|---|---|
| Incident Type accuracy | 0.644 |
| Incident Type macro F1 | 0.609 |
| Category Group accuracy | 0.808 |
| Category Group macro F1 | 0.790 |
| Axes agree | 82.2% of records |
| Type accuracy **when axes agree** | **0.700** |
| Type accuracy when axes disagree | 0.385 |

The disagreement flag is a free, calibrated confidence signal: **auto-accept the
82% where the axes agree (70% type accuracy), route the 13 disagreeing records
to the existing Needs Review queue.** That is a better use of a mismatch than
hiding it behind a forced-consistent label.

### Lift over the current prototype

| | Current (`comparison_results.json`) | This pipeline |
|---|---|---|
| Best F1 | 0.2578 (Decision Tree, n=15 test) | **0.609** macro F1 (LOOCV, n=73 real) |
| Category Group F1 | 0.0 on 3 of 5 groups | 0.790 macro over 7 groups |
| Classes with training data | ~5 after rare-class merging | 35 of 35 |
| Real test points | 15 | 73 |

---

## 7. Stage ⑥ — Integrity gates

All seven are gates, not warnings. Any FAIL blocks training. Current run: **7/7 PASS.**

| Gate | Guards against |
|---|---|
| `no_pii` | A real person token leaking into a "de-identified" synthetic row |
| `label_consistency` | `Category Group ≠ group(Incident Type)` anywhere in the corpus |
| `no_exact_dupes` | Duplicate narratives inflating accuracy for free |
| `split_disjoint` | The same core clause appearing in train and test |
| `real_in_test` | The headline metric quietly becoming synthetic-only |
| `class_coverage` | A canonical type with zero training rows |
| `no_real_mutation` | Generation silently editing a real narrative |

---

## 8. Limitations — state these before the panel does

1. **Synthetic:real ratio is 27.8:1.** The model is mostly learning synthetic
   language. This is why the headline metric is LOOCV on real records only; the
   synthetic-test number is a diagnostic, never the claim.
2. **Lexical diversity is bounded by the phrase bank.** 210 core clauses over
   35 types. Templates cover the *typical* phrasing of each incident type, not
   the tail. Real narratives that phrase an incident unusually will be missed —
   visible in the LOOCV per-class table (e.g. `Property Claim / Damage`
   F1 = 0.14, `Oral Defamation` F1 = 0.00).
3. **Two LOW-confidence bridge rows** (`Consumer / Fare Dispute`,
   `Property Deposit / Safekeeping`) are judgement calls awaiting adviser
   confirmation. They are flagged in `bantay_needs_review.csv`.
4. **One taxonomy extension** (`Peeping/Voyeurism`) beyond the approved CSV.
5. **Uniform synthetic dates mean the corpus carries no seasonal signal.** This
   is intentional (§4) but it means the temporal-trend module must be evaluated
   on real records only.
6. **60 rows per class is a ceiling on measurable per-class performance.** With
   ~10 synthetic test rows per class, a per-class F1 estimate has wide error
   bars. Report macro F1 with that caveat.

---

## 9. What to do next

1. Have the adviser rule on the 2 LOW bridge rows and the `Peeping/Voyeurism`
   extension. One decision each, both isolated to `label_bridge.py` /
   `taxonomy.py`.
2. Wire `DualAxisClassifier` into `bantay/ml/train.py`, and route
   axis-disagreement rows to the existing `bantay/routes/review.py` queue.
3. Drop `merge_rare_categories` from the training path — with 60 rows per class
   it now removes signal instead of noise.
4. If more real records get encoded, re-run the pipeline unchanged. The
   `60 − real_count` top-up rule automatically shrinks the synthetic share as
   real data grows, which is the property you want for the thesis narrative.

---

## 10. File map

```
run_pipeline.py                  one-shot runner
semisynth/taxonomy.py            35 canonical types, Category Group, PNP tier
semisynth/label_bridge.py        11 rewrite rules + rationale
semisynth/phrasebank.py          mined skeletons + 210 core clauses + slots
semisynth/generate.py            seeded generator, real-distribution sampling
semisynth/build_corpus.py        assembly + leakage-safe splitting
semisynth/model.py               flat / hierarchical / constrained / dual-axis
semisynth/verify.py              7 integrity gates
out/bantay_corpus.csv            2,100 rows, both provenances
out/bantay_{train,val,test}.csv  1,414 / 346 / 340
out/bantay_needs_review.csv      4 rows for human confirmation
out/label_bridge_audit.csv       every rewrite, defensible line by line
out/pipeline_report.json         checks + all metrics
out/dual_axis_loocv.json         the headline numbers
```
