# Frozen forward-holdout protocol

Protocol ID: `fci.forward-holdout.2024q2.v1`

Frozen on 2026-08-07 before acquiring or inspecting the 2024 Q2 holdout outcomes.

## Question

Does the word-and-character TF-IDF logistic-regression baseline that won the original
2024 Q1 comparison preserve its advantage over the frozen DistilBERT model on a later,
untouched quarter, and can a review threshold fitted only on the original calibration split
transfer without silently weakening its accuracy target?

This is a prospective temporal holdout for the follow-up question. The original Q1 test
labels and hard-prediction results have already been observed, so no new claim about that
test is confirmatory. The Q2 narratives, predictions, and outcomes must remain uninspected
until this protocol and its evaluation code exist in an earlier Git commit.

## Frozen data design

- Source: CFPB Consumer Complaint Database full daily CSV ZIP, filtered locally before
  sampling.
- Holdout window: 2024-04-01 through 2024-06-30, inclusive.
- Classes: the same ordered eight product labels in
  `src/complaint_intelligence/config.py`.
- Sampling: up to 500 narratives per product, selected at deterministic evenly spaced
  positions after sorting each filtered export by received date and complaint ID.
- Deduplication: normalized exact-narrative hashes. Same-label repeats keep the earliest
  record; all members of a cross-label conflict are excluded.
- Cross-period isolation: any Q2 narrative whose normalized hash appears in the original
  Q1 train, calibration, or test data is excluded before evaluation.
- No Q2 split, training, calibration, threshold selection, feature selection, or model
  selection is allowed.
- Historical database revisions are possible. The retrieval timestamp, source update
  timestamp, class counts, and local-content digest identify the exact evaluated snapshot.

Raw narratives, complaint IDs, issues, downloaded CSVs, model weights, and row-level
predictions remain local and gitignored. Only aggregate manifests, metrics, uncertainty
intervals, and plots may be published.

## Frozen models and calibration

Both models are the artifacts from the completed Q1 experiment. Their hashes are recorded
in the result artifact before evaluation.

### Baseline

1. Load the retained word-and-character TF-IDF one-vs-rest logistic-regression pipeline.
2. Align its probability columns to the frozen product order.
3. Convert the original Q1 calibration probabilities to log-probability scores.
4. Fit one scalar temperature by minimizing multiclass negative log likelihood on only the
   original Q1 calibration split.
5. On that same Q1 calibration split, choose the lowest confidence threshold that maximizes
   coverage while reaching at least 90% accepted accuracy. If no threshold qualifies, the
   policy accepts zero cases.
6. Apply the unchanged temperature and threshold to Q2.

The scalar temperature cannot change the baseline's predicted class; it only calibrates the
confidence used for review routing.

### Transformer

Load the retained DistilBERT weights and reuse the already-frozen Q1 temperature and review
policy. Do not refit either artifact for this follow-up. Apply the unchanged model,
temperature, and threshold to Q2.

## Endpoints and decision rules

### Primary endpoint

Report Q2 macro-F1 for both models and the paired difference:

`baseline macro-F1 - transformer macro-F1`

Use 5,000 paired percentile-bootstrap resamples with seed 53353. The baseline-advantage
hypothesis is supported only when the observed difference is positive and the lower bound
of its 95% paired bootstrap interval is above zero.

### Selective-policy transfer endpoint

Report Q2 accepted accuracy, Wilson 95% interval, accepted count, review count, and coverage
for both frozen policies. The baseline transfer rule is supported only when all of these are
true:

- the Q1 calibration procedure found an eligible threshold;
- Q2 accepted accuracy is at least 90%;
- the Q2 accepted-accuracy Wilson lower bound is at least 85%;
- Q2 coverage is at least 25%.

These thresholds are a portfolio experiment rule, not evidence that automated routing is
safe in a real complaint workflow.

### Secondary descriptive endpoints

- accuracy, weighted F1, per-class precision/recall/F1, and bootstrap macro-F1 intervals;
- negative log likelihood and 15-bin expected calibration error on Q2;
- empirical risk-coverage curves, published only at one-percentage-point coverage steps from
  100% down to 5% so row-level confidence traces remain local;
- reliability plots and class counts;
- runtime and artifact provenance needed to reproduce the local evaluation.

Secondary endpoints cannot overturn a failed primary rule.

## Change control

An outcome-affecting change after Q2 acquisition or prediction requires a new protocol ID
and a genuinely untouched future window. A code defect discovered before any Q2 outcome is
read may be corrected in a new pre-results commit with a written amendment. Failed or
unfavorable outcomes are retained and reported without post-hoc tuning.

### Pre-results acquisition amendment — 2026-08-07

The first acquisition attempt returned only HTTP 500 and 429 failures from the CFPB
filtered-export endpoint; it produced no holdout rows, predictions, or outcomes. CFPB
Release 23 also documents that JSON export is retired and filtered CSV export is capped.
Before any Q2 data was acquired or inspected, transport was amended to the official full
daily CSV ZIP, followed by local filtering to the same frozen quarter, narrative-present
rule, and product labels. Sorting, sample count, deduplication, cross-period exclusion,
models, metrics, and decision rules are unchanged. The full ZIP digest is recorded in the
aggregate manifest so the source snapshot remains identifiable.

## Boundaries

This experiment does not establish institution-specific validity, protected-group or
language performance, semantic deduplication, robustness to adversarial input, reviewer
workflow quality, latency or cost at production scale, or safety for consequential action.
CFPB complaints are self-selected public reports and are not a representative sample of
customers or verified accounts of events. Human review remains mandatory for every
consequential use.
