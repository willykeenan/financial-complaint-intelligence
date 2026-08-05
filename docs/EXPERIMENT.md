# Frozen experiment protocol

Frozen before model outcomes were observed on 2026-08-05.

## Question

Can a compact transformer improve classification of public financial complaint narratives while exposing calibrated uncertainty for human review?

## Hypothesis and decision rule

On a balanced, deduplicated eight-product sample from the completed 2024 Q1 CFPB window, a fixed two-epoch DistilBERT run will exceed a fixed word-and-character TF-IDF logistic-regression baseline by at least **0.02 macro-F1** on a chronological per-product test holdout.

The hypothesis is supported only if the locked test comparison reaches that threshold. A miss will be reported as a miss; no post-outcome tuning changes the primary result.

## Frozen design

- Source: CFPB Consumer Complaint Database search API, CC0 data.
- Window: 2024-01-01 through 2024-03-31.
- Classes: the eight product labels in `src/complaint_intelligence/config.py`.
- Sampling: up to 500 narratives per product from deterministic, evenly spaced rows within each official filtered CSV export.
- Privacy: raw narratives and complaint IDs remain local and are gitignored. Only aggregate manifests and derivative model artifacts may be published.
- Deduplication: normalized exact-narrative hashes; same-label repeats keep the earliest record, and cross-label conflicts are excluded.
- Split: within each product, earliest 70% train, next 15% calibration, latest 15% test.
- Baseline: fixed word + character TF-IDF FeatureUnion and logistic regression.
- Transformer: `distilbert/distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`, maximum length 192, two epochs, learning rate 2e-5, seed 53353.
- Primary metric: test macro-F1. Secondary: accuracy, per-class precision/recall/F1, NLL, ECE, confusion matrix, and bootstrap macro-F1 interval.
- Calibration: a single temperature fitted only on the calibration split.
- Review policy: select the lowest calibration confidence threshold that reaches 90% accepted accuracy, maximizing coverage. If none does, abstain on every item and report target failure.

## Boundaries

This is a portfolio experiment, not a production decision system. CFPB complaints are not a representative sample, narratives are not verified by CFPB, product labels are not customer intent labels, and reported metrics do not establish performance on institution-specific data, live traffic, protected subgroups, or future distributions. Human review remains mandatory for low-confidence predictions and any consequential action.

## Execution note

Before training or observing model outcomes, direct API pagination was found to repeat pages under the current CFPB search behavior. Acquisition was therefore changed to the official filtered CSV export, followed by deterministic evenly spaced row selection. The date window, labels, sample cap, deduplication, split, models, hyperparameters, metrics, and decision rule remained frozen. The manifest records the export counts and notes the CFPB 100,000-row export cap.
