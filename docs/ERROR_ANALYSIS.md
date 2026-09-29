# Aggregate error analysis

This report examines the completed locked test without redistributing complaint narratives or identifiers. All figures below come from the retained aggregate metrics and confusion counts. It is descriptive evidence, not a post-hoc change to the frozen hypothesis.

## Class-level results

| Product | Baseline F1 | DistilBERT F1 | Difference |
| --- | ---: | ---: | ---: |
| Credit reporting | 0.7044 | 0.6829 | -0.0215 |
| Debt collection | 0.6719 | 0.5354 | -0.1364 |
| Credit card | 0.7832 | 0.6562 | -0.1270 |
| Checking or savings | 0.7826 | 0.6816 | -0.1010 |
| Mortgage | 0.9116 | 0.7922 | -0.1194 |
| Student loan | 0.9444 | 0.9306 | -0.0139 |
| Money transfer or service | 0.8201 | 0.6230 | -0.1972 |
| Vehicle loan or lease | 0.7862 | 0.7568 | -0.0295 |

The baseline won every class. The smallest gaps were student loans, credit reporting, and vehicle loans; the largest was money transfer or service. This pattern argues against promoting the transformer on the strength of a few favorable categories.

## Dominant transformer confusions

The strongest off-diagonal errors on the 583-record test split were:

- **Money transfer → checking or savings:** 25 of 75 money-transfer records (33.3%). Transaction language can overlap across bank-account and payment-service contexts.
- **Debt collection → credit reporting:** 20 of 69 debt-collection records (29.0%). Collection narratives often discuss reports, bureaus, or disputed account records.
- **Credit card → credit reporting:** 10 of 71 records (14.1%).
- **Credit card → checking or savings:** 10 of 71 records (14.1%).

Student loans were the transformer's strongest class by recall: 67 of 75 test records were correct (89.3%). Money transfer was the weakest by recall: 38 of 75 were correct (50.7%). The confusion matrix is retained in [`artifacts/confusion_matrix.png`](../artifacts/confusion_matrix.png).

## What the errors suggest

1. **The label boundary is partly contextual.** Product names alone may not capture whether a payment narrative concerns the funding account, the transfer rail, or a later collection or reporting event.
2. **The frozen transformer run was deliberately modest.** It used 2,705 training records, two epochs, and a 192-token limit. The result does not prove that transformer models are generally inferior; it proves that this specific added complexity was not justified by this locked test.
3. **One global threshold is a coarse control.** Class-conditional thresholds may be better where error costs or confidence behavior differ, but they require a new calibration protocol and independent evaluation.
4. **Abstention changes the operating point, not the underlying label quality.** The 91.82% accepted accuracy applies only to the 46.14% of test cases above the frozen threshold. The remaining 53.86% still require review.

## Next experiments, in order

The first follow-up is complete: the baseline was calibrated only on the original Q1
calibration split, then both frozen models were evaluated on a preregistered Q2 holdout. The
baseline preserved a +0.0500 macro-F1 advantage (paired 95% bootstrap interval +0.0362 to
+0.0634) and transferred its review policy at 90.07% accepted accuracy and 77.48% coverage.
See the [forward-holdout results](FORWARD_HOLDOUT_RESULTS.md).

1. Test a hierarchical classifier that first separates account, credit, lending, and money-movement families, then predicts the detailed product.
2. Pre-register class-conditional review thresholds and evaluate both accuracy and coverage with uncertainty intervals.
3. Add later temporal slices and institution-external validation before any operational recommendation.
4. Measure end-to-end latency, memory, and cost so model quality is evaluated alongside workflow impact.

No error narrative is reproduced here. Qualitative text review would require a separate privacy-controlled process and explicit rules for handling potentially identifying content.
