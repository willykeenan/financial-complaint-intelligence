---
license: apache-2.0
language:
- en
pipeline_tag: text-classification
tags:
- financial-services
- calibrated-classification
- human-in-the-loop
- portfolio
datasets:
- cfpb/consumer-complaint-database
metrics:
- f1
---

# Financial Complaint Intelligence — DistilBERT experiment

This is an eight-class DistilBERT complaint-product classifier trained for a transparent portfolio experiment on public CFPB complaint narratives. It includes a held-out temperature and a confidence threshold that routes uncertain inputs to human review.

## Results

On the 583-record chronological locked test split:

- DistilBERT macro-F1: **0.7073** (95% bootstrap interval 0.6703–0.7393)
- DistilBERT accuracy: **0.7118**
- calibrated NLL: **0.8599**
- 15-bin ECE: **0.0430**
- calibration-selected threshold: **0.7417**
- accepted test accuracy: **0.9182** at **0.4614** coverage
- accepted-accuracy 95% Wilson interval: **0.8793–0.9454**

The precommitted hypothesis was not supported. A word-and-character TF-IDF logistic-regression baseline reached **0.8006 macro-F1**, exceeding this transformer by 0.0932. The transformer should not replace that simpler baseline based on this evidence.

## Training and evaluation

- source: CFPB Consumer Complaint Database, CC0
- date window: 2024-01-01 through 2024-03-31
- retained after validation/deduplication: 3,867 narratives
- split: earliest 70% train, next 15% calibration, latest 15% test within each class
- base model: `distilbert/distilbert-base-uncased`
- pinned revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`
- maximum length: 192 tokens
- training: 2 epochs, learning rate 2e-5, seed 53353
- raw narratives and complaint IDs are not redistributed

## Intended use

Research and portfolio demonstration of classification, calibration, selective prediction, and human-review routing. Use fictional or fully de-identified inputs only.

## Limitations

This is not a production or consequential decision system. CFPB complaints are self-selected and not representative of any institution's customers or future traffic. Narratives are not verified by CFPB. Product labels do not establish intent, validity, fault, urgency, fraud, eligibility, or customer outcome. The model has not been evaluated for protected groups, language varieties, institution-specific data, adversarial inputs, or distribution shift. Confidence is not a guarantee. Human review is required for uncertain cases and every consequential action.

See the repository experiment protocol, aggregate manifest, exact metrics, plots, and tests for the full evidence chain.
