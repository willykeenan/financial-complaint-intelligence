# Model card: [REQUIRED — model or artifact name]

> **Template status:** every bracketed item is an explicit placeholder. Replace it only with evidence from the exact retained run. Use `Not evaluated`, `Not applicable`, or `Unknown — [reason]` when that is the truthful state. Do not infer values, copy them from a different run, or delete unresolved sections.

## Document control

| Field | Value |
| --- | --- |
| Model-card version | [REQUIRED — version] |
| Model/artifact version | [REQUIRED — immutable version or digest] |
| Status | [REQUIRED — draft / experimental / candidate / released / retired] |
| Repository commit | [REQUIRED — full commit SHA] |
| Experiment/run identifier | [REQUIRED — durable identifier] |
| Data-manifest digest | [REQUIRED — algorithm and digest] |
| Model-artifact digest | [REQUIRED — algorithm and digest] |
| Evaluation-artifact digest | [REQUIRED — algorithm and digest] |
| Created | [REQUIRED — ISO 8601 timestamp with timezone] |
| Last reviewed | [REQUIRED — ISO 8601 timestamp with timezone] |
| Owner | [REQUIRED — accountable person or team] |
| Reviewers | [REQUIRED — privacy, domain, ML, security, or `Not reviewed`] |

## 1. Summary

**What it does:** [REQUIRED — one plain-language sentence]

**What it does not do:** [REQUIRED — one plain-language sentence]

**Evidence status:** [REQUIRED — distinguish protocol, local run, candidate artifact, deployed system, and independently verified state]

**Primary result:** [REQUIRED — result with split, denominator, uncertainty, and comparison, or `Not evaluated`]

**Human-review rule:** [REQUIRED — when review occurs, who reviews, and what happens if review is unavailable]

## 2. Model and system details

| Field | Value |
| --- | --- |
| Task | [REQUIRED — task definition] |
| Output labels | [REQUIRED — exact ordered label set and source] |
| Architecture | [REQUIRED — model family and head] |
| Base model | [REQUIRED — exact identifier, or `Not applicable`] |
| Base-model revision | [REQUIRED — immutable revision] |
| Tokenizer | [REQUIRED — identifier and immutable revision] |
| Maximum input length | [REQUIRED — tokens or characters] |
| Framework and versions | [REQUIRED — exact versions] |
| Inference wrapper | [REQUIRED — source path and commit] |
| Calibration method | [REQUIRED — method and fitted-parameter artifact] |
| Review-policy artifact | [REQUIRED — path and digest] |
| Supported hardware | [REQUIRED — verified devices only] |
| License | [REQUIRED — model, code, and data licenses separately] |

### System boundary

Describe what is included in the evaluated system and what sits outside it.

- Included: [REQUIRED — preprocessing, model, calibration, thresholding, label map]
- Excluded: [REQUIRED — UI, reviewer workflow, downstream actions, monitoring, or other components]
- External dependencies: [REQUIRED — APIs, model registries, runtime services, or `None`]
- Deployment state: [REQUIRED — not deployed / local / candidate / production, with evidence]

## 3. Intended use

### Intended users

[REQUIRED — user roles and required expertise]

### Supported use cases

- [REQUIRED — bounded use case]
- [OPTIONAL — additional bounded use case]

### Out-of-scope and prohibited uses

- [REQUIRED — consequential or unsupported decision]
- [REQUIRED — population, language, geography, channel, or time period outside evidence]
- [REQUIRED — automation forbidden without human review]
- [REQUIRED — misuse involving surveillance, eligibility, enforcement, or sensitive inference]

### Input requirements

- Accepted input: [REQUIRED — format, language, length, and validation]
- Rejected input: [REQUIRED — empty, malformed, unsupported, or sensitive cases]
- Sensitive-data handling: [REQUIRED — redaction, retention, access, and logging rules]
- Out-of-distribution behavior: [REQUIRED — detection or explicit lack of detection]

## 4. Data

### Source and authorization

| Field | Value |
| --- | --- |
| Dataset/source | [REQUIRED — exact source name and version/window] |
| Source URL or registry ID | [REQUIRED] |
| License/terms | [REQUIRED — verified terms and access date] |
| Collection window | [REQUIRED — inclusive dates] |
| Retrieval timestamp | [REQUIRED — ISO 8601] |
| Retrieval code | [REQUIRED — path, commit, and arguments] |
| Authorized uses | [REQUIRED — training/evaluation/publication boundaries] |
| Known consent limitations | [REQUIRED] |

### Population and sampling

- Source population: [REQUIRED — who or what can appear]
- Sampling method: [REQUIRED — deterministic/random, strata, caps, and offsets]
- Requested count: [REQUIRED — overall and per class]
- Retrieved count: [REQUIRED — overall and per class]
- Post-filter count: [REQUIRED — overall and per class]
- Exclusions: [REQUIRED — rules and counts]
- Representativeness: [REQUIRED — supported statement or `Not established`]

### Processing and leakage controls

- Text normalization: [REQUIRED]
- Deduplication: [REQUIRED — exact/semantic method, scope, and counts]
- Cross-label conflicts: [REQUIRED — handling and counts]
- Split method: [REQUIRED — chronology, grouping, stratification, or other]
- Split counts: [REQUIRED — train/calibration/test totals and per-class counts]
- Leakage checks: [REQUIRED — checks run and results]
- Known residual leakage risks: [REQUIRED]

### Privacy and sensitivity

- Potentially identifying fields: [REQUIRED]
- Raw-data storage: [REQUIRED — location class, encryption, and access]
- Retention/deletion policy: [REQUIRED]
- Publication policy: [REQUIRED — what is and is not shared]
- Memorization assessment: [REQUIRED — method and result, or `Not evaluated`]
- Privacy review: [REQUIRED — reviewer/date/evidence, or `Not completed`]

## 5. Training

| Field | Value |
| --- | --- |
| Training objective | [REQUIRED] |
| Random seed(s) | [REQUIRED] |
| Epochs/steps | [REQUIRED] |
| Batch size | [REQUIRED] |
| Learning rate and schedule | [REQUIRED] |
| Optimizer | [REQUIRED] |
| Regularization/clipping | [REQUIRED] |
| Class weighting/sampling | [REQUIRED] |
| Hyperparameter-selection process | [REQUIRED — include which split was used] |
| Hardware | [REQUIRED — exact verified device] |
| Training duration | [REQUIRED — measured, not estimated] |
| Energy/carbon measurement | [REQUIRED — value and method, or `Not measured`] |
| Training logs | [REQUIRED — retained artifact and digest] |

### Reproducibility caveats

[REQUIRED — nondeterministic operations, hardware variance, unavailable dependencies, network artifacts, or other limits]

## 6. Evaluation design

### Precommitted claim

- Question: [REQUIRED]
- Primary metric: [REQUIRED]
- Comparator: [REQUIRED]
- Decision threshold: [REQUIRED]
- Protocol freeze timestamp: [REQUIRED]
- Frozen-protocol artifact: [REQUIRED — path and digest]
- Test access policy: [REQUIRED — when and how often the locked test was evaluated]

### Evaluation dataset

- Split: [REQUIRED]
- Time window: [REQUIRED]
- Total denominator: [REQUIRED]
- Per-class denominators: [REQUIRED]
- Differences from training data: [REQUIRED]
- Evaluation exclusions/failures: [REQUIRED — counts and reasons]

### Primary comparison

| Metric | Baseline | Candidate | Difference | Uncertainty | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| [REQUIRED — primary metric] | [VALUE] | [VALUE] | [VALUE] | [METHOD AND INTERVAL] | [SUPPORTED / NOT SUPPORTED] |

If not evaluated, replace the row values with `Not evaluated`; do not remove the table.

### Classification performance

| Metric | Value | Denominator | Uncertainty/method |
| --- | ---: | ---: | --- |
| Accuracy | [VALUE / Not evaluated] | [N] | [INTERVAL OR METHOD] |
| Macro-F1 | [VALUE / Not evaluated] | [N] | [INTERVAL OR METHOD] |
| Weighted F1 | [VALUE / Not evaluated] | [N] | [INTERVAL OR METHOD] |

Per-class results: [REQUIRED — artifact path and digest]

Confusion matrix: [REQUIRED — artifact path and digest, or `Not generated`]

### Calibration

| Measure | Before calibration | After calibration | Split | Denominator |
| --- | ---: | ---: | --- | ---: |
| Negative log likelihood | [VALUE / Not evaluated] | [VALUE / Not evaluated] | [SPLIT] | [N] |
| Expected calibration error | [VALUE / Not evaluated] | [VALUE / Not evaluated] | [SPLIT AND BINNING] | [N] |

- Calibration method: [REQUIRED]
- Calibration parameter(s): [REQUIRED — value plus artifact digest]
- Reliability plot: [REQUIRED — artifact path and digest, or `Not generated`]
- Calibration limitations: [REQUIRED — sample size, bin sensitivity, drift, and classwise gaps]

Do not describe a model as calibrated solely because a calibration method was fitted. Report held-out calibration evidence.

### Human-review and selective-prediction policy

| Field | Calibration selection | Locked test evaluation |
| --- | ---: | ---: |
| Threshold | [VALUE / Not selected] | [SAME FROZEN VALUE] |
| Target accepted accuracy | [VALUE] | Not retuned on test |
| Accepted count | [N] | [N] |
| Coverage | [VALUE] | [VALUE] |
| Accepted accuracy | [VALUE] | [VALUE] |
| Accuracy interval | [METHOD AND INTERVAL] | [METHOD AND INTERVAL] |
| Target met | [YES / NO] | [YES / NO / NOT A TEST DECISION] |

- Fail-closed behavior: [REQUIRED — exact behavior if no threshold qualifies]
- Deferred-case destination: [REQUIRED — reviewer queue and fallback]
- Risk-coverage artifact: [REQUIRED — path and digest]
- Capacity analysis: [REQUIRED — review volume and staffing evidence, or `Not evaluated`]

The target is a selection criterion, not a future-performance guarantee.

## 7. Robustness, fairness, and error analysis

| Evaluation | Population/slice | Denominator | Metric/result | Status |
| --- | --- | ---: | --- | --- |
| Temporal robustness | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |
| Out-of-distribution inputs | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |
| Language/dialect | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |
| Protected-group analysis | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |
| Input perturbations | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |
| Abuse/adversarial testing | [SLICE] | [N] | [RESULT] | [COMPLETE / NOT EVALUATED] |

### Known error patterns

- [REQUIRED — observed confusion or `Not analyzed`]
- [REQUIRED — severity and affected use case]
- [REQUIRED — mitigation and residual risk]

Do not infer demographic fairness from an aggregate score or from the absence of protected attributes.

## 8. Limitations and risks

### Data limitations

- [REQUIRED]

### Model limitations

- [REQUIRED]

### Calibration and review-policy limitations

- [REQUIRED]

### Human-factors limitations

- [REQUIRED — automation bias, reviewer disagreement, capacity, escalation]

### Misuse and downstream harm

- [REQUIRED — plausible misuse]
- [REQUIRED — affected people]
- [REQUIRED — prevention and detection controls]
- [REQUIRED — residual risk]

## 9. Operational controls

Complete this section only for a candidate or deployed system. For an experiment, mark every item `Not implemented — experiment only` where appropriate.

- Access control: [REQUIRED]
- Input/output logging: [REQUIRED — include sensitive-data exclusions]
- Data retention: [REQUIRED]
- Rate/abuse controls: [REQUIRED]
- Human-review ownership and service level: [REQUIRED]
- Escalation and appeal: [REQUIRED]
- Monitoring metrics: [REQUIRED]
- Drift thresholds: [REQUIRED]
- Calibration revalidation trigger: [REQUIRED]
- Rollback procedure: [REQUIRED]
- Incident contact: [REQUIRED]
- Last operational test: [REQUIRED — date and evidence]

## 10. Reproduction and artifact inventory

### Environment

```text
Operating system: [REQUIRED]
Python: [REQUIRED]
Dependency lock or resolved environment digest: [REQUIRED]
Hardware: [REQUIRED]
Network-fetched dependencies and immutable revisions: [REQUIRED]
```

### Commands

```bash
# [REQUIRED — exact environment/setup command]
# [REQUIRED — exact data-retrieval command]
# [REQUIRED — exact training/evaluation command]
# [REQUIRED — exact verification command]
```

Never place credentials, tokens, private paths, raw narratives, or identifying records in this document.

### Artifacts

| Artifact | Purpose | Digest | Retention/access |
| --- | --- | --- | --- |
| [REQUIRED — data manifest] | [PURPOSE] | [DIGEST] | [POLICY] |
| [REQUIRED — metrics] | [PURPOSE] | [DIGEST] | [POLICY] |
| [REQUIRED — model] | [PURPOSE] | [DIGEST] | [POLICY] |
| [REQUIRED — calibration parameters] | [PURPOSE] | [DIGEST] | [POLICY] |
| [REQUIRED — review policy] | [PURPOSE] | [DIGEST] | [POLICY] |
| [REQUIRED — plots/logs] | [PURPOSE] | [DIGEST] | [POLICY] |

## 11. Review and release decision

| Review | Reviewer | Date | Evidence | Decision/open issues |
| --- | --- | --- | --- | --- |
| ML methodology | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |
| Domain | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |
| Privacy | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |
| Security | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |
| Human-review operations | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |
| Final accountable owner | [REQUIRED] | [DATE] | [ARTIFACT] | [DECISION] |

**Release state:** [REQUIRED — not approved / approved for bounded experiment / approved for specified deployment]

**Unresolved blockers:** [REQUIRED — list, or `None` with supporting review evidence]

Approval for an experiment, artifact publication, or demo does not imply approval for production or consequential use.

## 12. Change log

| Date | Model-card version | Model version | Change | Evidence | Author |
| --- | --- | --- | --- | --- | --- |
| [DATE] | [VERSION] | [VERSION] | [CHANGE] | [ARTIFACT/COMMIT] | [AUTHOR] |
