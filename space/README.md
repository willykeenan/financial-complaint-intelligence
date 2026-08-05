---
title: Financial Complaint Intelligence
emoji: 🧭
colorFrom: slate
colorTo: teal
sdk: gradio
sdk_version: 5.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Calibrated complaint classification with human-review routing
---

# Financial Complaint Intelligence

This Space presents a calibrated product-category prediction, the top three class scores,
and a fail-closed human-review route. It is a portfolio demonstration, not a production or
consequential decision system.

## Required model repository contract

Set the Space variable `MODEL_ID` to the Hugging Face repository containing the trained
sequence-classification model and tokenizer. The repository must also contain the frozen
experiment artifacts:

- `temperature.json` with a numeric `temperature` between `0.05` and `10.0`
- `review_policy.json` with a `threshold` from `0` to `1` and a boolean `target_met`
- `label_map.json` with contiguous string class IDs beginning at `0`

`MODEL_REVISION` is optional and can pin a branch, tag, or commit. For a private model
repository, store `HF_TOKEN` as a Space secret rather than a public variable. The app loads
the model lazily on the first prediction; importing the app does not fetch model files.

The interface uses the experiment’s fixed 192-token inference limit. It applies scalar
temperature scaling before reporting confidence and routes every item to human review when
the frozen policy did not meet its target or the calibrated confidence is below its threshold.

## Privacy and limitations

- Use only fictional or fully de-identified text. Do not enter names, account numbers, or
  other personal data.
- This application does not intentionally log, persist, or reproduce submitted complaint
  text. Hosting and network platform handling are outside the application’s controls.
- No complaint examples are bundled with the Space.
- Calibrated confidence is not a guarantee of correctness and may not remain calibrated
  outside the experiment data.
- The model predicts only its frozen product categories. It does not infer intent, urgency,
  wrongdoing, eligibility, or a resolution.
- Public complaint data may not represent other populations, protected subgroups, live
  traffic, or future conditions.
- Human review is mandatory for low-confidence items and before any consequential action.

## Local static checks

These checks do not download a model:

```bash
python -m compileall -q app.py inference.py tests
python -m unittest discover -s tests -v
```
