"""Gradio interface for calibrated financial-complaint classification."""

from __future__ import annotations

import threading
from typing import Any

import gradio as gr
from inference import (
    ComplaintRouter,
    ConfigurationError,
    InputValidationError,
    PredictionError,
    RuntimeConfig,
)

_ROUTER: ComplaintRouter | None = None
_ROUTER_KEY: tuple[str, str | None] | None = None
_ROUTER_LOCK = threading.Lock()


def get_router() -> ComplaintRouter:
    """Create one lazy model instance per configured repository revision."""
    global _ROUTER, _ROUTER_KEY
    config = RuntimeConfig.from_environment()
    key = (config.model_id, config.revision)
    with _ROUTER_LOCK:
        if _ROUTER is None or _ROUTER_KEY != key:
            _ROUTER = ComplaintRouter.from_repository(config)
            _ROUTER_KEY = key
    return _ROUTER


def classify_complaint(text: Any) -> tuple[str, str, str, list[list[str]], str]:
    """Return presentation-only fields; never return or log submitted text."""
    try:
        prediction = get_router().predict(text)
    except InputValidationError as exc:
        raise gr.Error(str(exc)) from None
    except ConfigurationError:
        raise gr.Error(
            "This demo is not configured correctly. The model repository or its support files "
            "are unavailable."
        ) from None
    except PredictionError:
        raise gr.Error("Prediction could not be completed. Please try again later.") from None
    except Exception:
        raise gr.Error("Prediction could not be completed. Please try again later.") from None

    if prediction.recommended_action == "human_review":
        route = "HUMAN REVIEW REQUIRED"
        route_detail = (
            f"{prediction.review_reason} Threshold: {prediction.review_threshold:.1%}. "
            "Do not route this item automatically."
        )
    else:
        route = "MODEL-ASSISTED ROUTE CANDIDATE"
        route_detail = (
            f"{prediction.review_reason} Threshold: {prediction.review_threshold:.1%}. "
            "Human confirmation is still required before any consequential action."
        )

    scores = [[score.label, f"{score.probability:.1%}"] for score in prediction.top_predictions]
    return (
        prediction.predicted_product,
        f"{prediction.confidence:.1%}",
        route,
        scores,
        route_detail,
    )


CSS = """
:root {
  --ink: #132238;
  --muted: #52657a;
  --line: #dbe4ea;
  --paper: #ffffff;
  --wash: #f4f8f8;
  --accent: #0f766e;
}
.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  color: var(--ink);
}
.hero {
  background: linear-gradient(135deg, #0f2735 0%, #154e55 58%, #0f766e 100%);
  border-radius: 24px;
  color: white;
  padding: 36px 38px 32px;
  margin: 14px 0 18px;
  box-shadow: 0 16px 45px rgba(15, 39, 53, 0.15);
}
.hero h1 { color: white; font-size: 2.2rem; margin: 8px 0 10px; }
.hero p { color: #d8eeee; font-size: 1.02rem; max-width: 760px; margin: 0; }
.eyebrow { font-size: .76rem; font-weight: 700; letter-spacing: .13em; opacity: .86; }
.privacy-note, .limitations {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--wash);
  padding: 16px 18px;
}
.privacy-note h3, .limitations h3 { margin-top: 0; color: var(--ink); }
.result-card { border: 1px solid var(--line) !important; border-radius: 18px !important; }
.route-note {
  border-left: 4px solid var(--accent);
  padding: 8px 14px;
  background: #f0fdfa;
  border-radius: 8px;
}
.footer-note { color: var(--muted); font-size: .86rem; text-align: center; margin-top: 12px; }
"""


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
    )
    with gr.Blocks(
        title="Financial Complaint Intelligence",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
        delete_cache=(3_600, 3_600),
    ) as demo:
        gr.HTML(
            """
            <section class="hero">
              <div class="eyebrow">PORTFOLIO DEMONSTRATION</div>
              <h1>Financial Complaint Intelligence</h1>
              <p>Explore a calibrated product-category prediction and see when uncertainty
              routes a complaint to human review.</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                complaint = gr.Textbox(
                    label="Complaint context",
                    placeholder="Describe a fictional or fully de-identified complaint…",
                    lines=9,
                    max_lines=14,
                    info=(
                        "20–6,000 characters. Do not include names, account numbers, "
                        "or other personal data."
                    ),
                )
                with gr.Row():
                    submit = gr.Button("Analyze complaint", variant="primary", scale=3)
                    clear = gr.ClearButton([complaint], value="Clear", scale=1)
                gr.HTML(
                    """
                    <aside class="privacy-note">
                      <h3>Privacy first</h3>
                      <p>Use fictional or fully de-identified text only. This app does not
                      intentionally log or persist submitted complaint text. Hosting and network
                      platform handling are outside this app’s controls.</p>
                    </aside>
                    """
                )

            with gr.Column(scale=5, elem_classes="result-card"):
                product = gr.Textbox(label="Predicted product category", interactive=False)
                confidence = gr.Textbox(label="Calibrated confidence", interactive=False)
                route = gr.Textbox(label="Review route", interactive=False)
                top_scores = gr.Dataframe(
                    headers=["Product category", "Calibrated score"],
                    datatype=["str", "str"],
                    row_count=(3, "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    label="Top three class scores",
                )
                route_detail = gr.Markdown(elem_classes="route-note")

        gr.HTML(
            """
            <section class="limitations">
              <h3>How to interpret this demo</h3>
              <ul>
                <li>Confidence is temperature-calibrated on the experiment’s held-out
                calibration split; it is not a guarantee that a prediction is correct.</li>
                <li>The classifier covers only its frozen product categories. It does not infer
                intent, urgency, eligibility, wrongdoing, or an appropriate resolution.</li>
                <li>Public complaint data may not represent other populations or future data,
                and performance can change when language or conditions change.</li>
                <li>This is not a production or consequential decision system. Human review is
                required for low-confidence items and before any consequential action.</li>
              </ul>
            </section>
            <p class="footer-note">No examples are preloaded to avoid presenting private or
            realistic complaint narratives as demo content.</p>
            """
        )

        outputs = [product, confidence, route, top_scores, route_detail]
        submit.click(
            fn=classify_complaint,
            inputs=complaint,
            outputs=outputs,
            api_name=False,
            show_progress="minimal",
        )
        complaint.submit(
            fn=classify_complaint,
            inputs=complaint,
            outputs=outputs,
            api_name=False,
            show_progress="minimal",
        )
        clear.add(outputs)

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20).launch(show_error=False)
