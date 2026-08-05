"""Offline checks for the Gradio presentation layer."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

SPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_DIR))

import app  # noqa: E402


def test_demo_builds_without_loading_a_model() -> None:
    assert isinstance(app.demo, gr.Blocks)
    assert app._ROUTER is None
