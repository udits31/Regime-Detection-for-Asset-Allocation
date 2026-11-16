"""Loads the frozen settings. Every module reads its parameters from here."""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path(__file__).with_name("settings.toml")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel.csv"


def load_settings() -> dict:
    with open(SETTINGS_PATH, "rb") as f:
        return tomllib.load(f)
