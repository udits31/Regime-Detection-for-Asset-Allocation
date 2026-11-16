"""Appends entries to results/config_log.csv so every configuration that is run is counted."""
from __future__ import annotations

import csv
import datetime as dt

from regime_eval.config import ROOT

LOG_PATH = ROOT / "results" / "config_log.csv"


def append_log(kind: str, item: str, detail: str, source: str, outcome: str) -> int:
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    next_id = max(int(r[0]) for r in rows[1:] if r and r[0].isdigit()) + 1
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerow([next_id, dt.date.today().isoformat(), kind, item, detail, source, outcome])
    return next_id
