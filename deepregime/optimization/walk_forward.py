# ============================================================
# optimization/walk_forward.py
# Walk-Forward Cross-Validation for Time-Series Models
#
# FIXED (vs original): This class was completely disconnected
# from the backtest engine.  It now:
#   • Returns price DataFrames (not feature DataFrames) so the
#     backtest can call FeatureBuilder inside each fold
#   • Supports both expanding and rolling windows
#   • Provides a fold summary report
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator
from dataclasses import dataclass
from loguru import logger


@dataclass
class FoldInfo:
    fold_idx:    int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_train:     int
    n_test:      int


class WalkForwardOptimizer:
    """
    Chronological walk-forward splitter for time-series backtesting.

    Prevents look-ahead bias by ensuring each test window is evaluated
    using only information available at the start of that window.

    Modes
    ─────
    Expanding (default):
      Train on all data from start to fold boundary.
      Each fold has more training data than the previous one.

    Rolling:
      Train on a fixed-size window ending at fold boundary.
      Useful for non-stationary processes where distant history hurts.

    Example (expanding, n_splits=3, test_window=63):

      |——————— Train 1 ———————|— Test 1 —|
      |————————— Train 2 —————————|— Test 2 —|
      |—————————————— Train 3 ——————————————|— Test 3 —|
    """

    def __init__(
        self,
        n_splits:     int  = 5,
        train_window: int  = 504,   # ~2 years (rolling mode)
        test_window:  int  = 63,    # ~3 months
        min_train:    int  = 252,   # Minimum 1 year
        expanding:    bool = True,
    ):
        self.n_splits     = n_splits
        self.train_window = train_window
        self.test_window  = test_window
        self.min_train    = min_train
        self.expanding    = expanding

    # ── Splitting ─────────────────────────────────────────────────────────────

    def split(
        self, data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate (train, test) price DataFrame pairs.

        Parameters
        ──────────
        data : Full price DataFrame with DatetimeIndex.

        Returns
        ───────
        List of (train_df, test_df) tuples, each a contiguous slice
        of `data` with no overlap between consecutive test windows.
        """
        n = len(data)
        if n < self.min_train + self.test_window:
            raise ValueError(
                f"Dataset too short: {n} rows, need ≥ "
                f"{self.min_train + self.test_window}"
            )

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        fold_infos: List[FoldInfo] = []

        # Calculate test window start points evenly spaced in the usable range
        usable_end   = n - self.test_window
        usable_start = self.min_train
        step = max(1, (usable_end - usable_start) // self.n_splits)

        for k in range(self.n_splits):
            test_start_idx = usable_start + k * step
            test_end_idx   = min(test_start_idx + self.test_window, n)

            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = max(0, test_start_idx - self.train_window)

            train_df = data.iloc[train_start_idx: test_start_idx]
            test_df  = data.iloc[test_start_idx: test_end_idx]

            if len(train_df) < self.min_train or len(test_df) == 0:
                continue

            folds.append((train_df, test_df))
            fold_infos.append(FoldInfo(
                fold_idx    = k,
                train_start = data.index[train_start_idx],
                train_end   = data.index[test_start_idx - 1],
                test_start  = data.index[test_start_idx],
                test_end    = data.index[test_end_idx - 1],
                n_train     = len(train_df),
                n_test      = len(test_df),
            ))

        logger.info(
            f"Walk-forward split | mode={'expanding' if self.expanding else 'rolling'} "
            f"| folds={len(folds)} | test_window={self.test_window}"
        )
        self._log_fold_summary(fold_infos)
        return folds

    def split_generator(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Generator version for memory-efficient iteration."""
        for fold in self.split(data):
            yield fold

    # ── Cross-validation evaluation ───────────────────────────────────────────

    def cross_validate(
        self,
        data:       pd.DataFrame,
        score_fn,
        train_fn,
    ) -> pd.DataFrame:
        """
        Run a scoring function across all folds.

        Parameters
        ──────────
        data     : Full price DataFrame.
        score_fn : Callable(train_df, test_df) → Dict[str, float]
                   Returns a dict of metric_name → value for this fold.
        train_fn : Callable(train_df) → model
                   Trains and returns the model for this fold.

        Returns
        ───────
        DataFrame with one row per fold and one column per metric.
        """
        rows = []
        for idx, (train_df, test_df) in enumerate(self.split(data)):
            try:
                model  = train_fn(train_df)
                scores = score_fn(model, train_df, test_df)
                scores["fold"]          = idx
                scores["train_start"]   = train_df.index[0].date()
                scores["test_end"]      = test_df.index[-1].date()
                rows.append(scores)
            except Exception as e:
                logger.warning(f"Fold {idx} evaluation failed: {e}")
                continue

        if not rows:
            raise RuntimeError("No folds evaluated successfully.")
        return pd.DataFrame(rows).set_index("fold")

    def fold_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame of fold boundaries without running models."""
        rows = []
        for idx, (train_df, test_df) in enumerate(self.split(data)):
            rows.append({
                "fold":         idx + 1,
                "train_start":  train_df.index[0].date(),
                "train_end":    train_df.index[-1].date(),
                "train_days":   len(train_df),
                "test_start":   test_df.index[0].date(),
                "test_end":     test_df.index[-1].date(),
                "test_days":    len(test_df),
            })
        return pd.DataFrame(rows)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_fold_summary(self, infos: List[FoldInfo]) -> None:
        for fi in infos:
            logger.debug(
                f"  Fold {fi.fold_idx+1} | "
                f"train: {fi.train_start.date()} → {fi.train_end.date()} "
                f"({fi.n_train}d) | "
                f"test:  {fi.test_start.date()} → {fi.test_end.date()} "
                f"({fi.n_test}d)"
            )
