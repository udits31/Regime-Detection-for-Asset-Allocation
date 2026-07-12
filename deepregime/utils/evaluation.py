# ============================================================
# utils/evaluation.py
# Phase IV: Evaluation Framework
#
# Reviewer's key statistical point:
# "The assumption that better forecasting automatically produces
# better portfolio performance is GENERALLY FALSE."
#
# A model may achieve lower RMSE while generating inferior
# portfolio returns. Forecasting and portfolio performance
# MUST be evaluated SEPARATELY.
#
# This module provides:
#   • ForecastEvaluator  — MAE, RMSE, MAPE, Directional Accuracy
#   • PortfolioEvaluator — Sharpe, Sortino, Calmar, VaR, CVaR
#   • CycleAnalyzer      — Performance across Indian economic cycles
#   • RegimeEvaluator    — Regime quality metrics
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


RISK_FREE = 0.065

# Economic cycles defined per reviewer's Phase IV requirements
ECONOMIC_CYCLES = {
    "Pre-GFC Bull":       ("2004-01-01", "2008-01-01"),
    "GFC Crisis":         ("2008-01-01", "2009-03-31"),
    "GFC Recovery":       ("2009-04-01", "2010-12-31"),
    "Taper Tantrum":      ("2013-05-01", "2013-09-30"),
    "Bull 2014-2018":     ("2014-05-01", "2018-09-30"),
    "Demonetisation":     ("2016-11-01", "2017-03-31"),
    "Pre-COVID Bull":     ("2019-01-01", "2020-01-31"),
    "COVID Crash":        ("2020-02-01", "2020-04-30"),
    "COVID Recovery":     ("2020-05-01", "2021-12-31"),
    "RBI Tightening":     ("2022-04-01", "2023-06-30"),
    "Post-Tightening":    ("2023-07-01", "2024-12-31"),
}


class ForecastEvaluator:
    """
    Evaluate regime prediction / return forecasting quality.
    Implements reviewer's recommended forecast metrics.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices:  Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute all forecasting metrics.

        Parameters
        ──────────
        y_true : Actual values (returns or regime label encoding)
        y_pred : Predicted values
        prices : Optional price series for directional accuracy

        Returns
        ───────
        MAE, RMSE, MAPE, Directional Accuracy, Hit Rate
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae  = float(np.abs(y_true - y_pred).mean())
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        # MAPE — skip zeros
        nonzero = y_true != 0
        mape = float(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]).mean()) if nonzero.any() else np.nan

        # Directional accuracy — fraction of correct sign predictions
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if len(y_true) > 1 else np.nan

        # Information Coefficient (rank correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_true, y_pred)

        return {
            "mae":               round(mae,     6),
            "rmse":              round(rmse,    6),
            "mape_pct":          round(mape * 100, 4) if not np.isnan(mape) else None,
            "directional_acc":   round(dir_acc, 4),
            "information_coeff": round(float(ic), 4) if not np.isnan(ic) else None,
            "n_observations":    int(len(y_true)),
            "note": (
                "Better MAE/RMSE does NOT guarantee better portfolio returns. "
                "Evaluate portfolio metrics separately."
            ),
        }


class PortfolioEvaluator:
    """
    Full portfolio performance evaluation.
    Implements reviewer's recommended portfolio metrics.
    """

    @staticmethod
    def evaluate(
        equity_curve:    pd.Series,
        benchmark:       Optional[pd.Series] = None,
        risk_free_rate:  float = RISK_FREE,
    ) -> Dict:
        """
        Compute comprehensive portfolio metrics.

        Returns
        ───────
        CAGR, Volatility, Sharpe, Sortino, Calmar,
        Max Drawdown, VaR 95/99, CVaR 95/99, Omega,
        Alpha/Beta vs benchmark (if provided)
        """
        ret   = equity_curve.pct_change().dropna()
        n_yr  = len(equity_curve) / 252
        cagr  = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/max(n_yr,0.1)) - 1
        vol   = ret.std() * np.sqrt(252)
        rf_d  = risk_free_rate / 252

        sharpe  = (cagr - risk_free_rate) / (vol + 1e-10)
        down    = ret[ret < rf_d]
        sortino = (cagr - risk_free_rate) / (down.std() * np.sqrt(252) + 1e-10)

        roll_max = equity_curve.cummax()
        dd       = (equity_curve - roll_max) / roll_max
        max_dd   = float(dd.min())
        calmar   = cagr / (abs(max_dd) + 1e-10)

        # VaR / CVaR
        var95  = float(np.percentile(ret, 5))
        cvar95 = float(ret[ret <= var95].mean()) if (ret <= var95).any() else var95
        var99  = float(np.percentile(ret, 1))
        cvar99 = float(ret[ret <= var99].mean()) if (ret <= var99).any() else var99

        # Omega ratio
        pos  = ret[ret > 0].sum()
        neg  = abs(ret[ret < 0].sum())
        omega = float(pos / neg) if neg > 1e-10 else float("inf")

        # Portfolio turnover proxy (via equity curve)
        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        win_rate = float((monthly > 0).mean())

        metrics = {
            "cagr":            round(cagr,    4),
            "annual_vol":      round(vol,     4),
            "sharpe_ratio":    round(sharpe,  4),
            "sortino_ratio":   round(sortino, 4),
            "calmar_ratio":    round(calmar,  4),
            "max_drawdown":    round(max_dd,  4),
            "var_95_daily":    round(var95,   4),
            "cvar_95_daily":   round(cvar95,  4),
            "var_99_daily":    round(var99,   4),
            "cvar_99_daily":   round(cvar99,  4),
            "omega_ratio":     round(omega,   4),
            "monthly_win_rate":round(win_rate,4),
            "final_value":     round(float(equity_curve.iloc[-1]), 2),
            "n_years":         round(n_yr,    2),
        }

        # Alpha / Beta vs benchmark
        if benchmark is not None:
            bench_ret = benchmark.pct_change().dropna()
            aligned   = pd.concat([ret, bench_ret], axis=1).dropna()
            if len(aligned) > 20:
                aligned.columns = ["port", "bench"]
                cov_   = aligned.cov().values
                beta   = cov_[0, 1] / (cov_[1, 1] + 1e-10)
                b_cagr = (benchmark.iloc[-1]/benchmark.iloc[0]) ** (1/max(n_yr,0.1)) - 1
                alpha  = cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate))
                metrics["alpha_annualised"] = round(alpha, 4)
                metrics["beta"]             = round(beta,  4)

        return metrics

    @staticmethod
    def cycle_breakdown(
        equity_curve: pd.Series,
        label:        str = "Strategy",
    ) -> pd.DataFrame:
        """Performance breakdown across Indian economic cycles."""
        rows = []
        for cycle, (start, end) in ECONOMIC_CYCLES.items():
            mask = (equity_curve.index >= start) & (equity_curve.index <= end)
            seg  = equity_curve[mask]
            if len(seg) < 5:
                continue
            ret = seg.pct_change().dropna()
            n   = len(seg) / 252
            cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/max(n,0.1)) - 1 if len(seg) > 1 else 0
            vol  = ret.std() * np.sqrt(252)
            rows.append({
                "Cycle":        cycle,
                "CAGR (%)":     round(cagr * 100, 2),
                "Vol (%)":      round(vol  * 100, 2),
                "Sharpe":       round((cagr - RISK_FREE) / (vol + 1e-10), 3),
                "n_days":       int(mask.sum()),
            })
        return pd.DataFrame(rows).set_index("Cycle")


class RegimeEvaluator:
    """
    Evaluate the quality of regime assignments.

    Metrics
    ───────
    • Regime persistence (mean duration)
    • Transition entropy (lower = more stable regimes)
    • Conditional Sharpe per regime
    • Economic cycle coverage
    """

    @staticmethod
    def regime_conditional_metrics(
        labels: pd.Series, returns: pd.Series
    ) -> pd.DataFrame:
        """Per-regime return statistics."""
        merged = pd.DataFrame({
            "regime":  labels,
            "returns": returns.reindex(labels.index),
        }).dropna()

        rows = []
        for regime, grp in merged.groupby("regime"):
            ann = grp["returns"].mean() * 252
            vol = grp["returns"].std()  * np.sqrt(252)
            dd  = grp["returns"].min()
            rows.append({
                "Regime":     regime,
                "Obs":        len(grp),
                "Weight (%)": round(len(grp) / len(merged) * 100, 1),
                "Ann Ret (%)":round(ann * 100, 2),
                "Ann Vol (%)":round(vol * 100, 2),
                "Sharpe":     round((ann - RISK_FREE) / (vol + 1e-10), 3),
                "Worst Day (%)": round(dd * 100, 2),
            })
        return pd.DataFrame(rows).set_index("Regime").sort_values("Ann Ret (%)", ascending=False)
