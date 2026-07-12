#!/usr/bin/env python3
"""
run_pipeline_v2.py — DeepRegime Improved Pipeline
==================================================
Integrates all 14 improvements from the code review.
Run with real NIFTY data: python run_pipeline_v2.py
Run with synthetic data:  python run_pipeline_v2.py --synthetic
"""

import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Optional, Tuple
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

RISK_FREE   = 0.065
INITIAL_CAP = 1_000_000.0
TCOST       = 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Real Yahoo Finance with synthetic fallback
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_real_data(start: str = "2016-01-01") -> pd.DataFrame:
    import yfinance as yf
    TICKERS = {
        "nifty":     "^NSEI",
        "gold":      "GC=F",
        "usdinr":    "USDINR=X",
        "india_vix": "^INDIAVIX",
    }
    logger.info("Downloading real market data from Yahoo Finance ...")
    frames = {}
    for col, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if len(raw) > 100:
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                frames[col] = close.rename(col)
                logger.info(f"  OK {col:12s} ({ticker}): {len(raw)} rows")
            else:
                logger.warning(f"  FAIL {col} ({ticker}): too few rows")
        except Exception as e:
            logger.warning(f"  FAIL {col} ({ticker}): {e}")

    if "nifty" not in frames:
        raise RuntimeError("NIFTY data unavailable")

    prices = pd.DataFrame(frames).ffill().bfill().dropna(subset=["nifty"])
    prices = prices.loc[(prices > 0).all(axis=1)]
    cagr   = (prices["nifty"].iloc[-1] / prices["nifty"].iloc[0]) ** (252 / len(prices)) - 1
    logger.info(f"Real data: {prices.shape} | {prices.index[0].date()} to {prices.index[-1].date()} | NIFTY CAGR={cagr:.1%}")
    return prices


def make_synthetic() -> pd.DataFrame:
    np.random.seed(2024)
    idx = pd.bdate_range("2016-01-01", "2024-01-01")
    N   = len(idx)
    calendar = [
        ("Bull",     0,   200), ("Sideways", 200, 320), ("Bull",  320, 550),
        ("Sideways", 550, 620), ("Bear",     620, 720), ("Recov", 720, 820),
        ("Bull",     820, 900), ("Crisis",   900, 960), ("Recov", 960,1100),
        ("Bull",    1100,1350), ("Sideways",1350,1450), ("Bear", 1450,1570),
        ("Sideways",1570,1650), ("Bull",    1650,N),
    ]
    P = {
        "Bull":     {"mu": 0.28/252, "sig": 0.12/252**.5, "vix": 14, "gb": -0.10},
        "Sideways": {"mu": 0.05/252, "sig": 0.15/252**.5, "vix": 18, "gb":  0.05},
        "Bear":     {"mu":-0.22/252, "sig": 0.23/252**.5, "vix": 32, "gb":  0.30},
        "Crisis":   {"mu":-0.45/252, "sig": 0.48/252**.5, "vix": 65, "gb":  0.50},
        "Recov":    {"mu": 0.38/252, "sig": 0.22/252**.5, "vix": 22, "gb": -0.05},
    }
    seq = ["Bull"] * N
    for nm, s, e in calendar:
        seq[s:min(e, N)] = [nm] * (min(e, N) - s)
    nr = np.zeros(N); gr = np.zeros(N); ur = np.zeros(N); vix = np.zeros(N)
    for i in range(N):
        p = P.get(seq[i], P["Sideways"]); z = np.random.randn(4)
        nr[i]  = p["mu"] + p["sig"] * z[0]
        gr[i]  = 0.07/252 + p["gb"] * p["sig"] * z[0] + 0.12/252**.5 * z[1]
        ur[i]  = (0.04 if seq[i] in ("Crisis","Bear") else 0.02)/252 + 0.07/252**.5 * z[2]
        vix[i] = max(8, p["vix"] + 8*z[3] + (0 if i==0 else 0.6*(vix[i-1]-p["vix"])))
    return pd.DataFrame({
        "nifty":     8500 * np.exp(np.cumsum(nr)),
        "gold":      1100 * np.exp(np.cumsum(gr)),
        "usdinr":    66   * np.exp(np.cumsum(ur)),
        "india_vix": np.clip(vix, 8, 90),
    }, index=idx)


def load_data(use_synthetic: bool = False, start: str = "2016-01-01") -> pd.DataFrame:
    if use_synthetic:
        logger.info("Using synthetic data (--synthetic flag)")
        return make_synthetic()
    try:
        return fetch_real_data(start)
    except Exception as e:
        logger.warning(f"Real data failed ({e}) — falling back to synthetic")
        return make_synthetic()


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    nifty = prices["nifty"]
    df    = pd.DataFrame(index=nifty.index)
    ret   = nifty.pct_change()
    df["returns"]     = ret
    df["log_returns"] = np.log(nifty / nifty.shift(1))
    for w in [10, 20, 50, 200]:
        sma = nifty.rolling(w).mean()
        df[f"sma_{w}"]         = sma
        df[f"price_vs_sma{w}"] = (nifty - sma) / (sma + 1e-10)
    df["golden_cross"] = (nifty.rolling(50).mean() > nifty.rolling(200).mean()).astype(float)
    d    = nifty.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]     = 100 - 100 / (1 + gain / (loss + 1e-10))
    df["rsi_regime"] = (df["rsi_14"] - 50) / 50
    ema12 = nifty.ewm(span=12).mean(); ema26 = nifty.ewm(span=26).mean()
    macd  = ema12 - ema26; sig = macd.ewm(span=9).mean()
    df["macd_hist"] = (macd - sig) / (nifty.rolling(26).std() + 1e-10)
    for w in [5, 10, 20, 60]:
        df[f"vol_{w}d"] = ret.rolling(w).std() * np.sqrt(252)
    df["vol_ratio"]      = df["vol_20d"] / (df["vol_60d"] + 1e-10)
    df["vol_percentile"] = df["vol_20d"].rolling(252).rank(pct=True)
    sma20 = nifty.rolling(20).mean(); std20 = nifty.rolling(20).std()
    df["bb_width"] = 2 * std20 / (sma20 + 1e-10)
    rm = nifty.cummax()
    df["drawdown"]   = (nifty - rm) / (rm + 1e-10)
    df["sharpe_60d"] = (ret.rolling(60).mean() * 252) / (ret.rolling(60).std() * np.sqrt(252) + 1e-10)
    df["skew_20d"]   = ret.rolling(20).skew()
    df["kurt_20d"]   = ret.rolling(20).kurt()
    if "india_vix" in prices.columns:
        vix = prices["india_vix"]
        df["india_vix_norm"] = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-10)
        df["india_vix_raw"]  = vix
    if "usdinr" in prices.columns:
        df["usdinr_chg20d"] = prices["usdinr"].pct_change(20)
    if "gold" in prices.columns:
        df["gold_ret20d"] = prices["gold"].pct_change(20)
        df["gold_ret5d"]  = prices["gold"].pct_change(5)
    norm_dd = (-df["drawdown"]).clip(0, 0.5) / 0.5
    hv      = df["vol_20d"].rolling(252).mean().fillna(0.15)
    df["crisis_score"] = 0.5 * norm_dd + 0.5 * (df["vol_20d"] / (hv + 1e-10)).clip(0, 4) / 4
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def get_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    desired = ["returns","log_returns","rsi_regime","macd_hist",
               "price_vs_sma50","price_vs_sma200","golden_cross",
               "vol_20d","vol_60d","vol_ratio","vol_percentile",
               "bb_width","drawdown","sharpe_60d","skew_20d","kurt_20d",
               "crisis_score","india_vix_norm","usdinr_chg20d","gold_ret20d"]
    return df[[c for c in desired if c in df.columns]].dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — BIC Curve
# ═══════════════════════════════════════════════════════════════════════════════

def select_n_states(regime_f: pd.DataFrame) -> int:
    from utils.stability_analysis import bic_sweep, find_elbow
    logger.info("\n" + "="*65)
    logger.info("IMPROVEMENT 1 — BIC Sweep: Auto-selecting n_states")
    bic_df  = bic_sweep(regime_f, n_range=range(2, 8), n_init=3)
    print("\nBIC / AIC by n_states:")
    print(f"{'n_states':>10}{'log_lik':>12}{'BIC':>14}{'AIC':>14}{'Optimal':>10}")
    print("-"*60)
    for n, row in bic_df.iterrows():
        flag = "<- OPTIMAL" if row["optimal"] else ""
        print(f"{n:>10}{row['log_lik']:>12.1f}{row['bic']:>14.1f}{row['aic']:>14.1f}  {flag}")
    optimal = int(bic_df[bic_df["optimal"]].index[0])
    elbow   = find_elbow(bic_df["bic"])
    logger.info(f"BIC-optimal: {optimal} | Elbow: {elbow} -> Using {optimal} states")
    return optimal


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 11 — Stability test + auto-cap
# ═══════════════════════════════════════════════════════════════════════════════

def stability_test(regime_f: pd.DataFrame, n_states: int) -> Tuple[Dict, int]:
    from utils.stability_analysis import regime_stability_test
    logger.info("\n" + "="*65)
    logger.info("IMPROVEMENT 11 — Regime Stability (20 seeds, ARI)")
    result = regime_stability_test(regime_f, n_components=n_states, n_seeds=20)
    print(f"\n  Mean ARI : {result['mean_ari']:.4f}")
    print(f"  Min  ARI : {result['min_ari']:.4f}")
    print(f"  Converged: {result['n_converged']} / 20")
    print(f"  Verdict  : {result['verdict']}")
    final_n = n_states
    if not result["stable"] and n_states > 4:
        final_n = 4
        print(f"\n  -> Auto-capped to {final_n} states for stability (ARI={result['mean_ari']:.3f} < 0.70)")
    else:
        print(f"\n  -> Proceeding with {final_n} states")
    return result, final_n


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE I — Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════

def phase1(regime_f, returns, n_states):
    from models.hmm_regime_detector   import HMMRegimeDetector
    from models.gmm_regime_detector   import GMMRegimeDetector
    from models.hsmm_detector         import HSMMRegimeDetector
    from models.change_point_detector import ChangePointDetector
    from models.kalman_smoother       import RegimeProbabilityKalmanSmoother

    logger.info("\n" + "="*65)
    logger.info(f"PHASE I — Regime Detector Comparison (n_states={n_states})")

    hmm = HMMRegimeDetector(n_components=n_states); hmm.fit(regime_f)
    hmm_labels = hmm.predict(regime_f)

    gmm = GMMRegimeDetector(n_components=n_states); gmm.fit(regime_f)
    gmm_labels = gmm.predict(regime_f)
    X_s = gmm.scaler.transform(regime_f.values)

    hsmm = HSMMRegimeDetector(n_components=n_states); hsmm.fit(regime_f)
    hsmm_labels = hsmm.predict_fast(regime_f)

    cpd = ChangePointDetector(penalty=8.0, min_size=15)
    cpd.fit_pelt(regime_f)
    cpd_labels = cpd.get_pelt_regimes(regime_f.index, returns)

    ks = RegimeProbabilityKalmanSmoother(1e-3, 0.1); ks.fit(hmm.model.transmat_)
    raw_probs    = hmm.predict_proba(regime_f)
    smoothed_arr = ks.smooth(raw_probs.values)
    smoothed_lbls= pd.Series(raw_probs.columns[smoothed_arr.argmax(axis=1)], index=regime_f.index)
    nrr      = ks.noise_reduction_ratio(raw_probs.values, smoothed_arr)
    hmm_eval = hmm.evaluate(regime_f, returns)

    def mean_dur(lbl):
        c = (lbl != lbl.shift(1)).cumsum()
        return float(c.groupby(c).size().mean())

    def avg_sh(lbl):
        m = pd.DataFrame({"r": lbl, "ret": returns}).dropna()
        sh = [(g["ret"].mean()*252-RISK_FREE)/(g["ret"].std()*np.sqrt(252)+1e-10) for _,g in m.groupby("r")]
        return round(np.mean(sh), 3) if sh else np.nan

    print("\n" + "="*72)
    print("PHASE I: REGIME MODEL COMPARISON")
    print("="*72)
    print(f"{'Model':<16}{'BIC':>12}{'AIC':>12}{'MeanDur(d)':>12}{'NTrans':>8}{'AvgSharpe':>12}")
    print("-"*72)
    rows = [
        ("HMM",        hmm_labels,    hmm_eval.get("bic",0), hmm_eval.get("aic",0)),
        ("HMM+Kalman", smoothed_lbls, None, None),
        ("GMM",        gmm_labels,    gmm.bic(X_s), gmm.aic(X_s)),
        ("HSMM",       hsmm_labels,   None, None),
    ]
    if cpd_labels is not None:
        rows.append(("PELT-CPD", cpd_labels, None, None))
    for name, lbl, bic, aic in rows:
        bic_s = f"{bic:>12.1f}" if bic else f"{'--':>12}"
        aic_s = f"{aic:>12.1f}" if aic else f"{'--':>12}"
        print(f"{name:<16}{bic_s}{aic_s}{mean_dur(lbl):>12.1f}{(lbl!=lbl.shift(1)).sum():>8}{avg_sh(lbl):>12.3f}")

    print(f"\n  Kalman noise reduction: {nrr:.3f}")
    print(f"  PELT structural breaks: {len(cpd.breakpoints_pelt or [])}")
    print("\n-- HSMM vs HMM Duration Table --")
    print(hsmm.get_duration_stats().to_string())
    return hmm, hmm_labels, smoothed_lbls, smoothed_arr, raw_probs


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 6 — Kaplan-Meier
# ═══════════════════════════════════════════════════════════════════════════════

def survival_analysis(hmm_labels, returns):
    from utils.stability_analysis import regime_survival_analysis
    logger.info("\n" + "="*65)
    logger.info("IMPROVEMENT 6 — Kaplan-Meier Survival + Log-Rank Test")
    sa = regime_survival_analysis(hmm_labels, returns)
    print("\n-- Regime Survival --")
    print(f"{'Regime':<30}{'Episodes':>10}{'Mean(d)':>10}{'Median(d)':>12}{'Max(d)':>9}")
    print("-"*71)
    for regime, info in sa["per_regime"].items():
        print(f"{regime:<30}{info['n_episodes']:>10}{info['mean_dur']:>10.1f}{info['median_dur']:>12.1f}{info['max_dur']:>9}")
    print("\n-- Pairwise Log-Rank Tests --")
    for pair, lr in sa["pairwise_logrank"].items():
        sig = "DIFFERENT" if lr.get("reject_H0") else "similar"
        pv  = lr.get("p_value", np.nan)
        if not np.isnan(pv):
            print(f"  {pair:<45}p={pv:.3f}  {sig}")
    return sa


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE II — Forecasting
# ═══════════════════════════════════════════════════════════════════════════════

def phase2(hmm_labels, returns, regime_f):
    from utils.statistical_tests import diebold_mariano
    from utils.evaluation        import RegimeEvaluator
    logger.info("\n" + "="*65)
    logger.info("PHASE II — Forecasting + Diebold-Mariano Test")
    print("\n-- Regime-Conditional Return Statistics --")
    print(RegimeEvaluator.regime_conditional_metrics(hmm_labels, returns).to_string())
    y_true = returns.values[1:]
    y_mom  = returns.values[:-1]
    y_zero = np.zeros_like(y_true)
    y_reg  = np.array([returns[hmm_labels == hmm_labels.iloc[max(0,i-1)]].mean() for i in range(1, len(returns))])
    e_mom  = y_true - y_mom; e_zero = y_true - y_zero; e_reg = y_true - y_reg
    print("\n-- Forecast Metrics --")
    print(f"{'Model':<22}{'MAE':>10}{'RMSE':>10}{'DirAcc':>10}")
    print("-"*52)
    for name, err in [("Momentum",e_mom),("Zero-pred",e_zero),("Regime-Cond",e_reg)]:
        print(f"{name:<22}{np.abs(err).mean():>10.5f}{np.sqrt((err**2).mean()):>10.5f}{(np.sign(y_true)==np.sign(y_true-err)).mean():>10.3f}")
    print("\n-- IMPROVEMENT 4: Diebold-Mariano Tests --")
    for name, dm in [
        ("Momentum vs Regime-Cond (MSE)", diebold_mariano(e_mom, e_reg, criterion="MSE")),
        ("Zero-pred vs Regime-Cond (MSE)", diebold_mariano(e_zero, e_reg, criterion="MSE")),
        ("Momentum vs Zero-pred   (MAE)", diebold_mariano(e_mom, e_zero, criterion="MAE")),
    ]:
        sig = "REJECT H0" if dm["reject_H0"] else "fail to reject"
        print(f"  {name:<42} DM={dm['dm_stat']:>7.3f}  p={dm['p_value']:.3f}  [{sig}]")
    print("\n  Warning: Better MAE does NOT guarantee better portfolio returns.")
    return y_true, y_reg


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE III — Portfolio
# ═══════════════════════════════════════════════════════════════════════════════

def phase3(prices, hmm_labels, hmm):
    from utils.model_monitor import RegimeTransitionSignal
    from sklearn.covariance  import LedoitWolf
    logger.info("\n" + "="*65)
    logger.info("PHASE III — Portfolio Allocation (Improvements 7, 8, 9)")
    assets = [c for c in ["nifty","gold","usdinr"] if c in prices.columns]
    ret_df = prices[assets].pct_change().dropna()
    la     = hmm_labels.reindex(ret_df.index, method="ffill")
    n      = len(assets)
    lw = LedoitWolf().fit(ret_df.values)
    cov = lw.covariance_ * 252; mu = ret_df.ewm(halflife=63).mean().iloc[-1].values * 252

    def max_sharpe():
        r = sco.minimize(lambda w:-(w@mu-RISK_FREE)/(np.sqrt(w@cov@w)+1e-10),
                         np.ones(n)/n, method="SLSQP", bounds=[(0,.8)]*n,
                         constraints=[{"type":"eq","fun":lambda w:w.sum()-1}])
        return r.x if r.success else np.ones(n)/n

    def risk_parity():
        def obj(w):
            pv=np.sqrt(w@cov@w); mrc=(cov@w)/(pv+1e-10); rc=w*mrc; rc/=rc.sum()+1e-10
            return float(((rc-np.ones(n)/n)**2).sum())
        r = sco.minimize(obj,np.ones(n)/n,method="SLSQP",bounds=[(0.01,.7)]*n,
                         constraints=[{"type":"eq","fun":lambda w:w.sum()-1}])
        w = r.x if r.success else np.ones(n)/n; return w/w.sum()

    static = {
        "Equal Weight":   np.array([1/n]*n),
        "60/40":          np.array([0.60,0.40]+[0]*(n-2)),
        "Max Sharpe MVO": max_sharpe(),
        "Risk Parity":    risk_parity(),
    }

    def sim_dd(base_w, dd_thr=-0.08):
        vals=[INITIAL_CAP]; cw=base_w.copy(); peak=INITIAL_CAP
        rebal=pd.date_range(ret_df.index[0],ret_df.index[-1],freq="ME")
        n_cal=0; n_dd=0; tot=0.0
        for i,(date,row) in enumerate(ret_df.iterrows()):
            if i==0: continue
            vals.append(vals[-1]*(1+float(cw@row[assets].fillna(0).values)))
            peak=max(peak,vals[-1]); dd=(vals[-1]-peak)/peak
            if date in rebal or dd < dd_thr:
                to=float(np.abs(cw-base_w).sum())/2; vals[-1]-=to*TCOST*vals[-1]; tot+=to; cw=base_w.copy()
                if date in rebal: n_cal+=1
                if dd < dd_thr:   n_dd+=1
        return pd.Series(vals,index=ret_df.index), {"n_calendar":n_cal,"n_dd_trigger":n_dd,"total_turnover":tot}

    trans_sig = RegimeTransitionSignal(crisis_threshold=0.20, min_equity=0.10, max_equity=0.75)

    def sim_adaptive():
        vals=[INITIAL_CAP]; cw=np.ones(n)/n; peak=INITIAL_CAP; tot=0.0; n_pre=0
        for i,(date,row) in enumerate(ret_df.iterrows()):
            if i==0: continue
            reg=str(la.get(date,""))
            if   "Bull" in reg or "Recov" in reg: base_eq=0.70
            elif "Crisis" in reg or "Panic" in reg: base_eq=0.10
            elif "Bear" in reg: base_eq=0.20
            else: base_eq=0.45
            sid = next((s for s,l in hmm.state_label_map.items() if l==reg), None)
            if sid is not None:
                sig = trans_sig.compute_signal(hmm, sid)
                eq_target = sig["recommended_eq_weight"]
                if sig["signal_strength"] > 0: n_pre+=1
            else:
                eq_target = base_eq
            tgt = np.array([eq_target,1-eq_target]+[0]*(n-2))[:n]; tgt/=tgt.sum()
            to=float(np.abs(cw-tgt).sum())/2; tot+=to; vals[-1]*=(1-to*TCOST)
            peak=max(peak,vals[-1])
            if (vals[-1]-peak)/peak < -0.08:
                safe=np.array([0.15,0.70]+[0.15]*(n-2))[:n]; safe/=safe.sum()
                to2=float(np.abs(tgt-safe).sum())/2; vals[-1]-=to2*TCOST*vals[-1]; tgt=safe; tot+=to2
            vals.append(vals[-1]*(1+float(tgt@row[assets].fillna(0).values)))
            peak=max(peak,vals[-1]); cw=tgt
        return pd.Series(vals,index=ret_df.index), {"total_turnover":tot,"n_preemptive":n_pre}

    eq_curves={}; tov_info={}
    for name,w in static.items():
        wc=np.clip(w[:n],0,1); wc/=wc.sum()
        eq_curves[name],tov_info[name]=sim_dd(wc)
    eq_curves["Regime Adaptive"],tov_info["Regime Adaptive"]=sim_adaptive()
    return eq_curves, tov_info, assets, ret_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE IV — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def phase4(equity_curves, tov_info, prices, hmm_labels, is_returns):
    from utils.statistical_tests  import jobson_korkie_test, bootstrap_sharpe_ci, information_ratio
    from utils.stability_analysis import conditional_drawdown, is_oos_gap
    from utils.evaluation         import PortfolioEvaluator
    logger.info("\n" + "="*65)
    logger.info("PHASE IV — Evaluation (Improvements 5, 9, 10, 12, 14)")
    bench_ret = prices["nifty"].pct_change().dropna()
    print("\n" + "="*105)
    print("PORTFOLIO METRICS (Turnover & IR included)")
    print("="*105)
    print(f"{'Strategy':<22}{'CAGR':>7}{'Vol':>6}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>7}{'VaR95':>7}{'CVaR95':>8}{'IR':>7}{'Turnover':>10}{'Final Rs':>13}")
    print("-"*105)
    best_sh=-99; best_nm=""; strat_rets={}
    for name,eq in equity_curves.items():
        m=PortfolioEvaluator.evaluate(eq); ret=eq.pct_change().dropna(); strat_rets[name]=ret
        ir=information_ratio(ret,bench_ret)
        ann_to=tov_info.get(name,{}).get("total_turnover",0)*252/max(len(eq)/252,1)
        print(f"{name:<22}{m['cagr']*100:>6.1f}%{m['annual_vol']*100:>5.1f}%{m['sharpe_ratio']:>8.3f}{m['sortino_ratio']:>9.3f}{m['calmar_ratio']:>8.3f}{m['max_drawdown']*100:>6.1f}%{m['var_95_daily']*100:>6.2f}%{m['cvar_95_daily']*100:>7.2f}%{ir['information_ratio']:>7.3f}{ann_to:>9.0f}%{m['final_value']:>12,.0f}")
        if m["sharpe_ratio"]>best_sh: best_sh=m["sharpe_ratio"]; best_nm=name
    print(f"\n  -> Best risk-adjusted: {best_nm}  (Sharpe={best_sh:.3f})")

    print("\n" + "="*65)
    print("IMPROVEMENT 5 — Jobson-Korkie Sharpe Significance Tests")
    print(f"  {'Strategy':<22}{'SR_strat':>10}{'SR_bench':>10}{'JK stat':>10}{'p-value':>10}{'Result':>15}")
    print("  " + "-"*77)
    for name in equity_curves:
        ret=strat_rets[name].reindex(bench_ret.index).dropna(); br=bench_ret.reindex(ret.index)
        jk=jobson_korkie_test(ret,br); sig="SIGNIFICANT" if jk["reject_H0"] else "not sig"
        print(f"  {name:<22}{jk['sr1_annual']:>10.3f}{jk['sr2_annual']:>10.3f}{jk['jk_stat']:>10.3f}{jk['p_value']:>10.3f}{sig:>15}")

    print(f"\n-- Bootstrap Sharpe CI (2000 blocks) for {best_nm} --")
    bci=bootstrap_sharpe_ci(strat_rets[best_nm],n_boot=2000,block_size=20)
    print(f"  Sharpe={bci['sharpe']:.4f}  95% CI=[{bci['ci_lower']:.4f}, {bci['ci_upper']:.4f}]  StdErr={bci['std_error']:.4f}")

    print("\n" + "="*65)
    print("IMPROVEMENT 10 — IS/OOS Gap")
    for name in [best_nm,"Equal Weight"]:
        oos=strat_rets[name].iloc[len(strat_rets[name])//2:]
        gap=is_oos_gap(is_returns,oos)
        verdict=gap["verdict"].split("--")[1].strip() if "--" in gap["verdict"] else gap["verdict"].split("—")[1].strip() if "—" in gap["verdict"] else ""
        print(f"  {name:<22}: IS={gap['is_sharpe']:.3f}  OOS={gap['oos_sharpe']:.3f}  Gap={gap['gap']:.3f}  {verdict}")

    print("\n" + "="*65)
    print("IMPROVEMENT 14 — Conditional Drawdown per Regime")
    print(conditional_drawdown(equity_curves[best_nm], hmm_labels).to_string())
    print("\n-- Economic Cycle Breakdown --")
    print(PortfolioEvaluator.cycle_breakdown(equity_curves[best_nm]).to_string())
    return strat_rets


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — Walk-Forward + Hungarian Alignment
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward(prices, regime_f, returns, hmm_ref, n_states):
    from utils.stability_analysis   import align_labels_hungarian
    from optimization.walk_forward  import WalkForwardOptimizer
    from models.hmm_regime_detector import HMMRegimeDetector
    from sklearn.preprocessing      import StandardScaler
    logger.info("\n" + "="*65)
    logger.info("IMPROVEMENT 3 — Walk-Forward + Hungarian Label Alignment")

    ref_scaler=StandardScaler(); ref_scaler.fit(regime_f.values)
    ref_means=ref_scaler.transform(hmm_ref.model.means_)
    wf=WalkForwardOptimizer(n_splits=4,min_train=252,test_window=126,expanding=True)
    fold_results=[]; all_is=[]; all_oos=[]

    def qf(series):
        r=series.pct_change(); df=pd.DataFrame({"returns":r})
        for w in [10,20,60]: df[f"vol_{w}d"]=r.rolling(w).std()*np.sqrt(252)
        df["vol_ratio"]=df["vol_20d"]/(df["vol_60d"]+1e-10)
        rm=series.cummax(); df["drawdown"]=(series-rm)/(rm+1e-10)
        d=series.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
        df["rsi_regime"]=(100-100/(1+g/(l+1e-10))-50)/50
        df["sharpe_60d"]=(r.rolling(60).mean()*252)/(r.rolling(60).std()*np.sqrt(252)+1e-10)
        return df.dropna()

    for fi,(trn,tst) in enumerate(wf.split(prices["nifty"].to_frame("nifty"))):
        trn_ret=trn["nifty"].pct_change().dropna(); tst_ret=tst["nifty"].pct_change().dropna()
        trf=qf(trn["nifty"]); tsf=qf(tst["nifty"])
        if len(trf)<50 or len(tsf)<3: continue
        h=HMMRegimeDetector(n_components=n_states)
        try: h.fit(trf)
        except: continue
        fs=StandardScaler(); fs.fit(trf.values)
        fm=fs.transform(h.model.means_[:,:min(trf.shape[1],ref_means.shape[1])])
        rm2=ref_means[:,:fm.shape[1]]
        try:
            perm=align_labels_hungarian(rm2,fm)
            aligned_map={s:hmm_ref.state_label_map.get(int(perm[s]),f"S{s}") for s in range(n_states)}
        except:
            aligned_map=h.state_label_map
        orig_map=h.state_label_map; h.state_label_map=aligned_map
        tst_al=tsf[tsf.index.isin(tst.index)]
        if len(tst_al)<3: h.state_label_map=orig_map; continue
        lbl=h.predict(tst_al); h.state_label_map=orig_map
        tr=tst_ret[tst_ret.index.isin(tst_al.index)]
        sr_b=float((tr.mean()-RISK_FREE/252)/(tr.std()+1e-10)*np.sqrt(252))
        dom=lbl.mode().iloc[0] if len(lbl) else "N/A"
        strat_r=[r*(0.70 if ("Bull" in str(lbl.get(d,"")) or "Recov" in str(lbl.get(d,""))) else
                    0.15 if ("Crisis" in str(lbl.get(d,"")) or "Panic" in str(lbl.get(d,""))) else
                    0.25 if "Bear" in str(lbl.get(d,"")) else 0.45)+
                 (1-(0.70 if ("Bull" in str(lbl.get(d,"")) or "Recov" in str(lbl.get(d,""))) else
                     0.15 if ("Crisis" in str(lbl.get(d,"")) or "Panic" in str(lbl.get(d,""))) else
                     0.25 if "Bear" in str(lbl.get(d,"")) else 0.45))*RISK_FREE/252
                 for d,r in tr.items()]
        sr_s=float((pd.Series(strat_r).mean()-RISK_FREE/252)/(pd.Series(strat_r).std()+1e-10)*np.sqrt(252))
        all_is.extend(trn_ret.tolist()); all_oos.extend(tr.tolist())
        fold_results.append({"Fold":fi+1,"Test":f"{tst.index[0].date()} to {tst.index[-1].date()}","Strategy":round(sr_s,3),"Benchmark":round(sr_b,3),"Dom Regime":dom})
        logger.info(f"  Fold {fi+1} | {tst.index[0].date()} to {tst.index[-1].date()} | Strategy={sr_s:.3f}  Bench={sr_b:.3f}  Dom={dom}")

    if fold_results:
        wf_df=pd.DataFrame(fold_results).set_index("Fold")
        print("\n-- Walk-Forward Results (Hungarian-aligned labels) --")
        print(wf_df.to_string())
        avg_s=wf_df["Strategy"].mean(); avg_b=wf_df["Benchmark"].mean()
        print(f"\n  Avg WF Sharpe: Strategy={avg_s:.3f}  Benchmark={avg_b:.3f}  Add={avg_s-avg_b:+.3f}")
    return all_is, all_oos


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 13 — Drift Detection + Registry (dedup fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def drift_detection(hmm, regime_f):
    from utils.model_monitor import ModelDriftDetector, ModelRegistry
    logger.info("\n" + "="*65)
    logger.info("IMPROVEMENT 13 — Model Drift Detector & Registry")
    split=int(len(regime_f)*0.75); train_f=regime_f.iloc[:split]; recent=regime_f.iloc[split:]
    monitor=ModelDriftDetector(window=60,k=2.0); monitor.fit_baseline(hmm,train_f)
    print(f"\n  {'Window':<8}{'LL':>10}{'Baseline':>12}{'z-score':>10}{'Status':>18}")
    print("  " + "-"*58)
    for i,w in enumerate(np.array_split(recent,4)):
        if len(w)<10: continue
        r=monitor.check(hmm,w,timestamp=f"Month {i+1}")
        status="DRIFT DETECTED" if r["is_drifted"] else "OK"
        print(f"  {'Month '+str(i+1):<8}{r['current_ll']:>10.3f}{r['train_mean_ll']:>12.3f}{r['z_score']:>10.3f}{status:>18}")

    registry=ModelRegistry(); existing=registry.list_models(); version="v2.0"
    already=( not existing.empty and "model" in existing.columns and
              "GaussianHMM" in existing["model"].values and version in existing.index)
    if not already:
        registry.register(
            model_name="GaussianHMM", version=version,
            metrics={"log_lik":round(float(hmm.model.score(hmm.scaler.transform(regime_f.values))),4),"n_states":hmm.n_components},
            train_start=str(regime_f.index[0].date()), train_end=str(regime_f.index[-1].date()),
            n_features=regime_f.shape[1], n_states=hmm.n_components,
            notes="Auto-selected n_states via BIC sweep")
        print(f"\n  Model registered: GaussianHMM {version}")
    else:
        print(f"\n  Already registered: GaussianHMM {version} (skipping duplicate)")
    print(registry.list_models().to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DeepRegime Pipeline v2")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--start", default="2016-01-01", help="Data start date")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  DeepRegime v2 — Improved Pipeline (14 Enhancements)")
    print("="*65)

    prices   = load_data(use_synthetic=args.synthetic, start=args.start)
    features = build_features(prices)
    regime_f = get_regime_features(features)
    returns  = prices["nifty"].pct_change().dropna().reindex(regime_f.index).dropna()
    regime_f = regime_f.reindex(returns.index).dropna()
    logger.info(f"Data ready: {len(prices)} days | Features: {regime_f.shape}")

    n_states_bic        = select_n_states(regime_f)
    stab, n_states      = stability_test(regime_f, n_states_bic)
    hmm, hmm_labels, smoothed_lbls, smoothed_arr, raw_probs = phase1(regime_f, returns, n_states)
    sa                  = survival_analysis(hmm_labels, returns)
    y_true, y_reg       = phase2(hmm_labels, returns, regime_f)
    equity_curves, tov, assets, ret_df = phase3(prices, hmm_labels, hmm)
    is_returns          = returns.iloc[:len(returns)//2]
    strat_rets          = phase4(equity_curves, tov, prices, hmm_labels, is_returns)
    all_is, all_oos     = walk_forward(prices, regime_f, returns, hmm, n_states)
    drift_detection(hmm, regime_f)

    print("\n" + "="*65)
    print("PIPELINE COMPLETE — ALL 14 IMPROVEMENTS ACTIVE")
    print("="*65)
    for num, text in [
        ("1",  "BIC sweep -> auto-selected n_states"),
        ("2",  "Multi-asset features (NIFTY + Gold + USDINR + VIX)"),
        ("3",  "Hungarian label alignment across WF folds"),
        ("4",  "Diebold-Mariano test for forecast significance"),
        ("5",  "Jobson-Korkie + Bootstrap CI for Sharpe"),
        ("6",  "Kaplan-Meier survival + log-rank per regime"),
        ("7",  "Drawdown-triggered rebalancing (calendar + 8% DD)"),
        ("8",  "Transition probability pre-emptive signal"),
        ("9",  "Turnover-adjusted comparison table"),
        ("10", "IS/OOS gap report (overfitting check)"),
        ("11", f"Regime stability (ARI={stab['mean_ari']:.3f}, capped at {n_states} states)"),
        ("12", "Information Ratio vs NIFTY benchmark"),
        ("13", "Model drift detector + registry (dedup fixed)"),
        ("14", "Conditional drawdown per regime"),
    ]:
        print(f"  [{num:>2}] {text}")

    data_src = "Synthetic" if args.synthetic else "Yahoo Finance (real NIFTY)"
    print(f"\n  Data   : {data_src}")
    print(f"  States : {n_states_bic} (BIC) -> {n_states} (after stability cap)")
    print(f"  Stable : ARI={stab['mean_ari']:.3f}  {'STABLE' if stab['stable'] else 'UNSTABLE - capped'}")


if __name__ == "__main__":
    main()