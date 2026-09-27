"""
dashboard/app.py — DeepRegime Streamlit Dashboard (Fixed)
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="DeepRegime — AI Market Intelligence",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

REGIME_COLORS = {
    "Bull Market":            "#00C851",
    "Bull Market II":         "#00A844",
    "Recovery":               "#00BCD4",
    "Sideways / Range-bound": "#FFB300",
    "High Volatility":        "#FF6D00",
    "Bear Market":            "#FF4444",
    "Bear Market II":         "#CC3333",
    "Crisis / Panic":         "#880E4F",
    "Crisis / Panic II":      "#6A0036",
    "Uncertain":              "#9E9E9E",
}
DEFAULT_COLOR = "#888888"


# ── Synthetic data ─────────────────────────────────────────────────────────────
def _make_synthetic():
    np.random.seed(2024)
    idx = pd.bdate_range("2016-01-01", "2024-01-01"); N = len(idx)
    calendar = [
        ("Bull",0,200),("Sideways",200,320),("Bull",320,550),("Sideways",550,620),
        ("Bear",620,720),("Recov",720,820),("Bull",820,900),("Crisis",900,960),
        ("Recov",960,1100),("Bull",1100,1350),("Sideways",1350,1450),
        ("Bear",1450,1570),("Sideways",1570,1650),("Bull",1650,N),
    ]
    P = {
        "Bull":    {"mu":0.28/252,"sig":0.12/252**.5,"vix":14,"gb":-0.10},
        "Sideways":{"mu":0.05/252,"sig":0.15/252**.5,"vix":18,"gb": 0.05},
        "Bear":    {"mu":-0.22/252,"sig":0.23/252**.5,"vix":32,"gb": 0.30},
        "Crisis":  {"mu":-0.45/252,"sig":0.48/252**.5,"vix":65,"gb": 0.50},
        "Recov":   {"mu":0.38/252,"sig":0.22/252**.5,"vix":22,"gb":-0.05},
    }
    seq = ["Bull"]*N
    for nm,s,e in calendar:
        seq[s:min(e,N)] = [nm]*(min(e,N)-s)
    nr=np.zeros(N); gr=np.zeros(N); ur=np.zeros(N); vix=np.zeros(N)
    for i in range(N):
        p=P.get(seq[i],P["Sideways"]); z=np.random.randn(4)
        nr[i]=p["mu"]+p["sig"]*z[0]
        gr[i]=0.07/252+p["gb"]*p["sig"]*z[0]+0.12/252**.5*z[1]
        ur[i]=(0.04 if seq[i] in ("Crisis","Bear") else 0.02)/252+0.07/252**.5*z[2]
        vix[i]=max(8,p["vix"]+8*z[3]+(0 if i==0 else 0.6*(vix[i-1]-p["vix"])))
    return pd.DataFrame({
        "nifty":8500*np.exp(np.cumsum(nr)), "gold":1100*np.exp(np.cumsum(gr)),
        "usdinr":66*np.exp(np.cumsum(ur)),  "india_vix":np.clip(vix,8,90),
    }, index=idx)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading market data …")
def load_data(start: str, use_synthetic: bool) -> pd.DataFrame:
    if use_synthetic:
        return _make_synthetic()
    try:
        import yfinance as yf
        tickers = {"nifty":"^NSEI","gold":"GC=F","usdinr":"USDINR=X","india_vix":"^INDIAVIX"}
        frames  = {}
        for col, tkr in tickers.items():
            try:
                raw = yf.download(tkr, start=start, auto_adjust=True, progress=False)
                if len(raw) > 100:
                    close = raw["Close"]
                    if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
                    frames[col] = close.rename(col)
            except Exception:
                pass
        if "nifty" not in frames:
            raise RuntimeError("NIFTY unavailable")
        prices = pd.DataFrame(frames).ffill().bfill().dropna(subset=["nifty"])
        return prices.loc[(prices>0).all(axis=1)]
    except Exception as e:
        st.warning(f"Yahoo Finance failed ({e}) — using synthetic data.")
        return _make_synthetic()


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(prices):
    nifty = prices["nifty"]
    df    = pd.DataFrame(index=nifty.index)
    ret   = nifty.pct_change()
    df["returns"]     = ret
    df["log_returns"] = np.log(nifty/nifty.shift(1))
    for w in [10,20,50,200]:
        sma=nifty.rolling(w).mean()
        df[f"sma_{w}"]=sma
        df[f"price_vs_sma{w}"]=(nifty-sma)/(sma+1e-10)
    df["golden_cross"]=(nifty.rolling(50).mean()>nifty.rolling(200).mean()).astype(float)
    d=nifty.diff(); gain=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]=100-100/(1+gain/(loss+1e-10))
    df["rsi_regime"]=(df["rsi_14"]-50)/50
    ema12=nifty.ewm(span=12).mean(); ema26=nifty.ewm(span=26).mean()
    macd=ema12-ema26; sig=macd.ewm(span=9).mean()
    df["macd_hist"]=(macd-sig)/(nifty.rolling(26).std()+1e-10)
    for w in [5,10,20,60]:
        df[f"vol_{w}d"]=ret.rolling(w).std()*np.sqrt(252)
    df["vol_ratio"]=df["vol_20d"]/(df["vol_60d"]+1e-10)
    df["vol_percentile"]=df["vol_20d"].rolling(252).rank(pct=True)
    sma20=nifty.rolling(20).mean(); std20=nifty.rolling(20).std()
    df["bb_width"]=2*std20/(sma20+1e-10)
    rm=nifty.cummax()
    df["drawdown"]=(nifty-rm)/(rm+1e-10)
    df["sharpe_60d"]=(ret.rolling(60).mean()*252)/(ret.rolling(60).std()*np.sqrt(252)+1e-10)
    df["skew_20d"]=ret.rolling(20).skew()
    df["kurt_20d"]=ret.rolling(20).kurt()
    if "india_vix" in prices.columns:
        vix=prices["india_vix"]
        df["india_vix_norm"]=(vix-vix.rolling(252).mean())/(vix.rolling(252).std()+1e-10)
    if "usdinr" in prices.columns:
        df["usdinr_chg20d"]=prices["usdinr"].pct_change(20)
    if "gold" in prices.columns:
        df["gold_ret20d"]=prices["gold"].pct_change(20)
    norm_dd=(-df["drawdown"]).clip(0,0.5)/0.5
    hv=df["vol_20d"].rolling(252).mean().fillna(0.15)
    df["crisis_score"]=0.5*norm_dd+0.5*(df["vol_20d"]/(hv+1e-10)).clip(0,4)/4
    df=df.replace([np.inf,-np.inf],np.nan).dropna()
    desired=["returns","log_returns","rsi_regime","macd_hist",
             "price_vs_sma50","price_vs_sma200","golden_cross",
             "vol_20d","vol_60d","vol_ratio","vol_percentile","bb_width",
             "drawdown","sharpe_60d","skew_20d","kurt_20d","crisis_score",
             "india_vix_norm","usdinr_chg20d","gold_ret20d"]
    regime_f=df[[c for c in desired if c in df.columns]].dropna()
    returns=nifty.pct_change().dropna().reindex(regime_f.index).dropna()
    regime_f=regime_f.reindex(returns.index).dropna()
    return df, regime_f, returns


# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training HMM regime model …")
def run_models(cache_key: str, n_states: int):
    # parse key
    parts        = cache_key.split("|")
    start        = parts[0]
    use_synthetic= parts[1] == "True"

    prices  = load_data(start, use_synthetic)
    _, regime_f, returns = build_features(prices)

    from models.hmm_regime_detector import HMMRegimeDetector
    from models.kalman_smoother     import RegimeProbabilityKalmanSmoother

    hmm = HMMRegimeDetector(n_components=n_states)
    hmm.fit(regime_f)
    hmm_labels = hmm.predict(regime_f)

    ks = RegimeProbabilityKalmanSmoother(1e-3, 0.1)
    ks.fit(hmm.model.transmat_)
    raw_probs    = hmm.predict_proba(regime_f)
    smoothed_arr = ks.smooth(raw_probs.values)
    smoothed_lbls= pd.Series(
        raw_probs.columns[smoothed_arr.argmax(axis=1)],
        index=regime_f.index
    )
    return prices, regime_f, returns, hmm, hmm_labels, smoothed_lbls, raw_probs, smoothed_arr


# ── Portfolio simulation (fixed: no JSON round-trip) ──────────────────────────
def run_portfolio(prices, smoothed_lbls, initial_capital):
    """Simulate strategies directly — no JSON serialization."""
    import scipy.optimize as sco
    from sklearn.covariance import LedoitWolf

    assets  = [c for c in ["nifty","gold","usdinr"] if c in prices.columns]
    ret_df  = prices[assets].pct_change().dropna()
    la      = smoothed_lbls.reindex(ret_df.index, method="ffill")
    n       = len(assets)

    lw  = LedoitWolf().fit(ret_df.values)
    cov = lw.covariance_ * 252
    mu  = ret_df.ewm(halflife=63).mean().iloc[-1].values * 252

    def sim(w_arr):
        w   = np.clip(w_arr[:n], 0, 1); w /= w.sum()
        vals= [initial_capital]; cw = w.copy()
        try:
            rebal = pd.date_range(ret_df.index[0], ret_df.index[-1], freq="ME")
        except Exception:
            rebal = pd.date_range(ret_df.index[0], ret_df.index[-1], freq="M")
        for i, (date, row) in enumerate(ret_df.iterrows()):
            if i == 0: continue
            vals.append(vals[-1] * (1 + float(cw @ row[assets].fillna(0).values)))
            if date in rebal: cw = w.copy()
        return pd.Series(vals, index=ret_df.index)

    def sim_adaptive():
        vals = [initial_capital]; cw = np.ones(n)/n
        for i, (date, row) in enumerate(ret_df.iterrows()):
            if i == 0: continue
            reg = str(la.get(date, ""))
            if   "Bull"   in reg or "Recov" in reg: eq = 0.70
            elif "Crisis" in reg or "Panic" in reg: eq = 0.10
            elif "Bear"   in reg:                   eq = 0.20
            else:                                   eq = 0.45
            tgt = np.array([eq, 1-eq] + [0]*(n-2))[:n]; tgt /= tgt.sum()
            vals.append(vals[-1] * (1 + float(tgt @ row[assets].fillna(0).values)))
            cw = tgt
        return pd.Series(vals, index=ret_df.index)

    res = sco.minimize(
        lambda w: -(w@mu - 0.065) / (np.sqrt(w@cov@w) + 1e-10),
        np.ones(n)/n, method="SLSQP", bounds=[(0,.8)]*n,
        constraints=[{"type":"eq","fun":lambda w: w.sum()-1}]
    )
    mvo_w = res.x if res.success else np.ones(n)/n

    curves = {
        "Equal Weight":    sim(np.array([1/n]*n)),
        "60/40":           sim(np.array([0.60,0.40]+[0]*(n-2))),
        "Max Sharpe MVO":  sim(mvo_w),
        "Regime Adaptive": sim_adaptive(),
    }

    def metrics(eq):
        r    = eq.pct_change().dropna(); ny = len(eq)/252
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/max(ny,0.1)) - 1
        vol  = r.std()*np.sqrt(252)
        sh   = (cagr - 0.065) / (vol + 1e-10)
        rm   = eq.cummax(); dd = float(((eq-rm)/rm).min())
        return {"CAGR (%)": round(cagr*100,2), "Vol (%)": round(vol*100,2),
                "Sharpe": round(sh,3), "Max DD (%)": round(dd*100,2),
                "Final (Rs)": round(float(eq.iloc[-1]),0)}

    return curves, {k: metrics(v) for k, v in curves.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 DeepRegime")
    st.caption("AI Market Intelligence Platform")
    st.divider()
    st.subheader("Data Settings")
    use_synthetic = st.toggle("Use Synthetic Data", value=False,
                              help="Enable if Yahoo Finance is unavailable")
    start_date    = st.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
    start_str     = str(start_date)
    st.divider()
    st.subheader("Model Settings")
    n_states    = st.slider("Number of Regimes", 2, 8, 4)
    initial_cap = st.number_input("Initial Capital (Rs)", value=1_000_000, step=100_000)
    st.divider()
    if st.button("Run / Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
cache_key = f"{start_str}|{use_synthetic}"
with st.spinner("Loading data and training HMM …"):
    try:
        prices, regime_f, returns, hmm, hmm_labels, smoothed_lbls, raw_probs, smoothed_arr = \
            run_models(cache_key, n_states)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

nifty    = prices["nifty"]
data_src = "Synthetic" if use_synthetic else "Yahoo Finance (Real NIFTY 50)"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("DeepRegime — AI Market Intelligence")
st.caption(f"Source: {data_src}  |  {prices.index[0].date()} to {prices.index[-1].date()}  |  {len(prices):,} days")

current_regime = str(smoothed_lbls.iloc[-1])
current_color  = REGIME_COLORS.get(current_regime, DEFAULT_COLOR)
ret_1d  = float(nifty.pct_change().iloc[-1]) * 100
ret_1m  = float(nifty.pct_change(21).iloc[-1]) * 100
vol_now = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
vix_now = float(prices["india_vix"].iloc[-1]) if "india_vix" in prices.columns else 0.0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current Regime",  current_regime)
c2.metric("NIFTY 1D",        f"{ret_1d:+.2f}%")
c3.metric("NIFTY 1M",        f"{ret_1m:+.2f}%")
c4.metric("India VIX",       f"{vix_now:.1f}")
c5.metric("Realised Vol",    f"{vol_now:.1f}%")

st.markdown(
    f'<div style="background:{current_color};color:white;padding:8px 16px;'
    f'border-radius:6px;font-weight:bold;display:inline-block;margin:8px 0;">'
    f'Active Regime: {current_regime}</div>',
    unsafe_allow_html=True
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Regime Timeline", "Regime Analysis", "Portfolio", "Risk Metrics", "Model Info"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Regime Timeline
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("NIFTY 50 with Regime Background")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55,0.22,0.23], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(x=nifty.index, y=nifty.values, name="NIFTY 50",
                             line=dict(color="#1976D2",width=1.5)), row=1, col=1)

    # Regime shading
    prev_date = smoothed_lbls.index[0]
    prev_reg  = smoothed_lbls.iloc[0]
    for i in range(1, len(smoothed_lbls)):
        reg  = smoothed_lbls.iloc[i]
        date = smoothed_lbls.index[i]
        if reg != prev_reg or i == len(smoothed_lbls)-1:
            color = REGIME_COLORS.get(str(prev_reg), DEFAULT_COLOR)
            for row in [1,2,3]:
                fig.add_vrect(x0=prev_date, x1=date, fillcolor=color,
                              opacity=0.12, layer="below", line_width=0, row=row, col=1)
            prev_date = date; prev_reg = reg

    dd = (nifty - nifty.cummax()) / nifty.cummax() * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown (%)",
                             line=dict(color="#FF4444",width=1),
                             fill="tozeroy", fillcolor="rgba(255,68,68,0.15)"), row=2, col=1)

    vol = returns.rolling(20).std() * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="Vol 20d (%)",
                             line=dict(color="#FF6D00",width=1.2)), row=3, col=1)

    fig.update_layout(height=580, hovermode="x unified", showlegend=True,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10,b=10))
    fig.update_yaxes(title_text="Price",   row=1, col=1)
    fig.update_yaxes(title_text="DD (%)",  row=2, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    dist = smoothed_lbls.value_counts(normalize=True).mul(100).round(1)
    fig2 = go.Figure(go.Bar(
        x=dist.index.tolist(), y=dist.values.tolist(),
        marker_color=[REGIME_COLORS.get(r,DEFAULT_COLOR) for r in dist.index],
        text=[f"{v:.1f}%" for v in dist.values], textposition="outside",
    ))
    fig2.update_layout(title="Regime Distribution", height=280, showlegend=False,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=30,b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regime Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Regime-Conditional Statistics")
    merged = pd.DataFrame({"regime": smoothed_lbls, "returns": returns}).dropna()
    stats  = merged.groupby("regime")["returns"].agg(
        Count   = "count",
        AnnRet  = lambda x: round(x.mean()*252*100, 2),
        AnnVol  = lambda x: round(x.std()*np.sqrt(252)*100, 2),
        Sharpe  = lambda x: round((x.mean()*252-0.065)/(x.std()*np.sqrt(252)+1e-10), 3),
        WorstDay= lambda x: round(x.min()*100, 2),
    )
    stats["Weight%"] = (stats["Count"]/stats["Count"].sum()*100).round(1)
    st.dataframe(stats.sort_values("AnnRet", ascending=False), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = go.Figure(go.Bar(
            x=stats.index.tolist(), y=stats["AnnRet"].tolist(),
            marker_color=[REGIME_COLORS.get(r,DEFAULT_COLOR) for r in stats.index],
            text=[f"{v:.1f}%" for v in stats["AnnRet"]], textposition="outside",
        ))
        fig3.update_layout(title="Ann. Return by Regime", height=320,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        try:
            trans = pd.DataFrame(
                hmm.model.transmat_,
                index  =[hmm.state_label_map.get(i,f"S{i}") for i in range(hmm.n_components)],
                columns=[hmm.state_label_map.get(i,f"S{i}") for i in range(hmm.n_components)],
            ).round(3)
            fig4 = px.imshow(trans, text_auto=True, color_continuous_scale="Blues",
                             aspect="auto", title="Transition Matrix")
            fig4.update_layout(height=320, margin=dict(t=40,b=10))
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.warning(f"Transition matrix error: {e}")

    st.subheader("Smoothed Regime Probability Stream")
    smoothed_df = pd.DataFrame(smoothed_arr, index=regime_f.index, columns=raw_probs.columns)
    fig5 = go.Figure()
    for col in smoothed_df.columns:
        fig5.add_trace(go.Scatter(
            x=smoothed_df.index, y=smoothed_df[col], name=col, stackgroup="one",
            fillcolor=REGIME_COLORS.get(col, DEFAULT_COLOR), line=dict(width=0),
        ))
    fig5.update_layout(title="P(regime | data)", height=280, hovermode="x unified",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=30,b=10))
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Portfolio
# ══════════════════════════════════════════════════════════════════════════════

# Regime allocation recommendations
ASSET_CLASSES = ["Indian Stocks", "Indian ETFs", "Gold", "Silver", "Commodities", "Govt Bonds", "Cash"]

REGIME_ALLOCATIONS = {
    "Bull Market": {
        "Indian Stocks": 60, "Indian ETFs": 15, "Gold": 5,
        "Silver": 3,  "Commodities": 5,  "Govt Bonds": 7,  "Cash": 5,
        "rationale": "Strong growth — maximise equity exposure. Defensive assets minimised."
    },
    "Recovery": {
        "Indian Stocks": 45, "Indian ETFs": 20, "Gold": 10,
        "Silver": 5,  "Commodities": 8,  "Govt Bonds": 7,  "Cash": 5,
        "rationale": "Gradually rotate back to equities. Keep some gold as uncertainty hedge."
    },
    "Sideways / Range-bound": {
        "Indian Stocks": 35, "Indian ETFs": 20, "Gold": 15,
        "Silver": 8,  "Commodities": 10, "Govt Bonds": 10, "Cash": 2,
        "rationale": "Balanced approach. Commodities and gold benefit from range-bound markets."
    },
    "High Volatility": {
        "Indian Stocks": 20, "Indian ETFs": 15, "Gold": 25,
        "Silver": 10, "Commodities": 10, "Govt Bonds": 15, "Cash": 5,
        "rationale": "Reduce equity risk. Increase gold and bonds as volatility hedge."
    },
    "Bear Market": {
        "Indian Stocks": 15, "Indian ETFs": 10, "Gold": 30,
        "Silver": 10, "Commodities": 10, "Govt Bonds": 20, "Cash": 5,
        "rationale": "Capital preservation mode. Heavy gold and bonds to protect downside."
    },
    "Crisis / Panic": {
        "Indian Stocks": 5,  "Indian ETFs": 5,  "Gold": 40,
        "Silver": 15, "Commodities": 5,  "Govt Bonds": 25, "Cash": 5,
        "rationale": "Maximum defensive positioning. Gold and bonds are safe havens."
    },
}

ASSET_COLORS = {
    "Indian Stocks": "#1976D2", "Indian ETFs": "#42A5F5",
    "Gold": "#FFD700",          "Silver": "#C0C0C0",
    "Commodities": "#FF6D00",   "Govt Bonds": "#4CAF50",
    "Cash": "#9E9E9E",
}

def get_allocation(regime_label):
    if regime_label in REGIME_ALLOCATIONS:
        return REGIME_ALLOCATIONS[regime_label]
    for key in REGIME_ALLOCATIONS:
        if key in regime_label:
            return REGIME_ALLOCATIONS[key]
    return REGIME_ALLOCATIONS["Sideways / Range-bound"]

with tab3:
    # SECTION 1: Regime-Based Allocation
    st.subheader("Regime-Based Asset Allocation Recommendations")
    st.caption("How to allocate your portfolio based on the currently detected market regime")

    active_alloc = get_allocation(current_regime)

    st.markdown(
        f'<div style="background:{current_color};color:white;padding:10px 18px;' +
        f'border-radius:8px;font-size:1.05em;font-weight:bold;margin-bottom:6px;">' +
        f'Active Regime: {current_regime}</div>',
        unsafe_allow_html=True
    )
    st.info(f"**Strategy Rationale:** {active_alloc.get('rationale', '')}")

    col_chart, col_table = st.columns([1.4, 1])

    with col_chart:
        alloc_vals = [active_alloc.get(a, 0) for a in ASSET_CLASSES]
        fig_alloc  = go.Figure(go.Bar(
            x=ASSET_CLASSES, y=alloc_vals,
            marker_color=[ASSET_COLORS[a] for a in ASSET_CLASSES],
            text=[f"{v}%" for v in alloc_vals],
            textposition="outside",
        ))
        fig_alloc.update_layout(
            title=f"Recommended Allocation — {current_regime}",
            height=340,
            yaxis=dict(title="Allocation (%)", range=[0, max(alloc_vals)+15]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    with col_table:
        st.markdown(f"**Allocation Table — {current_regime}**")
        rows = []
        for i, asset in enumerate(ASSET_CLASSES, 1):
            pct = active_alloc.get(asset, 0)
            inv = pct / 100 * float(initial_cap)
            rows.append({"#": i, "Asset Class": asset, "Allocation": f"{pct}%", "Amount (Rs)": f"Rs {inv:,.0f}"})
        st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, height=280)

    st.divider()

    # SECTION 2: All Regimes Comparison
    st.subheader("All Regime Allocations — Full Reference Table")
    st.caption("Complete allocation guide for every market regime")

    all_regimes = list(REGIME_ALLOCATIONS.keys())

    # Numbered table
    table_data = []
    for i, regime in enumerate(all_regimes, 1):
        row = {"#": i, "Regime": regime}
        for asset in ASSET_CLASSES:
            row[asset] = f"{REGIME_ALLOCATIONS[regime].get(asset, 0)}%"
        table_data.append(row)
    full_df = pd.DataFrame(table_data).set_index("#")
    st.dataframe(full_df, use_container_width=True, height=260)

    # Grouped bar chart
    fig_grouped = go.Figure()
    for regime in all_regimes:
        vals  = [REGIME_ALLOCATIONS[regime].get(a, 0) for a in ASSET_CLASSES]
        color = REGIME_COLORS.get(regime, DEFAULT_COLOR)
        fig_grouped.add_trace(go.Bar(
            name=regime, x=ASSET_CLASSES, y=vals, marker_color=color,
            text=[f"{v}%" if v > 0 else "" for v in vals], textposition="outside",
            textfont=dict(size=9),
        ))
    fig_grouped.update_layout(
        title="Asset Allocation by Regime — Grouped View",
        barmode="group", height=420,
        yaxis=dict(title="Allocation (%)", range=[0, 75]),
        legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=130),
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # Stacked bar chart
    fig_stacked = go.Figure()
    for asset in ASSET_CLASSES:
        vals = [REGIME_ALLOCATIONS[r].get(asset, 0) for r in all_regimes]
        fig_stacked.add_trace(go.Bar(
            name=asset, x=all_regimes, y=vals, marker_color=ASSET_COLORS[asset],
            text=[f"{v}%" if v >= 5 else "" for v in vals],
            textposition="inside", textfont=dict(size=11, color="white"),
        ))
    fig_stacked.update_layout(
        title="Stacked Allocation — Each Bar Totals 100%",
        barmode="stack", height=380,
        yaxis=dict(title="Allocation (%)", range=[0, 105]),
        legend=dict(orientation="h", y=-0.30, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=100),
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    st.divider()
    st.subheader("Portfolio Strategy Comparison")
    with st.spinner("Simulating portfolios …"):
        try:
            eq_curves, port_metrics = run_portfolio(prices, smoothed_lbls, float(initial_cap))
        except Exception as e:
            st.error(f"Portfolio failed: {e}")
            st.stop()

    STRAT_COLORS = {
        "Equal Weight":"#9E9E9E","60/40":"#1976D2",
        "Max Sharpe MVO":"#FF6D00","Regime Adaptive":"#00C851",
    }
    fig6 = go.Figure()
    for name, eq in eq_curves.items():
        fig6.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
                                  line=dict(color=STRAT_COLORS.get(name,"#888"),width=2)))
    fig6.update_layout(title="Portfolio Value", height=380, hovermode="x unified",
                       yaxis_title="Value (Rs)",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=30,b=10))
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Performance Metrics")
    mdf = pd.DataFrame(port_metrics).T
    st.dataframe(
        mdf.style
           .background_gradient(subset=["Sharpe"], cmap="Greens")
           .background_gradient(subset=["Max DD (%)"], cmap="Reds_r"),
        use_container_width=True,
    )

    best = max(port_metrics, key=lambda k: port_metrics[k]["Sharpe"])
    st.subheader(f"Monthly Returns Heatmap — {best}")
    monthly = eq_curves[best].resample("ME").last().pct_change().dropna() * 100
    mdf2    = pd.DataFrame({
        "Year": monthly.index.year, "Month": monthly.index.strftime("%b"), "Ret": monthly.values
    })
    pivot = mdf2.pivot_table(index="Year", columns="Month", values="Ret")
    order = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
             if m in pivot.columns]
    pivot = pivot[order]
    fig7  = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn",
                      color_continuous_midpoint=0, aspect="auto")
    fig7.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Risk Dashboard")
    ret = returns.dropna()
    ann_vol = float(ret.std() * np.sqrt(252))
    var95   = float(np.percentile(ret, 5))
    cvar95  = float(ret[ret <= var95].mean()) if (ret <= var95).any() else var95
    rm_dd   = nifty.cummax()
    max_dd  = float(((nifty - rm_dd) / rm_dd).min())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Volatility",  f"{ann_vol*100:.2f}%")
    c2.metric("VaR 95% (daily)",    f"{var95*100:.3f}%")
    c3.metric("CVaR 95% (daily)",   f"{cvar95*100:.3f}%")
    c4.metric("Max Drawdown",       f"{max_dd*100:.2f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Return Distribution")
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(x=ret.values*100, nbinsx=80,
                                    marker_color="#1976D2", opacity=0.75, name="Returns"))
        fig8.add_vline(x=var95*100,  line_dash="dash", line_color="red",   annotation_text="VaR 95%")
        fig8.add_vline(x=cvar95*100, line_dash="dash", line_color="#880E4F", annotation_text="CVaR 95%")
        fig8.update_layout(height=320, xaxis_title="Daily Return (%)",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig8, use_container_width=True)

    with col_b:
        st.subheader("Volatility by Regime")
        vol20 = ret.rolling(20).std() * np.sqrt(252) * 100
        mv    = pd.DataFrame({"vol":vol20,"regime":smoothed_lbls}).dropna()
        rv    = mv.groupby("regime")["vol"].mean().sort_values()
        fig9  = go.Figure(go.Bar(
            x=rv.index.tolist(), y=rv.values.tolist(),
            marker_color=[REGIME_COLORS.get(r,DEFAULT_COLOR) for r in rv.index],
            text=[f"{v:.1f}%" for v in rv.values], textposition="outside",
        ))
        fig9.update_layout(height=320, yaxis_title="Mean 20d Vol (%)",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Rolling 60-day Sharpe Ratio")
    rs = (ret.rolling(60).mean()-0.065/252) / (ret.rolling(60).std()+1e-10) * np.sqrt(252)
    fig10 = go.Figure()
    fig10.add_hline(y=0, line_dash="dash", line_color="gray")
    fig10.add_hline(y=1, line_dash="dot",  line_color="#00C851", annotation_text="Sharpe=1")
    fig10.add_trace(go.Scatter(x=rs.index, y=rs.values, name="Rolling Sharpe",
                               line=dict(color="#1976D2",width=1.5),
                               fill="tozeroy", fillcolor="rgba(25,118,210,0.1)"))
    fig10.update_layout(height=280, yaxis_title="Sharpe (60d)",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10,b=10))
    st.plotly_chart(fig10, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("HMM Model Information")

    c1,c2,c3 = st.columns(3)
    c1.metric("Number of States",  hmm.n_components)
    c2.metric("Training Samples",  len(regime_f))
    c3.metric("Features Used",     regime_f.shape[1])

    st.subheader("State Label Map")
    st.dataframe(
        pd.DataFrame([{"State ID":k,"Label":v}
                      for k,v in sorted(hmm.state_label_map.items())]),
        use_container_width=True,
    )

    st.subheader("Model Fit Statistics")
    try:
        eval_r = hmm.evaluate(regime_f, returns)
        c1,c2,c3 = st.columns(3)
        c1.metric("Log-Likelihood", f"{eval_r.get('log_likelihood',0):.4f}")
        c2.metric("AIC",            f"{eval_r.get('aic',0):.1f}")
        c3.metric("BIC",            f"{eval_r.get('bic',0):.1f}")

        st.subheader("Stationary Distribution")
        sd = eval_r.get("stationary_dist", [])
        if len(sd):
            for i, prob in enumerate(sd):
                label = hmm.state_label_map.get(i, f"State {i}")
                st.progress(float(np.clip(prob,0,1)), text=f"{label}: {float(prob):.3f}")

        st.subheader("Mean Regime Durations (days)")
        dur = eval_r.get("mean_regime_duration", {})
        if dur:
            fig11 = go.Figure(go.Bar(
                x=list(dur.keys()), y=list(dur.values()),
                marker_color=[REGIME_COLORS.get(k,DEFAULT_COLOR) for k in dur.keys()],
                text=[f"{v:.1f}d" for v in dur.values()], textposition="outside",
            ))
            fig11.update_layout(height=280, yaxis_title="Days",
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(t=10,b=10))
            st.plotly_chart(fig11, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load model stats: {e}")

    st.subheader("Features Used")
    st.write(", ".join(regime_f.columns.tolist()))