# ============================================================
# models/explainability.py
# SHAP-based AI Explainability for Regime Predictions
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class RegimeExplainer:
    """
    Explains WHY the AI detected a particular market regime.
    Uses SHAP values to identify the most influential features.

    Example output:
    "Bull Market detected because:
     1. Golden Cross signal (SMA50 > SMA200): +0.34 impact
     2. RSI at 61 (healthy momentum zone): +0.28 impact
     3. Low volatility (11.4% annualized): +0.22 impact
     ..."
    """

    def __init__(self):
        self._explainer = None
        self._initialized = False

    def initialize(self, model, background_data: np.ndarray):
        """Initialize SHAP explainer with trained model and background."""
        try:
            import shap

            # Use KernelExplainer for model-agnostic explanations
            # (works with both HMM and LSTM)
            sample_size = min(100, len(background_data))
            background  = shap.sample(background_data, sample_size)

            self._explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model,
                background
            )
            self._initialized = True
            logger.info("✅ SHAP explainer initialized")
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}")

    def explain_prediction(
        self,
        features:      pd.DataFrame,
        feature_names: List[str],
        regime:        str,
        top_n:         int = 8
    ) -> Dict:
        """
        Generate SHAP-based explanation for a regime prediction.

        Returns:
            {
                regime, top_features, explanation_text,
                shap_values, feature_importance
            }
        """
        if not self._initialized:
            return self._rule_based_explanation(features, regime, feature_names, top_n)

        try:
            import shap
            X = features[feature_names].tail(1).values
            shap_values = self._explainer.shap_values(X)

            # For multi-class, get values for predicted regime class
            if isinstance(shap_values, list):
                # Multi-class: pick the regime's class index
                class_idx = 0  # Simplified; full impl maps regime → class index
                sv = shap_values[class_idx][0]
            else:
                sv = shap_values[0]

            # Build feature importance
            importance = pd.Series(sv, index=feature_names).abs().sort_values(ascending=False)

            top_features = []
            for feat, imp in importance.head(top_n).items():
                feat_val = float(features[feat].iloc[-1]) if feat in features.columns else 0
                direction = "↑" if sv[list(feature_names).index(feat)] > 0 else "↓"
                top_features.append({
                    "feature":   feat,
                    "impact":    round(float(imp), 4),
                    "value":     round(feat_val, 4),
                    "direction": direction,
                    "description": self._describe_feature(feat, feat_val, direction)
                })

            return {
                "regime":        regime,
                "top_features":  top_features,
                "explanation":   self._build_explanation_text(regime, top_features),
                "shap_values":   dict(zip(feature_names, sv.tolist())),
                "method":        "SHAP KernelExplainer"
            }

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}. Using rule-based fallback.")
            return self._rule_based_explanation(features, regime, feature_names, top_n)

    def _rule_based_explanation(
        self,
        features:      pd.DataFrame,
        regime:        str,
        feature_names: List[str],
        top_n:         int
    ) -> Dict:
        """
        Rule-based explanation when SHAP is unavailable.
        Uses domain knowledge to explain detections.
        """
        latest = features.iloc[-1] if not features.empty else pd.Series()
        explanations = []

        # Check key indicators
        if "rsi_14" in latest.index:
            rsi = latest["rsi_14"]
            if rsi > 60:
                explanations.append({
                    "feature": "RSI (14)", "value": round(rsi, 1),
                    "impact": 0.28, "direction": "↑",
                    "description": f"RSI at {rsi:.0f} — momentum in bullish territory"
                })
            elif rsi < 40:
                explanations.append({
                    "feature": "RSI (14)", "value": round(rsi, 1),
                    "impact": 0.28, "direction": "↓",
                    "description": f"RSI at {rsi:.0f} — oversold, bearish pressure"
                })

        if "golden_cross" in latest.index:
            gc = bool(latest["golden_cross"])
            explanations.append({
                "feature": "Golden Cross (SMA50/200)", "value": int(gc),
                "impact": 0.34, "direction": "↑" if gc else "↓",
                "description": "SMA50 > SMA200 — long-term uptrend confirmed" if gc
                              else "SMA50 < SMA200 — long-term downtrend"
            })

        if "vol_20d" in latest.index:
            vol = latest["vol_20d"] * 100
            if vol > 25:
                explanations.append({
                    "feature": "Annualized Volatility (20D)", "value": round(vol, 1),
                    "impact": 0.22, "direction": "↑",
                    "description": f"Elevated volatility at {vol:.1f}% — risk-off conditions"
                })
            else:
                explanations.append({
                    "feature": "Annualized Volatility (20D)", "value": round(vol, 1),
                    "impact": 0.18, "direction": "↓",
                    "description": f"Low volatility at {vol:.1f}% — calm market environment"
                })

        if "macd_hist" in latest.index:
            macd = latest["macd_hist"]
            explanations.append({
                "feature": "MACD Histogram", "value": round(macd, 3),
                "impact": 0.19, "direction": "↑" if macd > 0 else "↓",
                "description": f"MACD hist {'positive' if macd > 0 else 'negative'} — "
                              f"{'bullish' if macd > 0 else 'bearish'} momentum"
            })

        if "drawdown" in latest.index:
            dd = latest["drawdown"] * 100
            if dd < -10:
                explanations.append({
                    "feature": "Current Drawdown", "value": round(dd, 1),
                    "impact": 0.25, "direction": "↓",
                    "description": f"Portfolio down {dd:.1f}% from peak — stress signal"
                })

        # Sort by impact
        explanations.sort(key=lambda x: x["impact"], reverse=True)

        return {
            "regime":        regime,
            "top_features":  explanations[:top_n],
            "explanation":   self._build_explanation_text(regime, explanations[:top_n]),
            "shap_values":   {},
            "method":        "Rule-Based (SHAP unavailable)"
        }

    def _describe_feature(self, feature: str, value: float, direction: str) -> str:
        """Human-readable description of a feature's contribution."""
        descriptions = {
            "rsi_14":       f"RSI at {value:.1f} — {'overbought' if value > 70 else 'oversold' if value < 30 else 'neutral momentum'}",
            "golden_cross": f"{'Golden cross active' if value else 'Death cross active'} — long-term trend signal",
            "vol_20d":      f"Volatility: {value*100:.1f}% annualized — {'high' if value > 0.25 else 'normal'}",
            "macd_hist":    f"MACD histogram: {value:+.3f} — {'bullish' if value > 0 else 'bearish'} momentum",
            "drawdown":     f"Current drawdown: {value*100:.1f}% from peak",
            "bb_width":     f"Bollinger width: {value:.3f} — {'squeeze' if value < 0.05 else 'expansion'}",
            "sharpe_60d":   f"Rolling Sharpe (60D): {value:.2f}",
        }
        return descriptions.get(feature, f"{feature}: {value:.4f}")

    def _build_explanation_text(self, regime: str, features: List[Dict]) -> str:
        """Build a natural language explanation paragraph."""
        regime_intros = {
            "Bull Market":           "A Bull Market has been detected. The market shows strong upward momentum driven by:",
            "Bear Market":           "A Bear Market has been identified. Downward pressure is being driven by:",
            "High Volatility":       "High Volatility conditions detected. The main drivers are:",
            "Sideways / Range-bound": "The market is range-bound with mixed signals. Key factors:",
            "Recovery Phase":        "A Recovery Phase is underway. Supporting evidence includes:",
            "Crisis / Panic Phase":  "Crisis/Panic conditions detected. Risk factors include:",
        }
        intro = regime_intros.get(regime, f"{regime} detected. Key factors:")
        lines = [intro]
        for i, f in enumerate(features[:5], 1):
            lines.append(f"  {i}. {f['description']} (impact: {f['impact']:.2f})")
        return "\n".join(lines)

    def plot_shap_summary(self, shap_values: Dict, top_n: int = 10):
        """Generate a SHAP waterfall plot (returns plotly figure)."""
        try:
            import plotly.graph_objects as go
            if not shap_values:
                return None

            items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            features = [i[0] for i in items]
            values   = [i[1] for i in items]
            colors   = ["#00e676" if v > 0 else "#ff1744" for v in values]

            fig = go.Figure(go.Bar(
                x=values, y=features, orientation="h",
                marker=dict(color=colors, opacity=0.85),
                text=[f"{v:+.3f}" for v in values],
                textposition="outside"
            ))
            fig.update_layout(
                title="SHAP Feature Importance — Regime Detection",
                xaxis_title="SHAP Value (Impact on Prediction)",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
                font=dict(family="Space Mono", size=9, color="#e8f0fe"),
                margin=dict(l=0, r=60, t=40, b=0),
                height=350
            )
            return fig
        except Exception:
            return None
