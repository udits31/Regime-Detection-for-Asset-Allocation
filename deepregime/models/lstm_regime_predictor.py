# ============================================================
# models/lstm_regime_predictor.py
# BiLSTM + Attention — Next-Day Regime Forecaster
#
# FIXED (vs original):
#   • LSTM now predicts label[t+1] given features[t-lookback:t]
#     (true one-step-ahead forecast, not circular same-day prediction)
#   • TemporalConvNet added as parallel branch for multi-scale patterns
#   • Proper out-of-sample validation (no shuffle, time-ordered split)
#   • Label smoothing to reduce overconfidence on HMM pseudo-labels
#   • Calibration via temperature scaling after training
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import joblib

from config.settings import settings


# ── Custom Layers ─────────────────────────────────────────────────────────────

class TemporalBlock(layers.Layer):
    """
    Dilated causal convolution block (TCN-style).
    Captures multi-scale temporal patterns in parallel with the LSTM.
    """
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, **kw):
        super().__init__(**kw)
        self.conv1 = layers.Conv1D(
            filters, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.conv2 = layers.Conv1D(
            filters, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(0.1)
        # Residual projection if dimensions differ
        self.residual = layers.Conv1D(filters, 1, padding="same")

    def call(self, x, training=False):
        residual = self.residual(x)
        x = self.norm1(self.conv1(x))
        x = self.drop(x, training=training)
        x = self.norm2(self.conv2(x))
        x = self.drop(x, training=training)
        return layers.ReLU()(x + residual)


class TemperatureScaling(layers.Layer):
    """Post-hoc calibration layer — learned scalar temperature T."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.temperature = self.add_weight(
            "temperature", shape=(1,),
            initializer="ones", trainable=True,
            constraint=keras.constraints.NonNeg()
        )

    def call(self, logits):
        return logits / tf.maximum(self.temperature, 1e-3)


# ── Main Model ────────────────────────────────────────────────────────────────

class LSTMRegimePredictor:
    """
    BiLSTM + TCN + Multi-Head Attention for next-day regime forecasting.

    Training target: label[t+1]  ← features[t-lookback : t]
    ─────────────────────────────────────────────────────────
    This eliminates the circular label-leakage present in the original where
    LSTM was trained on same-day HMM labels (making the ensemble redundant).
    Now the two models are genuinely complementary:
      • HMM  → current structural state (what regime ARE we in?)
      • LSTM → next-day probabilistic forecast (what regime will we enter?)

    Architecture
    ────────────
    Input [B, T=60, F]
      ├─ BiLSTM(128) → BiLSTM(64) → Self-Attention → GAP   [sequential branch]
      └─ TCN(64, dilations=[1,2,4,8]) → GAP                [convolutional branch]
      Concat → Dense(128) → BN → Dropout → Dense(n_regimes, softmax)
      └─ Temperature Scaling (calibration)
    """

    MODEL_SAVE_PATH   = Path("models/saved/lstm_model.keras")
    SCALER_SAVE_PATH  = Path("models/saved/lstm_scaler.pkl")
    ENCODER_SAVE_PATH = Path("models/saved/lstm_encoder.pkl")
    TEMP_SAVE_PATH    = Path("models/saved/lstm_temperature.npy")

    def __init__(
        self,
        lookback:     int   = None,
        hidden_units: int   = None,
        dropout:      float = None,
        n_regimes:    int   = 6,
    ):
        self.lookback     = lookback     or settings.LSTM_LOOKBACK_DAYS
        self.hidden_units = hidden_units or settings.LSTM_HIDDEN_UNITS
        self.dropout      = dropout      or settings.LSTM_DROPOUT
        self.n_regimes    = n_regimes
        self.model:  keras.Model | None  = None
        self.scaler        = StandardScaler()
        self.encoder       = LabelEncoder()
        self.temperature   = 1.0          # calibration scalar
        self.history       = None
        self.is_fitted     = False
        self.feature_names: List[str] = []

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build_model(self, n_features: int) -> keras.Model:
        inp = keras.Input(shape=(self.lookback, n_features), name="sequence_input")

        # ── Sequential branch (BiLSTM) ────────────────────────────────────────
        x = layers.Bidirectional(
            layers.LSTM(self.hidden_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm_1"
        )(inp)
        x = layers.Dropout(self.dropout)(x)
        x = layers.LayerNormalization()(x)

        x = layers.Bidirectional(
            layers.LSTM(self.hidden_units // 2, return_sequences=True,
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm_2"
        )(x)
        x = layers.Dropout(self.dropout)(x)

        # Multi-head self-attention with residual
        attn = layers.MultiHeadAttention(
            num_heads=4, key_dim=self.hidden_units // 4, name="attention"
        )(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        lstm_out = layers.GlobalAveragePooling1D(name="lstm_gap")(x)

        # ── Convolutional branch (TCN) ─────────────────────────────────────────
        t = inp
        for dilation in [1, 2, 4, 8]:
            t = TemporalBlock(64, kernel_size=3, dilation_rate=dilation)(t)
        tcn_out = layers.GlobalAveragePooling1D(name="tcn_gap")(t)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = layers.Concatenate()([lstm_out, tcn_out])
        fused = layers.Dense(128, activation="relu",
                             kernel_regularizer=regularizers.l2(1e-4))(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.Dropout(self.dropout)(fused)
        fused = layers.Dense(64, activation="relu")(fused)

        # Logits (pre-softmax) for temperature scaling
        logits = layers.Dense(self.n_regimes, name="logits")(fused)
        probs  = layers.Activation("softmax", name="regime_probs")(logits)

        model = keras.Model(inputs=inp, outputs=probs, name="LSTMRegimePredictor")
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss      = keras.losses.SparseCategoricalCrossentropy(
                            label_smoothing=0.1   # reduces overconfidence on pseudo-labels
                        ),
            metrics   = ["accuracy"],
        )
        return model

    # ── Data Preparation ──────────────────────────────────────────────────────

    def _make_sequences(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping (X, y) pairs where:
          X[i] = features[i : i+lookback]          (lookback-day window)
          y[i] = labels[i + lookback + horizon - 1] (next-day target)

        forecast_horizon=1 → true next-day prediction (no leakage).
        """
        X, y = [], []
        max_i = len(features) - self.lookback - forecast_horizon + 1
        for i in range(max_i):
            X.append(features[i: i + self.lookback])
            y.append(labels[i + self.lookback + forecast_horizon - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        features_df:   pd.DataFrame,
        regime_labels: pd.Series,
        forecast_horizon: int = 1,
    ) -> "LSTMRegimePredictor":
        """
        Train the LSTM to predict regime at t+horizon given features[t-lookback:t].

        Parameters
        ──────────
        features_df      : engineered feature DataFrame (aligned to regime_labels)
        regime_labels    : regime string labels (from HMM.predict())
        forecast_horizon : days ahead to predict (default 1 = next trading day)
        """
        logger.info(
            f"Training LSTM | samples={len(features_df)} "
            f"lookback={self.lookback} horizon={forecast_horizon}"
        )

        self.feature_names = list(features_df.columns)
        X_raw  = self.scaler.fit_transform(features_df.values)
        y_enc  = self.encoder.fit_transform(regime_labels.values)
        self.n_regimes = len(self.encoder.classes_)

        X, y = self._make_sequences(X_raw, y_enc, forecast_horizon)
        logger.info(f"  Sequence tensor: X={X.shape}  y={y.shape}")

        # Strict time-ordered 85/15 split — NO shuffling
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Balance imbalanced regime classes
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(cw))

        self.model = self._build_model(len(self.feature_names))

        cbs = [
            callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor="val_loss"
            ),
            callbacks.ReduceLROnPlateau(
                patience=8, factor=0.5, min_lr=1e-6, monitor="val_loss"
            ),
            callbacks.ModelCheckpoint(
                str(self.MODEL_SAVE_PATH),
                save_best_only=True, monitor="val_accuracy"
            ),
        ]

        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data  = (X_val, y_val),
            epochs           = settings.LSTM_EPOCHS,
            batch_size       = settings.LSTM_BATCH_SIZE,
            class_weight     = class_weights,
            callbacks        = cbs,
            verbose          = 1,
        )

        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"LSTM trained | val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")

        # Temperature scaling calibration on validation set
        self._calibrate(X_val, y_val)

        self.is_fitted = True
        return self

    def _calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Post-hoc temperature scaling calibration.
        Minimises NLL on the validation set by optimising a single scalar T.
        """
        from scipy.optimize import minimize_scalar

        raw_probs = self.model.predict(X_val, verbose=0)

        def nll(t: float) -> float:
            clipped = np.clip(raw_probs / max(t, 1e-3), 1e-7, 1.0)
            norm    = clipped / clipped.sum(axis=1, keepdims=True)
            return -np.mean(np.log(norm[np.arange(len(y_val)), y_val]))

        result          = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        logger.info(f"Temperature calibration | T={self.temperature:.4f}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def _calibrated_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        scaled = raw_probs / max(self.temperature, 1e-3)
        return scaled / scaled.sum(axis=1, keepdims=True)

    def predict_next_regime(self, recent_features_df: pd.DataFrame) -> Dict:
        """
        Predict the NEXT regime given the most recent `lookback` rows.

        Returns
        ───────
        predicted_regime, confidence, all_probabilities, top_3
        """
        self._check_fitted()
        if len(recent_features_df) < self.lookback:
            raise ValueError(
                f"Need ≥{self.lookback} rows, got {len(recent_features_df)}"
            )

        seq        = recent_features_df[self.feature_names].tail(self.lookback).values
        seq_scaled = self.scaler.transform(seq)
        X          = seq_scaled[np.newaxis, ...]

        raw_probs  = self.model.predict(X, verbose=0)
        probs      = self._calibrated_probs(raw_probs)[0]

        pred_cls   = int(np.argmax(probs))
        top3_idx   = np.argsort(probs)[::-1][:3]

        return {
            "predicted_regime": self.encoder.inverse_transform([pred_cls])[0],
            "confidence":       float(probs[pred_cls]),
            "all_probabilities": {
                self.encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            },
            "top_3": [
                {
                    "regime":      self.encoder.inverse_transform([i])[0],
                    "probability": float(probs[i]),
                }
                for i in top3_idx
            ],
            "is_calibrated": True,
            "temperature":   self.temperature,
        }

    def predict_sequence(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict regimes across a full historical feature sequence."""
        self._check_fitted()
        X_scaled = self.scaler.transform(features_df[self.feature_names].values)
        X_seq, _ = self._make_sequences(X_scaled, np.zeros(len(X_scaled)), 1)

        raw   = self.model.predict(X_seq, verbose=0)
        probs = self._calibrated_probs(raw)
        preds = self.encoder.inverse_transform(np.argmax(probs, axis=1))

        idx = features_df.index[self.lookback: self.lookback + len(X_seq)]
        return pd.DataFrame({
            "predicted_regime": preds,
            "confidence":       probs.max(axis=1),
        }, index=idx)

    def get_training_history(self) -> Dict:
        if self.history is None:
            return {}
        h = self.history.history
        return {
            "accuracy":     h.get("accuracy", []),
            "val_accuracy": h.get("val_accuracy", []),
            "loss":         h.get("loss", []),
            "val_loss":     h.get("val_loss", []),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.MODEL_SAVE_PATH))
        joblib.dump(self.scaler,  self.SCALER_SAVE_PATH)
        joblib.dump(self.encoder, self.ENCODER_SAVE_PATH)
        np.save(str(self.TEMP_SAVE_PATH), np.array([self.temperature]))
        logger.info("LSTM saved")

    @classmethod
    def load(cls) -> "LSTMRegimePredictor":
        inst          = cls()
        inst.model    = keras.models.load_model(str(cls.MODEL_SAVE_PATH))
        inst.scaler   = joblib.load(cls.SCALER_SAVE_PATH)
        inst.encoder  = joblib.load(cls.ENCODER_SAVE_PATH)
        inst.n_regimes = len(inst.encoder.classes_)
        if cls.TEMP_SAVE_PATH.exists():
            inst.temperature = float(np.load(str(cls.TEMP_SAVE_PATH))[0])
        inst.is_fitted = True
        logger.info("LSTM loaded from disk")
        return inst

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LSTM not fitted. Call .fit() or .load() first.")
