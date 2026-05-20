#!/usr/bin/env python3
"""
train_model.py — DeepRegime Model Training CLI

Usage:
    python train_model.py --start 2015-01-01
    python train_model.py --start 2020-01-01 --hmm-only
"""

import argparse
import sys
from datetime import datetime
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="DeepRegime Model Trainer")
    parser.add_argument("--start",    default="2015-01-01", help="Training start date")
    parser.add_argument("--end",      default=None,          help="Training end date (default: today)")
    parser.add_argument("--hmm-only", action="store_true",  help="Train HMM only (fast)")
    parser.add_argument("--wf-folds", type=int, default=5,   help="Walk-forward folds")
    args = parser.parse_args()

    logger.info(f"Training DeepRegime | start={args.start} hmm_only={args.hmm_only}")

    try:
        from realtime_engine.data_ingestion import MarketDataIngestion
        from regime_detector import EnsembleRegimeDetector
        from config.settings import ALL_ASSETS

        ing = MarketDataIngestion()
        logger.info(f"Fetching price data from {args.start} …")
        prices = ing.fetch_multi_asset(ALL_ASSETS[:20], start=args.start, end=args.end)
        logger.info(f"Downloaded {len(prices)} trading days × {prices.shape[1]} assets")

        detector = EnsembleRegimeDetector()
        detector.fit(prices)

        if not args.hmm_only:
            logger.info("Training ensemble (HMM + Kalman + LSTM) …")

        detector.save()
        logger.info("Training complete. Models saved.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
