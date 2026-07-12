# ============================================================
# models/sentiment_analyzer.py
# Financial News Sentiment Analysis using FinBERT + VADER
# ============================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime, timedelta
import requests
import os


class SentimentAnalyzer:
    """
    Multi-source financial sentiment analyzer.

    Sources:
    - News headlines (NewsAPI / Finnhub)
    - FinBERT (HuggingFace) for deep NLP sentiment
    - VADER for quick rule-based fallback

    Output:
    - Sentiment score [-1, +1]
    - Market sentiment label (Bullish/Neutral/Bearish)
    - Topic extraction
    - Sentiment momentum (trend)
    """

    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self._finbert_pipeline = None
        self._vader = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of NLP models."""
        if self._initialized:
            return

        # VADER (fast, always available)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("✓ VADER sentiment initialized")
        except ImportError:
            logger.warning("VADER not available — install vaderSentiment")

        # FinBERT (slower, more accurate for finance)
        if self.use_finbert:
            try:
                from transformers import pipeline
                self._finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    max_length=512,
                    truncation=True
                )
                logger.info("✓ FinBERT pipeline initialized")
            except Exception as e:
                logger.warning(f"FinBERT not available: {e}. Using VADER fallback.")
                self._finbert_pipeline = None

        self._initialized = True

    # ── News Fetching ──────────────────────────────────────

    def fetch_market_news(
        self,
        query:    str = "Indian stock market NIFTY Sensex",
        days:     int = 7,
        max_articles: int = 50
    ) -> List[Dict]:
        """Fetch financial news from available sources."""
        articles = []

        # Try NewsAPI
        news_api_key = os.getenv("NEWS_API_KEY", "")
        if news_api_key:
            articles.extend(self._fetch_newsapi(query, days, max_articles, news_api_key))

        # Try Finnhub
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if finnhub_key and len(articles) < 10:
            articles.extend(self._fetch_finnhub(days, finnhub_key))

        if not articles:
            logger.warning("No news API keys configured. Using mock sentiment data.")
            return self._mock_news_data()

        return articles[:max_articles]

    def _fetch_newsapi(
        self, query: str, days: int, max_articles: int, api_key: str
    ) -> List[Dict]:
        """Fetch from NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = (
                f"https://newsapi.org/v2/everything?q={query}"
                f"&from={from_date}&language=en&sortBy=publishedAt"
                f"&pageSize={max_articles}&apiKey={api_key}"
            )
            resp = requests.get(url, timeout=10)
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title":       a.get("title", ""),
                    "description": a.get("description", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "published":   a.get("publishedAt", ""),
                    "url":         a.get("url", "")
                })
            return articles
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def _fetch_finnhub(self, days: int, api_key: str) -> List[Dict]:
        """Fetch market news from Finnhub."""
        try:
            from_ts = int((datetime.now() - timedelta(days=days)).timestamp())
            to_ts   = int(datetime.now().timestamp())
            url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return [
                {
                    "title":       a.get("headline", ""),
                    "description": a.get("summary", ""),
                    "source":      a.get("source", ""),
                    "published":   datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                }
                for a in data[:30] if a.get("headline")
            ]
        except Exception as e:
            logger.warning(f"Finnhub fetch failed: {e}")
            return []

    def _mock_news_data(self) -> List[Dict]:
        """Return mock news for demo/testing."""
        return [
            {"title": "NIFTY 50 surges to record highs on strong FII inflows", "description": "Foreign institutional investors pumped ₹8,500 crore into Indian equities today."},
            {"title": "RBI holds repo rate at 6.5%, signals neutral stance", "description": "The Reserve Bank of India kept rates unchanged, boosting market sentiment."},
            {"title": "IT sector leads gains as TCS, Infosys report strong Q3 results", "description": "Large-cap IT stocks surged 3-4% after better-than-expected earnings."},
            {"title": "Global recession fears weigh on emerging markets", "description": "Concerns about US Fed policy impact on developing economies."},
            {"title": "Gold hits all-time high as safe-haven demand spikes", "description": "Precious metals surge amid geopolitical tensions."},
        ]

    # ── Sentiment Analysis ─────────────────────────────────

    def analyze(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment of a list of news articles.
        Returns DataFrame with sentiment scores per article.
        """
        self.initialize()
        results = []

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            text = text.strip()[:512]

            if not text:
                continue

            # FinBERT (preferred)
            if self._finbert_pipeline:
                score, label = self._finbert_score(text)
            elif self._vader:
                score, label = self._vader_score(text)
            else:
                score, label = 0.0, "Neutral"

            results.append({
                "title":     article.get("title", ""),
                "source":    article.get("source", ""),
                "published": article.get("published", ""),
                "score":     round(score, 3),
                "label":     label,
                "model":     "FinBERT" if self._finbert_pipeline else "VADER"
            })

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _finbert_score(self, text: str) -> tuple:
        """Run FinBERT inference."""
        try:
            result = self._finbert_pipeline(text)[0]
            label  = result["label"]   # positive/negative/neutral
            conf   = result["score"]

            score_map = {"positive": conf, "negative": -conf, "neutral": 0.0}
            score = score_map.get(label.lower(), 0.0)
            label_map = {"positive": "Bullish", "negative": "Bearish", "neutral": "Neutral"}
            return score, label_map.get(label.lower(), "Neutral")
        except Exception:
            return 0.0, "Neutral"

    def _vader_score(self, text: str) -> tuple:
        """Run VADER sentiment analysis."""
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        if compound > 0.05:
            return compound, "Bullish"
        elif compound < -0.05:
            return compound, "Bearish"
        return compound, "Neutral"

    # ── Aggregate Sentiment ────────────────────────────────

    def get_market_sentiment(
        self,
        query: str = "Indian market NIFTY Sensex BSE",
        days: int = 3
    ) -> Dict:
        """
        Get aggregated market sentiment from recent news.
        Returns a sentiment summary dict.
        """
        articles = self.fetch_market_news(query, days)
        if not articles:
            return self._neutral_sentiment()

        df = self.analyze(articles)
        if df.empty:
            return self._neutral_sentiment()

        avg_score = float(df["score"].mean())
        label_counts = df["label"].value_counts().to_dict()
        dominant = df["label"].mode()[0] if not df.empty else "Neutral"

        # Sentiment momentum (earlier vs later articles)
        if len(df) >= 4:
            mid = len(df) // 2
            early_score = df.iloc[:mid]["score"].mean()
            recent_score = df.iloc[mid:]["score"].mean()
            momentum = float(recent_score - early_score)
        else:
            momentum = 0.0

        return {
            "sentiment_score":   round(avg_score, 3),
            "dominant_label":    dominant,
            "label_distribution": label_counts,
            "article_count":     len(df),
            "sentiment_momentum": round(momentum, 3),
            "interpretation":    self._interpret_sentiment(avg_score),
            "timestamp":         datetime.utcnow().isoformat(),
            "top_headlines":     df["title"].head(5).tolist()
        }

    def _interpret_sentiment(self, score: float) -> str:
        if score > 0.3:   return "Strongly Bullish — Market optimism is high"
        if score > 0.1:   return "Mildly Bullish — Cautious optimism"
        if score > -0.1:  return "Neutral — Mixed signals"
        if score > -0.3:  return "Mildly Bearish — Caution advised"
        return "Strongly Bearish — Risk-off sentiment dominant"

    def _neutral_sentiment(self) -> Dict:
        return {
            "sentiment_score":   0.0,
            "dominant_label":    "Neutral",
            "label_distribution": {"Neutral": 1},
            "article_count":     0,
            "sentiment_momentum": 0.0,
            "interpretation":    "Insufficient data for sentiment analysis",
            "timestamp":         datetime.utcnow().isoformat(),
            "top_headlines":     []
        }
