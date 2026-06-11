"""
Sentiment Data Provider Implementation
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import random
import os

from src.domain.entities.sentiment import Sentiment, SentimentSource


class SentimentProvider:
    """Sentiment data provider"""
    
    def __init__(
        self,
        twitter_client=None,
        news_client=None,
        llm_analyzer=None,
        hf_analyzer=None,
        nitter_scraper=None
    ):
        """
        Args:
            twitter_client: Optional TwitterClient for real data
            news_client: Optional NewsClient for real data
            llm_analyzer: Optional LLMSentimentAnalyzer for LLM-based analysis
            hf_analyzer: Optional HuggingFaceSentimentAnalyzer for free LLM analysis
            nitter_scraper: Optional NitterScraper for free Twitter data
        """
        self.twitter_client = twitter_client
        self.news_client = news_client
        self.llm_analyzer = llm_analyzer
        self.hf_analyzer = hf_analyzer
        self.nitter_scraper = nitter_scraper
    
    def fetch_sentiment(
        self,
        symbol: str,
        source: SentimentSource,
        limit: int = 100
    ) -> List[Sentiment]:
        """
        Fetch sentiment data
        
        Args:
            symbol: Trading symbol
            source: Sentiment source
            limit: Number of sentiment records
        
        Returns:
            List of Sentiment entities
        """
        # Use real Twitter client if available
        if source == SentimentSource.TWITTER and self.twitter_client:
            query = f"${symbol} OR #{symbol}"
            tweets = self.twitter_client.fetch_tweets(query, max_results=limit)
          Use real News client if available
        if source == SentimentSource.NEWS and self.news_client:
            news = self.news_client.fetch_crypto_news(symbol, days_back=7, limit=limit)
            return self.news_client.analyze_sentiment_from_news(news, symbol)
        
        #   return self.twitter_client.analyze_sentiment_from_tweets(tweets, symbol)
        
        # Otherwise use simplified version for demo
        return self._generate_mock_sentiment(symbol, source, limit)
    
    def _generate_mock_sentiment(
        self,
        symbol: str,
        source: SentimentSource,
        limit: int
    ) -> List[Sentiment]:
        """Generate mock sentiment data for demo"""
        sentiments = []
        for i in range(limit):
            score = random.uniform(-1.0, 1.0)
            positive_ratio = max(0, score) if score > 0 else random.uniform(0, 0.3)
            negative_ratio = abs(min(0, score)) if score < 0 else random.uniform(0, 0.3)
            volume = random.randint(100, 10000)
            
            sentiment = Sentiment(
                source=source,
                score=score,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                volume=volume,
                timestamp=datetime.now() - timedelta(hours=i),
                symbol=symbol,
                metadata={}
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    def fetch_twitter_sentiment(self, symbol: str, limit: int = 100) -> List[Sentiment]:
        """Fetch Twitter sentiment (placeholder)"""
        return self.fetch_sentiment(symbol, SentimentSource.TWITTER, limit)
    
    def fetch_reddit_sentiment(self, symbol: str, limit: int = 100) -> List[Sentiment]:
        """Fetch Reddit sentiment (placeholder)"""
        return self.fetch_sentiment(symbol, SentimentSource.REDDIT, limit)
    
    def fetch_news_sentiment(self, symbol: str, limit: int = 100) -> List[Sentiment]:
        """Fetch News sentiment (placeholder)"""
        return self.fetch_sentiment(symbol, SentimentSource.NEWS, limit)
