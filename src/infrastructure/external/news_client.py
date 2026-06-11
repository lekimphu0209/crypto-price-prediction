"""
Crypto News API Client for sentiment analysis
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.domain.entities.sentiment import Sentiment, SentimentSource


class NewsClient:
    """Crypto news API client for fetching news articles"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news client
        
        Args:
            api_key: Optional API key for news service
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"  # Example using NewsAPI
    
    def fetch_crypto_news(
        self,
        symbol: str,
        days_back: int = 7,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch crypto news articles
        
        Args:
            symbol: Trading symbol
            days_back: Number of days to look back
            limit: Maximum number of articles
        
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            # Return mock data if no API key
            return self._generate_mock_news(symbol, limit)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.base_url}/everything"
            params = {
                'q': f"{symbol} OR cryptocurrency OR bitcoin OR ethereum",
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': limit,
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            return [
                {
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'publishedAt': article['publishedAt'],
                    'source': article['source']['name']
                }
                for article in data.get('articles', [])
            ]
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._generate_mock_news(symbol, limit)
    
    def _generate_mock_news(self, symbol: str, limit: int) -> List[Dict]:
        """Generate mock news data for demo"""
        mock_titles = [
            f"{symbol} price surges as institutional adoption increases",
            f"Market volatility continues for {symbol}",
            f"Analysts predict {symbol} could reach new highs",
            f"{symbol} faces regulatory scrutiny",
            f"Trading volume for {symbol} hits record levels"
        ]
        
        news = []
        for i in range(limit):
            news.append({
                'title': mock_titles[i % len(mock_titles)],
                'description': f"Market analysis for {symbol}",
                'publishedAt': (datetime.now() - timedelta(hours=i)).isoformat(),
                'source': 'CryptoNews'
            })
        
        return news
    
    def analyze_sentiment_from_news(
        self,
        news: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze sentiment from news articles (simplified version)
        In production, use LLM or NLP library for sentiment analysis
        
        Args:
            news: List of news article dictionaries
            symbol: Trading symbol
        
        Returns:
            List of Sentiment entities
        """
        sentiments = []
        
        for article in news:
            # Simplified sentiment analysis (in production, use actual NLP)
            title = article['title'].lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            # Simple keyword-based sentiment
            positive_keywords = ['surge', 'rally', 'gain', 'profit', 'bull', 'up', 'high', 'growth', 'adoption']
            negative_keywords = ['crash', 'dump', 'loss', 'bear', 'down', 'low', 'fall', 'decline', 'scrutiny', 'fear']
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            total = positive_count + negative_count
            if total == 0:
                score = 0
                positive_ratio = 0.5
                negative_ratio = 0.5
            else:
                score = (positive_count - negative_count) / total
                positive_ratio = positive_count / total
                negative_ratio = negative_count / total
            
            sentiment = Sentiment(
                source=SentimentSource.NEWS,
                score=score,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                volume=1,  # News articles don't have volume in the same way
                timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                symbol=symbol,
                metadata={
                    'title': article['title'],
                    'source': article['source']
                }
            )
            sentiments.append(sentiment)
        
        return sentiments
