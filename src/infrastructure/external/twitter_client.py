"""
Twitter/X.com API Client for sentiment analysis
"""
import tweepy
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.domain.entities.sentiment import Sentiment, SentimentSource


class TwitterClient:
    """Twitter/X.com API client for fetching tweets"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str
    ):
        """
        Initialize Twitter client with API credentials
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_token_secret: Twitter access token secret
        """
        self.client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
    
    def fetch_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Fetch tweets based on query
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: Number of days to look back
        
        Returns:
            List of tweet dictionaries
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'public_metrics', 'text']
            )
            
            return [
                {
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                }
                for tweet in tweets.data
            ] if tweets.data else []
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return []
    
    def analyze_sentiment_from_tweets(
        self,
        tweets: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze sentiment from tweets (simplified version)
        In production, use LLM or NLP library for sentiment analysis
        
        Args:
            tweets: List of tweet dictionaries
            symbol: Trading symbol
        
        Returns:
            List of Sentiment entities
        """
        sentiments = []
        
        for tweet in tweets:
            # Simplified sentiment analysis (in production, use actual NLP)
            text = tweet['text'].lower()
            
            # Simple keyword-based sentiment
            positive_keywords = ['bull', 'up', 'buy', 'pump', 'moon', 'gain', 'profit', 'good']
            negative_keywords = ['bear', 'down', 'sell', 'dump', 'crash', 'loss', 'bad', 'fear']
            
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
                source=SentimentSource.TWITTER,
                score=score,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                volume=tweet['likes'] + tweet['retweets'] + tweet['replies'],
                timestamp=tweet['created_at'],
                symbol=symbol,
                metadata={'tweet_text': tweet['text']}
            )
            sentiments.append(sentiment)
        
        return sentiments
