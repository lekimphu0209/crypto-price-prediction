"""
Hugging Face Sentiment Analyzer - Free LLM-based sentiment analysis
"""
from typing import List, Dict, Optional
from datetime import datetime

from src.domain.entities.sentiment import Sentiment, SentimentSource


class HuggingFaceSentimentAnalyzer:
    """Hugging Face sentiment analyzer using pre-trained models (free)"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Initialize Hugging Face sentiment analyzer
        
        Args:
            model_name: Model name (default: twitter-roberta-base-sentiment)
        """
        self.model_name = model_name
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the model lazily"""
        try:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name)
            print(f"Loaded Hugging Face model: {self.model_name}")
        except ImportError:
            print("transformers not installed, install with: pip install transformers")
            self.pipeline = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pipeline = None
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using Hugging Face
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with score, positive_ratio, negative_ratio
        """
        if self.pipeline is None:
            return self._analyze_with_keywords(text)
        
        try:
            result = self.pipeline(text)[0]
            
            # Convert Hugging Face labels to our format
            # Labels: POSITIVE, NEGATIVE, NEUTRAL
            label = result['label']
            confidence = result['score']
            
            if label == 'POSITIVE':
                score = confidence
                positive_ratio = confidence
                negative_ratio = 1 - confidence
            elif label == 'NEGATIVE':
                score = -confidence
                positive_ratio = 1 - confidence
                negative_ratio = confidence
            else:  # NEUTRAL
                score = 0
                positive_ratio = 0.5
                negative_ratio = 0.5
            
            return {
                'score': score,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio
            }
            
        except Exception as e:
            print(f"Error with Hugging Face analysis: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> Dict:
        """Analyze sentiment using keywords (fallback)"""
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'up', 'buy', 'pump', 'moon', 'gain', 'profit', 'good',
            'surge', 'rally', 'growth', 'strong', 'positive', 'increase',
            'high', 'peak', 'breakthrough', 'success', 'win'
        ]
        
        negative_keywords = [
            'bear', 'down', 'sell', 'dump', 'crash', 'loss', 'bad', 'fear',
            'drop', 'fall', 'decline', 'weak', 'negative', 'decrease',
            'low', 'crash', 'fail', 'loss', 'risk', 'danger'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            score = 0
            positive_ratio = 0.5
            negative_ratio = 0.5
        else:
            score = (positive_count - negative_count) / total
            positive_ratio = positive_count / total
            negative_ratio = negative_count / total
        
        return {
            'score': score,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        }
    
    def analyze_twitter_sentiment(
        self,
        tweets: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze Twitter sentiment using Hugging Face
        
        Args:
            tweets: List of tweet dictionaries
            symbol: Trading symbol
        
        Returns:
            List of Sentiment entities
        """
        sentiments = []
        
        for tweet in tweets:
            text = tweet['text']
            result = self.analyze_text_sentiment(text)
            
            sentiment = Sentiment(
                source=SentimentSource.TWITTER,
                score=result['score'],
                positive_ratio=result['positive_ratio'],
                negative_ratio=result['negative_ratio'],
                volume=tweet.get('likes', 0) + tweet.get('retweets', 0) + tweet.get('replies', 0),
                timestamp=tweet.get('created_at', datetime.now()),
                symbol=symbol,
                metadata={'tweet_text': text, 'analyzer': 'HuggingFace'}
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    def analyze_news_sentiment(
        self,
        news: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze news sentiment using Hugging Face
        
        Args:
            news: List of news article dictionaries
            symbol: Trading symbol
        
        Returns:
            List of Sentiment entities
        """
        sentiments = []
        
        for article in news:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            result = self.analyze_text_sentiment(text)
            
            sentiment = Sentiment(
                source=SentimentSource.NEWS,
                score=result['score'],
                positive_ratio=result['positive_ratio'],
                negative_ratio=result['negative_ratio'],
                volume=1,
                timestamp=datetime.fromisoformat(article.get('publishedAt', datetime.now().isoformat()).replace('Z', '+00:00')),
                symbol=symbol,
                metadata={
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'analyzer': 'HuggingFace'
                }
            )
            sentiments.append(sentiment)
        
        return sentiments
