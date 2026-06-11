"""
LLM-based Sentiment Analyzer
"""
from typing import List, Dict, Optional
from datetime import datetime

from src.domain.entities.sentiment import Sentiment, SentimentSource


class LLMSentimentAnalyzer:
    """LLM-based sentiment analyzer using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM analyzer
        
        Args:
            api_key: OpenAI API key (optional, will use mock if not provided)
        """
        self.api_key = api_key
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("OpenAI not installed, using mock analyzer")
                self.client = None
        else:
            self.client = None
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using LLM
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with score, positive_ratio, negative_ratio
        """
        if self.client:
            return self._analyze_with_llm(text)
        else:
            return self._analyze_with_keywords(text)
    
    def _analyze_with_llm(self, text: str) -> Dict:
        """Analyze sentiment using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analyzer for cryptocurrency market. "
                                  "Analyze the sentiment and return a JSON with score (-1 to 1), "
                                  "positive_ratio (0 to 1), and negative_ratio (0 to 1)."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze sentiment: {text}"
                    }
                ],
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error with LLM analysis: {e}")
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
        Analyze Twitter sentiment using LLM
        
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
                metadata={'tweet_text': text, 'analyzer': 'LLM'}
            )
            sentiments.append(sentiment)
        
        return sentiments
    
    def analyze_news_sentiment(
        self,
        news: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze news sentiment using LLM
        
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
                    'analyzer': 'LLM'
                }
            )
            sentiments.append(sentiment)
        
        return sentiments
