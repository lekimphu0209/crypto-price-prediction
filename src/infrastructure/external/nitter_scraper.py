"""
Nitter Scraper - Free Twitter/X scraping without API key
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from src.domain.entities.sentiment import Sentiment, SentimentSource


class NitterScraper:
    """Nitter scraper for fetching tweets without API key"""
    
    def __init__(self, instance: str = "https://nitter.net"):
        """
        Initialize Nitter scraper
        
        Args:
            instance: Nitter instance URL (can use alternative instances)
        """
        self.instance = instance
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Fetch tweets using Nitter
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: Number of days to look back
        
        Returns:
            List of tweet dictionaries
        """
        try:
            url = f"{self.instance}/search"
            params = {
                'q': query,
                'f': 'tweets',
                'e': 'tweets'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            tweets = []
            tweet_elements = soup.find_all('div', class_='tweet')
            
            for tweet in tweet_elements[:max_results]:
                try:
                    text = tweet.find('div', class_='tweet-content').get_text(strip=True)
                    time_str = tweet.find('span', class_='tweet-date').get('title', '')
                    
                    # Parse time
                    try:
                        tweet_time = datetime.strptime(time_str, '%b %d, %Y · %I:%M %p UTC')
                    except:
                        tweet_time = datetime.now()
                    
                    likes = tweet.find('span', class_='icon-comment').parent.get_text(strip=True) or '0'
                    retweets = tweet.find('span', class_='icon-retweet').parent.get_text(strip=True) or '0'
                    replies = tweet.find('span', class_='icon-quote').parent.get_text(strip=True) or '0'
                    
                    tweets.append({
                        'text': text,
                        'created_at': tweet_time,
                        'likes': self._parse_number(likes),
                        'retweets': self._parse_number(retweets),
                        'replies': self._parse_number(replies)
                    })
                    
                except Exception as e:
                    continue
            
            return tweets[:max_results]
            
        except Exception as e:
            print(f"Error fetching tweets from Nitter: {e}")
            return []
    
    def _parse_number(self, text: str) -> int:
        """Parse number from text (e.g., '1.2K' -> 1200)"""
        try:
            text = text.upper().replace('K', '000').replace('M', '000000').replace(',', '')
            return int(float(text))
        except:
            return 0
    
    def analyze_sentiment_from_tweets(
        self,
        tweets: List[Dict],
        symbol: str
    ) -> List[Sentiment]:
        """
        Analyze sentiment from tweets
        
        Args:
            tweets: List of tweet dictionaries
            symbol: Trading symbol
        
        Returns:
            List of Sentiment entities
        """
        sentiments = []
        
        for tweet in tweets:
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
                metadata={'tweet_text': tweet['text'], 'source': 'Nitter'}
            )
            sentiments.append(sentiment)
        
        return sentiments
