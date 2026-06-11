"""
External API Clients
"""
from .twitter_client import TwitterClient
from .news_client import NewsClient
from .llm_sentiment_analyzer import LLMSentimentAnalyzer
from .nitter_scraper import NitterScraper
from .huggingface_sentiment import HuggingFaceSentimentAnalyzer

__all__ = ['TwitterClient', 'NewsClient', 'LLMSentimentAnalyzer', 'NitterScraper', 'HuggingFaceSentimentAnalyzer']
