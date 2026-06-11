"""
Test NewsAPI integration
"""
import sys
sys.path.append('src')

from src.infrastructure.external.news_client import NewsClient


def main():
    """Test NewsAPI client"""
    
    print("=" * 80)
    print("TESTING NEWSAPI INTEGRATION")
    print("=" * 80)
    
    # Initialize client (will use API key from config)
    client = NewsClient()
    
    print(f"\nAPI Key: {'***' + client.api_key[-4:] if client.api_key else 'Not set'}")
    
    # Fetch crypto news
    print("\nFetching crypto news for BTC...")
    news = client.fetch_crypto_news("BTC", days_back=3, limit=10)
    
    print(f"\nFound {len(news)} news articles")
    
    if news:
        print("\n" + "-" * 80)
        for i, article in enumerate(news[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['publishedAt']}")
            if article.get('description'):
                print(f"   Description: {article['description'][:100]}...")
        
        # Analyze sentiment
        print("\n" + "=" * 80)
        print("ANALYZING SENTIMENT")
        print("=" * 80)
        
        sentiments = client.analyze_sentiment_from_news(news[:5], "BTC")
        
        print(f"\nAnalyzed {len(sentiments)} articles")
        print("\n" + "-" * 80)
        for i, sentiment in enumerate(sentiments, 1):
            print(f"\n{i}. Score: {sentiment.score:.3f}")
            print(f"   Positive: {sentiment.positive_ratio:.2%}")
            print(f"   Negative: {sentiment.negative_ratio:.2%}")
            print(f"   Source: {sentiment.metadata.get('source', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
