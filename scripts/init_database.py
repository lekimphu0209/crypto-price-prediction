"""
Initialize Database Script
Creates SQLite database with all tables
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.infrastructure.database import init_database


def main():
    """Initialize database"""
    print("Initializing database...")
    
    # Create database in data directory
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(db_path, exist_ok=True)
    
    db_url = f'sqlite:///{os.path.join(db_path, "crypto_prediction.db")}'
    engine = init_database(db_url)
    
    print(f"Database initialized successfully at: {db_url}")
    print("Tables created:")
    print("- predictions")
    print("- model_performance")
    print("- backtest_results")
    print("- trading_signals")
    print("- market_data")
    print("- sentiment_data")
    print("- system_logs")
    print("- model_versions")


if __name__ == "__main__":
    main()
