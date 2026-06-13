"""
Simple Training Script with Database Integration
Uses only LinearRegression to demonstrate database functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load .env from config directory
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

from datetime import datetime, timedelta
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.data_providers.yfinance_provider import YahooFinanceProvider
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.infrastructure.models.linear_regression import LinearRegressionModel
from src.application.use_cases.collect_data import CollectDataUseCase
from src.infrastructure.database.factory import DatabaseFactory


def main():
    """Simple training with database integration"""
    
    print("=" * 80)
    print("SIMPLE TRAINING WITH DATABASE INTEGRATION")
    print("=" * 80)
    
    # Initialize database repository
    print("\n0. Initializing database repository...")
    db_repo = DatabaseFactory.create_repository()
    print(f"   Database type: MongoDB")
    print(f"   Database: crypto_prediction")
    
    # Initialize providers
    print("\n1. Initializing data providers...")
    binance_provider = BinanceProvider()
    yfinance_provider = YahooFinanceProvider()
    data_repository = CSVDataRepository(data_dir="data")
    
    # Collect data
    print("\n2. Collecting data...")
    collect_use_case = CollectDataUseCase(
        binance_provider=binance_provider,
        yfinance_provider=yfinance_provider,
        data_repository=data_repository
    )
    
    # Collect BTC data
    symbol = "BTCUSDT"
    interval = "1d"
    
    try:
        data = collect_use_case.execute(
            symbol=symbol,
            interval=interval,
            source="yahoo",
            limit=1000,
            days_back=730
        )
        print(f"   Collected {len(data)} candles for {symbol}")
    except Exception as e:
        print(f"   Error: {e}")
        try:
            data = collect_use_case.execute(
                symbol="BTC-USD",
                interval="1d",
                source="yahoo",
                limit=1000,
                days_back=730
            )
            print(f"   Collected {len(data)} candles for BTC-USD")
            symbol = "BTC-USD"
        except Exception as e2:
            print(f"   Error: {e2}")
            db_repo.close()
            return
    
    # Initialize model
    print("\n3. Initializing model...")
    model = LinearRegressionModel()
    print(f"   Model: {model.name}")
    
    # Train model
    print("\n4. Training model...")
    try:
        # Simple training with close prices
        import pandas as pd
        df = pd.DataFrame(data)
        df['returns'] = df['close'].pct_change()
        df = df.dropna()
        
        # Prepare features and target
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].values
        
        # Train
        model.train(X, y)
        print(f"   Training completed")
        
        # Evaluate
        metrics = model.evaluate(X, y)
        print(f"   MAE: {metrics['mae']:.2f}")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   R2: {metrics['r2']:.4f}")
        
        # Save to database
        print("\n5. Saving results to database...")
        db_repo.save_model_performance(
            model_name=model.name,
            symbol="BTC",
            metric_name="MAE",
            metric_value=metrics['mae'],
            training_samples=len(X)
        )
        db_repo.save_model_performance(
            model_name=model.name,
            symbol="BTC",
            metric_name="RMSE",
            metric_value=metrics['rmse'],
            training_samples=len(X)
        )
        db_repo.save_model_performance(
            model_name=model.name,
            symbol="BTC",
            metric_name="R2",
            metric_value=metrics['r2'],
            training_samples=len(X)
        )
        print("   Results saved to database")
        
        # Save prediction
        print("\n6. Making prediction and saving to database...")
        last_features = X[-1:].reshape(1, -1)
        prediction = model.predict(last_features)[0]
        target_date = datetime.utcnow() + timedelta(days=1)
        
        db_repo.save_prediction(
            symbol="BTC",
            model_name=model.name,
            predicted_price=float(prediction),
            target_date=target_date,
            confidence=0.75
        )
        print(f"   Prediction: {prediction:.2f}")
        print(f"   Prediction saved to database")
        
        # Log event
        db_repo.log_system_event(
            log_level="INFO",
            message="Simple training completed with database integration",
            module="train_simple_and_save_db",
            details=f"Model: {model.name}, Samples: {len(X)}"
        )
        print("   Event logged to database")
        
    except Exception as e:
        print(f"   Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Close database
    db_repo.close()


if __name__ == "__main__":
    main()
