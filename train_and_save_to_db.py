"""
Train Models and Save Results to Database
"""
import sys
sys.path.append('src')

from datetime import datetime, timedelta
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.data_providers.yfinance_provider import YahooFinanceProvider
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.infrastructure.models.linear_regression import LinearRegressionModel
from src.infrastructure.models.rnn_model import SimpleRNNModel
from src.infrastructure.models.lstm_model import LSTMModel
from src.infrastructure.models.bilstm_model import BiLSTMModel
from src.application.use_cases.collect_data import CollectDataUseCase
from src.application.use_cases.train_model import TrainModelUseCase
from src.infrastructure.database.factory import DatabaseFactory


def main():
    """Main function to train models and save results to database"""
    
    print("=" * 80)
    print("CRYPTO PREDICTION - TRAINING MODELS AND SAVING TO DATABASE")
    print("=" * 80)
    
    # Initialize database repository
    print("\n0. Initializing database repository...")
    db_repo = DatabaseFactory.create_repository()
    print("   Database: MongoDB (crypto_prediction)")
    
    # Initialize providers
    print("\n1. Initializing data providers...")
    binance_provider = BinanceProvider()
    yfinance_provider = YahooFinanceProvider()
    data_repository = CSVDataRepository(data_dir="data")
    
    # Check availability
    print(f"   Binance available: {binance_provider.is_available()}")
    print(f"   Yahoo Finance available: {yfinance_provider.is_available()}")
    
    # Collect data
    print("\n2. Collecting data from Yahoo Finance...")
    collect_use_case = CollectDataUseCase(
        binance_provider=binance_provider,
        yfinance_provider=yfinance_provider,
        data_repository=data_repository
    )
    
    # Collect macro data
    print("\n3. Collecting macro data...")
    macro_data = collect_use_case.collect_macro_data(['GLD', 'UUP'], period="2y")
    print(f"   Gold data points: {len(macro_data.get('GLD', []))}")
    print(f"   DXY data points: {len(macro_data.get('UUP', []))}")
    
    # Collect BTC data
    print("\n4. Collecting crypto data...")
    symbol = "BTCUSDT"
    interval = "1d"
    
    try:
        data = collect_use_case.execute(
            symbol=symbol,
            interval=interval,
            source="yahoo",
            limit=1000,
            days_back=730  # 2 years
        )
        print(f"   Collected {len(data)} candles for {symbol}")
    except Exception as e:
        print(f"   Error collecting data: {e}")
        print("   Using Yahoo Finance instead...")
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
    
    # Initialize models
    print("\n5. Initializing models...")
    models = [
        LinearRegressionModel(),
        SimpleRNNModel(input_shape=(30, 1), rnn_units=50, dropout_rate=0.2),
        LSTMModel(sequence_length=30, lstm_units=64, dropout_rate=0.2),
        BiLSTMModel(sequence_length=30, lstm_units=64, dropout_rate=0.2)
    ]
    
    for model in models:
        print(f"   - {model.name}")
    
    # Train models
    print("\n6. Training models...")
    
    # Create pipeline with macro data
    from src.application.pipelines.feature_pipeline import create_default_pipeline
    feature_pipeline = create_default_pipeline(macro_data=macro_data)
    
    train_use_case = TrainModelUseCase(
        models=models,
        data_repository=data_repository,
        feature_pipeline=feature_pipeline
    )
    
    results = train_use_case.execute_with_ensemble(
        symbol=symbol,
        interval=interval,
        train_size=0.8,
        save_models=True,
        models_dir="models",
        ensemble_weights=[0.3, 0.3, 0.2, 0.2]  # LR, RNN, LSTM, BiLSTM
    )
    
    # Print results and save to database
    print("\n7. Training Results:")
    print("-" * 80)
    for model_name, result in results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"   MAE:  {metrics['mae']:.2f}")
            print(f"   RMSE: {metrics['rmse']:.2f}")
            print(f"   R2:   {metrics['r2']:.4f}")
            
            # Save to database
            print(f"   Saving to database...")
            try:
                db_repo.save_model_performance(
                    model_name=model_name,
                    symbol="BTC",
                    metric_name="MAE",
                    metric_value=metrics['mae'],
                    training_samples=int(len(data) * 0.8)
                )
                db_repo.save_model_performance(
                    model_name=model_name,
                    symbol="BTC",
                    metric_name="RMSE",
                    metric_value=metrics['rmse'],
                    training_samples=int(len(data) * 0.8)
                )
                db_repo.save_model_performance(
                    model_name=model_name,
                    symbol="BTC",
                    metric_name="R2",
                    metric_value=metrics['r2'],
                    training_samples=int(len(data) * 0.8)
                )
                print(f"   [OK] Saved to database")
            except Exception as e:
                print(f"   [ERROR] Failed to save to database: {e}")
        else:
            print(f"\n{model_name}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Log training completion
    try:
        db_repo.log_system_event(
            log_level="INFO",
            message="Model training completed",
            module="train_and_save_to_db",
            details=f"Trained {len(models)} models on {len(data)} data points"
        )
        print("\n[OK] Training event logged to database")
    except Exception as e:
        print(f"\n[ERROR] Failed to log to database: {e}")
    
    print("\nTraining complete with Ensemble!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check database: Use MongoDB Compass to view saved results")
    print("2. Run dashboard: streamlit run dashboard/streamlit_app.py")
    
    # Close database connection
    db_repo.close()


if __name__ == "__main__":
    main()
