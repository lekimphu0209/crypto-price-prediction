"""
Train Models Script - Collect real data and train models
"""
import sys
sys.path.append('src')

from datetime import datetime, timedelta
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.data_providers.yfinance_provider import YahooFinanceProvider
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.infrastructure.models.linear_regression import LinearRegressionModel
from src.infrastructure.models.xgboost_model import XGBoostModel
from src.infrastructure.models.lstm_model import LSTMModel
from src.infrastructure.models.bilstm_model import BiLSTMModel
from src.application.use_cases.collect_data import CollectDataUseCase
from src.application.use_cases.train_model import TrainModelUseCase


def main():
    """Main function to collect data and train models"""
    
    print("=" * 80)
    print("CRYPTO PREDICTION - TRAINING MODELS ON REAL DATA")
    print("=" * 80)
    
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
            return
    
    # Initialize models
    print("\n5. Initializing models...")
    models = [
        LinearRegressionModel(),
        XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1),
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
    
    results = train_use_case.execute(
        symbol=symbol,
        interval=interval,
        train_size=0.8,
        save_models=True,
        models_dir="models"
    )
    
    # Print results
    print("\n7. Training Results:")
    print("-" * 80)
    for model_name, result in results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"   MAE:  {metrics['mae']:.2f}")
            print(f"   RMSE: {metrics['rmse']:.2f}")
            print(f"   R²:   {metrics['r2']:.4f}")
        else:
            print(f"\n{model_name}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Test trained models with predict_models.py")
    print("2. Run dashboard: streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
