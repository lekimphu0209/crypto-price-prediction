"""
Example script demonstrating Clean Architecture implementation
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.container import container
from src.application.use_cases.predict import PredictUseCase


def main():
    """Main function demonstrating Clean Architecture"""
    
    print("=" * 80)
    print("CLEAN ARCHITECTURE DEMO - Crypto Prediction System")
    print("=" * 80)
    
    # Get use case from container (Dependency Injection)
    predict_use_case = container.predict_use_case()
    
    print("\n1. Dependency Injection:")
    print(f"   - PredictUseCase injected with {len(predict_use_case.models)} model(s)")
    print(f"   - Data repository: {type(predict_use_case.data_repository).__name__}")
    
    print("\n2. Domain Layer Examples:")
    from src.domain.value_objects.symbol import Symbol
    from src.domain.value_objects.interval import Interval
    from src.domain.entities.ohlcv import OHLCV
    from datetime import datetime
    
    # Value Objects
    symbol = Symbol("BTCUSDT")
    interval = Interval("1h")
    print(f"   - Symbol: {symbol} (base={symbol.base}, quote={symbol.quote})")
    print(f"   - Interval: {interval} ({interval.minutes} minutes)")
    
    # Entity
    ohlcv = OHLCV(
        timestamp=datetime.now(),
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        symbol="BTCUSDT"
    )
    print(f"   - OHLCV Entity: close=${ohlcv.close}, is_bullish={ohlcv.is_bullish}")
    
    print("\n3. Infrastructure Layer Examples:")
    from src.infrastructure.models.linear_regression import LinearRegressionModel
    
    model = LinearRegressionModel()
    print(f"   - Model: {model.name}, is_trained={model.is_trained}")
    
    print("\n4. Application Layer Example:")
    print(f"   - Use Case: {type(predict_use_case).__name__}")
    print(f"   - This use case orchestrates the prediction flow")
    
    print("\n5. Architecture Benefits:")
    print("   ✓ SOLID Principles applied")
    print("   ✓ Dependency Inversion: High-level modules don't depend on low-level")
    print("   ✓ Easy to test: Can mock dependencies")
    print("   ✓ Easy to extend: Add new models without changing existing code")
    print("   ✓ Separation of concerns: Each layer has clear responsibility")
    
    print("\n" + "=" * 80)
    print("Clean Architecture setup complete!")
    print("=" * 80)
    
    print("\nNext steps:")
    print("1. Add more model implementations (RNN, LSTM, Transformer)")
    print("2. Implement data sources (Binance, Yahoo Finance)")
    print("3. Create Presentation layer (API, CLI)")
    print("4. Write unit tests for each layer")
    print("5. Migrate existing functionality to new architecture")


if __name__ == "__main__":
    main()
