"""
Test MongoDB Repository
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from datetime import datetime, timedelta
from src.infrastructure.repositories.mongo_repository import MongoDatabaseRepository


def test_mongodb_repository():
    """Test MongoDB repository operations"""
    print("🧪 Testing MongoDB Repository...")
    
    try:
        # Create repository
        repo = MongoDatabaseRepository(
            connection_string="mongodb://localhost:27017/",
            database_name="crypto_prediction"
        )
        print("✅ Repository created")
        
        # Test 1: Save prediction
        print("\n📝 Test 1: Save prediction...")
        prediction_id = repo.save_prediction(
            symbol="BTC",
            model_name="RNN",
            predicted_price=110000.0,
            target_date=datetime.utcnow() + timedelta(days=1),
            confidence=0.85,
            features_used={"rsi": 0.65, "macd": 0.72}
        )
        print(f"✅ Prediction saved with ID: {prediction_id}")
        
        # Test 2: Get predictions
        print("\n📋 Test 2: Get predictions...")
        predictions = repo.get_predictions(symbol="BTC", days=1)
        print(f"✅ Retrieved {len(predictions)} predictions")
        if predictions:
            print(f"   Latest: {predictions[0]}")
        
        # Test 3: Save model performance
        print("\n📊 Test 3: Save model performance...")
        perf_id = repo.save_model_performance(
            model_name="RNN",
            symbol="BTC",
            metric_name="RMSE",
            metric_value=980.5,
            training_samples=50000
        )
        print(f"✅ Performance saved with ID: {perf_id}")
        
        # Test 4: Save trading signal
        print("\n📈 Test 4: Save trading signal...")
        signal_id = repo.save_trading_signal(
            symbol="BTC",
            signal_type="BUY",
            model_name="RNN",
            price_at_signal=108500.0,
            signal_strength=0.9,
            target_price=112000.0,
            stop_loss=107000.0
        )
        print(f"✅ Signal saved with ID: {signal_id}")
        
        # Test 5: Get active signals
        print("\n🔔 Test 5: Get active signals...")
        signals = repo.get_active_signals(symbol="BTC")
        print(f"✅ Retrieved {len(signals)} active signals")
        
        # Test 6: Log system event
        print("\n📝 Test 6: Log system event...")
        log_id = repo.log_system_event(
            log_level="INFO",
            message="MongoDB repository test completed",
            module="test_mongodb_repository"
        )
        print(f"✅ Log saved with ID: {log_id}")
        
        # Test 7: Save market data
        print("\n💹 Test 7: Save market data...")
        data_id = repo.save_market_data(
            symbol="BTC",
            timestamp=datetime.utcnow(),
            open_price=108000.0,
            high_price=109000.0,
            low_price=107500.0,
            close_price=108500.0,
            volume=1000000.0,
            source="binance"
        )
        print(f"✅ Market data saved with ID: {data_id}")
        
        # Close connection
        repo.close()
        print("\n🎉 All MongoDB repository tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB repository test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mongodb_repository()
