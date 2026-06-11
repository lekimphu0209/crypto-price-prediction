"""
Realtime Crypto Price Predictor
Script chính để chạy hệ thống dự đoán giá theo thời gian thực
"""

import asyncio
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.realtime.data_collector import OHLCVCollector
from src.realtime.predictor import RealtimePredictor


class RealtimePredictionSystem:
    """Hệ thống dự đoán giá theo thời gian thực"""
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        """
        Args:
            symbol: Symbol trading pair
            interval: Khoảng thời gian (1m, 5m, 15m, 1h)
        """
        self.symbol = symbol
        self.interval = interval
        
        # Components
        self.collector = OHLCVCollector(symbol, interval)
        self.predictor = RealtimePredictor(symbol)
        
        # State
        self.is_running = False
        self.last_prediction = None
        self.prediction_count = 0
    
    def on_new_candle(self, candle_data: dict):
        """Callback khi có candle mới"""
        try:
            print(f"\n[{datetime.now()}] Candle mới: {candle_data['timestamp']}")
            print(f"  Open: {candle_data['open']}, High: {candle_data['high']}, "
                  f"Low: {candle_data['low']}, Close: {candle_data['close']}")
            
            # Lấy candles gần nhất
            candles = self.collector.get_latest_candles(100)
            
            if len(candles) >= 30:
                # Predict
                result = self.predictor.predict_realtime(candles)
                
                self.last_prediction = result
                self.prediction_count += 1
                
                # Hiển thị kết quả
                print(f"\n  === PREDICTION #{self.prediction_count} ===")
                print(f"  Current Price: ${result['current_price']:,.2f}")
                
                if 'ensemble' in result['predictions']:
                    pred_price = result['predictions']['ensemble']['prediction']
                    change_pct = ((pred_price - result['current_price']) / result['current_price']) * 100
                    print(f"  Predicted Price: ${pred_price:,.2f} ({change_pct:+.2f}%)")
                
                if 'linear_regression' in result['predictions']:
                    lr_pred = result['predictions']['linear_regression']['last_prediction']
                    print(f"  Linear Regression: ${lr_pred:,.2f}")
                
                if 'rnn' in result['predictions']:
                    rnn_pred = result['predictions']['rnn']['last_prediction']
                    print(f"  RNN: ${rnn_pred:,.2f}")
                
                print(f"  Data Points: {result['data_points']}")
            else:
                print(f"  Đang thu thập dữ liệu... ({len(candles)}/30 candles)")
                
        except Exception as e:
            print(f"Lỗi khi xử lý prediction: {e}")
    
    async def start(self):
        """Bắt đầu hệ thống"""
        print("=" * 80)
        print("REALTIME CRYPTO PRICE PREDICTION SYSTEM")
        print("=" * 80)
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Models: Linear Regression, RNN")
        print("=" * 80)
        print("\nĐang khởi động...")
        
        # Set callback
        self.collector.set_callback(self.on_new_candle)
        
        # Start collector
        self.is_running = True
        await self.collector.start()
    
    def stop(self):
        """Dừng hệ thống"""
        self.is_running = False
        self.collector.stop()
        print("\nĐã dừng hệ thống")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Realtime Crypto Price Predictor')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                        help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--interval', type=str, default='1m',
                        help='Interval (1m, 5m, 15m, 1h)')
    
    args = parser.parse_args()
    
    # Create system
    system = RealtimePredictionSystem(args.symbol, args.interval)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\n\nĐang dừng...")
        system.stop()
    except Exception as e:
        print(f"\nLỗi: {e}")
        system.stop()


if __name__ == "__main__":
    asyncio.run(main())
