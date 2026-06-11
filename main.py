"""
Main Script - Crypto Price Prediction Pipeline
Chạy toàn bộ pipeline: Data Collection -> Feature Engineering -> Model Training -> Evaluation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_collector import DataCollector
from src.feature_engineering import FeatureEngineer
from src.my_models import LinearRegressionModel, RNNModel, plot_predictions, plot_training_history

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Main function để chạy toàn bộ pipeline"""
    
    print("=" * 80)
    print("CRYPTO PRICE PREDICTION - BASIC REGRESSION & RNN")
    print("=" * 80)
    
    # ========================================
    # STEP 1: DATA COLLECTION
    # ========================================
    print("\n[STEP 1] DATA COLLECTION")
    print("-" * 80)
    
    collector = DataCollector()
    
    # Thu thập dữ liệu BTC
    print("\nĐang thu thập dữ liệu Bitcoin (BTC)...")
    btc_df = collector.fetch_binance_ohlcv("BTCUSDT", "1d", days=730)  # 2 năm dữ liệu
    print(f"BTC Data shape: {btc_df.shape}")
    
    if btc_df.empty:
        print("Lỗi: Không thể thu thập dữ liệu BTC!")
        return
    
    # Thu thập dữ liệu ETH
    print("\nĐang thu thập dữ liệu Ethereum (ETH)...")
    eth_df = collector.fetch_binance_ohlcv("ETHUSDT", "1d", days=730)
    print(f"ETH Data shape: {eth_df.shape}")
    
    # Thu thập dữ liệu macro (Gold, DXY)
    print("\nĐang thu thập dữ liệu Macro (Gold, DXY)...")
    gold_df, dxy_df = collector.fetch_macro_data(days=730)
    print(f"Gold Data shape: {gold_df.shape}")
    print(f"DXY Data shape: {dxy_df.shape}")
    
    # Merge dữ liệu
    print("\nĐang merge dữ liệu...")
    btc_merged = collector.merge_data(btc_df, gold_df, dxy_df)
    eth_merged = collector.merge_data(eth_df, gold_df, dxy_df)
    
    print(f"Merged BTC Data shape: {btc_merged.shape}")
    print(f"Merged ETH Data shape: {eth_merged.shape}")
    
    # Lưu dữ liệu thô
    collector.save_data(btc_merged, "btc_raw_data.csv")
    collector.save_data(eth_merged, "eth_raw_data.csv")
    
    # ========================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================
    print("\n[STEP 2] FEATURE ENGINEERING")
    print("-" * 80)
    
    engineer = FeatureEngineer()
    
    # Xử lý BTC
    print("\nĐang tạo features cho BTC...")
    btc_features = engineer.add_technical_indicators(btc_merged)
    btc_features = engineer.add_macro_features(btc_features)
    btc_features = engineer.add_lag_features(btc_features, lag_periods=[1, 2, 3, 7, 14])
    btc_features = engineer.prepare_target(btc_features, forecast_horizon=1)
    btc_features = engineer.clean_data(btc_features)
    print(f"BTC Features shape: {btc_features.shape}")
    
    # Xử lý ETH
    print("\nĐang tạo features cho ETH...")
    eth_features = engineer.add_technical_indicators(eth_merged)
    eth_features = engineer.add_macro_features(eth_features)
    eth_features = engineer.add_lag_features(eth_features, lag_periods=[1, 2, 3, 7, 14])
    eth_features = engineer.prepare_target(eth_features, forecast_horizon=1)
    eth_features = engineer.clean_data(eth_features)
    print(f"ETH Features shape: {eth_features.shape}")
    
    # Lưu dữ liệu đã xử lý
    collector.save_data(btc_features, "btc_processed_data.csv")
    collector.save_data(eth_features, "eth_processed_data.csv")
    
    # ========================================
    # STEP 3: MODEL TRAINING - BTC
    # ========================================
    print("\n[STEP 3] MODEL TRAINING - BITCOIN")
    print("-" * 80)
    
    # Chuẩn bị data cho BTC
    feature_cols_btc = engineer.get_feature_columns(btc_features)
    X_btc = btc_features[feature_cols_btc]
    y_btc = btc_features['target']
    
    # Split data (không shuffle để giữ time series)
    split_idx = int(len(X_btc) * 0.8)
    X_btc_train, X_btc_test = X_btc.iloc[:split_idx], X_btc.iloc[split_idx:]
    y_btc_train, y_btc_test = y_btc.iloc[:split_idx], y_btc.iloc[split_idx:]
    
    print(f"\nBTC Training set size: {X_btc_train.shape}")
    print(f"BTC Test set size: {X_btc_test.shape}")
    
    # 3.1 Linear Regression
    print("\n--- 3.1 Linear Regression ---")
    lr_btc = LinearRegressionModel()
    lr_btc.train(X_btc_train, y_btc_train)
    lr_btc_metrics = lr_btc.evaluate(X_btc_test, y_btc_test)
    
    print("\nLinear Regression Metrics (BTC):")
    for metric, value in lr_btc_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 10 Feature Importance:")
    importance = lr_btc.get_feature_importance()
    print(importance.head(10))
    
    # Save model
    lr_btc.save_model("models/linear_regression_btc.pkl")
    
    # 3.2 RNN
    print("\n--- 3.2 RNN Model ---")
    rnn_btc = RNNModel(sequence_length=30, units=64)
    history_btc = rnn_btc.train(X_btc_train, y_btc_train, epochs=50, batch_size=32)
    rnn_btc_metrics = rnn_btc.evaluate(X_btc_test, y_btc_test)
    
    print("\nRNN Metrics (BTC):")
    for metric, value in rnn_btc_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    rnn_btc.save_model("models/rnn_btc.h5")
    
    # ========================================
    # STEP 4: MODEL TRAINING - ETH
    # ========================================
    print("\n[STEP 4] MODEL TRAINING - ETHEREUM")
    print("-" * 80)
    
    # Chuẩn bị data cho ETH
    feature_cols_eth = engineer.get_feature_columns(eth_features)
    X_eth = eth_features[feature_cols_eth]
    y_eth = eth_features['target']
    
    # Split data
    split_idx = int(len(X_eth) * 0.8)
    X_eth_train, X_eth_test = X_eth.iloc[:split_idx], X_eth.iloc[split_idx:]
    y_eth_train, y_eth_test = y_eth.iloc[:split_idx], y_eth.iloc[split_idx:]
    
    print(f"\nETH Training set size: {X_eth_train.shape}")
    print(f"ETH Test set size: {X_eth_test.shape}")
    
    # 4.1 Linear Regression
    print("\n--- 4.1 Linear Regression ---")
    lr_eth = LinearRegressionModel()
    lr_eth.train(X_eth_train, y_eth_train)
    lr_eth_metrics = lr_eth.evaluate(X_eth_test, y_eth_test)
    
    print("\nLinear Regression Metrics (ETH):")
    for metric, value in lr_eth_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    lr_eth.save_model("models/linear_regression_eth.pkl")
    
    # 4.2 RNN
    print("\n--- 4.2 RNN Model ---")
    rnn_eth = RNNModel(sequence_length=30, units=64)
    history_eth = rnn_eth.train(X_eth_train, y_eth_train, epochs=50, batch_size=32)
    rnn_eth_metrics = rnn_eth.evaluate(X_eth_test, y_eth_test)
    
    print("\nRNN Metrics (ETH):")
    for metric, value in rnn_eth_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    rnn_eth.save_model("models/rnn_eth.h5")
    
    # ========================================
    # STEP 5: MODEL COMPARISON
    # ========================================
    print("\n[STEP 5] MODEL COMPARISON")
    print("=" * 80)
    
    # Tạo bảng so sánh
    comparison = pd.DataFrame({
        'Model': ['Linear Regression (BTC)', 'RNN (BTC)', 
                  'Linear Regression (ETH)', 'RNN (ETH)'],
        'MAE': [lr_btc_metrics['MAE'], rnn_btc_metrics['MAE'],
                lr_eth_metrics['MAE'], rnn_eth_metrics['MAE']],
        'RMSE': [lr_btc_metrics['RMSE'], rnn_btc_metrics['RMSE'],
                 lr_eth_metrics['RMSE'], rnn_eth_metrics['RMSE']],
        'R²': [lr_btc_metrics['R²'], rnn_btc_metrics['R²'],
               lr_eth_metrics['R²'], rnn_eth_metrics['R²']],
        'Direction_Accuracy': [lr_btc_metrics['Direction_Accuracy'], rnn_btc_metrics['Direction_Accuracy'],
                               lr_eth_metrics['Direction_Accuracy'], rnn_eth_metrics['Direction_Accuracy']]
    })
    
    print("\nBảng so sánh các mô hình:")
    print(comparison.to_string(index=False))
    
    # Lưu bảng so sánh
    comparison.to_csv("data/model_comparison.csv", index=False)
    
    # ========================================
    # STEP 6: VISUALIZATION
    # ========================================
    print("\n[STEP 6] VISUALIZATION")
    print("-" * 80)
    
    # 6.1 Predictions vs Actual cho BTC
    print("\nĐang vẽ predictions vs actual cho BTC...")
    lr_pred_btc = lr_btc.predict(X_btc_test)
    rnn_pred_btc = rnn_btc.predict(X_btc_test)
    
    # Align RNN predictions with test data
    y_btc_test_aligned = y_btc_test.iloc[30:].values
    X_btc_test_aligned = X_btc_test.iloc[30:]
    lr_pred_btc_aligned = lr_pred_btc[30:]
    
    plt.figure(figsize=(15, 8))
    plt.plot(y_btc_test_aligned, label='Actual', alpha=0.7)
    plt.plot(lr_pred_btc_aligned, label='Linear Regression', alpha=0.7)
    plt.plot(rnn_pred_btc, label='RNN', alpha=0.7)
    plt.title('Bitcoin Price Prediction - Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/btc_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6.2 Predictions vs Actual cho ETH
    print("\nĐang vẽ predictions vs actual cho ETH...")
    lr_pred_eth = lr_eth.predict(X_eth_test)
    rnn_pred_eth = rnn_eth.predict(X_eth_test)
    
    y_eth_test_aligned = y_eth_test.iloc[30:].values
    lr_pred_eth_aligned = lr_pred_eth[30:]
    
    plt.figure(figsize=(15, 8))
    plt.plot(y_eth_test_aligned, label='Actual', alpha=0.7)
    plt.plot(lr_pred_eth_aligned, label='Linear Regression', alpha=0.7)
    plt.plot(rnn_pred_eth, label='RNN', alpha=0.7)
    plt.title('Ethereum Price Prediction - Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/eth_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6.3 Training history cho RNN
    print("\nĐang vẽ training history...")
    plot_training_history(history_btc, "RNN Training History - Bitcoin")
    plt.savefig('plots/rnn_btc_training_history.png', dpi=150, bbox_inches='tight')
    
    plot_training_history(history_eth, "RNN Training History - Ethereum")
    plt.savefig('plots/rnn_eth_training_history.png', dpi=150, bbox_inches='tight')
    
    # 6.4 Feature importance visualization
    print("\nĐang vẽ feature importance...")
    plt.figure(figsize=(12, 8))
    top_features = importance.head(15)
    plt.barh(top_features['feature'], top_features['abs_coefficient'])
    plt.title('Top 15 Feature Importance - Linear Regression (BTC)')
    plt.xlabel('Absolute Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========================================
    # STEP 7: SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nĐã hoàn thành toàn bộ pipeline!")
    print("\nCác file đã tạo:")
    print("  - data/btc_raw_data.csv: Dữ liệu thô BTC")
    print("  - data/eth_raw_data.csv: Dữ liệu thô ETH")
    print("  - data/btc_processed_data.csv: Dữ liệu đã xử lý BTC")
    print("  - data/eth_processed_data.csv: Dữ liệu đã xử lý ETH")
    print("  - data/model_comparison.csv: Bảng so sánh mô hình")
    print("  - models/linear_regression_btc.pkl: Model Linear Regression BTC")
    print("  - models/linear_regression_eth.pkl: Model Linear Regression ETH")
    print("  - models/rnn_btc.h5: Model RNN BTC")
    print("  - models/rnn_eth.h5: Model RNN ETH")
    print("  - plots/: Các biểu đồ visualization")
    
    print("\nKết quả chính:")
    print(f"  - Linear Regression BTC - R²: {lr_btc_metrics['R²']:.4f}, Direction Acc: {lr_btc_metrics['Direction_Accuracy']:.4f}")
    print(f"  - RNN BTC - R²: {rnn_btc_metrics['R²']:.4f}, Direction Acc: {rnn_btc_metrics['Direction_Accuracy']:.4f}")
    print(f"  - Linear Regression ETH - R²: {lr_eth_metrics['R²']:.4f}, Direction Acc: {lr_eth_metrics['Direction_Accuracy']:.4f}")
    print(f"  - RNN ETH - R²: {rnn_eth_metrics['R²']:.4f}, Direction Acc: {rnn_eth_metrics['Direction_Accuracy']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Tạo thư mục plots nếu chưa có
    os.makedirs("plots", exist_ok=True)
    
    # Chạy main
    main()
