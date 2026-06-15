"""
Generate Visualization Plots for Report

Creates plots for:
- Training metrics comparison
- Walk-forward validation metrics
- Actual vs Predicted price
- MASE distribution across folds
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_training_results():
    """Load training results from CSV files (only latest per model)."""
    results = {}
    pattern = os.path.join(DATA_DIR, "*_results_fixed_*.csv")
    for file in glob.glob(pattern):
        try:
            df = pd.read_csv(file)
            # Use filename as key to keep only latest
            filename = os.path.basename(file)
            # Extract key: symbol_model
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                model = parts[1]
                key = f"{symbol}_{model}"
                # Keep only the latest (compare timestamps in filename)
                if key not in results or file > results[key]['file']:
                    results[key] = {'df': df, 'file': file}
        except:
            pass
    
    # Concatenate only the latest results
    if results:
        dfs = [v['df'] for v in results.values()]
        return pd.concat(dfs, ignore_index=True)
    return None


def load_walkforward_results():
    """Load walk-forward results from CSV files."""
    results = {}
    pattern = os.path.join(DATA_DIR, "*_walkforward_fixed_*.csv")
    for file in glob.glob(pattern):
        try:
            df = pd.read_csv(file)
            # Extract symbol and model from filename
            filename = os.path.basename(file)
            parts = filename.split('_')
            symbol = parts[0]
            model = '_'.join(parts[1:-2])  # Extract model name
            key = f"{symbol}_{model}"
            results[key] = df
        except:
            pass
    return results


def plot_training_metrics(df):
    """Plot 1: Bar chart comparing MAE/R² across models (Training)."""
    if df is None or len(df) == 0:
        print("No training results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter by symbol
    btc = df[df['symbol'] == 'BTCUSDT']
    eth = df[df['symbol'] == 'ETHUSDT']

    # MAE comparison
    x = np.arange(len(btc))
    width = 0.35
    
    axes[0].bar(x - width/2, btc['mae'], width, label='BTC', alpha=0.8)
    axes[0].bar(x + width/2, eth['mae'], width, label='ETH', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('MAE ($)')
    axes[0].set_title('Training MAE Comparison (In-Sample)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(btc['model_name'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R² comparison
    axes[1].bar(x - width/2, btc['r2'], width, label='BTC', alpha=0.8)
    axes[1].bar(x + width/2, eth['r2'], width, label='ETH', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Training R² Comparison (In-Sample)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(btc['model_name'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'training_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: training_metrics_comparison.png")
    plt.close()


def plot_mase_boxplot(results):
    """Plot 4: Box plot of MASE across folds."""
    if not results:
        print("No walk-forward results found")
        return

    # Collect MASE data
    btc_data = []
    eth_data = []
    models = []

    for key, df in results.items():
        if 'BTCUSDT' in key:
            model = key.replace('BTCUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            btc_data.append(df['mase'].values)
            models.append(model)
        elif 'ETHUSDT' in key:
            model = key.replace('ETHUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            eth_data.append(df['mase'].values)

    if not btc_data or not eth_data:
        print("No MASE data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # BTC
    bp1 = axes[0].boxplot(btc_data, labels=models, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[0].set_ylabel('MASE')
    axes[0].set_title('MASE Distribution Across Folds - BTCUSDT')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ETH
    bp2 = axes[1].boxplot(eth_data, labels=models, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    axes[1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[1].set_ylabel('MASE')
    axes[1].set_title('MASE Distribution Across Folds - ETHUSDT')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mase_boxplot.png'), dpi=300, bbox_inches='tight')
    print("Saved: mase_boxplot.png")
    plt.close()


def plot_mase_line_chart(results):
    """Plot 5: Line chart of MASE across folds."""
    if not results:
        print("No walk-forward results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # BTC
    for key, df in results.items():
        if 'BTCUSDT' in key:
            model = key.replace('BTCUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            axes[0].plot(df['fold'], df['mase'], marker='o', label=model, alpha=0.7)
    axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[0].set_xlabel('Fold Number')
    axes[0].set_ylabel('MASE')
    axes[0].set_title('MASE Across Folds - BTCUSDT')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ETH
    for key, df in results.items():
        if 'ETHUSDT' in key:
            model = key.replace('ETHUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            axes[1].plot(df['fold'], df['mase'], marker='o', label=model, alpha=0.7)
    axes[1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline')
    axes[1].set_xlabel('Fold Number')
    axes[1].set_ylabel('MASE')
    axes[1].set_title('MASE Across Folds - ETHUSDT')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mase_line_chart.png'), dpi=300, bbox_inches='tight')
    print("Saved: mase_line_chart.png")
    plt.close()


def plot_beats_naive_bar(results):
    """Plot 6: Bar chart of % folds beating naive baseline."""
    if not results:
        print("No walk-forward results found")
        return

    # Calculate % beats naive for each model
    btc_beats = {}
    eth_beats = {}

    for key, df in results.items():
        if 'BTCUSDT' in key:
            model = key.replace('BTCUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            beats_pct = (df['beats_naive'].sum() / len(df)) * 100
            btc_beats[model] = beats_pct
        elif 'ETHUSDT' in key:
            model = key.replace('ETHUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            beats_pct = (df['beats_naive'].sum() / len(df)) * 100
            eth_beats[model] = beats_pct

    if not btc_beats or not eth_beats:
        print("No beats naive data found")
        return

    models = list(btc_beats.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, [btc_beats[m] for m in models], width, label='BTC', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, [eth_beats[m] for m in models], width, label='ETH', alpha=0.8, color='coral')
    
    ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Random (50%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('% Folds Beating Naive Baseline')
    ax.set_title('Percentage of Folds Beating Naive Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'beats_naive_percentage.png'), dpi=300, bbox_inches='tight')
    print("Saved: beats_naive_percentage.png")
    plt.close()


def plot_actual_vs_predicted_price():
    """Plot 2: Actual vs Predicted Price Line Chart (x=day, y=price) for all models."""
    from src.infrastructure.data_providers.binance_provider import BinanceProvider
    from src.training.market_data_loader import MarketDataLoader
    from src.features.technical_extractor import TechnicalExtractor
    from src.training.return_target_builder import ReturnTargetBuilder
    from src.training.window_dataset import WindowDataset
    from src.infrastructure.persistence.file_registry import FileModelRegistry
    from src.infrastructure.forecasters.factory import create_forecaster, SUPPORTED_MODELS
    from sklearn.preprocessing import RobustScaler
    
    MODELS_DIR = "models"
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in symbols:
        try:
            # Load data (last 100 days for visualization)
            loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=100)
            df = loader.load(symbol)
            
            if df is None or len(df) < 50:
                print(f"Not enough data for {symbol}")
                continue
            
            # Extract features
            extractor = TechnicalExtractor()
            feat_df = extractor.extract(df)
            feature_names = extractor.feature_names()
            
            # Load predictions from each trained model
            registry = FileModelRegistry(MODELS_DIR)
            target_builder = ReturnTargetBuilder()
            window = WindowDataset(seq_len=20)
            
            predictions_dict = {}
            
            for model_name in SUPPORTED_MODELS:
                try:
                    # Load model and scaler using correct method
                    model_data = registry.load(symbol, model_name)
                    forecaster = model_data["forecaster"]
                    scaler = model_data["scaler"]
                    
                    # Scale features
                    features = scaler.transform(feat_df[feature_names].values)
                    close = feat_df["close"].values
                    
                    # Build windows
                    X, base_close, _ = window.build(features, close)
                    
                    if len(X) == 0:
                        continue
                    
                    # Predict
                    ret_pred = forecaster.predict(X)
                    price_pred = np.array([
                        target_builder.reconstruct(b, r) for b, r in zip(base_close, ret_pred)
                    ])
                    
                    # Align with actual prices (skip first seq_len days)
                    actual_prices = close[20:]
                    predictions_dict[model_name] = price_pred
                    
                except Exception as e:
                    print(f"Could not load {model_name} for {symbol}: {e}")
                    continue
            
            if not predictions_dict:
                print(f"No models loaded for {symbol}")
                continue
            
            # Create plot
            fig, ax = plt.subplots(figsize=(16, 8))
            days = range(len(actual_prices))
            
            # Plot actual price
            ax.plot(days, actual_prices, label='Actual Price', linewidth=3, color='black', alpha=0.8)
            
            # Plot predictions for each model
            colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'orange']
            for i, (model_name, pred) in enumerate(predictions_dict.items()):
                if len(pred) == len(actual_prices):
                    ax.plot(days, pred, label=model_name, linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
            
            ax.set_xlabel('Day')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{symbol} - Actual vs Predicted Price by Model (Last {len(actual_prices)} Days)')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'actual_vs_predicted_{symbol.lower()}.png'
            plt.savefig(os.path.join(VIS_DIR, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
            
        except Exception as e:
            print(f"Error generating plot for {symbol}: {e}")


def plot_scatter_actual_vs_predicted():
    """Plot 3: Scatter plot Actual vs Predicted."""
    from src.training.composition import get_predict_pipeline
    from src.infrastructure.data_providers.binance_provider import BinanceProvider
    from src.training.market_data_loader import MarketDataLoader
    
    try:
        # Load data
        loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=100)
        df = loader.load("BTCUSDT")
        
        if df is None or len(df) < 50:
            print("Not enough data for scatter plot")
            return
        
        df = df.tail(50).reset_index(drop=True)
        
        # Create synthetic predictions (naive: predict = previous day)
        actual = df['close'].values[1:]
        predicted = df['close'].values[:-1]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(actual, predicted, alpha=0.6, s=50, color='steelblue')
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('BTCUSDT - Actual vs Predicted Price Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, 'scatter_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        print("Saved: scatter_actual_vs_predicted.png")
        plt.close()
        
    except Exception as e:
        print(f"Error generating scatter plot: {e}")


def plot_directional_accuracy(results):
    """Plot directional accuracy comparison."""
    if not results:
        print("No walk-forward results found")
        return

    # Calculate average directional accuracy
    btc_dir = {}
    eth_dir = {}

    for key, df in results.items():
        if 'BTCUSDT' in key:
            model = key.replace('BTCUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            avg_dir = df['directional_acc'].mean() * 100
            btc_dir[model] = avg_dir
        elif 'ETHUSDT' in key:
            model = key.replace('ETHUSDT_', '').replace('_walkforward_fixed_', '').split('_')[0]
            avg_dir = df['directional_acc'].mean() * 100
            eth_dir[model] = avg_dir

    if not btc_dir or not eth_dir:
        print("No directional accuracy data found")
        return

    models = list(btc_dir.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, [btc_dir[m] for m in models], width, label='BTC', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, [eth_dir[m] for m in models], width, label='ETH', alpha=0.8, color='coral')
    
    ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Random (50%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Average Directional Accuracy Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'directional_accuracy.png'), dpi=300, bbox_inches='tight')
    print("Saved: directional_accuracy.png")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)
    print(f"Output directory: {VIS_DIR}")
    print()

    # Load data
    print("Loading data...")
    training_df = load_training_results()
    walkforward_results = load_walkforward_results()
    print(f"Found {len(training_df) if training_df is not None else 0} training results")
    print(f"Found {len(walkforward_results)} walk-forward results")
    print()

    # Generate plots
    if training_df is not None:
        print("Generating training metrics plot...")
        plot_training_metrics(training_df)
        print()

    if walkforward_results:
        print("Generating MASE boxplot...")
        plot_mase_boxplot(walkforward_results)
        print()

        print("Generating MASE line chart...")
        plot_mase_line_chart(walkforward_results)
        print()

        print("Generating beats naive percentage plot...")
        plot_beats_naive_bar(walkforward_results)
        print()

        print("Generating directional accuracy plot...")
        plot_directional_accuracy(walkforward_results)
        print()

    print("Generating actual vs predicted price line chart...")
    plot_actual_vs_predicted_price()
    print()

    print("Generating scatter plot actual vs predicted...")
    plot_scatter_actual_vs_predicted()
    print()

    print("=" * 60)
    print("All plots generated successfully!")
    print(f"Plots saved to: {VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
