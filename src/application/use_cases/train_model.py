"""
Train Model Use Case
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import os

from src.domain.interfaces.model import IModel
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.application.pipelines.feature_pipeline import FeaturePipeline, create_default_pipeline
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class TrainModelUseCase:
    """Use case for training models"""
    
    def __init__(
        self,
        models: List[IModel],
        data_repository: CSVDataRepository,
        feature_pipeline: FeaturePipeline = None
    ):
        """
        Args:
            models: List of models to train
            data_repository: Data repository
            feature_pipeline: Feature engineering pipeline
        """
        self.models = models
        self.data_repository = data_repository
        self.feature_pipeline = feature_pipeline or create_default_pipeline()
    
    def execute(
        self,
        symbol: str,
        interval: str,
        train_size: float = 0.8,
        save_models: bool = True,
        models_dir: str = "models"
    ) -> Dict[str, Dict]:
        """
        Execute training for all models
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            train_size: Training set ratio
            save_models: Whether to save trained models
            models_dir: Directory to save models
        
        Returns:
            Dictionary of training results for each model
        """
        # Load data
        symbol_vo = Symbol(symbol)
        interval_vo = Interval(interval)
        
        ohlcv_data = self.data_repository.get_latest_data(symbol_vo, interval_vo, limit=2000)
        
        if len(ohlcv_data) < 100:
            raise ValueError(f"Not enough data for training (need 100, got {len(ohlcv_data)})")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': o.timestamp,
            'open': o.open,
            'high': o.high,
            'low': o.low,
            'close': o.close,
            'volume': o.volume
        } for o in ohlcv_data])
        
        # Apply feature engineering pipeline
        df_processed = self.feature_pipeline.execute(df)
        
        # Drop NaN values
        df_processed = df_processed.dropna()
        
        # Prepare features and target
        # Target: next day's close price
        df_processed['target'] = df_processed['close'].shift(-1)
        df_processed = df_processed.dropna()
        
        # Select feature columns (exclude non-numeric columns)
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'target']
        
        X = df_processed[feature_cols].values
        y = df_processed['target'].values
        
        # Train-test split
        split_idx = int(len(X) * train_size)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Feature count: {len(feature_cols)}")
        
        # Train each model
        results = {}
        
        for model in self.models:
            print(f"\nTraining {model.name}...")
            
            try:
                # Train model
                model.train(X_train, y_train)
                
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                
                # Save model if requested
                if save_models:
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = os.path.join(models_dir, f"{symbol}_{model.name.lower()}.pkl")
                    model.save(model_path)
                    print(f"Model saved to: {model_path}")
                
                results[model.name] = {
                    'metrics': metrics,
                    'status': 'success'
                }
                
                print(f"{model.name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"Error training {model.name}: {e}")
                results[model.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def train_single_model(
        self,
        model: IModel,
        symbol: str,
        interval: str,
        train_size: float = 0.8,
        save_model: bool = True
    ) -> Dict:
        """
        Train a single model
        
        Args:
            model: Model to train
            symbol: Trading symbol
            interval: Time interval
            train_size: Training set ratio
            save_model: Whether to save trained model
        
        Returns:
            Training results
        """
        return self.execute(symbol, interval, train_size, save_model)[model.name]
