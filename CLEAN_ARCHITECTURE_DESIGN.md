# Clean Architecture Design - Quantitative Trading Platform

## Tổng Quan

Thiết kế hệ thống Quantitative Trading Platform thu nhỏ cho dự đoán giá crypto theo Clean Architecture, SOLID principles, dễ mở rộng và ít phụ thuộc.

## Mục Tiêu Hệ Thống

Dự đoán giá BTC/ETH trong tương lai với đa nguồn dữ liệu và đa mô hình.

**Inputs:**
- Historical OHLCV
- Technical Indicators
- Gold Price
- DXY (US Dollar Index)
- News Sentiment
- X.com/Twitter Sentiment

**Outputs:**
- Giá dự đoán
- Xu hướng tăng/giảm
- Độ tin cậy

**Models:**
- Linear Regression
- XGBoost
- LSTM
- BiLSTM
- Transformer

## Kiến Trúc Clean Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                       │
│  (FastAPI, CLI, Dashboard, WebSocket)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
│  (Use Cases, DTOs, Application Services)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           Domain Layer                           │
│  (Entities, Value Objects, Domain Services, Interfaces)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Infrastructure Layer                      │
│  (Data Sources, External APIs, Models, Database)                │
└─────────────────────────────────────────────────────────────────┘
```

## Các Nguyên Tắc SOLID

### 1. Single Responsibility Principle (SRP)
- Mỗi class chỉ có một trách nhiệm
- DataCollector chỉ thu thập dữ liệu
- FeatureEngineer chỉ xử lý features
- Predictor chỉ dự đoán
- Presenter chỉ hiển thị

### 2. Open/Closed Principle (OCP)
- Mở rộng cho các models mới mà không sửa code cũ
- Thêm data source mới mà không sửa existing collectors
- Plugin architecture cho models

### 3. Liskov Substitution Principle (LSP)
- Các implementations của interfaces có thể thay thế nhau
- IModel interface với LinearRegression, RNN, LSTM implementations

### 4. Interface Segregation Principle (ISP)
- Các interfaces nhỏ, focused
- IDataReader, IDataWriter, IModel, IPredictor
- Không bắt clients implement methods không dùng

### 5. Dependency Inversion Principle (DIP)
- High-level modules không phụ thuộc low-level modules
- Cả hai đều phụ thuộc vào abstractions (interfaces)
- Dependency Injection container

## Cấu Trúc Thư Mục (Mở Rộng)

```
DAMH/
├── src/
│   ├── domain/                      # Domain Layer (Không phụ thuộc external libs)
│   │   ├── __init__.py
│   │   ├── entities/                # Business entities
│   │   │   ├── __init__.py
│   │   │   ├── ohlcv.py            # OHLCV entity
│   │   │   ├── prediction.py       # Prediction entity
│   │   │   ├── model_metrics.py    # Model metrics entity
│   │   │   ├── sentiment.py        # Sentiment entity
│   │   │   ├── feature_vector.py   # Feature vector entity
│   │   │   └── trade_signal.py     # Trade signal entity
│   │   ├── value_objects/          # Value objects (immutable)
│   │   │   ├── __init__.py
│   │   │   ├── symbol.py           # Symbol value object
│   │   │   ├── interval.py         # Interval value object
│   │   │   ├── direction.py        # Direction (UP/DOWN)
│   │   │   └── confidence.py       # Confidence level
│   │   ├── services/               # Domain services
│   │   │   ├── __init__.py
│   │   │   ├── feature_service.py  # Feature calculation logic
│   │   │   ├── model_evaluator.py  # Model evaluation logic
│   │   │   └── backtest_service.py # Backtesting logic
│   │   ├── interfaces/             # Domain interfaces (abstractions)
│   │   │   ├── __init__.py
│   │   │   ├── data_provider.py    # IDataProvider interface
│   │   │   ├── feature_generator.py # IFeatureGenerator interface
│   │   │   ├── sentiment_analyzer.py # ISentimentAnalyzer interface
│   │   │   ├── predictor.py        # IPredictor interface
│   │   │   ├── ensemble.py         # IEnsemble interface
│   │   │   ├── backtester.py      # IBacktester interface
│   │   │   └── model_repository.py # IModelRepository interface
│   │   └── repositories/           # Repository interfaces
│   │       ├── __init__.py
│   │       ├── data_repository.py  # IDataRepository interface
│   │       └── model_repository.py # IModelRepository interface
│   │
│   ├── application/                # Application Layer
│   │   ├── __init__.py
│   │   ├── pipelines/              # Data pipelines
│   │   │   ├── __init__.py
│   │   │   ├── feature_pipeline.py # Feature engineering pipeline
│   │   │   ├── missing_value_handler.py
│   │   │   ├── outlier_handler.py
│   │   │   ├── normalizer.py
│   │   │   └── technical_feature_generator.py
│   │   ├── use_cases/              # Use cases
│   │   │   ├── __init__.py
│   │   │   ├── collect_data.py     # CollectDataUseCase
│   │   │   ├── train_model.py      # TrainModelUseCase
│   │   │   ├── predict.py          # PredictUseCase
│   │   │   ├── evaluate_model.py   # EvaluateModelUseCase
│   │   │   ├── backtest.py         # BacktestUseCase
│   │   │   └── ensemble_predict.py # EnsemblePredictUseCase
│   │   ├── dtos/                   # Data Transfer Objects
│   │   │   ├── __init__.py
│   │   │   ├── prediction_dto.py   # Prediction DTO
│   │   │   ├── model_dto.py        # Model DTO
│   │   │   ├── backtest_dto.py     # Backtest DTO
│   │   │   └── trade_dto.py        # Trade DTO
│   │   └── services/               # Application services
│   │       ├── __init__.py
│   │       ├── prediction_service.py # Prediction orchestration
│   │       ├── model_service.py      # Model management
│   │       └── ensemble_service.py   # Ensemble orchestration
│   │
│   ├── infrastructure/             # Infrastructure Layer
│   │   ├── __init__.py
│   │   ├── data_providers/         # Data provider implementations
│   │   │   ├── __init__.py
│   │   │   ├── binance_provider.py # Binance data provider
│   │   │   ├── yfinance_provider.py # Yahoo Finance data provider
│   │   │   ├── coingecko_provider.py # CoinGecko data provider
│   │   │   └── sentiment_provider.py # Sentiment data provider
│   │   ├── feature_generators/     # Feature generator implementations
│   │   │   ├── __init__.py
│   │   │   ├── technical_generator.py # Technical indicators
│   │   │   ├── sentiment_generator.py # Sentiment features
│   │   │   └── macro_generator.py     # Macro features
│   │   ├── repositories/           # Repository implementations
│   │   │   ├── __init__.py
│   │   │   ├── csv_repository.py   # CSV data repository
│   │   │   ├── model_repository.py # Model file repository
│   │   │   └── trade_repository.py # Trade history repository
│   │   ├── models/                 # ML model implementations
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py       # Base model class
│   │   │   ├── linear_regression.py # Linear Regression
│   │   │   ├── xgboost_model.py    # XGBoost
│   │   │   ├── rnn_model.py        # RNN
│   │   │   ├── lstm_model.py       # LSTM
│   │   │   ├── bilstm_model.py     # BiLSTM
│   │   │   └── transformer_model.py # Transformer
│   │   ├── ensemble/               # Ensemble implementations
│   │   │   ├── __init__.py
│   │   │   ├── weighted_ensemble.py # Weighted average
│   │   │   └── stacking_ensemble.py  # Stacking ensemble
│   │   ├── backtesting/            # Backtesting implementations
│   │   │   ├── __init__.py
│   │   │   ├── simple_backtester.py # Simple backtesting
│   │   │   └── advanced_backtester.py # Advanced backtesting
│   │   ├── external/               # External API clients
│   │   │   ├── __init__.py
│   │   │   ├── binance_client.py   # Binance API client
│   │   │   ├── twitter_client.py   # Twitter API client
│   │   │   └── news_client.py      # News API client
│   │   └── persistence/            # Persistence
│   │       ├── __init__.py
│   │       ├── file_storage.py     # File storage
│   │       └── database.py         # Database (optional)
│   │
│   ├── presentation/               # Presentation Layer
│   │   ├── __init__.py
│   │   ├── api/                    # REST API (FastAPI)
│   │   │   ├── __init__.py
│   │   │   ├── routes/             # API routes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prediction.py   # Prediction endpoints
│   │   │   │   ├── model.py        # Model endpoints
│   │   │   │   ├── data.py         # Data endpoints
│   │   │   │   ├── backtest.py     # Backtest endpoints
│   │   │   │   └── ensemble.py     # Ensemble endpoints
│   │   │   ├── schemas/            # Pydantic schemas
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prediction.py   # Prediction schemas
│   │   │   │   └── model.py        # Model schemas
│   │   │   └── app.py              # FastAPI app
│   │   ├── cli/                    # Command Line Interface
│   │   │   ├── __init__.py
│   │   │   └── commands.py         # CLI commands
│   │   ├── dashboard/              # Dashboard (Streamlit)
│   │   │   ├── __init__.py
│   │   │   ├── app.py              # Streamlit app
│   │   │   ├── pages/              # Dashboard pages
│   │   │   │   ├── __init__.py
│   │   │   │   ├── home.py         # Home page
│   │   │   │   ├── prediction.py   # Prediction page
│   │   │   │   ├── backtest.py     # Backtest page
│   │   │   │   └── models.py       # Models comparison page
│   │   │   └── components/         # Reusable components
│   │   │       ├── __init__.py
│   │   │       ├── charts.py       # Chart components
│   │   │       └── tables.py       # Table components
│   │   └── websocket/              # WebSocket handlers
│   │       ├── __init__.py
│   │       └── handler.py          # WebSocket handler
│   │
│   └── core/                       # Core (cross-cutting concerns)
│       ├── __init__.py
│       ├── config.py               # Configuration
│       ├── exceptions.py           # Custom exceptions
│       ├── logging.py              # Logging setup
│       └── container.py            # Dependency Injection container
│
├── tests/                          # Tests
│   ├── unit/                       # Unit tests
│   │   ├── domain/                 # Domain layer tests
│   │   ├── application/            # Application layer tests
│   │   └── infrastructure/         # Infrastructure layer tests
│   ├── integration/                # Integration tests
│   └── e2e/                        # End-to-end tests
│
├── config/                         # Configuration files
│   ├── settings.yaml               # Application settings
│   ├── logging.yaml                # Logging configuration
│   └── models.yaml                # Model configurations
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_analysis.ipynb
│
├── main.py                         # Entry point
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## Data Collection Layer

### Historical Data
**Symbols:** BTC, ETH

**Fields:**
- Open, High, Low, Close, Volume

**Sources:**
- Binance API
- CoinGecko
- Yahoo Finance

### Macro Data
**Gold Price**
- GLD ETF (via Yahoo Finance)

**DXY (US Dollar Index)**
- UUP ETF (via Yahoo Finance)

**Interest Rate**
- Federal Reserve data

**CPI (Consumer Price Index)**
- Bureau of Labor Statistics

### Sentiment Data
**Twitter/X.com**
- Real-time tweets
- Sentiment analysis using LLM

**Reddit**
- r/cryptocurrency, r/Bitcoin
- Sentiment scores

**Crypto News**
- Major crypto news sites
- News impact scoring

## Feature Engineering Layer

Đây là phần quyết định độ chính xác của dự đoán.

### Feature Nhóm 1: OHLCV
- close, open, high, low, volume
- price changes, returns

### Feature Nhóm 2: Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- MA20, MA50 (Moving Averages)
- EMA20, EMA50 (Exponential Moving Averages)
- Bollinger Bands

### Feature Nhóm 3: Sentiment
- Average Sentiment Score
- Positive Ratio
- Negative Ratio
- News Impact Score

### Feature Nhóm 4: Macro
- Gold Return
- DXY Return
- Interest Rate
- CPI

### Kết quả Feature Vector
```
[
    close, volume,
    RSI, MACD, ATR,
    MA20, MA50,
    gold_return, dxy_return,
    news_sentiment, twitter_sentiment
]
```

## Data Pipeline

Tách riêng từng bước để dễ mở rộng.

### FeaturePipeline Components
```
FeaturePipeline
├── MissingValueHandler
├── OutlierHandler
├── TechnicalFeatureGenerator
├── SentimentFeatureGenerator
├── MacroFeatureGenerator
└── Normalizer
```

### Ví dụ Pipeline
```python
pipeline = [
    MissingValueHandler(),
    TechnicalFeatureGenerator(),
    SentimentFeatureGenerator(),
    Normalizer()
]
```

**Lợi ích:** Sau này thêm `WhaleTransactionFeatureGenerator()` không cần sửa code cũ (Open/Closed Principle).

## Model Layer

### Interface Chung
```python
class IPredictor:
    train()
    predict()
    save()
    load()
```

### Model Implementations
- LinearRegressionPredictor
- XGBoostPredictor
- LSTMPredictor
- BiLSTMPredictor
- TransformerPredictor

## Deep Learning Architectures

### LSTM Architecture
```
Input: 30 ngày gần nhất
├── Day1, Day2, ..., Day30
├── LSTM Layer (128 units)
├── Dropout (0.2)
├── LSTM Layer (64 units)
├── Dropout (0.2)
├── Dense (32)
└── Output: BTC ngày 31
```

### BiLSTM Architecture
```
Input Layer
├── BiLSTM Layer (128 units)
├── Dropout (0.3)
├── BiLSTM Layer (64 units)
├── Dropout (0.3)
├── Dense (32)
└── Output: Prediction
```

### Transformer Architecture
```
Input
├── Embedding Layer
├── Positional Encoding
├── Multi-Head Attention (4 heads)
├── Feed Forward Network
├── Layer Normalization
├── Dense (64)
└── Output: Prediction
```

## Ensemble Layer

### Weighted Average Ensemble
```
LSTM (weight: 0.3)
├── BiLSTM (weight: 0.3)
├── Transformer (weight: 0.25)
└── XGBoost (weight: 0.15)
    ↓
Weighted Average
    ↓
Final Prediction
```

### Stacking Ensemble (Meta Learner)
```
LSTM Prediction
├── BiLSTM Prediction
├── Transformer Prediction
└── XGBoost Prediction
    ↓
Meta Learner (Linear Regression)
    ↓
Final Prediction
```

## Evaluation Layer

### Regression Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (R-squared)

### Direction Accuracy (Quan trọng trong trading)
```
Prediction ↑, Reality ↑ → Correct
Prediction ↓, Reality ↓ → Correct
Prediction ↑, Reality ↓ → Wrong
Prediction ↓, Reality ↑ → Wrong
```

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score

## Backtesting Layer

Mô phỏng giao dịch để đánh giá khả năng sinh lợi.

### Trading Strategy
```
Nếu prediction > 2% → BUY
Nếu prediction < -2% → SELL
```

### Backtesting Metrics
- Total Profit
- PnL (Profit and Loss)
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Average Win/Loss Ratio

**Lợi ích:** Phần này giúp báo cáo được đánh giá cao vì không chỉ dự đoán mà còn kiểm tra khả năng sinh lợi thực tế.

## Dashboard

### Tech Stack
- **Backend:** FastAPI
- **Frontend:** Streamlit

### Hiển thị
- Current BTC/ETH Price
- Prediction Tomorrow
- Prediction 7 Days
- Sentiment Trend
- Latest Crypto News
- Model Comparison
- Backtesting Results

## Chi Tách Các Layer

### Domain Layer (Core Business Logic)

**Entities:**
- `OHLCV`: Dữ liệu giá (Open, High, Low, Close, Volume)
- `Prediction`: Kết quả dự đoán
- `ModelMetrics`: Metrics của model (MAE, RMSE, R²)
- `ModelInfo`: Thông tin model (name, version, trained_at)

**Value Objects:**
- `Symbol`: Trading pair (BTCUSDT, ETHUSDT) - immutable
- `Interval`: Time interval (1m, 5m, 1h) - immutable
- `Timestamp`: DateTime wrapper - immutable

**Domain Services:**
- `FeatureCalculator`: Tính technical indicators
- `ModelEvaluator`: Đánh giá model performance

**Interfaces (Abstractions):**
- `IDataReader`: Đọc dữ liệu
- `IDataWriter`: Ghi dữ liệu
- `IModel`: Interface cho tất cả models
- `IPredictor`: Interface cho prediction logic
- `IDataSource`: Interface cho data sources
- `IDataRepository`: Interface cho data storage
- `IModelRepository`: Interface cho model storage

### Application Layer (Orchestration)

**Use Cases:**
- `CollectDataUseCase`: Thu thập dữ liệu từ các sources
- `TrainModelUseCase`: Train model với dữ liệu
- `PredictUseCase`: Dự đoán giá
- `EvaluateModelUseCase`: Đánh giá model

**DTOs:**
- `PredictionDTO`: Data transfer object cho predictions
- `ModelDTO`: Data transfer object cho models
- `DataDTO`: Data transfer object cho data

**Application Services:**
- `PredictionService`: Orchestrate prediction flow
- `ModelService`: Manage models lifecycle

**Ports:**
- `IOutputPort`: Interfaces cho output (API, CLI, etc.)

### Infrastructure Layer (Implementation Details)

**Data Sources:**
- `BinanceDataSource`: Implement IDataSource cho Binance
- `YFinanceDataSource`: Implement IDataSource cho Yahoo Finance
- `SentimentDataSource`: Implement IDataSource cho sentiment

**Repositories:**
- `CSVDataRepository`: Implement IDataRepository với CSV
- `ModelFileRepository`: Implement IModelRepository với files

**Models:**
- `LinearRegressionModel`: Implement IModel
- `RNNModel`: Implement IModel
- `LSTMModel`: Implement IModel
- `TransformerModel`: Implement IModel

**External Clients:**
- `BinanceAPIClient`: Binance API wrapper
- `TwitterAPIClient`: Twitter API wrapper

### Presentation Layer (UI/API)

**API:**
- FastAPI routes cho predictions, models, data
- Pydantic schemas cho validation
- Swagger documentation

**CLI:**
- Click/Argparse commands
- Interactive menus

**WebSocket:**
- Realtime data streaming
- Prediction updates

**Dashboard:**
- Web interface
- Visualization

## Dependency Injection

Sử dụng Dependency Injection container để quản lý dependencies:

```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Data sources
    binance_source = providers.Singleton(
        BinanceDataSource,
        api_key=config.binance.api_key,
        api_secret=config.binance.api_secret
    )
    
    yfinance_source = providers.Singleton(
        YFinanceDataSource
    )
    
    # Repositories
    data_repository = providers.Singleton(
        CSVDataRepository,
        data_dir=config.data.directory
    )
    
    model_repository = providers.Singleton(
        ModelFileRepository,
        model_dir=config.models.directory
    )
    
    # Models
    lr_model = providers.Factory(
        LinearRegressionModel,
        model_repository=model_repository
    )
    
    rnn_model = providers.Factory(
        RNNModel,
        model_repository=model_repository
    )
    
    # Use cases
    collect_data_use_case = providers.Factory(
        CollectDataUseCase,
        data_sources=[binance_source, yfinance_source],
        data_repository=data_repository
    )
    
    predict_use_case = providers.Factory(
        PredictUseCase,
        models=[lr_model, rnn_model],
        data_repository=data_repository
    )
```

## Ví Dụ Code

### Domain Interface

```python
# domain/interfaces/model.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class IModel(ABC):
    """Interface cho tất cả ML models"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model"""
        pass
```

### Infrastructure Implementation

```python
# infrastructure/models/linear_regression.py
from sklearn.linear_model import LinearRegression as SKLearnLR
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from src.domain.interfaces.model import IModel

class LinearRegressionModel(IModel):
    """Linear Regression implementation"""
    
    def __init__(self):
        self.model = SKLearnLR()
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred = self.predict(X)
        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
    
    def save(self, path: str) -> None:
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
```

### Application Use Case

```python
# application/use_cases/predict.py
from src.domain.interfaces.model import IModel
from src.domain.interfaces.data_repository import IDataRepository
from src.application.dtos.prediction_dto import PredictionDTO

class PredictUseCase:
    """Use case cho prediction"""
    
    def __init__(
        self,
        models: list[IModel],
        data_repository: IDataRepository
    ):
        self.models = models
        self.data_repository = data_repository
    
    def execute(self, symbol: str, interval: str) -> PredictionDTO:
        """Execute prediction use case"""
        # Get data
        data = self.data_repository.get_latest_data(symbol, interval)
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Predict with each model
        predictions = {}
        for model in self.models:
            pred = model.predict(features)
            predictions[model.__class__.__name__] = pred
        
        # Ensemble
        ensemble_pred = self._ensemble(predictions)
        
        return PredictionDTO(
            symbol=symbol,
            predictions=predictions,
            ensemble=ensemble_pred,
            timestamp=datetime.now()
        )
```

## Lợi Ích

1. **Testability**: Dễ test vì dependencies có thể mock
2. **Maintainability**: Mỗi layer có trách nhiệm rõ ràng
3. **Extensibility**: Dễ thêm features mới (models, data sources)
4. **Flexibility**: Dễ thay đổi implementations
5. **Independence**: Domain logic không phụ thuộc frameworks
6. **Reusability**: Components có thể reuse ở nhiều nơi

## Các Bước Refactor

1. Tạo cấu trúc thư mục mới
2. Định nghĩa Domain interfaces
3. Move business logic sang Domain layer
4. Create Use cases trong Application layer
5. Implement Infrastructure layer
6. Create Presentation layer
7. Setup Dependency Injection
8. Migrate existing code
9. Write tests
10. Update documentation
