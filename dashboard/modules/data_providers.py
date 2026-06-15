"""
Data Providers Module
Handles data fetching and conversion for dashboard
"""
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.data_providers.yfinance_provider import YahooFinanceProvider
import streamlit as st


# Initialize data providers
@st.cache_resource
def get_data_providers():
    """Get cached data providers"""
    binance = BinanceProvider()
    yfinance = YahooFinanceProvider()
    return binance, yfinance


def ohlcv_to_dataframe(ohlcv_list):
    """Convert OHLCV list to DataFrame"""
    if not ohlcv_list:
        return pd.DataFrame()
    df = pd.DataFrame([{
        'timestamp': o.timestamp,
        'open': o.open,
        'high': o.high,
        'low': o.low,
        'close': o.close,
        'volume': o.volume
    } for o in ohlcv_list])
    df.set_index('timestamp', inplace=True)
    return df


@st.cache_data(ttl=60)
def get_real_prices(_binance):
    """Fetch real-time prices from Binance"""
    try:
        btc_ohlcv = _binance.fetch_ohlcv("BTCUSDT", "1d", limit=2)
        btc_df = ohlcv_to_dataframe(btc_ohlcv)
        btc_current = btc_df['close'].iloc[-1] if len(btc_df) > 0 else None
        btc_prev = btc_df['close'].iloc[-2] if len(btc_df) > 1 else None
        btc_change = ((btc_current - btc_prev) / btc_prev * 100) if btc_prev and btc_current else 0
        
        eth_ohlcv = _binance.fetch_ohlcv("ETHUSDT", "1d", limit=2)
        eth_df = ohlcv_to_dataframe(eth_ohlcv)
        eth_current = eth_df['close'].iloc[-1] if len(eth_df) > 0 else None
        eth_prev = eth_df['close'].iloc[-2] if len(eth_df) > 1 else None
        eth_change = ((eth_current - eth_prev) / eth_prev * 100) if eth_prev and eth_current else 0
        
        return {
            'btc_price': btc_current,
            'btc_change': btc_change,
            'eth_price': eth_current,
            'eth_change': eth_change
        }
    except Exception as e:
        return None


TIMEFRAME_DAYS = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90,
}


def timeframe_to_days(timeframe: str) -> int:
    """Convert sidebar timeframe label to number of days."""
    return TIMEFRAME_DAYS.get(timeframe, 30)


@st.cache_data(ttl=300)
def get_historical_data(_binance, symbol_binance, days=30):
    """Fetch historical data from Binance"""
    try:
        ohlcv_data = _binance.fetch_ohlcv(symbol_binance, "1d", limit=days)
        df = ohlcv_to_dataframe(ohlcv_data)
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def get_macro_data(_yfinance):
    """Fetch macro data from Yahoo Finance"""
    try:
        gold_symbols = ["GC=F", "GLD", "XAUUSD=X"]
        dxy_symbols = ["DX-Y.NYG", "UUP", "DXY"]
        
        gold_df = None
        dxy_df = None
        
        for gold_sym in gold_symbols:
            try:
                gold_ohlcv = _yfinance.fetch_ohlcv(gold_sym, "1d", limit=30)
                gold_df = ohlcv_to_dataframe(gold_ohlcv)
                if gold_df is not None and len(gold_df) > 0:
                    break
            except:
                continue
        
        for dxy_sym in dxy_symbols:
            try:
                dxy_ohlcv = _yfinance.fetch_ohlcv(dxy_sym, "1d", limit=30)
                dxy_df = ohlcv_to_dataframe(dxy_ohlcv)
                if dxy_df is not None and len(dxy_df) > 0:
                    break
            except:
                continue
        
        if gold_df is not None and len(gold_df) > 0 and dxy_df is not None and len(dxy_df) > 0:
            return gold_df, dxy_df
    except:
        pass
    
    # Fallback to mock data
    try:
        from datetime import datetime, timezone, timedelta
        import numpy as np
        
        dates = pd.date_range(start=datetime.now(timezone.utc) - timedelta(days=30), periods=30, freq='D', tz='UTC')
        
        gold_prices = np.random.randn(30).cumsum() * 5 + 235
        gold_df = pd.DataFrame({
            'open': gold_prices + np.random.randn(30) * 2,
            'high': gold_prices + np.random.rand(30) * 5,
            'low': gold_prices - np.random.rand(30) * 5,
            'close': gold_prices,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        
        dxy_prices = np.random.randn(30).cumsum() * 0.5 + 105
        dxy_df = pd.DataFrame({
            'open': dxy_prices + np.random.randn(30) * 0.2,
            'high': dxy_prices + np.random.rand(30) * 0.5,
            'low': dxy_prices - np.random.rand(30) * 0.5,
            'close': dxy_prices,
            'volume': np.random.randint(500000, 2000000, 30)
        }, index=dates)
        
        return gold_df, dxy_df
    except Exception as e:
        return None, None
