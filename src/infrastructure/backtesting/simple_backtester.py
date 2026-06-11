"""
Simple Backtesting Implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    """Result of backtesting"""
    total_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class SimpleBacktester:
    """Simple backtesting engine for trading strategies"""
    
    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.001
    ):
        """
        Args:
            buy_threshold: Buy when predicted change > threshold
            sell_threshold: Sell when predicted change < threshold
            initial_capital: Initial capital for backtesting
            transaction_fee: Transaction fee per trade
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
    
    def backtest(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> BacktestResult:
        """
        Run backtest
        
        Args:
            predictions: Predicted prices or returns
            actual_prices: Actual prices
            timestamps: Optional timestamps for each prediction
        
        Returns:
            BacktestResult object
        """
        capital = self.initial_capital
        position = 0.0  # Amount of crypto held
        trades = []
        
        for i in range(len(predictions) - 1):
            current_price = actual_prices[i]
            next_price = actual_prices[i + 1]
            
            # Calculate predicted change
            predicted_change = (predictions[i] - current_price) / current_price
            
            # Trading logic
            if predicted_change > self.buy_threshold and position == 0:
                # Buy signal
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'timestamp': timestamps[i] if timestamps else i
                })
            
            elif predicted_change < self.sell_threshold and position > 0:
                # Sell signal
                capital = position * current_price * (1 - self.transaction_fee)
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'timestamp': timestamps[i] if timestamps else i
                })
        
        # Close final position
        if position > 0:
            capital = position * actual_prices[-1] * (1 - self.transaction_fee)
            position = 0
        
        # Calculate metrics
        total_profit = capital - self.initial_capital
        total_trades = len(trades) // 2  # Buy + sell pairs
        
        winning_trades = 0
        losing_trades = 0
        wins = []
        losses = []
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_price = trades[i]['price']
                sell_price = trades[i + 1]['price']
                profit_pct = (sell_price - buy_price) / buy_price - self.transaction_fee
                
                if profit_pct > 0:
                    winning_trades += 1
                    wins.append(profit_pct)
                else:
                    losing_trades += 1
                    losses.append(abs(profit_pct))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(actual_prices)):
            returns.append((actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1])
        
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        equity_curve = [self.initial_capital]
        current_equity = self.initial_capital
        for trade in trades:
            if trade['type'] == 'sell':
                current_equity = current_equity * (1 + 0.02)  # Simplified
                equity_curve.append(current_equity)
        
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return BacktestResult(
            total_profit=total_profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor
        )
