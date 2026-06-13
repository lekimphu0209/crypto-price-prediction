"""
Trading Performance Evaluator

Evaluates trading strategy performance using backtesting results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    total_return: float
    total_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'Total Return (%)': self.total_return,
            'Total Profit ($)': self.total_profit,
            'Total Trades': self.total_trades,
            'Win Rate (%)': self.win_rate,
            'Profit Factor': self.profit_factor,
            'Sharpe Ratio': self.sharpe_ratio,
            'Max Drawdown (%)': self.max_drawdown,
            'Avg Win ($)': self.avg_win,
            'Avg Loss ($)': self.avg_loss
        }


class TradingEvaluator:
    """Evaluates trading strategy performance."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize Trading Evaluator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.metrics_history = []
    
    def calculate_total_return(self, initial_capital: float, final_capital: float) -> float:
        """Calculate total return percentage."""
        return ((final_capital - initial_capital) / initial_capital) * 100
    
    def calculate_win_rate(self, winning_trades: int, total_trades: int) -> float:
        """Calculate win rate percentage."""
        return (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    def calculate_profit_factor(self, total_profit: float, total_loss: float) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        return total_profit / abs(total_loss) if total_loss != 0 else float('inf')
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, periods: int = 252) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Array of periodic returns
            periods: Number of periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods)
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            Maximum drawdown as percentage
        """
        peak = equity_curve[0]
        max_drawdown = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown * 100
    
    def evaluate_from_trades(self, trades: List[Dict], 
                            initial_capital: float = 10000) -> TradingMetrics:
        """
        Evaluate trading performance from a list of trades.
        
        Args:
            trades: List of trade dictionaries with 'profit' key
            initial_capital: Starting capital
            
        Returns:
            TradingMetrics object
        """
        total_trades = len(trades)
        profits = [trade['profit'] for trade in trades]
        
        winning_trades = sum(1 for p in profits if p > 0)
        losing_trades = sum(1 for p in profits if p < 0)
        
        total_profit = sum(profits)
        total_loss = sum(p for p in profits if p < 0)
        
        final_capital = initial_capital + total_profit
        
        metrics = TradingMetrics(
            total_return=self.calculate_total_return(initial_capital, final_capital),
            total_profit=total_profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=self.calculate_win_rate(winning_trades, total_trades),
            profit_factor=self.calculate_profit_factor(total_profit, total_loss),
            sharpe_ratio=0.0,  # Need returns array for this
            max_drawdown=0.0,   # Need equity curve for this
            avg_win=np.mean([p for p in profits if p > 0]) if winning_trades > 0 else 0,
            avg_loss=np.mean([p for p in profits if p < 0]) if losing_trades > 0 else 0
        )
        
        return metrics
    
    def evaluate_from_equity_curve(self, equity_curve: np.ndarray,
                                   initial_capital: float = 10000) -> TradingMetrics:
        """
        Evaluate trading performance from equity curve.
        
        Args:
            equity_curve: Array of portfolio values over time
            initial_capital: Starting capital
            
        Returns:
            TradingMetrics object
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_profit = equity_curve[-1] - initial_capital
        total_trades = len(equity_curve) - 1  # Assuming one trade per period
        
        # Simple trade counting based on direction changes
        winning_periods = sum(1 for r in returns if r > 0)
        losing_periods = sum(1 for r in returns if r < 0)
        
        metrics = TradingMetrics(
            total_return=self.calculate_total_return(initial_capital, equity_curve[-1]),
            total_profit=total_profit,
            total_trades=total_trades,
            winning_trades=winning_periods,
            losing_trades=losing_periods,
            win_rate=self.calculate_win_rate(winning_periods, total_trades),
            profit_factor=0.0,  # Need trade-level data for this
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            max_drawdown=self.calculate_max_drawdown(equity_curve),
            avg_win=np.mean(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0,
            avg_loss=np.mean(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
        )
        
        return metrics
    
    def compare_models(self, results: Dict[str, TradingMetrics]) -> pd.DataFrame:
        """
        Compare trading performance of multiple models.
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison table
        """
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                **metrics.to_dict()
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by Total Return (most important)
        df = df.sort_values('Total Return (%)', ascending=False)
        
        return df
    
    def print_report(self, metrics: TradingMetrics, model_name: str = "Strategy"):
        """Print a formatted report of trading metrics."""
        print(f"\n{'='*50}")
        print(f"Trading Performance Report: {model_name}")
        print(f"{'='*50}")
        print(f"Total Return:      {metrics.total_return:.2f}%")
        print(f"Total Profit:      ${metrics.total_profit:.2f}")
        print(f"Total Trades:      {metrics.total_trades}")
        print(f"Win Rate:          {metrics.win_rate:.2f}%")
        print(f"Profit Factor:     {metrics.profit_factor:.2f}")
        print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown:      {metrics.max_drawdown:.2f}%")
        print(f"Avg Win:           ${metrics.avg_win:.2f}")
        print(f"Avg Loss:          ${metrics.avg_loss:.2f}")
        print(f"{'='*50}\n")
