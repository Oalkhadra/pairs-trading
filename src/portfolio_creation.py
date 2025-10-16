"""
Portfolio-level aggregation and analysis for pairs trading strategies.
Combines multiple strategy results into weighted portfolios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


PAIRS = ['KMX', 'ATO', 'DE', 'TRMB', 'REG', 'FRT']

def create_portfolio_returns(strategy_results: Dict[str, pd.DataFrame], 
                             weights: Dict[str, float]) -> pd.Series:
    """
    Combine individual strategy returns into portfolio returns using weights.
    
    Args:
        strategy_results: Dict mapping strategy name to DataFrame with 'returns' column
        weights: Dict mapping strategy (candidate) name to portfolio weight (should sum to 1.0)
    
    Returns:
        pd.Series: Portfolio returns (weighted sum of individual strategy returns)
    
    TODO: 
        1. For each strategy, multiply its returns by its weight
        2. Sum all weighted returns to get portfolio returns
    """
    # Extract, weight, and sum
    first_ticker = list(strategy_results.keys())[0]
    dates = strategy_results[first_ticker]['Date']

    # Extract, weight, and sum
    portfolio_column = sum(
        df['Ret'] * weights[ticker] 
        for ticker, df in strategy_results.items()
    )

    # Create DataFrame with Date column
    portfolio_returns = pd.DataFrame({
        'Date': dates,
        'Ret': portfolio_column
    })

    return portfolio_returns

def calculate_performance_metrics(returns_df: pd.DataFrame, 
                                  date_ranges: List[Tuple[str, str, str]] = None) -> pd.DataFrame:
    """
    Calculate performance metrics for returns, optionally broken down by sub-periods.
    
    Args:
        returns: Dataframe containing returns
        date_ranges: Optional list of (period_name, start_date, end_date) tuples
                    If None, calculates for full period only
                    Example: [("Full Period", "2024-07-01", "2025-06-30"),
                             ("H2 2024", "2024-07-01", "2024-12-31"),
                             ("H1 2025", "2025-01-01", "2025-06-30")]
    
    Returns:
        pd.DataFrame: One row per period with columns:
            - period: name of period
            - total_return: cumulative return
            - annualized_return: annualized return
            - volatility: annualized volatility
            - sharpe_ratio: annualized Sharpe (assume rf=0)
            - max_drawdown: maximum drawdown (negative value)
            - win_rate: % of positive days
    
    TODO:
        1. If date_ranges is None, create one range for full period
        2. For each date range, slice the returns
        3. Calculate all metrics for that period:
           - Total return: (1 + returns).prod() - 1
           - Annualized return: returns.mean() * 252
           - Volatility: returns.std() * sqrt(252)
           - Sharpe: (ann_return / ann_vol)
           - Max drawdown: see helper below
           - Win rate: (returns > 0).mean()
        4. Compile into DataFrame
    
    Hint for max drawdown:
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
    """
    # Extract returns series, indexed on date
    returns = returns_df.set_index('Date')['Ret']

    if date_ranges is None:
        date_ranges = [("Full Period", returns.index.min(), returns.index.max())]

    results = []
    
    for period_name, start_date, end_date in date_ranges:
        period_returns = returns.loc[start_date:end_date]

        # Total return (compound returns)
        total_return = (1 + period_returns).prod() - 1

        # Annualized return (geometric annualization)
        annualized_return = (1 + period_returns).prod() ** (252 / len(period_returns)) - 1

        # Volatility
        volatility = period_returns.std() * np.sqrt(252)

        # Sharpe (Assumes a Risk-Free rate of 4.1%)
        sharpe = (annualized_return - 0.041) / volatility
        
        # Maximum DrawDown
        cum_returns = (1 + period_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        results.append({
                'period': period_name,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Define results dictionary
    results_dict = {}

    # Read in all results for strategies
    for pair in PAIRS:
        return_df = pd.read_csv(f"./data/processed/strategy_returns_XOM_{pair}.csv")
        results_dict[pair] = return_df

    # Define weights for strategies (equal-weighted)
    weights = {'KMX': 1/6,          
                'ATO': 1/6,  # Pairs hedge
                'DE': 1/6,   # Pairs hedge
                'TRMB': 1/6, 
                'REG': 1/6,  # Pairs hedge
                'FRT': 1/6}
    
    # Create portfolio returns, and extract cummulative returns
    portfolio_returns = create_portfolio_returns(results_dict, weights)
    
    # Extract metrics for all periods (Train period (for validity), Testing period (This is the heart of the resutls), Entire period (For lookback assessment, least signficant test))
    date_ranges = [('Full Testing Period', '2024-07-01', '2025-06-30'),
                   ('1st Half Testing', '2024-07-01', '2024-12-31'),
                   ('2nd Half Testing', '2025-01-01', '2025-06-30')]
    
    portfolio_metrics = calculate_performance_metrics(portfolio_returns, date_ranges)

    print(portfolio_metrics)