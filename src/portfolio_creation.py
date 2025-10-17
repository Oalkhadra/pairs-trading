import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

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
    
    Returns:
        pd.DataFrame: One row per period with columns:
            - period: name of period
            - total_return: cumulative return
            - annualized_return: annualized return
            - volatility: annualized volatility
            - sharpe_ratio: annualized Sharpe
            - max_drawdown: maximum drawdown
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
        
        # Maximum Drawdown
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
