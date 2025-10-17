import pandas as pd
import yfinance as yf
from datetime import datetime
from math import sqrt 

# Paths
BENCHMARK_PRICES_PATH = './data/raw/benchmark_prices.csv'

# Date ranges
TRAIN_START = '2021-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-06-30'

def calculate_benchmark_returns(prices_df, period_start, period_end):
    """
    Calculate buy-and-hold returns for each benchmark over a period
    
    Parameters:
    -----------
    prices_df : DataFrame - Benchmark prices with [Date, Ticker, Ret]
    period_start : str - Period start date
    period_end : str - Period end date
    
    Returns:
    --------
    DataFrame with columns [Ticker, total_return, annualized_return, 
                           annualized_volatility, sharpe_ratio]

    """
    # Slice dataframe based on dates
    prices_sliced = prices_df[(prices_df['Date'] >= period_start) & (prices_df['Date'] <= period_end)]
    num_days = len(prices_sliced)

    # Calculate benchmark metrics
    total_return = (1 + prices_sliced['Ret']).prod() - 1
    annualized_return = (1 + total_return) ** (252 / num_days) - 1
    annualized_volatility = prices_sliced['Ret'].std() * sqrt(252)

    return total_return, annualized_return, annualized_volatility

if __name__ == '__main__':
    # Read in benchmark prices dataframe, extract benchmark
    benchmark_prices_df = pd.read_csv(BENCHMARK_PRICES_PATH)
    qqq = benchmark_prices_df[benchmark_prices_df['Ticker'] == 'QQQ']
    spy = benchmark_prices_df[benchmark_prices_df['Ticker'] == 'SPY']

    # Calculate benchmark returns
    tr_qqq, ar_qqq, v_qqq = calculate_benchmark_returns(qqq,
                                                    TEST_START,
                                                    TEST_END) 

    tr_spy, ar_spy, v_spy = calculate_benchmark_returns(spy,
                                                    TEST_START,
                                                    TEST_END)
                    
    
    print(f"QQQ Total Return: {tr_qqq}")
    print(f"QQQ Annualized Return: {ar_qqq}")   
    print(f"QQQ Annualized Volatility: {v_qqq}")   

    print(f"SPY Total Return: {tr_spy}")
    print(f"SPY Annualized Return: {ar_spy}")
    print(f"SPY Annualized Volatility: {v_spy}")
