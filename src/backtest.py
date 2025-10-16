# src/backtest.py

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

TRAIN_PRICES_PATH = './data/raw/prices_train.csv'
TEST_PRICES_PATH = './data/raw/prices_test.csv'
PAIRS_PATH = './data/processed/stationary_pairs_XOM.csv'

# Hyperparameters
PAIRS = [0, 4, 13, 16, 31, 32]   # Indeces of chosen pairs

def calculate_portfolio_returns(base_returns, candidate_returns, signals, beta):
    """
    Calculate portfolio returns from pairs trading signals
    
    Strategy returns when:
    - Long spread (signal=1): long candidate, short base
    - Short spread (signal=-1): short candidate, long base
    - Flat (signal=0): no position
    
    Parameters:
    -----------
    base_returns : Series - Daily returns of base stock
    candidate_returns : Series - Daily returns of candidate stock
    signals : DataFrame - Trading signals (1, -1, 0)
    beta : float - Hedge ratio from cointegration
    
    Returns:
    --------
    Series of daily portfolio returns
    
    Hints:
    ------
    - Spread return = candidate_return - beta * base_return
    - When signal=1 (long spread): portfolio_return = spread_return
    - When signal=-1 (short spread): portfolio_return = -spread_return
    - When signal=0: portfolio_return = 0
    - Use lagged signals (shift by 1) to avoid look-ahead bias
    """
    # Shift signals dataframe by 1, then merge with returns
    shifted_signals = signals.shift(1)
    merged_df = pd.merge(base_returns, candidate_returns,   
                         left_index=True, right_index=True, how='inner',
                         suffixes=('_base', '_candidate'))

    # Join signals with return dataframe
    merged_df = pd.merge(merged_df, shifted_signals,    
                         left_index=True, right_index=True, how='inner')

    merged_df['Ret_spread'] = (merged_df['Ret_candidate'] - beta * merged_df['Ret_base']) * merged_df['Signal']

    return merged_df


def calculate_transaction_costs(signals, transaction_cost=0.001):
    """
    Calculate transaction costs based on position changes
    
    Each position change requires trading BOTH legs of the pair:
    - Entering/exiting long spread: 2 trades (buy candidate, sell base)
    - Entering/exiting short spread: 2 trades (sell candidate, buy base)
    - Flipping positions: 4 trades total
    
    Parameters:
    -----------
    signals : DataFrame - Trading signals
    transaction_cost : float - Cost per trade as decimal (0.001 = 0.1% per leg)
    
    Returns:
    --------
    Series of daily transaction costs (as negative returns)
    
    Hints:
    ------
    - Detect position changes: signals != signals.shift(1)
    - Each change costs transaction_cost * 2 (two legs)
    - Return as negative values to subtract from returns
    """
    # Initialize transaction series
    transactions = pd.Series(0, index=signals.index)

    # Iterate through entire Series, leverage if statements to determine transaction cost at each point
    for i in range(len(signals)):
        if i == 0:
            transaction = 0
            continue
        
        current_signal = signals.iloc[i]
        previous_signal = signals.iloc[i-1]

        # If previous position is flat, and current position is not flat, only 2*transaction regardless
        if previous_signal == 0:
            if current_signal != 0:
                transaction = transaction_cost * 2
            else:
                transaction = 0

        # If previous positition is not 0, and current position is opposite of previous, then transaction * 4
        elif previous_signal != 0:
            if current_signal == 0:
                transaction = transaction_cost * 2
            elif current_signal != previous_signal:
                transaction = transaction_cost * 4
            else:
                transaction = 0 

        # Fill transactions series    
        transactions[i] = transaction

    return transactions


def calculate_performance_metrics(returns, signals):
    """
    Calculate strategy performance metrics
    
    Parameters:
    -----------
    returns : Series - Daily portfolio returns (after costs)
    signals : DataFrame - Trading signals
    
    Returns:
    --------
    Dictionary with performance metrics:
    - total_return: Cumulative return over period
    - annualized_return: Geometric mean return * 252
    - annualized_volatility: Std dev of returns * sqrt(252)
    - sharpe_ratio: (annualized_return) / annualized_volatility
    - max_drawdown: Maximum peak-to-trough decline
    - num_trades: Total number of position changes
    - win_rate: Percentage of profitable trades
    - avg_trade_return: Mean return per trade
    
    Hints:
    ------
    - Cumulative return: (1 + returns).cumprod() - 1
    - Annualized return: (1 + total_return) ** (252 / len(returns)) - 1
    - Max drawdown: max((cumulative_max - cumulative) / cumulative_max)
    - For trade-level metrics, segment returns by position changes
    """
    pass


def backtest_strategy(base_prices, candidate_prices, signals, beta, 
                     initial_capital=100000, transaction_cost=0.001):
    """
    Run complete backtest of pairs trading strategy
    
    Parameters:
    -----------
    base_prices : Series - Base stock prices
    candidate_prices : Series - Candidate stock prices
    signals : DataFrame - Trading signals from generate_signals()
    beta : float - Hedge ratio from cointegration
    initial_capital : float - Starting capital (for plotting only)
    transaction_cost : float - Cost per trade as decimal
    
    Returns:
    --------
    Tuple of (results_df, metrics_dict):
    - results_df: DataFrame with columns [date, signal, returns, cumulative_returns, drawdown, equity_curve]
    - metrics_dict: Dictionary of performance metrics
    
    Hints:
    ------
    - Calculate returns using calculate_portfolio_returns()
    - Subtract transaction costs using calculate_transaction_costs()
    - Calculate cumulative returns: (1 + returns).cumprod()
    - Calculate equity curve: cumulative_returns * initial_capital
    - Calculate drawdown at each point: (peak - current) / peak
    - Get metrics using calculate_performance_metrics()
    """
    pass


def plot_backtest_results(results_df, metrics_dict, base, candidate):
    """
    Visualize backtest results
    
    Creates 3-panel plot:
    1. Equity curve with drawdown shading
    2. Cumulative returns vs buy-and-hold benchmark
    3. Underwater plot (drawdown over time)
    
    Parameters:
    -----------
    results_df : DataFrame - Output from backtest_strategy()
    metrics_dict : dict - Performance metrics
    base : str - Base ticker symbol
    candidate : str - Candidate ticker symbol
    
    Returns:
    --------
    Matplotlib figure
    
    Hints:
    ------
    - Plot equity curve as line chart
    - Shade drawdown regions in red
    - Add text box with key metrics (Sharpe, max DD, total return)
    - Include horizontal line at initial capital for reference
    """
    pass


if __name__ == '__main__':
    # Read necessary data files
    train_prices_df = pd.read_csv(TRAIN_PRICES_PATH)
    test_prices_df = pd.read_csv(TEST_PRICES_PATH)
    pairs_df = pd.read_csv(PAIRS_PATH)

    for pair_idx in PAIRS:
        # Select pair to test
        pair = pairs_df.iloc[pair_idx]    # Ensure that this pair is the same as the pair in signals.py!!!!!!!
        base = pair['base']
        candidate = pair['candidate']
        beta = pair['beta']

        # Read generated signals from df
        TRAIN_SIGNALS_PATH = f'./data/processed/train_signals_{base}_{candidate}.csv'
        TEST_SIGNALS_PATH = f'./data/processed/test_signals_{base}_{candidate}.csv'
        train_signals_df = pd.read_csv(TRAIN_SIGNALS_PATH)
        test_signals_df = pd.read_csv(TEST_SIGNALS_PATH)

        # ------------------------------------------------------------------------------------------------------------ Train Backtest ------------------------------------------------------------------------------------------------------------
        
        # Extract returns from prices df
        train_base_returns = train_prices_df[train_prices_df['Ticker'] == base].set_index('Date')['Ret']
        train_candidate_returns = train_prices_df[train_prices_df['Ticker'] == candidate].set_index('Date')['Ret']

        # Extract signals indexed on date
        train_signals = train_signals_df.set_index('Date')['Signal']

        # Calculate portfolio returns and transaction costs
        train_gross_returns = calculate_portfolio_returns(train_base_returns, train_candidate_returns, train_signals, beta)
        train_transaction_costs = calculate_transaction_costs(train_signals, transaction_cost=0.0015)

        # Calculate net and gross returns
        train_net_returns = train_gross_returns['Ret_spread'] - train_transaction_costs
        train_cumulative_return = (1 + train_gross_returns['Ret_spread'].dropna()).prod() - 1
        train_net = (1 + train_net_returns.dropna()).prod() - 1

        # Calculate position changes and total
        train_position_changes = (train_signals != train_signals.shift(1)).sum()
        train_total_costs = train_transaction_costs.sum()

        # ------------------------------------------------------------------------------------------------------------ Test Backtest ------------------------------------------------------------------------------------------------------------

        # Extract returns from prices df
        test_base_returns = test_prices_df[test_prices_df['Ticker'] == base].set_index('Date')['Ret']
        test_candidate_returns = test_prices_df[test_prices_df['Ticker'] == candidate].set_index('Date')['Ret']

        # Extract signals indexed on date
        test_signals = test_signals_df.set_index('Date')['Signal']

        # Calculate portfolio returns and transaction costs
        test_gross_returns = calculate_portfolio_returns(test_base_returns, test_candidate_returns, test_signals, beta)
        test_transaction_costs = calculate_transaction_costs(test_signals, transaction_cost=0.0015)

        # Calculate net and gross returns
        test_net_returns = test_gross_returns['Ret_spread'] - test_transaction_costs
        test_cumulative_return = (1 + test_gross_returns['Ret_spread'].dropna()).prod() - 1
        test_net = (1 + test_net_returns.dropna()).prod() - 1

        # Calculate position changes and total
        test_position_changes = (test_signals != test_signals.shift(1)).sum()
        test_total_costs = test_transaction_costs.sum()

        # Print results
        print("\n" + "="*60)
        print("TRAINING PERIOD")
        print("="*60)
        print(f"Gross return: {train_cumulative_return:.2%}")
        print(f"Net return: {train_net:.2%}")

        print("\n" + "="*60)
        print("TESTING PERIOD")
        print("="*60)
        print(f"Gross return: {test_cumulative_return:.2%}")
        print(f"Net return: {test_net:.2%}")
        print(f"Total position changes: {test_position_changes}")
        print(f"Total transaction costs: {test_total_costs:.4f}")
        print(f"Average cost per trade: {abs(test_total_costs) / test_position_changes:.4f}")

        # ------------------------------------------------------------------------------------------------------------ 
        # Save Strategy Returns for Benchmark Comparison
        # ------------------------------------------------------------------------------------------------------------

        # Prepare training returns for saving
        train_strategy_returns = pd.DataFrame({
            'Date': train_net_returns.index,
            'Ticker': 'Strategy',
            'Ret': train_net_returns.values
        })

        # Prepare test returns for saving
        test_strategy_returns = pd.DataFrame({
            'Date': test_net_returns.index,
            'Ticker': 'Strategy',
            'Ret': test_net_returns.values
        })

        # Combine train and test
        full_strategy_returns = pd.concat([train_strategy_returns, test_strategy_returns], 
                                        ignore_index=True)

        # Save to CSV
        strategy_returns_path = f'./data/processed/strategy_returns_{base}_{candidate}.csv'
        full_strategy_returns.to_csv(strategy_returns_path, index=False)