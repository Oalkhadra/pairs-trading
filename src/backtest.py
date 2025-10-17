import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define base ticker
BASE = 'XOM'

# Define paths
TEST_PRICES_PATH = './data/raw/prices_test.csv'
PAIRS_PATH = f'./data/processed/stationary_pairs_{BASE}.csv'

# Hyperparameters
PAIRS = [0, 4, 13, 16, 31, 32]   # Indices of chosen pairs

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

def calculate_transaction_costs(signals, transaction_cost=0.0015):
    """
    Calculate transaction costs based on position changes
    
    Each position change requires trading BOTH legs of the pair:
    - Entering/exiting long spread: 2 trades (buy candidate, sell base)
    - Entering/exiting short spread: 2 trades (sell candidate, buy base)
    - Flipping positions: 4 trades total
    
    Parameters:
    -----------
    signals : DataFrame - Trading signals
    transaction_cost : float - Cost per trade, meant to account for slippage, trade costs, short interest (Assume conservative: 15 bps)
    
    Returns:
    --------
    Series of daily transaction costs (as negative returns)

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

if __name__ == '__main__':
    # Read necessary data files
    test_prices_df = pd.read_csv(TEST_PRICES_PATH)
    pairs_df = pd.read_csv(PAIRS_PATH)

    for pair_idx in PAIRS:
        # Select pair to test
        pair = pairs_df.iloc[pair_idx]
        base = pair['base']
        candidate = pair['candidate']
        beta = pair['beta']

        # Read generated signals from df
        TEST_SIGNALS_PATH = f'./data/processed/test_signals_{base}_{candidate}.csv'
        test_signals_df = pd.read_csv(TEST_SIGNALS_PATH)
    
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
        print("\n" + "="*50)
        print("TRAINING PERIOD ANALYSIS")
        print("="*50)
        print(f"Cointegration Test ADF: {pair['adf_stat']:.2}")
        print(f"Mean Reversion Half Life: {pair['half_life']:.4}")
        print(f"Hedge Ratio: {beta:.4}")

        print("\n" + "="*50)
        print("ENTIRE TESTING PERIOD")
        print("="*50)
        print(f"Gross return: {test_cumulative_return:.2%}")
        print(f"Net return: {test_net:.2%}")
        print(f"Total position changes: {test_position_changes}")
        print(f"Total transaction costs: {test_total_costs:.4f}")
        print(f"Average cost per trade: {abs(test_total_costs) / test_position_changes:.4f}")

        # ------------------------------------------------------------------------------------------------------------ 
        # Save Strategy Returns for Benchmark Comparison
        # ------------------------------------------------------------------------------------------------------------
        # Prepare test returns for saving
        strategy_returns = pd.DataFrame({
            'Date': test_net_returns.index,
            'Ticker': 'Strategy',
            'Ret': test_net_returns.values
        })

        # Save to CSV
        strategy_returns.to_csv(f'./data/processed/strategy_returns_{base}_{candidate}.csv', index=False)