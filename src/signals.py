import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define base ticker
BASE = 'XOM'

# Define paths
TRAIN_PRICES_PATH = './data/raw/prices_train.csv'
TEST_PRICES_PATH = './data/raw/prices_test.csv'
PAIRS_PATH = f'./data/processed/stationary_pairs_{BASE}.csv'

# Define hyperparameters
LOOKBACK = 23                   # Days (Extracted from pair_selection.ipynb)
ENTRY_THRESHOLD = 2.0           # Z-score (σ)
EXIT_THRESHOLD = 0.0            # Z-score (σ)
STOP_BUFFER = 1.0               # Z-score (σ)
PAIRS = [0, 4, 13, 16, 31, 32]   # Indices of chosen pairs 

def build_spread(prices_df, base, candidate, beta): # (Spread = candidate - β * base)
    """
    Construct the spread between two assets using the hedge ratio
    
    Parameters:
    -----------
    prices_df : DataFrame with columns [Date, Ticker, LogClose]
    base : Base ticker symbol
    candidate : Candidate ticker symbol  
    beta : Hedge ratio from cointegration
    
    Returns:
    --------
    DataFrame with columns [Date, spread]
    """
    # Extract individual dataframes and combine for base and candidate tickers
    base_df = prices_df[prices_df['Ticker'] == base][['Date', 'LogClose']]
    candidate_df = prices_df[prices_df['Ticker'] == candidate][['Date', 'LogClose']]
    merged_df = pd.merge(base_df, candidate_df, 
                         how='inner', 
                         on='Date',
                         suffixes=(f"_{base}", f"_{candidate}"))

    # Calculate spread 
    merged_df['Spread'] = merged_df[f'LogClose_{candidate}'] - beta * merged_df[f'LogClose_{base}'] 
    spread_df = merged_df[['Date', 'Spread']].set_index('Date')

    return spread_df

def calculate_zscore(spread, lookback=20):
    """
    Calculate rolling z-score of the spread
    
    Parameters:
    -----------
    spread : Series or DataFrame column containing spread values
    lookback : Rolling window size for mean/std calculation
    
    Returns:
    --------
    Series with z-scores
    """
    # Calculate rolling mean and standard devation
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_stdev = spread.rolling(window=lookback).std()

    rolling_z = (spread - rolling_mean) / rolling_stdev

    return rolling_z

def generate_signals(zscore, entry_threshold=2.5, exit_threshold=0.0, stop_buffer=0.5):
    """
    Generate trading signals based on z-score thresholds
    
    Parameters:
    -----------
    zscore : Series of z-scores
    entry_threshold : Absolute z-score value to enter position
    exit_threshold : Z-score value to exit position (typically 0)
    stop_loss : Absolute z-score value to stop out
    
    Returns:
    --------
    DataFrame with columns [Date, signal, position]
    signal: 1 ong spread)(l, -1 (short spread), 0 (no position)
    position: current position state
    """
    # Initialize signals series and base position (0)
    signals = pd.Series(0, index=zscore.index)
    position = 0

    for i in range(len(zscore)):

        current_z = zscore.iloc[i][0]

        # For rolling Z, start with flat positions until Z scores are computed
        if current_z is None:
            position = 0

        # If flat, options are to short or long depending on z score of spread
        if position == 0:
            if current_z >= entry_threshold:
                position = -1
            elif current_z <= -entry_threshold:
                position = 1

        # If short, option is to long when opposite threshold is reach, or leave position when stop threshold is engaged
        elif position == -1:
            if current_z >= (entry_threshold + stop_buffer): # If spread continues to grow, stop loss should be engaged
                position = 0
            elif current_z <= (-entry_threshold - stop_buffer): # If spread flips past negative stop threshold, stop loss should also be engaged
                position = 0
            elif current_z <= exit_threshold:
                position = 0 # If the z_score passes the mean, but not fully swings to opposite position, take profits // Reason is to stay semi-aggressive
            elif current_z <= -entry_threshold: # If spread reaches long threshold, switch positions
                position = 1
            
        # If long, option is to short when opposite threshold is reach, or leave position when stop threshold is engaged
        elif position == 1:
            if current_z <= (-entry_threshold - stop_buffer): # If spread continues to grow, stop loss should be engaged
                position = 0
            elif current_z >= (entry_threshold + stop_buffer): # If spread flips past negative stop threshold, stop loss should also be engaged
                position = 0
            elif current_z >= entry_threshold: # If spread reaches short threshold, switch positions
                position = -1


        signals.iloc[i] = position
    
    signals = pd.DataFrame(signals)
    signals.index = zscore.index
    signals.columns = ['signal']

    return signals


if __name__ == '__main__':
    # Load data
    prices_train = pd.read_csv(TRAIN_PRICES_PATH)
    prices_test = pd.read_csv(TEST_PRICES_PATH)
    pairs_df = pd.read_csv(PAIRS_PATH)
    
    for pair_idx in PAIRS:
        pair = pairs_df.iloc[pair_idx]
        base = pair['base']
        candidate = pair['candidate']
        beta = pair['beta']
        print(f"\nCandidate ticker is {candidate}, with a hedge ratio of {beta}\n")

        # Build spread on test data
        test_spread = build_spread(prices_test, base=base, candidate=candidate, beta=beta)

        # Calculate z-score     
        test_z = calculate_zscore(test_spread, 
                                    lookback=LOOKBACK) # Define lookback window (days)
        
        # Generate signals
        test_signals = generate_signals(test_z,
                                entry_threshold=ENTRY_THRESHOLD,
                                exit_threshold=EXIT_THRESHOLD,
                                stop_buffer=STOP_BUFFER)

        # Combine into single dataframe       
        test_results_df = test_spread.copy()
        test_results_df['Z_score'] = test_z
        test_results_df['Signal'] = test_signals['signal']

        # Save to CSV
        test_results_df.to_csv(f'./data/processed/test_signals_{base}_{candidate}.csv')

        print(f"Generated signals for {base}-{candidate}")
        print(f"Saved to: ./data/signals/test_signals_{base}_{candidate}.csv")