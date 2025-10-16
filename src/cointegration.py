# src/cointegration.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
import statsmodels.api as sm

TRAIN_PRICES_PATH = './data/raw/prices_train.csv'

def cointegration_test(df, ticker_1, ticker_2):
    # Extract log prices for both tickers
    ticker_1_prices = df[df['Ticker'] == ticker_1][["Date", "LogClose"]]
    ticker_2_prices = df[df['Ticker'] == ticker_2][["Date", "LogClose"]]

    # Merge data and rename columns
    merged = pd.merge(ticker_1_prices, ticker_2_prices, on='Date')
    merged = merged[["LogClose_x", "LogClose_y"]]
    merged = merged.rename(columns={'LogClose_x': f'LogClose_{ticker_1}', 'LogClose_y': f'LogClose_{ticker_2}'})

    try:
        coint_result = coint_johansen(merged, det_order=0, k_ar_diff=1)
    except np.linalg.LinAlgError:
        return {"status": "Johansen failed", "beta": None}

    is_cointegrated = coint_result.lr1[0] > coint_result.cvt[0][2]
    
    # Calculate OLS hedge ratio
    X = merged.iloc[:, 0].values  # ticker_1
    Y = merged.iloc[:, 1].values  # ticker_2
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(Y, X_with_const).fit()
    ols_beta = ols_model.params[1]
    
    result = {
        "base": ticker_1,
        "candidate": ticker_2,
        "status": "Candidate Pair" if is_cointegrated else "Not a candidate",
        "test_statistic": coint_result.lr1[0],
        "critical_value_5pct": coint_result.cvt[0][1],
        "beta": ols_beta  # More interpretable for trading
    }
    
    return result

def run_cointegration(df, base, candidates):
    # Initiate results dictionary
    cointegration_results = {}

    for ticker in tqdm(candidates, desc="Running cointegration test..."):
        cointegration_results[f"{base} and {ticker}"] = cointegration_test(df, base, ticker)

    return cointegration_results
    
def half_life(spread):
    spread_lag = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    
    # Align indexes
    spread_lag = spread_lag.loc[delta_spread.index]
    
    # Regress Δz_t on z_{t-1}
    model = sm.OLS(delta_spread, sm.add_constant(spread_lag))
    res = model.fit()
    beta = res.params.iloc[1]
                               
    # If beta >= 0, no mean reversion
    if beta >= 0:
        return np.inf, res.rsquared
    
    halflife = -np.log(2) / beta

    return halflife, res.rsquared

def stationarity_half_life(df, coint_results):
    # Pre-filter to only cointegrated pairs
    cointegrated_pairs = {
        pair: result for pair, result in coint_results.items() 
        if result["status"] == "Candidate Pair"
    }
    print(f"✅ Found {len(cointegrated_pairs)} cointegrated pairs. Running stationarity tests...")

    # Define list of results
    results = []
    
    for pair, result in tqdm(cointegrated_pairs.items(), desc='Checking for stationarity...'):
        if result["status"] != "Candidate Pair":
            continue
        
        # Extract metadata
        base = result["base"]
        cand = result["candidate"]
        beta = result["beta"]

        # Merge log prices
        base_prices = df[df['Ticker'] == base][["Date", "LogClose"]].rename(columns={'LogClose': f'LogClose_{base}'})
        cand_prices = df[df['Ticker'] == cand][["Date", "LogClose"]].rename(columns={'LogClose': f'LogClose_{cand}'})
        merged = pd.merge(base_prices, cand_prices, on="Date")

        # Build spread
        spread = merged[f'LogClose_{cand}'] - beta * merged[f'LogClose_{base}']
       
        # Run ADF and Half Life regression
        try:
            adf_stat, pval, _, _, crit_vals, _ = adfuller(spread)
            halflife, r2 = half_life(spread)
            
            results.append({
                "pair": pair,
                "base": base,
                "candidate": cand,
                "beta": beta,
                "adf_stat": adf_stat,
                "pval": pval,
                "crit_1pct": crit_vals["1%"],
                "is_stationary": pval < 0.01,  # reject unit root at 1% (I want 99% confidence)
                "half_life": halflife,
                "r2": r2
            })
        except Exception as e:
            results.append({
                "pair": pair,
                "error": str(e)
            })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Define ticker you are looking for cointegrated pair candidates with
    base_ticker = 'XOM'

    # Read data and identify all other tickers
    prices_df = pd.read_csv(TRAIN_PRICES_PATH)
    candidate_tickers = prices_df['Ticker'].unique().tolist() 
    candidate_tickers.remove(base_ticker)

    # Run cointegration test on base ticker and all other tickers
    coint_results = run_cointegration(prices_df, base_ticker, candidate_tickers)

    # Run ADF test on candidate pairs to decipher tradeable pairs
    stationarity_test = stationarity_half_life(prices_df, coint_results=coint_results)
    
    # Sort by half life and save results
    output_path = f'./data/processed/stationary_pairs_{base_ticker}.csv'
    stationary_pairs = stationarity_test[stationarity_test['is_stationary'] == True]
    stationary_pairs = stationary_pairs.sort_values(by='half_life')

    stationary_pairs.to_csv(output_path, index=False)
    print(f"\n✅ Found {len(stationary_pairs)} stationary pairs")
    print(f"💾 Saved to {output_path}")
    print("\nStationary Pairs:")
    print(stationary_pairs[['base', 'candidate', 'beta', 'pval', 'half_life']])