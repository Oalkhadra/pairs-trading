# src/data_loader.py
import yfinance as yf
import requests
import pandas as pd
from io import StringIO
import numpy as np

def ticker_scrape(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)

    # Parse tables
    tables = pd.read_html(StringIO(resp.text))

    sp500 = tables[0]
    tickers = sorted(sp500["Symbol"].tolist())

    df = pd.DataFrame(tickers, columns=["Ticker"])
    
    # Extract prices for Index Benchmarks for comparison
    benchmarks = pd.DataFrame(['QQQ', 'SPY'], columns=['Ticker'])
    df = pd.concat([df, benchmarks], ignore_index=True)

    csv_path = './data/raw/ticker_list.csv'

    # Store tickers as .csv list    
    df.to_csv(csv_path, index=False)

    print(f"✅ Extracted {len(df)} tickers")

    return csv_path

def load_price_data(ticker_list_path):
    # Read ticker data
    ticker_list = pd.read_csv(ticker_list_path)["Ticker"].tolist()
    
    # Initialize empty list of dataframes and output path
    print(f"⬇️ Downloading {len(ticker_list)} tickers...")
    
    # Download data for all tickers
    df = yf.download(ticker_list, 
                     start='2021-01-01', 
                     end='2025-06-30', 
                     group_by="ticker", 
                     auto_adjust=True,
                     progress=True)

    df = df.stack(level="Ticker", future_stack=True).reset_index()

    # Ensure sorted by Date within each ticker
    df = df.sort_values(["Ticker", "Date"])

    # Compute log return, return, and log close
    df["Ret"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x / x.shift(1) - 1)
    df["LogRet"] = df.groupby("Ticker")["Close"].transform(
        lambda x: np.log(x / x.shift(1)))
    df["LogClose"] = df.groupby("Ticker")["Close"].transform(lambda x: np.log(x))

    # Save to csv
    out_path = './data/raw/raw_prices.csv'
    df.to_csv(out_path)

    print(f"✅ Saved {len(df):,} rows for {df['Ticker'].nunique()} tickers to {out_path}")

    return df

if __name__ == '__main__':
    # Leverage wikipedia of S&P 500 list
    sp500_ticker_list = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Extract tickers and store as .csv
    ticker_list_path = ticker_scrape(sp500_ticker_list)

    # Load price data for all tickers in list
    prices_df = load_price_data(ticker_list_path)

    # Split data into train and test by date (Use first 4 years to estimate output in final year)
    train_start = '2021-01-01'
    train_end = '2024-06-30'
    test_start = '2024-07-01'
    test_end = '2025-06-30'

    print("Splitting data into train and test with the following ranges:\n")
    print(f"Training period between {train_start} and {train_end}")
    print(f"Testing period between {test_start} and {test_end}")

    prices_train = prices_df[(prices_df['Date'] >= train_start) & (prices_df['Date'] <= train_end)]
    prices_test = prices_df[(prices_df['Date'] >= test_start) & (prices_df['Date'] <= test_end)]
    
    prices_train.to_csv( './data/raw/prices_train.csv')
    prices_test.to_csv( './data/raw/prices_test.csv')
    print(f"Successfully split data into {len(prices_train)} rows for training and {len(prices_test)} rows for testing")


    # Store benchmark prices
    benchmark_prices = prices_df[(prices_df['Ticker'] == 'QQQ') | (prices_df['Ticker'] == 'SPY')]
    benchmark_prices.to_csv( './data/raw/benchmark_prices.csv')
    print(f"Successfully extracted {len(benchmark_prices)} rows for benchmarks")