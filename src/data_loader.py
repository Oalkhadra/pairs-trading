import yfinance as yf
import requests
import pandas as pd
from io import StringIO
import os

def ticker_scrape(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)

    # Parse tables
    tables = pd.read_html(StringIO(resp.text))

    sp500 = tables[0]
    tickers = sorted(sp500["Symbol"].tolist())

    df = pd.DataFrame(tickers, columns=["Ticker"])

    csv_path = './data/raw/ticker_list.csv'
    # Store tickers as .csv list    
    df.to_csv(csv_path, index=False)

    print(f"✅ Extracted {len(tickers)} tickers")

    return csv_path

def load_price_data(ticker_list_path):
    # Read ticker data
    ticker_list = pd.read_csv(ticker_list_path)["Ticker"].tolist()
    
    # Initialize empty list of dataframes and output path
    print(f"⬇️ Downloading {len(ticker_list)} tickers...")
    
    # Download data for all tickers
    df = yf.download(ticker_list, 
                     start='2010-01-01', 
                     end='2025-09-01', 
                     group_by="ticker", 
                     auto_adjust=True,
                     progress=True)

    df = df.stack(level="Ticker", future_stack=True).reset_index()
    df = df.set_index('Date')

    out_path = './data/raw/raw_prices.csv'

    # Save to csv
    df.to_csv(out_path)

    print(f"✅ Saved {len(df):,} rows for {df['Ticker'].nunique()} tickers to {out_path}")

    return df

if __name__ == '__main__':
    # Leverage wikipedia of S&P 500 list
    sp500_ticker_list = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Extract tickers and store as .csv
    ticker_list_path = ticker_scrape(sp500_ticker_list)

    # Load price data for all tickers in list
    price_df = load_price_data(ticker_list_path)

