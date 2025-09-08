# config.py
# Central configuration for the portfolio analysis tool.

# --- ASSET DEFINITIONS ---
# Define the tickers for the assets you want to analyze.
ASSETS = {
    'Equities': '^NSEI',       # Nifty 50 Index for India
    'Gold': 'GC=F',           # Gold Futures
    'REITs': 'VNQ',           # Vanguard Real Estate ETF (a global proxy)
    'Bitcoin': 'BTC-USD'       # Bitcoin in USD
}

# --- SENTIMENT PROXY ---
# Define the ticker for our market sentiment proxy
SENTIMENT_TICKER = '^INDIAVIX'

# --- DATE RANGE ---
# Define the start and end dates for historical data analysis.
START_DATE = '2015-01-01'
END_DATE = '2025-09-05' # Use the current date

