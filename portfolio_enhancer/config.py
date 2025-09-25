# config.py
# Central configuration for the portfolio analysis tool.

# --- ASSET DEFINITIONS ---
ASSETS = {
    'Equities': '^NSEI',   # Nifty 50 Index (India)
    'Gold': 'GC=F',        # Gold Futures
    'REITs': 'VNQ',        # Vanguard Real Estate ETF (US proxy)
    'Bitcoin': 'BTC-USD'   # Bitcoin in USD (24/7)
}

# --- SENTIMENT PROXY ---
SENTIMENT_TICKER = '^INDIAVIX'

# --- DATE RANGE ---
# START_DATE fixed; END_DATE can be static OR None (rolling to today).
START_DATE = '2015-01-01'
END_DATE = None  # None => the app will treat END_DATE as "today (UTC)" on each refresh
