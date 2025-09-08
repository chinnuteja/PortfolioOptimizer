# main.py
# Final, definitive version that orchestrates the full, intelligent workflow.

# --- Import Project Modules ---
import config
import data_fetcher
import data_preparer
import forecaster
import optimizer
import visualizer
import data_saver
from api_client import AssetSentimentAPI

 

def run_portfolio_analysis():
    """
    Executes the final, robust, and intelligent workflow.
    """
    print("--- Starting Intelligent Riskfolio Workflow ---")

    # 1. Data Fetching and Preparation
    asset_prices, _ = data_fetcher.get_data(config.ASSETS, config.SENTIMENT_TICKER, config.START_DATE, config.END_DATE)
    if asset_prices.empty: return
        
    data_saver.save_data_to_csv(asset_prices, 'historical_prices.csv')
    asset_returns = data_preparer.calculate_returns(asset_prices)
    if asset_returns.empty: return
        
    visualizer.plot_performance(asset_prices)
    visualizer.plot_correlation_matrix(asset_returns)

    # 2. Live Sentiment Analysis via hosted API (Dynamically for active assets)
    print("\nFetching live asset sentiments from backend…")
    api = AssetSentimentAPI()
    # Request sentiments for the configured asset names
    sentiments = api.analyze_and_get_sentiments(assets=list(config.ASSETS.keys()), wait_s=60)
    # Fallback to neutral if any missing
    asset_sentiment_scores = {a: sentiments.get(a, 0.0) for a in config.ASSETS.keys()}
    print("Live sentiments:", asset_sentiment_scores)
    
    # 3. Generate Intelligent Forecasts (using the sentiment scores)
    expected_returns = forecaster.generate_forecasted_returns(asset_returns, asset_sentiment_scores)

    # 4. Optimization for different risk profiles
    profiles = {
        "BALANCED (MAX SHARPE RATIO)": "Sharpe",
        "CONSERVATIVE (MINIMUM RISK)": "MinRisk",
        "AGGRESSIVE (MAXIMUM RETURN)": "MaxRet"
    }

    for name, objective in profiles.items():
        weights, performance = optimizer.get_optimal_portfolio(asset_returns, expected_returns, objective=objective)
        if not weights.empty:
            print(f"\n--- {name} ---")
            print((weights * 100).to_string(float_format="%.2f%%"))
            if performance:
                print(f"\nExpected annual return: {performance[0]*100:.2f}%")
                print(f"Annual volatility: {performance[1]*100:.2f}%")
                print(f"Sharpe Ratio: {performance[2]:.2f}")
        
    print("\n--- Workflow Finished ---")

if __name__ == "__main__":
    run_portfolio_analysis()

