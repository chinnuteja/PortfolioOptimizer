# visualizer.py
# This module is responsible for creating data visualizations.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_performance(prices_df: pd.DataFrame):
    """
    Plots the normalized historical performance of assets.

    Args:
        prices_df (pd.DataFrame): DataFrame with historical asset prices.
    """
    print("\nVisualizing historical data...")
    try:
        # Normalize prices to a starting value of 100
        normalized_prices = (prices_df / prices_df.iloc[0] * 100)
        
        plt.figure(figsize=(14, 8))
        for column in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[column], label=column)
        
        plt.title('Historical Asset Performance (Normalized to 100)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('historical_performance.png')
        plt.close()
        print("Chart of historical performance saved to 'historical_performance.png'")
    except Exception as e:
        print(f"An error occurred during performance plotting: {e}")

def plot_correlation_matrix(returns_df: pd.DataFrame):
    """
    Plots a heatmap of the asset correlation matrix.

    Args:
        returns_df (pd.DataFrame): DataFrame with daily asset returns.
    """
    try:
        correlation_matrix = returns_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Asset Correlation Matrix')
        plt.savefig('correlation_matrix.png')
        plt.close()
        print("Correlation matrix heatmap saved to 'correlation_matrix.png'")
        print("Visualization successful.")
    except Exception as e:
        print(f"An error occurred during correlation plotting: {e}")