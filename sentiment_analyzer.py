# sentiment_analyzer.py
# This module fetches live news and calculates a sentiment score using FinBERT.

from transformers import pipeline
import requests

# --- Initialize the FinBERT model once when the module is loaded ---
# This is efficient as it prevents reloading the model on every call.
try:
    print("Loading FinBERT model... (This may take a moment on first run)")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("FinBERT model loaded successfully.")
except Exception as e:
    print(f"Could not load FinBERT model. Please ensure you have an internet connection and have run 'pip install transformers torch'. Error: {e}")
    sentiment_pipeline = None
# --------------------------------------------------------------------

def get_daily_sentiment_score(api_key: str, keywords: str) -> (float, int):
    """
    Fetches today's news headlines, analyzes them with FinBERT, and returns
    an aggregated sentiment score.

    Args:
        api_key (str): Your free API key from newsapi.org.
        keywords (str): The search keywords for the news articles.

    Returns:
        tuple(float, int): A tuple containing:
            - The average sentiment score for the day (-1 to +1).
            - The number of articles analyzed.
    """
    if not sentiment_pipeline:
        print("Sentiment pipeline not available.")
        return 0.0, 0

    # --- 1. Fetch News Headlines ---
    url = f'https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        
        if not articles:
            # print("No news articles found for today.")
            return 0.0, 0
            
        headlines = [article['title'] for article in articles if article['title']]
        if not headlines:
            return 0.0, 0

    except requests.exceptions.RequestException as e:
        print(f"Could not fetch news from API: {e}")
        return 0.0, 0

    # --- 2. Analyze Sentiment ---
    results = sentiment_pipeline(headlines)
    
    # --- 3. Aggregate Scores ---
    # Convert labels to numerical scores: positive=1, negative=-1, neutral=0
    total_score = 0
    for res in results:
        if res['label'] == 'positive':
            total_score += res['score']
        elif res['label'] == 'negative':
            total_score -= res['score']
            # Neutral scores are treated as 0 and don't change the total_score
            
    # Return the average score
    average_score = total_score / len(headlines) if headlines else 0.0
    
    return average_score, len(headlines)

