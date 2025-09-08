# Portfolio Optimizer

An intelligent portfolio optimization tool that combines historical market data with real-time sentiment analysis to provide personalized investment recommendations.

## Features

- **Risk-Based Optimization**: Three profiles (Conservative, Balanced, Aggressive)
- **Real-Time Sentiment**: Live market sentiment analysis via hosted API
- **Smart Rebalancing**: Turnover-controlled rebalancing with user anchoring
- **Explainable AI**: Natural language explanations for allocation decisions
- **Interactive UI**: Modern web interface with charts and visualizations

## Quick Deploy

### Railway (Recommended)
1. Fork this repository
2. Go to [railway.app](https://railway.app)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Railway will automatically detect and deploy the Python app

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd portfolio_enhancer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Architecture

- **Backend**: Flask API with portfolio optimization engine
- **Frontend**: HTML/CSS/JavaScript with Chart.js
- **Data**: Yahoo Finance for historical data
- **Sentiment**: Hosted sentiment analysis API
- **AI**: OpenRouter integration for explanations

## API Endpoints

- `GET /` - Main application interface
- `POST /get_portfolio` - Get portfolio recommendations
- `GET /health` - Health check
- `GET /download/<filename>` - Download CSV exports

## Configuration

### Environment Variables
Create a `.env` file (copy from `env.example`):
```bash
OPENROUTER_API_KEY=your_api_key_here
FLASK_ENV=development
```

### Risk Profiles
The application uses the following risk profiles:
- **MinRisk**: Conservative, 30% max per asset, 8% turnover cap
- **Sharpe**: Balanced, 35% max per asset, 20% turnover cap  
- **MaxRet**: Aggressive, 40% max per asset, 35% turnover cap

### Security
- **NEVER commit API keys to version control**
- Use environment variables for all sensitive data
- The app will fall back to rule-based explanations if no API key is provided

## License

MIT License - See LICENSE file for details
