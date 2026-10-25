"""
Dashboard configuration.

Adding a new page = create pages/<name>.py with a render() function and add
one entry to PAGES below. No other file needs to change.
"""

API_BASE_URL = "http://localhost:8502"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

SYMBOL_COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
}

TIMEFRAME_DAYS = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90,
}

# Page registry: sidebar label -> "pages.<module>"
# Each page module must expose a render() function.
PAGES = {
    "🏠 Overview": "pages.overview",
    "📈 Trend Prediction": "pages.predict",
    "🤖 Model Comparison": "pages.compare",
    "🧪 Insights (Demo)": "pages.insights",
    "⚙ System Status": "pages.status",
}
