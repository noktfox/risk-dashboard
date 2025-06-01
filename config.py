import os
from datetime import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API CONFIGURATION
API_KEY      = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
API_TIMEOUT  = int(os.getenv("API_TIMEOUT", 10))

BASE_DIR = Path(__file__).parent

# LOGGING CONFIGURATION
LOG_LEVEL    = "INFO"
LOG_DIR      = BASE_DIR / "logs"
LOG_FILENAME = "app.log"

# DATA PATHS
RAW_DATA_DIR        = BASE_DIR / "data" / "raw"
CACHE_DATA_DIR      = BASE_DIR / "data" / "cache"
TICKERS_FILENAME    = "tickers.csv"
PRICE_DATA_PERIOD   = "1y"
PRICE_DATA_INTERVAL = "1d"

# MODEL PATHS
MODEL_DIR      = BASE_DIR / "models"
MODEL_FILENAME = "risk_dashboard_kmeans.joblib"

# MARKET TRADING
MARKET_CLOSE     = time(hour=20, minute=30)
TRADING_DAYS     = 252
MARKET_TZ        = "America/New_York"
BENCHMARK_TICKER = "SPY"
TICKERS_URL      =  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# PROGRAM CONSTANTS
DEFAULT_N_PEERS = 4