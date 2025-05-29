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

# DATA PATHS
BASE_DIR       = Path(__file__).parent
RAW_DATA_DIR   = BASE_DIR / "data" / "raw"
CACHE_DATA_DIR = BASE_DIR / "data" / "cache"
TICKERS_FILENAME = "tickers.csv"

# MODEL & CLUSTERING
MODEL_DIR      = BASE_DIR / "models"
MODEL_FILENAME = "risk_dashboard_kmeans.joblib"

# LOGGING
LOG_LEVEL    = "INFO"
LOG_FILENAME = "app.log"
LOG_DIR      = BASE_DIR / "logs"

# MARKET TRADING
MARKET_CLOSE = time(hour=20, minute=30)
MARKET_TZ = "America/New_York"
BENCHMARK_TICKER = "SPY"
TICKERS_URL =  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# OTHER CONSTANTS
DEFAULT_N_RECOMMEND = 4