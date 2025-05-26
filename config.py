import os
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
CACHE_DATA_DIR = BASE_DIR / "data" / "cache"

# MODEL & CLUSTERING
MODEL_DIR      = BASE_DIR / "models"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "risk_dashboard_kmeans.joblib")

# OTHER CONSTANTS
DEFAULT_N_RECOMMEND = int(os.getenv("DEFAULT_N_RECOMMEND", 4))
LOG_LEVEL             = os.getenv("LOG_LEVEL", "INFO")