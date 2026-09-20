import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Ensure data and weights directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Database Config
DB_PATH = os.path.join(DATA_DIR, "spip.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Model Config
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best.pt")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "yolo11s.pt")

if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = FALLBACK_MODEL_PATH

# JWT & Security Config
SECRET_KEY = os.getenv("SECRET_KEY", "spip-super-secret-key-2026-production-ready")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# API Service Config
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# OpenWeatherMap API (Optional / Mocked fallback)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "mock_key")

# OpenAI API Key (Optional / Mocked fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
