from pathlib import Path
#root directory of the project (2 levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent[1]
#data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#outputs folders
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

#ensure folders exist
for folder in [MODEL_DIR,FIGURE_DIR,LOG_DIR,PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE=42
DECIMAL_PRECISION=5
SAMPLE_RATE_HZ=2000

#Sindy model configuration
SINDY_CONFIG = {
    "degree":3,
    "poly_order":3,
    "optimizer":"STLSQ",
    "threshold":0.1,
    "include_interaction":True,
    "normalize":True,
    "discrete_time":False,
}

#plotting configuration
FIGURE_DIR={
    "dpi":150,
    "figsize":(7,5),
    "font":"Inter",
    "linewidth":1.8,
    "color_fxn":"#1f77b4",
    "color_fy": "#ff7f0e",  # orange
    "color_fz": "#2ca02c",  # green
    "grid_alpha": 0.3,
}

#logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

#project information
__version__ = "0.1.0"
__author__ = "Long Nguyen Khac"
__email__ = "nk.long723@gmail.com"
__description__ = "Configuration for SINDy Cutting Force Prediction project."