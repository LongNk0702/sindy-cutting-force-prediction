from .data import load_taguchi_csv, load_force_signals, merge_experiment_data
from .features import build_feature_matrix
from .sindy_model import fit_sindy
from .evaluation import regression_metricc
from .config import PROJECT_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, SINDY_CONFIG

__all__ = [
    #config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "SINDY_CONFIG",
    #data
    "load_taguchi_csv",
    "load_force_signals",
    "merge_experiment_data",
    #features
    "build_feature_matrix",
    #sindy model
    "fit_sindy",
    #evaluation
    "regression_metricc",
]

__version__ = "0.1.2"
__author__ = "Long Nguyen Khac"
__email__ = "nk.long723@gmail.com"