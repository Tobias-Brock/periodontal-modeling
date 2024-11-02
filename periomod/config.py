"""File containing paths."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_BEHAVIOR_DIR = PROCESSED_DATA_DIR / "behavior"
PROCESSED_BASE_DIR = PROCESSED_DATA_DIR / "base"
TRAINING_DATA_DIR = PROCESSED_BASE_DIR / "training"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"