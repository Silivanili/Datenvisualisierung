# \Datenvisualisierung\app\config.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

CACHE_DIR = os.path.join(BASE_DIR, ".cache")
CACHE_DEFAULT_TIMEOUT = 60 * 60

MAX_SCATTER_POINTS = 20_000
