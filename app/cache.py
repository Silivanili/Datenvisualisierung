# \Datenvisualisierung\app\cache.py
from flask_caching import Cache
from .config import CACHE_DIR, CACHE_DEFAULT_TIMEOUT

cache = None

def init_cache(server):
    global cache
    cache = Cache(server, config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": CACHE_DIR,
        "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
    })
    return cache
