# app/utils.py

from functools import wraps
from typing import Any, Callable, Iterable, List, Optional, Tuple
import logging
import plotly.express as px
from app.data.processing import get_dataset

logger = logging.getLogger(__name__)


def get_df_from_meta(df_meta: Optional[dict]) -> Optional[object]:

    if not df_meta:
        return None
    dataset_id = df_meta.get("dataset_id")
    return get_dataset(dataset_id) if dataset_id else None


def get_df_or_none(df_meta: Optional[dict]):
    """
    Centralized helper used by callbacks:
    given the dict from dcc.Store, return the cached DataFrame or None.
    """
    return get_df_from_meta(df_meta)


def empty_fig(msg: str = "No data", kind: str = "scatter"):

    if kind == "line":
        return px.line(title=msg)
    if kind == "bar":
        return px.bar(title=msg)
    if kind == "box":
        return px.box(title=msg)
    return px.scatter(title=msg)


def ensure_list(x: Any) -> List:
    """
    Return a list for common inputs:
      - None -> []
      - str -> [str]
      - iterable -> list(iterable)
      - otherwise -> [x]
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def ensure_df(empty_return=None):
    """
    Decorator that expects the wrapped function to accept a DataFrame as first arg.
    The decorator will accept df_meta (dict with 'dataset_id') instead, fetch the DataFrame,
    and call the wrapped function with the DataFrame. If dataset is missing, returns empty_return.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(df_meta, *args, **kwargs):
            df = get_df_or_none(df_meta)
            if df is None:
                return empty_return
            return func(df, *args, **kwargs)
        return wrapper
    return decorator
