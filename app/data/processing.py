# \Datenvisualisierung\app\data\processing.py
import os
import uuid
from werkzeug.utils import secure_filename
import ast
import pandas as pd
import numpy as np
import json
from ..cache import cache
from ..config import DATA_DIR
import re
from typing import Optional

# --- Parsing helpers (vectorized-friendly) ---

def parse_genre(x):
    try:
        if isinstance(x, str):
            genres = ast.literal_eval(x)
            if isinstance(genres, list) and len(genres) > 0:
                return genres[0]
            elif isinstance(genres, str):
                return genres
        return None
    except (ValueError, SyntaxError):
        return None

def estimated_owners_vectorized(series: pd.Series):
    """Vectorized parse of estimated_owners strings -> (mid, low, high) Series."""
    s = series.fillna("").astype(str)
    s = s.str.replace(r"[–—−]", "-", regex=True)
    two_nums = s.str.extract(r"^\s*([0-9,]+(?:\.\d+)?)\s*(?:-\s*([0-9,]+(?:\.\d+)?))?\s*$")
    two_nums = two_nums.replace("", np.nan)
    low = pd.to_numeric(two_nums[0].str.replace(",", ""), errors="coerce")
    high = pd.to_numeric(two_nums[1].str.replace(",", ""), errors="coerce")
    high = high.fillna(low)
    mid = (low + high) / 2.0
    return mid, low, high

def estimated_owners_to_numeric_series(series: pd.Series):
    return estimated_owners_vectorized(series)

# --- secure path helpers ---

def resolve_local_path(user_path: str) -> str:

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    filename = secure_filename(user_path)
    candidate = os.path.abspath(os.path.join(DATA_DIR, filename))
    if not candidate.startswith(os.path.abspath(DATA_DIR)):
        raise ValueError(f"Invalid path or path traversal detected: {user_path}")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Local dataset not found: {candidate}")
    return candidate

# --- dataset load / cache ---

def load_and_cache_dataset(path_or_url: str):

    if not path_or_url:
        raise ValueError("No path_or_url provided")

    if not (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        path_or_url = resolve_local_path(path_or_url)

    df = pd.read_csv(path_or_url, low_memory=False)

    if "genres" in df.columns:
        df["main_genre"] = df["genres"].apply(parse_genre)
    else:
        df["main_genre"] = None

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    else:
        df["release_year"] = pd.NA

    for col in ("average_playtime_forever", "price", "peak_ccu"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "main_genre" in df.columns:
        df["main_genre"] = df["main_genre"].astype("category")

    dataset_id = str(uuid.uuid4())
    cache.set(dataset_id, df)

    return {"dataset_id": dataset_id, "nrows": int(len(df))}

def get_dataset(dataset_id: str):
    """Return DataFrame from cache or None."""
    if not dataset_id:
        return None
    return cache.get(dataset_id)

# --- inspection helpers ---

def y_axis_options_from_df(df):

    if df is None:
        return [], None

    playtime_cols = [
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
    ]

    opts = []
    for c in playtime_cols:
        if c in df.columns:
            opts.append({"label": c, "value": c})

    if "estimated_owners" in df.columns:
        opts.append(
            {"label": "estimated_owners (range)", "value": "estimated_owners"}
        )

    default = opts[0]["value"] if opts else None
    return opts, default


def genres_from_df(df):
    """Return sorted list of unique genres (strings)."""
    if df is None or "main_genre" not in df.columns:
        return []
    g = sorted(df["main_genre"].dropna().unique().tolist())
    return g


# ---------------------------
# Cached aggregation functions
# ---------------------------
# Note: cache.memoize stores result in the flask-caching backend keyed by arguments.
# Provide a reasonable timeout for each expensive aggregation.

@cache.memoize(timeout=60 * 60)
def compute_mean_by_genre_json(dataset_id: str, y_var: str, top_n: int = 15, selected_genres: tuple = None):

    df = cache.get(dataset_id)
    if df is None:
        return None

    if selected_genres:
        sel = [str(g) for g in selected_genres]
        df = df[df["main_genre"].astype(str).isin(sel)].copy()

    if y_var == "estimated_owners":
        if "estimated_owners" not in df.columns:
            return None
        # compute mid, low, high series (vectorized)
        mid, low, high = estimated_owners_to_numeric_series(df["estimated_owners"])
        df2 = df.assign(estimated_owners_mid=mid, estimated_owners_low=low, estimated_owners_high=high)

        # aggregate means per genre for mid, low and high so we can show asymmetric error bars
        g = (
            df2.groupby("main_genre", observed=True)
               .agg(
                   estimated_owners_mid_mean=("estimated_owners_mid", "mean"),
                   estimated_owners_low_mean=("estimated_owners_low", "mean"),
                   estimated_owners_high_mean=("estimated_owners_high", "mean"),
               )
               .dropna()
               .sort_values(by="estimated_owners_mid_mean", ascending=False)
               .head(int(top_n))
               .reset_index()
        )

        # ensure strings for main_genre to avoid json surprises
        g["main_genre"] = g["main_genre"].astype(str)
        return g.to_json(orient="split")
    else:
        if y_var not in df.columns:
            return None
        series = pd.to_numeric(df[y_var], errors="coerce")
        g = (
            df.assign(_y_numeric=series)
              .groupby("main_genre", observed=True)["_y_numeric"]
              .mean()
              .dropna()
              .sort_values(ascending=False)
              .head(int(top_n))
              .reset_index()
              .rename(columns={"_y_numeric": y_var})
        )
        g["main_genre"] = g["main_genre"].astype(str)
        return g.to_json(orient="split")



@cache.memoize(timeout=60 * 60)
def compute_games_per_year_counts_json(dataset_id: str, genres_tuple, year_min=None, year_max=None):

    df = cache.get(dataset_id)
    if df is None:
        return None
    if "release_year" not in df.columns:
        return None

    if genres_tuple:
        genres = [str(g) for g in list(genres_tuple)]
    else:
        if "main_genre" in df.columns:
            genres = [str(g) for g in df["main_genre"].value_counts().head(4).index.tolist()]
        else:
            genres = []

    df_f = df[df["main_genre"].astype(str).isin(genres)].copy()

    if year_min is not None:
        df_f = df_f[df_f["release_year"] >= int(year_min)]
    if year_max is not None:
        df_f = df_f[df_f["release_year"] <= int(year_max)]

    if df_f.empty:
        return None

    try:
        counts = df_f.groupby(["release_year", "main_genre"], observed=True).size().reset_index(name="count")
    except TypeError:
        counts = df_f.groupby(["release_year", "main_genre"]).size().reset_index(name="count")

    counts["main_genre"] = counts["main_genre"].astype(str)

    return counts.to_json(orient="split")


@cache.memoize(timeout=60 * 60)
def compute_peak_ccu_by_year_json(dataset_id: str, genres_tuple, year_min=None, year_max=None):
    """
    Sum peak_ccu per release_year and main_genre; return JSON (orient='split').
    """
    df = cache.get(dataset_id)
    if df is None:
        return None
    if "release_year" not in df.columns or "peak_ccu" not in df.columns:
        return None

    if genres_tuple:
        genres = [str(g) for g in list(genres_tuple)]
    else:
        if "main_genre" in df.columns:
            genres = [str(g) for g in df["main_genre"].value_counts().head(4).index.tolist()]
        else:
            genres = []

    df_f = df[df["main_genre"].astype(str).isin(genres)].copy()
    if year_min is not None:
        df_f = df_f[df_f["release_year"] >= int(year_min)]
    if year_max is not None:
        df_f = df_f[df_f["release_year"] <= int(year_max)]

    if df_f.empty:
        return None

    try:
        agg = df_f.groupby(["release_year", "main_genre"], observed=True)["peak_ccu"].sum().reset_index(name="peak_ccu_sum")
    except TypeError:
        agg = df_f.groupby(["release_year", "main_genre"])["peak_ccu"].sum().reset_index(name="peak_ccu_sum")

    agg["main_genre"] = agg["main_genre"].astype(str)
    return agg.to_json(orient="split")


# --- robust tag parser + top_tags_from_df replacement ---

_TAG_CLEAN_RE = re.compile(r"""^\s*['"]?\s*([^'":\(\)\d]+?)(?:\s*[:\(\[]?.*)?$""", re.UNICODE)

def _clean_tag_token(tok: str) -> Optional[str]:
    """
    Simplified, robust cleaning for a single token that should represent a tag name.
    - strips quotes/whitespace
    - splits on common separators like ':' '(' '[' and takes the left part
    - removes leading/trailing digits or stray punctuation
    """
    if tok is None:
        return None
    if not isinstance(tok, str):
        tok = str(tok)
    # strip surrounding whitespace/quotes
    tok = tok.strip().strip("'\"")
    if not tok:
        return None
    # split off common separators (counts, parentheses, etc.)
    tok = re.split(r"[:\(\[\]\)]+", tok)[0]
    # if commas separate items, take the first piece (fallback)
    tok = tok.split(",")[0].strip()
    # final cleanup: remove stray digits and surrounding punctuation
    tok = re.sub(r"^[\d\W_]+|[\d\W_]+$", "", tok).strip()
    return tok if tok else None


def parse_tags_cell(x):
    """
    Parse a cell that may contain tags in many formats:
      - a dict: {'FPS': 90076, 'Shooter': 64786, ...}  -> return the dict keys
      - a list/tuple of strings or dicts
      - a JSON string representing a list/dict
      - a Python literal string (ast.literal_eval)
      - a comma-separated string like "FPS, Shooter, Multiplayer"

    Always returns a list of cleaned tag strings (unique, order-preserving).
    """
    if pd.isna(x):
        return []

    # If it's already a dict: keys are the tags
    if isinstance(x, dict):
        keys = [ _clean_tag_token(k) for k in x.keys() ]
        return [k for k in keys if k]

    # If it's already a list/tuple, we'll process elements below
    items = None
    if isinstance(x, (list, tuple)):
        items = list(x)
    else:
        # Try Python literal parsing (handles stringified dict/list), then JSON
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, dict):
                return [ _clean_tag_token(k) for k in parsed.keys() if _clean_tag_token(k) ]
            if isinstance(parsed, (list, tuple)):
                items = list(parsed)
            else:
                # scalar -> single token
                tok = _clean_tag_token(parsed)
                return [tok] if tok else []
        except Exception:
            try:
                parsed = json.loads(x)
                if isinstance(parsed, dict):
                    return [ _clean_tag_token(k) for k in parsed.keys() if _clean_tag_token(k) ]
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
                else:
                    tok = _clean_tag_token(parsed)
                    return [tok] if tok else []
            except Exception:
                # fallback: comma-split the string
                items = [part.strip() for part in str(x).split(",") if part.strip()]

    cleaned = []
    for it in items:
        if it is None:
            continue

        # If element is a dict: prefer known keys 'tag','name','label'; otherwise take dict keys
        if isinstance(it, dict):
            # prefer explicit tag-like keys
            cand = None
            for k in ("tag", "name", "label"):
                if k in it:
                    cand = it[k]
                    break
            if cand is not None:
                tok = _clean_tag_token(cand)
                if tok:
                    cleaned.append(tok)
                continue
            # otherwise, take the dict's keys as tag names
            keys = [ _clean_tag_token(k) for k in it.keys() ]
            cleaned.extend([k for k in keys if k])
            continue

        # If element is list/tuple, try first item
        if isinstance(it, (list, tuple)):
            if len(it) == 0:
                continue
            tok = _clean_tag_token(it[0])
            if tok:
                cleaned.append(tok)
            continue

        # Otherwise, treat as scalar
        tok = _clean_tag_token(it)
        if tok:
            cleaned.append(tok)

    # unique while preserving order
    seen = set()
    unique = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def top_tags_from_df(df, top_n=10, tags_col="tags"):
    """
    Collect top tags from a DataFrame column using parse_tags_cell.
    Returns a DataFrame with columns ['tag', 'count'] sorted by count desc then tag asc.
    """
    if df is None or tags_col not in df.columns:
        return pd.DataFrame(columns=["tag", "count"])

    tags_series = df[tags_col].dropna().map(parse_tags_cell)
    if tags_series.empty:
        return pd.DataFrame(columns=["tag", "count"])

    exploded = tags_series.explode().dropna().astype(str)
    if exploded.empty:
        return pd.DataFrame(columns=["tag", "count"])

    counts = exploded.value_counts().reset_index()
    counts.columns = ["tag", "count"]
    counts = counts.sort_values(by=["count", "tag"], ascending=[False, True]).reset_index(drop=True)
    return counts.head(int(top_n))
