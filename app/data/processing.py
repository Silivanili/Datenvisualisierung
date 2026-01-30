# \Datenvisualisierung\app\data\processing.py
import os
import uuid
import ast
import json
import re
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from ..cache import cache
from ..config import DATA_DIR
from functools import lru_cache

@lru_cache(maxsize=128)
def parse_genre(x):
    try:
        if isinstance(x, str):
            val = ast.literal_eval(x)
            if isinstance(val, list) and val:
                return val[0]
            if isinstance(val, str):
                return val
        return None
    except (ValueError, SyntaxError):
        return None

@lru_cache(maxsize=128)
def estimated_owners_vectorized(series: pd.Series):
    s = series.fillna("").astype(str)
    s = s.str.replace(r"[–—−]", "-", regex=True)
    two = s.str.extract(r"^\s*([0-9,]+(?:\.\d+)?)\s*(?:-\s*([0-9,]+(?:\.\d+)?))?\s*$")
    low = pd.to_numeric(two[0].str.replace(",", ""), errors="coerce")
    high = pd.to_numeric(two[1].str.replace(",", ""), errors="coerce").fillna(low)
    mid = (low + high) / 2.0
    return mid, low, high

def estimated_owners_to_numeric_series(series: pd.Series):
    return estimated_owners_vectorized(series)

def resolve_local_path(user_path: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = secure_filename(user_path)
    candidate = os.path.abspath(os.path.join(DATA_DIR, filename))
    if not candidate.startswith(os.path.abspath(DATA_DIR)):
        raise ValueError(f"Invalid path or path traversal detected: {user_path}")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Local dataset not found: {candidate}")
    return candidate

def load_and_cache_dataset(path_or_url: str):
    if not path_or_url:
        raise ValueError("No path_or_url provided")
    if not path_or_url.lower().startswith(("http://", "https://")):
        path_or_url = resolve_local_path(path_or_url)
    df = pd.read_csv(path_or_url, low_memory=False)
    if "genres" in df.columns:
        df["main_genre"] = df["genres"].apply(parse_genre)
    else:
        df["main_genre"] = None
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year.astype("Int64")
    else:
        df["release_year"] = pd.NA
    for col in ("average_playtime_forever", "price", "peak_ccu"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "main_genre" in df.columns:
        df["main_genre"] = df["main_genre"].astype("category")
    dataset_id = str(uuid.uuid4())
    cache.set(dataset_id, df)
    meta = {
        "dataset_id": dataset_id,
        "nrows": int(len(df)),
    }
    return meta

def get_dataset(dataset_id: str):
    if not dataset_id:
        return None
    return cache.get(dataset_id)

def y_axis_options_from_df(df):
    if df is None:
        return [], None
    playtime = [
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
    ]
    opts = [{"label": c, "value": c} for c in playtime if c in df.columns]
    if "estimated_owners" in df.columns:
        opts.append({"label": "estimated_owners (range)", "value": "estimated_owners"})
    default = opts[0]["value"] if opts else None
    return opts, default

def genres_from_df(df):
    if df is None or "main_genre" not in df.columns:
        return []
    return sorted(df["main_genre"].dropna().unique().tolist())

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
        mid, low, high = estimated_owners_to_numeric_series(df["estimated_owners"])
        agg = (
            df.assign(mid=mid, low=low, high=high)
            .groupby("main_genre", observed=True)
            .agg(
                estimated_owners_mid_mean=("mid", "mean"),
                estimated_owners_low_mean=("low", "mean"),
                estimated_owners_high_mean=("high", "mean"),
            )
            .dropna()
            .sort_values("estimated_owners_mid_mean", ascending=False)
            .head(int(top_n))
            .reset_index()
        )
        agg["main_genre"] = agg["main_genre"].astype(str)
        return agg.to_json(orient="split")
    if y_var not in df.columns:
        return None
    series = pd.to_numeric(df[y_var], errors="coerce")
    agg = (
        df.assign(_y_numeric=series)
        .groupby("main_genre", observed=True)["_y_numeric"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(int(top_n))
        .reset_index()
        .rename(columns={"_y_numeric": y_var})
    )
    agg["main_genre"] = agg["main_genre"].astype(str)
    return agg.to_json(orient="split")

@cache.memoize(timeout=60 * 60)
def compute_games_per_year_counts_json(dataset_id: str, genres_tuple, year_min=None, year_max=None):
    df = cache.get(dataset_id)
    if df is None or "release_year" not in df.columns:
        return None
    if genres_tuple:
        genres = [str(g) for g in genres_tuple]
    else:
        genres = (
            df["main_genre"].value_counts().head(4).index.astype(str).tolist()
            if "main_genre" in df.columns
            else []
        )
    df_f = df[df["main_genre"].astype(str).isin(genres)].copy()
    if year_min is not None:
        df_f = df_f[df_f["release_year"] >= int(year_min)]
    if year_max is not None:
        df_f = df_f[df_f["release_year"] <= int(year_max)]
    if df_f.empty:
        return None
    counts = df_f.groupby(["release_year", "main_genre"], observed=True).size().reset_index(name="count")
    counts["main_genre"] = counts["main_genre"].astype(str)
    return counts.to_json(orient="split")

@cache.memoize(timeout=60 * 60)
def compute_peak_ccu_by_year_json(dataset_id: str, genres_tuple, year_min=None, year_max=None):
    df = cache.get(dataset_id)
    if df is None or "release_year" not in df.columns or "peak_ccu" not in df.columns:
        return None
    if genres_tuple:
        genres = [str(g) for g in genres_tuple]
    else:
        genres = (
            df["main_genre"].value_counts().head(4).index.astype(str).tolist()
            if "main_genre" in df.columns
            else []
        )
    df_f = df[df["main_genre"].astype(str).isin(genres)].copy()
    if year_min is not None:
        df_f = df_f[df_f["release_year"] >= int(year_min)]
    if year_max is not None:
        df_f = df_f[df_f["release_year"] <= int(year_max)]
    if df_f.empty:
        return None
    agg = df_f.groupby(["release_year", "main_genre"], observed=True)["peak_ccu"].sum().reset_index(name="peak_ccu_sum")
    agg["main_genre"] = agg["main_genre"].astype(str)
    return agg.to_json(orient="split")

_TAG_CLEAN_RE = re.compile(r"""^\s*['"]?\s*([^'":\(\)\d]+?)(?:\s*[:\(\[]?.*)?$""", re.UNICODE)

def _clean_tag_token(tok: str):
    if not isinstance(tok, str):
        tok = str(tok)
    tok = tok.strip().strip("'\"")
    if not tok:
        return None
    tok = re.split(r"[:\(\[\]\)]+", tok)[0]
    tok = tok.split(",")[0].strip()
    tok = re.sub(r"^[\d\W_]+|[\d\W_]+$", "", tok).strip()
    return tok if tok else None

def parse_tags_cell(x):
    if pd.isna(x):
        return []
    if isinstance(x, dict):
        return [_clean_tag_token(k) for k in x.keys() if _clean_tag_token(k)]
    items = None
    if isinstance(x, (list, tuple)):
        items = list(x)
    else:
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, dict):
                return [_clean_tag_token(k) for k in parsed.keys() if _clean_tag_token(k)]
            if isinstance(parsed, (list, tuple)):
                items = list(parsed)
            else:
                token = _clean_tag_token(parsed)
                return [token] if token else []
        except Exception:
            try:
                parsed = json.loads(x)
                if isinstance(parsed, dict):
                    return [_clean_tag_token(k) for k in parsed.keys() if _clean_tag_token(k)]
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
                else:
                    token = _clean_tag_token(parsed)
                    return [token] if token else []
            except Exception:
                items = [part.strip() for part in str(x).split(",") if part.strip()]
    cleaned = []
    for it in items:
        if isinstance(it, dict):
            for key in ("tag", "name", "label"):
                if key in it:
                    tok = _clean_tag_token(it[key])
                    if tok:
                        cleaned.append(tok)
                    break
            else:
                cleaned.extend([_clean_tag_token(k) for k in it.keys() if _clean_tag_token(k)])
        elif isinstance(it, (list, tuple)):
            if it:
                tok = _clean_tag_token(it[0])
                if tok:
                    cleaned.append(tok)
        else:
            tok = _clean_tag_token(it)
            if tok:
                cleaned.append(tok)
    seen = set()
    return [t for t in cleaned if t not in seen and not seen.add(t)]

def top_tags_from_df(df: pd.DataFrame, top_n: int = 10, tags_col: str = "tags"):
    """
    Extract top N tags based on occurrences in the dataframe.
    """
    if df is None or tags_col not in df.columns:
        return pd.DataFrame()

    # Ensure the column is clean
    tags_series = df[tags_col].dropna()

    # Handle 'tags' format: clean structured tags like {'Tag1': value, 'Tag2': value}
    def extract_tags(value):
        if isinstance(value, dict):  # Parse dictionary keys
            return list(value.keys())
        elif isinstance(value, str):  # Parse strings (e.g., JSON-style dictionary)
            try:
                parsed = eval(value)
                return list(parsed.keys()) if isinstance(parsed, dict) else []
            except Exception:
                return []  # Return empty on parse failure
        elif isinstance(value, list):  # Already a list of tags
            return value
        else:
            return []  # Return empty otherwise

    # Apply tag extraction and flatten
    all_tags = pd.Series([tag for tags_list in tags_series.map(extract_tags) for tag in tags_list])

    # Group by tag occurrences and sort by frequency
    top_tags = all_tags.value_counts().nlargest(top_n).reset_index()
    top_tags.columns = ["tag", "count"]

    return top_tags
