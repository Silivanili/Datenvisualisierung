# \Datenvisualisierung\app\plots.py
from typing import Optional
import pandas as pd
import plotly.express as px
from app.data.processing import estimated_owners_to_numeric_series
from app.config import MAX_SCATTER_POINTS
from app.utils import empty_fig, ensure_list
import numpy as np

STEAM_COLORS = [
    "#1b2838",
    "#66c0f4",
    "#2a475e",
    "#5c7e10",
    "#b7c3c7",
    "#4a6d8c",
    "#8cc63f",
]

STEAM_COLORS_HIGH_CONTRAST = [
    "#000000",
    "#ff0000",
    "#ffff00",
    "#00ff00",
    "#00ffff",
    "#ff00ff",
    "#ffffff",
]

def games_per_year_by_genre_fig_from_counts_df(
    counts_df: Optional[pd.DataFrame],
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    genres_order=None,
    color_swap: bool = False,
):
    if counts_df is None or counts_df.empty:
        return empty_fig("No data for games per year by genre")
    df = counts_df
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    year_min = year_min or int(df["release_year"].min())
    year_max = year_max or int(df["release_year"].max())
    df["main_genre"] = df["main_genre"].astype(str)
    pivot = df.pivot(index="release_year", columns="main_genre", values="count")
    if genres_order:
        desired = [str(g) for g in genres_order if str(g) in pivot.columns]
        if desired:
            pivot = pivot.reindex(columns=desired)
    pivot = pivot.reindex(range(year_min, year_max + 1), fill_value=0)
    melt = pivot.reset_index().melt(id_vars="release_year", var_name="main_genre", value_name="count")
    fig = px.line(
        melt,
        x="release_year",
        y="count",
        color="main_genre",
        markers=True,
        color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS,
        title="Number of Games per Year by Genre",
    )
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Games")
    return fig

def histogram_fig_for_column(df: pd.DataFrame, col: str, bins: int = 50, log_x: bool = False, color_swap=False):
    if df is None or col not in df.columns:
        return empty_fig(f"Column '{col}' not found")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return empty_fig(f"No numeric data in '{col}'")
    fig = px.histogram(
        series,
        nbins=bins,
        color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS,
        title=f"Histogram of {col}",
    )
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig

def violin_playtime_by_genre(df: pd.DataFrame, playtime_col: str, top_n: int = 10, color_swap: bool = False):
    if df is None or playtime_col not in df.columns:
        return empty_fig("No data")
    local = df
    local[playtime_col] = pd.to_numeric(local[playtime_col], errors="coerce")
    local = local.dropna(subset=[playtime_col, "main_genre"])
    if local.empty:
        return empty_fig("No data after cleaning")
    top = local["main_genre"].value_counts().head(top_n).index
    local = local[local["main_genre"].isin(top)]
    fig = px.violin(
        local,
        x="main_genre",
        y=playtime_col,
        box=True,
        points="outliers",
        color="main_genre",
        color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS,
        title=f"{playtime_col} distribution by Genre (Top {top_n})",
    )
    fig.update_layout(xaxis_title="Genre", yaxis_title=playtime_col)
    return fig

def stratified_sample(df: pd.DataFrame, by: str, n: int, random_state: Optional[int] = 1) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df
    if by not in df.columns:
        return df.sample(min(len(df), n), random_state=random_state)
    sizes = df[by].value_counts(sort=False)
    props = sizes / sizes.sum()
    exact = props * n
    base = np.floor(exact).astype(int)
    remainder = int(n - base.sum())
    if remainder:
        fracs = exact - base
        add_idx = fracs.sort_values(ascending=False).index[:remainder]
        base.loc[add_idx] += 1
    picks = base.clip(upper=sizes).astype(int).loc[lambda s: s > 0]
    frames = [
        df[df[by] == g].sample(k, random_state=random_state)
        for g, k in zip(picks.index, picks)
    ]
    result = pd.concat(frames, ignore_index=True)
    return result.sample(n, random_state=random_state) if len(result) > n else result

def _ensure_release_year(data: pd.DataFrame) -> pd.DataFrame:
    if "release_year" in data.columns:
        data["release_year"] = data["release_year"].astype(str).fillna("")
        return data
    if "release_date" in data.columns:
        try:
            yrs = pd.to_datetime(data["release_date"], errors="coerce").dt.year
            data["release_year"] = yrs.fillna("").astype("Int64").astype(object).where(yrs.notna(), "")
        except Exception:
            data["release_year"] = data["release_date"].astype(str).fillna("")
        return data
    data["release_year"] = ""
    return data

def scatter_release_vs_fig(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hide_zero: bool = False,
    operator: Optional[str] = None,
    threshold: Optional[float] = None,
    selected_genres=None,
    max_points: int = MAX_SCATTER_POINTS,
    color_by_genre: bool = True,
    color_swap: bool = False,
):
    if df is None:
        return empty_fig("No data for scatter")
    if x_col not in df.columns or y_col not in df.columns:
        return empty_fig(f"Column '{x_col}' or '{y_col}' not found")
    data = df.dropna(subset=[x_col, y_col])
    data = _ensure_release_year(data)
    if selected_genres:
        sel = ensure_list(selected_genres)
        if "main_genre" in data.columns:
            data = data[data["main_genre"].isin(sel)]
    if data.empty:
        return empty_fig("No data for selected axes")
    if y_col == "estimated_owners":
        mid, low, high = estimated_owners_to_numeric_series(data[y_col])
        data = data.assign(y_mid=mid, low=low, high=high).dropna(subset=["y_mid"])
        if hide_zero:
            data = data[data["y_mid"] != 0]
        if operator and threshold is not None:
            thr = float(threshold)
            ops = {
                "eq": data["y_mid"] == thr,
                "ge": data["y_mid"] >= thr,
                "le": data["y_mid"] <= thr,
                "gt": data["y_mid"] > thr,
                "lt": data["y_mid"] < thr,
            }
            data = data[ops.get(operator, slice(None))]
        if data.empty:
            return empty_fig("No points after filters")
        if len(data) > max_points:
            data = stratified_sample(data, "main_genre", max_points) if "main_genre" in data.columns else data.sample(max_points, random_state=1)
        hover = ["name", "appid", "main_genre", "release_year"]
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color="main_genre" if color_by_genre and "main_genre" in data.columns else None,
            color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS,
            hover_data=[c for c in hover if c in data.columns],
            title=f"{y_col} vs {x_col}",
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        if not color_by_genre:
            fig.update_traces(showlegend=False)
        return fig
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col])
    if hide_zero:
        data = data[data[y_col] != 0]
    if operator and threshold is not None:
        thr = float(threshold)
        ops = {
            "eq": data[y_col] == thr,
            "ge": data[y_col] >= thr,
            "le": data[y_col] <= thr,
            "gt": data[y_col] > thr,
            "lt": data[y_col] < thr,
        }
        data = data[ops.get(operator, slice(None))]
    if data.empty:
        return empty_fig("No points after filters")
    if len(data) > max_points:
        data = stratified_sample(data, "main_genre", max_points) if "main_genre" in data.columns else data.sample(max_points, random_state=1)
    color = "main_genre" if (color_by_genre and "main_genre" in data.columns) else None
    hover = ["name", "appid", "main_genre", "release_year"]
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color,
        color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS if color else None,
        hover_data=[c for c in hover if c in data.columns],
        title=f"{y_col} vs {x_col}",
    )
    if not color_by_genre:
        fig.update_traces(showlegend=False)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig
