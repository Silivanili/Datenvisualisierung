# app/plots.py
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.data.processing import estimated_owners_to_numeric_series
from app.config import MAX_SCATTER_POINTS
from app.utils import empty_fig
import logging

logger = logging.getLogger(__name__)


def games_per_year_by_genre_fig_from_counts_df(counts_df: Optional[pd.DataFrame], year_min: Optional[int] = None, year_max: Optional[int] = None, genres_order=None):
    if counts_df is None or counts_df.empty:
        return empty_fig("No data for games per year by genre", kind="line")
    # ensure numeric year
    counts_df = counts_df.copy()
    counts_df["release_year"] = pd.to_numeric(counts_df["release_year"], errors="coerce").astype("Int64")
    if year_min is None:
        year_min = int(counts_df["release_year"].min())
    if year_max is None:
        year_max = int(counts_df["release_year"].max())

    counts_df["main_genre"] = counts_df["main_genre"].astype(str)

    pivot = counts_df.pivot(index="release_year", columns="main_genre", values="count")

    if genres_order:
        desired = [str(g) for g in genres_order if str(g) in pivot.columns]
        if desired:
            pivot = pivot.reindex(columns=desired)

    pivot = pivot.reindex(range(year_min, year_max + 1), fill_value=0)

    df_melt = pivot.reset_index().melt(id_vars="release_year", var_name="main_genre", value_name="count")

    fig = px.line(df_melt, x="release_year", y="count", color="main_genre", markers=True,
                  title="Number of Games per Year by Genre")
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Games")
    return fig


def histogram_fig_for_column(df: pd.DataFrame, col: str, bins: int = 50, log_x: bool = False):
    if df is None or col not in df.columns:
        return empty_fig(f"Column '{col}' not found", kind="bar")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return empty_fig(f"No numeric data in '{col}'", kind="bar")
    fig = px.histogram(series, nbins=bins, title=f"Histogram of {col}")
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig


def violin_playtime_by_genre(df: pd.DataFrame, playtime_col: str, top_n: int = 10):
    if df is None or playtime_col not in df.columns:
        return empty_fig("No data", kind="box")
    df_local = df.copy()
    df_local[playtime_col] = pd.to_numeric(df_local[playtime_col], errors="coerce")
    df_local = df_local.dropna(subset=[playtime_col, "main_genre"])
    if df_local.empty:
        return empty_fig("No data after cleaning", kind="box")
    top_genres = df_local["main_genre"].value_counts().head(top_n).index.tolist()
    df_local = df_local[df_local["main_genre"].isin(top_genres)]
    fig = px.violin(df_local, x="main_genre", y=playtime_col, box=True, points="outliers",
                    title=f"{playtime_col} distribution by Genre (Top {top_n})")
    fig.update_layout(xaxis_title="Genre", yaxis_title=playtime_col)
    return fig


# -------------------------
# Stratified sampling (robust)
# -------------------------
def stratified_sample(df: pd.DataFrame, by: str, n: int, random_state: Optional[int] = 1) -> pd.DataFrame:
    """
    Stratified sampling that:
    - computes proportions per group,
    - takes floor(props * n) as base picks,
    - distributes the remaining picks by largest fractional parts,
    - clips picks to group sizes,
    - samples deterministically using random_state.

    Returns a DataFrame with up to n rows (fewer only if dataset smaller).
    """
    if n <= 0 or df is None or df.empty:
        return df

    if by not in df.columns:
        return df.sample(min(len(df), n), random_state=random_state)

    sizes = df[by].value_counts(sort=False)
    total = int(sizes.sum())
    if total == 0:
        return df.sample(min(len(df), n), random_state=random_state)

    props = sizes / total
    exact = props * n

    # base allocation: floor
    base = np.floor(exact).astype(int)

    remainder = int(n - base.sum())

    if remainder > 0:
        fracs = exact - base
        # select top fractional parts to add 1
        add_idx = fracs.sort_values(ascending=False).index[:remainder]
        increments = pd.Series(0, index=base.index, dtype=int)
        increments.loc[add_idx] = 1
        picks = base + increments
    else:
        picks = base.copy()

    # clip to group sizes (can't pick more than available)
    picks = picks.clip(upper=sizes).astype(int)

    # remove zero picks
    picks = picks[picks > 0]

    sampled_frames = []
    for g, k in picks.items():
        gdf = df.loc[df[by] == g]
        k = min(len(gdf), int(k))
        if k <= 0:
            continue
        sampled_frames.append(gdf.sample(k, random_state=random_state))

    if not sampled_frames:
        return df.sample(min(len(df), n), random_state=random_state)

    result = pd.concat(sampled_frames, ignore_index=True)

    # If we somehow overshot, downsample to n
    if len(result) > n:
        result = result.sample(n, random_state=random_state)

    return result


# -------------------------
# Scatter / plotting logic
# -------------------------
def _ensure_release_year_column(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'release_year' column exists (string-like) for hover/tooltips."""
    if "release_year" in data.columns:
        data["release_year"] = data["release_year"].astype(str).fillna("")
        return data
    if "release_date" in data.columns:
        try:
            yrs = pd.to_datetime(data["release_date"], errors="coerce").dt.year
            if yrs.notna().any():
                data["release_year"] = yrs.fillna("").astype("Int64").astype(object).where(yrs.notna(), "")
            else:
                data["release_year"] = data["release_date"].astype(str).fillna("")
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
):
    # validate x and y columns
    if df is None:
        return empty_fig("No data for scatter")
    if x_col not in df.columns:
        return empty_fig(f"Column '{x_col}' not found")
    if y_col not in df.columns:
        return empty_fig(f"Column '{y_col}' not found")

    data = df.dropna(subset=[x_col, y_col]).copy()
    data = _ensure_release_year_column(data)

    # apply selected_genres filter only if provided
    if selected_genres:
        if isinstance(selected_genres, str):
            selected_genres = [selected_genres]
        if "main_genre" in data.columns:
            data = data[data["main_genre"].isin(selected_genres)]

    if data.empty:
        return empty_fig("No data for selected axes")

    # ----- estimated_owners special branch (y_col) -----
    if y_col == "estimated_owners":
        mid, low, high = estimated_owners_to_numeric_series(data[y_col])
        data = data.assign(y_mid=mid, low=low, high=high).dropna(subset=["y_mid"])
        if hide_zero:
            data = data[data["y_mid"] != 0]

        if operator and threshold is not None:
            try:
                thr = float(threshold)
                if operator == "eq":
                    data = data[data["y_mid"] == thr]
                elif operator == "ge":
                    data = data[data["y_mid"] >= thr]
                elif operator == "le":
                    data = data[data["y_mid"] <= thr]
                elif operator == "gt":
                    data = data[data["y_mid"] > thr]
                elif operator == "lt":
                    data = data[data["y_mid"] < thr]
            except Exception:
                logger.exception("Error applying operator filter on estimated_owners")

        if data.empty:
            return empty_fig("No points after filters")

        # sampling
        if len(data) > max_points:
            if "main_genre" in data.columns:
                data = stratified_sample(data, "main_genre", max_points)
            else:
                data = data.sample(max_points, random_state=1)

        # Build hover content and plot
        if color_by_genre and "main_genre" in data.columns:
            fig = go.Figure()
            for genre, gdf in data.groupby("main_genre"):
                customdata = [
                    [
                        row.get("name", ""),
                        row.get("appid", ""),
                        f"{int(row['low']):,} - {int(row['high']):,}" if pd.notna(row.get("low")) and pd.notna(row.get("high")) else "",
                        row.get("release_year", ""),
                    ]
                    for _, row in gdf.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=gdf[x_col],
                    y=gdf["y_mid"],
                    mode="markers",
                    name=str(genre),
                    customdata=customdata,
                    hovertemplate="Name: %{customdata[0]}<br>AppID: %{customdata[1]}<br>Owners range: %{customdata[2]}<br>Year: %{customdata[3]}<br>Owners mid: %{y:,}<extra></extra>",
                    error_y=dict(
                        type="data",
                        array=(gdf["high"] - gdf["y_mid"]).abs().tolist(),
                        arrayminus=(gdf["y_mid"] - gdf["low"]).abs().tolist(),
                        visible=True,
                    ),
                    marker=dict(size=6),
                ))
        else:
            df_plot = data.copy()
            fig = px.scatter(df_plot, x=x_col, y="y_mid", hover_data=[c for c in ["name", "appid", "release_year"] if c in df_plot.columns or c == "release_year"], title=f"estimated_owners midpoint vs {x_col}")
            fig.update_traces(showlegend=False)

        fig.update_layout(
            title=f"{y_col} (midpoint) vs {x_col}",
            xaxis_title=x_col, yaxis_title="estimated_owners (midpoint)"
        )
        return fig

    # ----- Normal numeric branch -----
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data = data.dropna(subset=[y_col, x_col])
    if hide_zero:
        data = data[data[y_col] != 0]

    if operator and threshold is not None:
        try:
            thr = float(threshold)
            if operator == "eq":
                data = data[data[y_col] == thr]
            elif operator == "ge":
                data = data[data[y_col] >= thr]
            elif operator == "le":
                data = data[data[y_col] <= thr]
            elif operator == "gt":
                data = data[data[y_col] > thr]
            elif operator == "lt":
                data = data[data[y_col] < thr]
        except Exception:
            logger.exception("Error applying operator filter on numeric y_col")

    if data.empty:
        return empty_fig("No points after filters")

    if len(data) > max_points:
        if "main_genre" in data.columns:
            data = stratified_sample(data, "main_genre", max_points)
        else:
            data = data.sample(max_points, random_state=1)

    color = "main_genre" if (color_by_genre and "main_genre" in data.columns) else None

    hover_cols = [c for c in ["name", "appid", "main_genre", "release_year"] if c in data.columns or c == "release_year"]
    if not color_by_genre and "main_genre" in hover_cols:
        hover_cols = [h for h in hover_cols if h != "main_genre"]

    fig = px.scatter(data, x=x_col, y=y_col, color=color, hover_data=hover_cols, title=f"{y_col} vs {x_col}")
    if not color_by_genre:
        fig.update_traces(showlegend=False)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig
