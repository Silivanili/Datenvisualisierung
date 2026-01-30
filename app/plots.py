# \Datenvisualisierung\app\plots.py
from typing import Optional
import pandas as pd
import plotly.express as px
from app.data.processing import estimated_owners_to_numeric_series
from app.config import MAX_SCATTER_POINTS
from app.utils import empty_fig, ensure_list
import numpy as np
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from math import ceil
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

    # Convert release_date column to datetime format if applicable
    if col == "release_date":
        df = df.copy()  # Avoid modifying original dataframe
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col])  # Remove invalid dates

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return empty_fig(f"No numeric data in '{col}'")

    title = "New Releases on Steam over Time" if col == "release_date" else f"Histogram of {col}"
    fig = px.histogram(
        df,
        x=col,
        nbins=bins,
        color_discrete_sequence=STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS,
        title=title,
    )

    if log_x:
        fig.update_xaxes(type="log")
    elif col == "release_date":
        fig.update_xaxes(title="Release Date")  # Update x-axis title for release_date specifically

    fig.update_layout(yaxis_title="Count")
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
    color_by_genre: bool = False,
    color_swap: bool = False,
):
    # Define units for recognized columns
    column_units = {
        "price": "USD",
        "metacritic_score": "Score (0-100)",
        "user_score": "Score",
        "positive": "Count",
        "negative": "Count",
        "pct_pos_total": "Percentage (%)",
        "average_playtime_forever": "Minutes",
        "median_playtime_forever": "Minutes",
        "release_date": "Release Date",
    }

    x_label = f"{x_col} ({column_units.get(x_col, x_col)})"
    y_label = f"{y_col} ({column_units.get(y_col, y_col)})"

    # Handle release_date as a datetime column
    if x_col == "release_date":
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")  # Ensure proper datetime format
        df = df.dropna(subset=[x_col])  # Remove rows with invalid datetime values

    # Default filtering
    data = df.dropna(subset=[x_col, y_col])
    if hide_zero:
        data = data[data[y_col] != 0]

    # Apply operator filtering if threshold is provided
    if operator and threshold is not None:
        thr = float(threshold)
        op_map = {
            "eq": data[y_col] == thr,
            "ge": data[y_col] >= thr,
            "le": data[y_col] <= thr,
            "gt": data[y_col] > thr,
            "lt": data[y_col] < thr,
        }
        data = data[op_map.get(operator, slice(None))]

    # Apply sampling for large datasets
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)

    if data.empty:
        return empty_fig("No data available for the selected axes")

    # Generate the scatter plot
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="main_genre" if color_by_genre and "main_genre" in data.columns else None,
        opacity=0.6,  # Reduce opacity for readability
        hover_data=["name", "appid"] if "appid" in data.columns else None,
        title=f"Scatterplot for all games",
    )
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

    # Explicitly set the x-axis type as datetime for release_date
    if x_col == "release_date":
        fig.update_xaxes(type="date")  # Ensure proper datetime handling for x-axis

    return fig


def genre_releases_subplots(df: pd.DataFrame, max_genres: int = 10):
    """
    Create subplots for the number of new game releases across genres by year.
    Arrange the plots in two columns side by side with improved spacing for clarity.
    """
    if df is None or "release_year" not in df.columns or "main_genre" not in df.columns:
        return empty_fig("No valid data for genre releases.")

    # Ensure release_year is numeric and main_genre exists
    df = df.copy()
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year", "main_genre"])

    # Aggregate data: count total releases per genre
    genre_total_counts = df["main_genre"].value_counts().reset_index()
    genre_total_counts.columns = ["main_genre", "total_releases"]

    # Select top genres based on total release counts
    top_genres = genre_total_counts.nlargest(max_genres, "total_releases")["main_genre"]

    # Group the data by genre and year
    genre_year_counts = df[df["main_genre"].isin(top_genres)].groupby(
        ["main_genre", "release_year"]).size().reset_index(name="release_count")

    # Determine number of rows needed for two columns
    num_rows = ceil(len(top_genres) / 2)  # Two columns -> One row for every two genres

    # Create subplots with improved spacing
    fig = make_subplots(
        rows=num_rows,
        cols=2,
        shared_xaxes=True,  # Share X-axis across columns
        vertical_spacing=0.15,  # Increased spacing between rows
        horizontal_spacing=0.075,  # Increased spacing between columns
        subplot_titles=[f"{genre}" for genre in top_genres],
    )

    # Add traces for each genre
    for i, genre in enumerate(top_genres):
        genre_data = genre_year_counts[genre_year_counts["main_genre"] == genre]

        # Determine row and column placement
        row = (i // 2) + 1  # Divide into pairs for a new row
        col = (i % 2) + 1  # Alternate between column 1 and 2

        trace = go.Scatter(
            x=genre_data["release_year"],
            y=genre_data["release_count"],
            mode="lines+markers",
            name=genre,
            showlegend=False,  # Turn off legend for subplots
        )
        fig.add_trace(trace, row=row, col=col)

    # Update axis titles
    for r in range(1, num_rows + 1):
        fig.update_xaxes(title_text="Year", row=r, col=1)  # Add X-axis title only for the first column
    fig.update_yaxes(title_text="New Releases")

    # Adjust overall layout with margins
    fig.update_layout(
        height=250 * num_rows,  # Adjust height dynamically based on number of rows
        margin=dict(t=50, l=50, r=50, b=50),  # Added margins for clarity
        title_text="Genre growth based on amount of releases",
        title_x=0.5,
        template="plotly_white",
    )

    return fig


def genre_metric_subplots(agg_df: pd.DataFrame, metric_name: str = "Number of Releases", max_genres: int = 10,) -> go.Figure:
    
    if agg_df is None or agg_df.empty:
        return empty_fig("No data for selected metric / year range")

    top_genres = (
        agg_df.groupby("main_genre")["value"]
        .sum()
        .nlargest(max_genres)
        .index
        .tolist()
    )
    df_plot = agg_df[agg_df["main_genre"].isin(top_genres)]

    num_rows = ceil(len(top_genres) / 2)

    fig = make_subplots(
        rows=num_rows,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.15,
        horizontal_spacing=0.075,
        subplot_titles=[f"{g}" for g in top_genres],
    )

    for i, genre in enumerate(top_genres):
        genre_data = df_plot[df_plot["main_genre"] == genre]

        row = (i // 2) + 1
        col = (i % 2) + 1

        trace = go.Scatter(
            x=genre_data["release_year"],
            y=genre_data["value"],
            mode="lines+markers",
            name=genre,
            showlegend=False,
        )
        fig.add_trace(trace, row=row, col=col)

    for r in range(1, num_rows + 1):
        fig.update_xaxes(title_text="Year", row=r, col=1)  
        fig.update_yaxes(title_text=metric_name, row=r, col=1)

    fig.update_layout(
        height=250 * num_rows,
        margin=dict(t=60, l=60, r=60, b=60),
        title_text=f"Genre growth – {metric_name}",
        title_x=0.5,
        template="plotly_white",
    )
    return fig
