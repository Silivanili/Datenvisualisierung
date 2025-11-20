# app/callbacks.py
import logging
from dash import Input, Output, State, dcc, html
from app import app
from app.data.processing import (
    load_and_cache_dataset,
    get_dataset,
    y_axis_options_from_df,
    genres_from_df,
    compute_mean_by_genre_json,
    compute_games_per_year_counts_json,
    compute_peak_ccu_by_year_json,
    estimated_owners_to_numeric_series,
    top_tags_from_df,
)
from app.plots import (
    scatter_release_vs_fig,
    histogram_fig_for_column,
    games_per_year_by_genre_fig_from_counts_df,
    violin_playtime_by_genre,
)
from app.layout import developer_page_layout, game_page_layout, genre_page_layout
from app.utils import get_df_or_none, empty_fig
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------
# Upload / store callbacks
# ---------------------------
@app.callback(
    Output("df-store", "data"),
    Input("load-dataset", "n_clicks"),
    State("dataset-path", "value"),
    prevent_initial_call=True,
)
def on_load_dataset(n_clicks, path_or_url):
    if not path_or_url:
        return None
    try:
        meta = load_and_cache_dataset(path_or_url)
    except Exception:
        logger.exception("Failed to load dataset: %s", path_or_url)
        return None
    return meta


# ---------------------------
# Populate controls callbacks
# ---------------------------
@app.callback(
    Output("scatter-y-select", "options"),
    Output("scatter-y-select", "value"),
    Input("df-store", "data"),
)
def populate_scatter_y_options(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return [], None

    curated = [
        "price",
        "metacritic_score",
        "user_score",
        "positive",
        "negative",
        "pct_pos_total",
        "average_playtime_forever",
        "median_playtime_forever",
        "peak_ccu",
        "estimated_owners",
        "num_reviews_total",
    ]
    options = [{"label": c, "value": c} for c in curated if c in df.columns]
    default = options[0]["value"] if options else None
    return options, default


@app.callback(
    Output("genre-y-select", "options"),
    Output("genre-y-select", "value"),
    Input("df-store", "data"),
)
def populate_genre_y_options(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return [], None

    candidates = [
        "average_playtime_forever",
        "median_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_2weeks",
        "estimated_owners",
    ]
    options = [{"label": c, "value": c} for c in candidates if c in df.columns]
    default = options[0]["value"] if options else None
    return options, default


@app.callback(
    Output("genre-filter", "options"),
    Output("genre-filter", "value"),
    Input("df-store", "data"),
)
def populate_genres(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return [], ["Action"]

    genres = genres_from_df(df)
    options = [{"label": g, "value": g} for g in genres]
    default = ["Action"] if "Action" in genres else ([genres[0]] if genres else [])
    return options, default


@app.callback(
    Output("release-year-range", "min"),
    Output("release-year-range", "max"),
    Output("release-year-range", "marks"),
    Output("release-year-range", "value"),
    Input("df-store", "data"),
)
def populate_release_year(df_meta):
    default_min, default_max = 1970, 2025
    default_marks = {default_min: str(default_min), default_max: str(default_max)}
    default_value = [2010, 2020]

    df = get_df_or_none(df_meta)
    if df is None or "release_year" not in df.columns:
        return default_min, default_max, default_marks, default_value

    yrs = pd.to_numeric(df["release_year"].dropna(), errors="coerce")
    if yrs.empty:
        return default_min, default_max, default_marks, default_value

    min_year, max_year = int(yrs.min()), int(yrs.max())
    span = max_year - min_year
    step = 1 if span <= 10 else (2 if span <= 40 else 5)
    marks = {y: str(y) for y in range(min_year, max_year + 1, step)}
    default_value = [max(min_year, max_year - 10), max_year]
    return min_year, max_year, marks, default_value


# ---------------------------
# Metadata panel
# ---------------------------
@app.callback(
    Output("dataset-size-text", "children"),
    Output("dataset-rows-text", "children"),
    Output("dataset-top-tags", "children"),
    Input("df-store", "data"),
)
def update_dataset_meta(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return "Size in Memory: ", "Games: ", "Top Tags: "

    # memory footprint
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_str = f"{mem_bytes / (1024 ** 2):.2f} MB"
    rows_str = f"{len(df)}"

    # compute top tags: try 'tags', fall back to 'genres' or other candidate columns
    tags_df = None
    if "tags" in df.columns:
        tags_df = top_tags_from_df(df, top_n=5, tags_col="tags")

    if (tags_df is None or tags_df.empty) and "genres" in df.columns:
        tags_df = top_tags_from_df(df, top_n=5, tags_col="genres")

    if (tags_df is None or tags_df.empty):
        candidate_cols = [c for c in df.columns if any(k in c.lower() for k in ("tag", "tags", "genre", "category", "top"))]
        for c in candidate_cols:
            if c in ("tags", "genres"):
                continue
            tags_df = top_tags_from_df(df, top_n=5, tags_col=c)
            if tags_df is not None and not tags_df.empty:
                break

    tags_str = "N/A" if (tags_df is None or tags_df.empty) else ", ".join(f"{row['tag']} ({row['count']})" for _, row in tags_df.iterrows())

    return (
        f"Size in Memory: {mem_str}",
        f"Games: {rows_str}",
        f"Top Tags: {tags_str}",
    )


# ---------------------------
# Navigation callback
# ---------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/":
        return game_page_layout()
    if pathname == "/genre":
        return genre_page_layout()
    if pathname == "/developer":
        return developer_page_layout()
    return html.H1("404: Page not found", className="text-danger text-center mt-5")


# ---------------------------
# Game page callbacks
# ---------------------------
@app.callback(
    Output("game-plot1", "figure"),
    Input("df-store", "data"),
    Input("scatter-y-select", "value"),
    Input("scatter-x-select", "value"),
    Input("hide-zero-reviews", "value"),
    Input("y-filter-operator", "value"),
    Input("y-filter-value", "value"),
)
def update_game_scatter(df_meta, scatter_y_input, scatter_x_input, hide_zero, operator, threshold):
    df = get_df_or_none(df_meta)
    if df is None:
        return empty_fig("No data loaded")

    hide_zero_bool = bool(hide_zero and "hide" in hide_zero)
    if not scatter_x_input or not scatter_y_input:
        return empty_fig("Select both X and Y variables for the scatter plot")

    try:
        fig = scatter_release_vs_fig(
            df,
            scatter_x_input,
            scatter_y_input,
            hide_zero=hide_zero_bool,
            operator=operator,
            threshold=threshold,
            selected_genres=None,
            color_by_genre=False,
        )
        return fig
    except Exception:
        logger.exception("Failed to build game scatter for x=%s y=%s", scatter_x_input, scatter_y_input)
        return empty_fig("Failed to build scatter")


@app.callback(
    Output("game-plot2", "figure"),
    Input("df-store", "data"),
    Input("game-hist-select", "value"),
    Input("view-settings", "value"),
)
def update_game_histogram(df_meta, hist_col, view_settings):
    df = get_df_or_none(df_meta)
    if df is None:
        return empty_fig("No data loaded")

    if not hist_col:
        return empty_fig("No column selected for game histogram")

    swap_hist = ("swap_hist" in view_settings) if isinstance(view_settings, (list, tuple, set)) else (view_settings == "swap_hist")

    if swap_hist:
        if hist_col not in df.columns:
            return empty_fig(f"Column '{hist_col}' not found", "box")
        col_ser = pd.to_numeric(df[hist_col], errors="coerce").dropna()
        if col_ser.empty:
            return empty_fig(f"No numeric data in '{hist_col}'", "box")
        df_box = pd.DataFrame({hist_col: col_ser})
        fig = px.box(df_box, y=hist_col, points="outliers", title=f"{hist_col} distribution (Box-plot)")
        fig.update_layout(yaxis_title=hist_col, xaxis_title=None, showlegend=False)
        return fig

    return histogram_fig_for_column(df, hist_col, bins=50)


@app.callback(
    Output("game-plot3", "figure"),
    Input("df-store", "data"),
)
def update_game_top_tags(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return empty_fig("No data loaded")
    tags_df = top_tags_from_df(df, top_n=10)
    if tags_df.empty:
        return empty_fig("No tags found", "bar")
    fig = px.bar(tags_df, x="tag", y="count", title="Top Tags")
    return fig


# ---------------------------
# Genre page callbacks
# ---------------------------
@app.callback(
    Output("genre-plot1", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-y-select", "value"),
)
def update_genre_mean(df_meta, selected_genres, genre_y_input):
    df = get_df_or_none(df_meta)
    if df is None or genre_y_input is None:
        return empty_fig("No data loaded")

    if isinstance(selected_genres, str):
        selected_genres = [selected_genres]

    top_n = 10
    genre_tuple = tuple(selected_genres) if selected_genres else None
    json_res = compute_mean_by_genre_json(dataset_id=df_meta.get("dataset_id"), y_var=genre_y_input, top_n=top_n, selected_genres=genre_tuple)

    # cache miss handling (special-case estimated_owners)
    if not json_res:
        if genre_y_input == "estimated_owners":
            if "estimated_owners" not in df.columns:
                return empty_fig("No data for estimated_owners", "bar")

            mid, low, high = estimated_owners_to_numeric_series(df["estimated_owners"])
            df2 = df.assign(estimated_mid=mid, estimated_low=low, estimated_high=high)

            agg = (
                df2.groupby("main_genre", observed=True)
                .agg(
                    estimated_owners_mid_mean=("estimated_mid", "mean"),
                    estimated_owners_low_mean=("estimated_low", "mean"),
                    estimated_owners_high_mean=("estimated_high", "mean"),
                )
                .dropna()
                .sort_values(by="estimated_owners_mid_mean", ascending=False)
                .head(top_n)
                .reset_index()
            )

            if agg.empty:
                return empty_fig("No data for the selected genre(s)", "bar")

            agg["estimated_owners_mid_mean"] = pd.to_numeric(agg["estimated_owners_mid_mean"], errors="coerce")
            agg["estimated_owners_low_mean"] = pd.to_numeric(agg["estimated_owners_low_mean"], errors="coerce")
            agg["estimated_owners_high_mean"] = pd.to_numeric(agg["estimated_owners_high_mean"], errors="coerce")

            err_plus = (agg["estimated_owners_high_mean"] - agg["estimated_owners_mid_mean"]).fillna(0).clip(lower=0).tolist()
            err_minus = (agg["estimated_owners_mid_mean"] - agg["estimated_owners_low_mean"]).fillna(0).clip(lower=0).tolist()

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=agg["main_genre"].astype(str),
                        y=agg["estimated_owners_mid_mean"],
                        error_y=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True),
                        name="estimated_owners (mid mean)",
                    )
                ]
            )
            fig.update_layout(title=f"Mean estimated_owners (mid) by Genre (Top {len(agg)})", xaxis_tickangle=-45, showlegend=False)
            return fig

        # regular numeric column (no cache result)
        df["_y_numeric"] = pd.to_numeric(df[genre_y_input], errors="coerce")
        agg = (
            df.groupby("main_genre")["_y_numeric"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        if agg.empty:
            return empty_fig("No data for the selected genre(s)", "bar")
        fig = px.bar(agg, x="main_genre", y="_y_numeric", title=f"Mean {genre_y_input} by Genre (Top {len(agg)})")
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        return fig

    # cached path: parse JSON and render (handles estimated_owners or numeric)
    try:
        agg_df = pd.read_json(json_res, orient="split")
    except Exception:
        logger.exception("Failed to parse cached JSON for compute_mean_by_genre_json")
        return empty_fig("Cache parse error", "bar")

    if agg_df.empty:
        return empty_fig("No data for the selected genre(s)", "bar")

    if "estimated_owners_mid_mean" in agg_df.columns:
        agg_df["estimated_owners_mid_mean"] = pd.to_numeric(agg_df["estimated_owners_mid_mean"], errors="coerce")
        agg_df["estimated_owners_low_mean"] = pd.to_numeric(agg_df.get("estimated_owners_low_mean", pd.Series([None] * len(agg_df))), errors="coerce")
        agg_df["estimated_owners_high_mean"] = pd.to_numeric(agg_df.get("estimated_owners_high_mean", pd.Series([None] * len(agg_df))), errors="coerce")

        err_plus = (agg_df["estimated_owners_high_mean"] - agg_df["estimated_owners_mid_mean"]).fillna(0).clip(lower=0).tolist()
        err_minus = (agg_df["estimated_owners_mid_mean"] - agg_df["estimated_owners_low_mean"]).fillna(0).clip(lower=0).tolist()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=agg_df["main_genre"].astype(str),
                    y=agg_df["estimated_owners_mid_mean"],
                    error_y=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True),
                    name="estimated_owners (mid mean)",
                )
            ]
        )
        fig.update_layout(title=f"Mean {genre_y_input} by Genre (Top {len(agg_df)})", xaxis_tickangle=-45, showlegend=False)
        return fig

    if "main_genre" not in agg_df.columns:
        return empty_fig("No data for the selected genre(s)", "bar")
    ycol = [c for c in agg_df.columns if c != "main_genre"][0]
    fig = px.bar(agg_df, x="main_genre", y=ycol, title=f"Mean {genre_y_input} by Genre (Top {len(agg_df)})")
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig


@app.callback(
    Output("genre-plot2", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("release-year-range", "value"),
    Input("year-metric-select", "value"),
)
def update_genre_yearly(df_meta, selected_genres, year_range, metric):
    df = get_df_or_none(df_meta)
    if df is None:
        return empty_fig("No data loaded")

    if isinstance(selected_genres, str):
        selected_genres = [selected_genres]

    if selected_genres:
        genres_tuple = tuple(selected_genres)
    else:
        if "main_genre" in df.columns:
            genres_tuple = tuple(df["main_genre"].value_counts().head(6).index.tolist())
        else:
            genres_tuple = ()

    year_min, year_max = None, None
    if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
        try:
            year_min = int(year_range[0]); year_max = int(year_range[1])
        except Exception:
            year_min, year_max = None, None

    if metric == "count":
        json_counts = compute_games_per_year_counts_json(df_meta.get("dataset_id"), genres_tuple, year_min, year_max)
        if not json_counts:
            return empty_fig("No data for selected metric", "line")
        counts_df = pd.read_json(json_counts, orient="split")
        return games_per_year_by_genre_fig_from_counts_df(counts_df, year_min, year_max, genres_order=genres_tuple)

    if metric == "peak_ccu":
        json_ccu = compute_peak_ccu_by_year_json(df_meta.get("dataset_id"), genres_tuple, year_min, year_max)
        if not json_ccu:
            return empty_fig("No data for selected metric", "line")
        ccu_df = pd.read_json(json_ccu, orient="split")
        ccu_df["release_year"] = ccu_df["release_year"].astype(int)
        year_min = year_min if year_min is not None else int(ccu_df["release_year"].min())
        year_max = year_max if year_max is not None else int(ccu_df["release_year"].max())

        pivot = ccu_df.pivot(index="release_year", columns="main_genre", values="peak_ccu_sum")
        if genres_tuple:
            desired = [g for g in genres_tuple if g in pivot.columns]
            pivot = pivot.reindex(columns=desired, fill_value=0)
        pivot = pivot.reindex(range(year_min, year_max + 1), fill_value=0)
        agg_df = pivot.reset_index().melt(id_vars="release_year", var_name="main_genre", value_name="peak_ccu_sum")
        fig = px.line(agg_df, x="release_year", y="peak_ccu_sum", color="main_genre", markers=True, title="Peak CCU per Year by Genre")
        return fig

    return empty_fig("Unknown metric", "line")


@app.callback(
    Output("genre-plot3", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-y-select", "value"),
)
def update_genre_histogram(df_meta, selected_genres, genre_y_input):
    df = get_df_or_none(df_meta)
    if df is None or not genre_y_input:
        return empty_fig("No data loaded")

    if isinstance(selected_genres, str):
        selected_genres = [selected_genres]
    if selected_genres:
        df = df[df["main_genre"].isin(selected_genres)]

    playtime_cols = [
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
    ]
    if genre_y_input in playtime_cols:
        return violin_playtime_by_genre(df, genre_y_input, top_n=8)
    return histogram_fig_for_column(df, genre_y_input, bins=50)


@app.callback(
    Output("game-hist-select", "options"),
    Output("game-hist-select", "value"),
    Input("df-store", "data"),
)
def populate_game_hist_options(df_meta):
    df = get_df_or_none(df_meta)
    if df is None:
        return [], None

    num_cols = []
    for c in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                num_cols.append(c)
        except Exception:
            try:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.dropna().shape[0] > 0:
                    num_cols.append(c)
            except Exception:
                continue

    options = [{"label": col, "value": col} for col in num_cols]
    default = num_cols[0] if num_cols else None
    return options, default
