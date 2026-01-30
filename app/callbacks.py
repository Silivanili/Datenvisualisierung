# \Datenvisualisierung\app\callbacks.py
import logging
from dash import Input, Output, State, dcc, html
import plotly.express as px
import pandas as pd 
from app import app
import io
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
    genre_releases_subplots,
    genre_metric_subplots,
    histogram_fig_for_column,
    games_per_year_by_genre_fig_from_counts_df,
    STEAM_COLORS,
    STEAM_COLORS_HIGH_CONTRAST,
)
from app.layout import developer_page_layout, game_page_layout, genre_page_layout
from app.utils import get_df_or_none, empty_fig, ensure_list, json_str_to_df
import plotly.graph_objects as go

log = logging.getLogger(__name__)

@app.callback(
    Output("df-store", "data"),
    Input("load-dataset", "n_clicks"),
    State("dataset-path", "value"),
    prevent_initial_call=False,
)
def on_load_dataset(_, path_or_url):
    if not path_or_url:
        return None
    try:
        meta = load_and_cache_dataset(path_or_url)
        df = get_dataset(meta["dataset_id"])
        meta["scatter_y_opts"], meta["scatter_y_default"] = y_axis_options_from_df(df)
        meta["genre_y_opts"], meta["genre_y_default"] = y_axis_options_from_df(df)
        meta["genre_opts"] = [{"label": g, "value": g} for g in genres_from_df(df)]
        meta["game_hist_opts"] = [
            {"label": c, "value": c}
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        return meta
    except Exception:
        log.exception("Failed to load dataset: %s", path_or_url)
        return None

@app.callback(
    Output("scatter-x-select", "options"),
    Output("scatter-x-select", "value"),
    Output("scatter-y-select", "options"),
    Output("scatter-y-select", "value"),
    Input("df-store", "data"),
)
def populate_scatter_y(df_meta):
    if not df_meta:
        return [], "release_date", [], "pct_pos_total"  # Default to release_date and pct_pos_total

    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return [], "release_date", [], "pct_pos_total"  # Default fallback

    # Define units for recognized columns
    column_units = {
        "appid": "Identifier",
        "required_age": "Integer",
        "dlc_count": "Integer",
        "recommendations": "Integer",
        "num_reviews_total": "Number of reviews (total)",
        "num_reviews_recent": "Number of reviews (recent)",
        "main_genre": "String",
        "release_year": "Integer",
        "price": "USD",
        "tags": "List of Strings",
        "peak_ccu": "Peak Concurrent Users (Integer)",
        "metacritic_score": "Integer Score (0-100)",
        "user_score": "Integer Score (0-100)",
        "positive": "Positive Reviews (Integer)",
        "negative": "Negative Reviews (Integer)",
        "pct_pos_total": "Percentage positive reviews (total)",
        "pct_pos_recent": "Percentage positive reviews (recent)",
        "average_playtime_forever": "Minutes",
        "median_playtime_forever": "Minutes",
        "average_playtime_2weeks": "Minutes",
        "median_playtime_2weeks": "Minutes",
        "release_date": "Datetime",
    }

    # Use only columns that have corresponding units defined in column_units
    options = [
        {"label": f"{col} ({column_units[col]})", "value": col}
        for col in df.columns if col in column_units
    ]

    return options, "release_date", options, "pct_pos_total"

@app.callback(
    Output("genre-y-select", "options"),
    Output("genre-y-select", "value"),
    Input("df-store", "data"),
)
def populate_genre_y(df_meta):
    if not df_meta:
        return [], None
    return df_meta.get("genre_y_opts", []), df_meta.get("genre_y_default")

@app.callback(
    Output("genre-filter", "options"),
    Output("genre-filter", "value"),
    Input("df-store", "data"),
)
def populate_genres(df_meta):
    if not df_meta:
        return [], ["Action"]
    opts = df_meta.get("genre_opts", [])
    default = ["Action"] if any(o["value"] == "Action" for o in opts) else (opts[0]["value"] if opts else [])
    return opts, default

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
    if not df_meta:
        return default_min, default_max, default_marks, default_value
    df = get_dataset(df_meta["dataset_id"])
    if df is None or "release_year" not in df.columns:
        return default_min, default_max, default_marks, default_value
    yrs = pd.to_numeric(df["release_year"].dropna(), errors="coerce")
    if yrs.empty:
        return default_min, default_max, default_marks, default_value
    min_y, max_y = int(yrs.min()), int(yrs.max())
    span = max_y - min_y
    step = 1 if span <= 10 else (2 if span <= 40 else 5)
    marks = {y: str(y) for y in range(min_y, max_y + 1, step)}
    default_value = [max(min_y, max_y - 10), max_y]
    return min_y, max_y, marks, default_value

@app.callback(
    Output("dataset-size-text", "children"),
    Output("dataset-rows-text", "children"),
    Output("dataset-top-tags", "children"),
    Input("df-store", "data"),
)
def update_dataset_meta(df_meta):
    if not df_meta:
        return "Size in Memory: ", "Games: ", "Top Tags: "
    df = get_dataset(df_meta["dataset_id"])
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    rows = len(df)
    tags_df = top_tags_from_df(df, top_n=5, tags_col="tags")
    if tags_df.empty and "genres" in df.columns:
        tags_df = top_tags_from_df(df, top_n=5, tags_col="genres")
    if tags_df.empty:
        cand = [c for c in df.columns if any(k in c.lower() for k in ("tag", "genre", "category"))]
        for c in cand:
            tags_df = top_tags_from_df(df, top_n=5, tags_col=c)
            if not tags_df.empty:
                break
    tags_str = "N/A" if tags_df.empty else ", ".join(f"{r['tag']} ({r['count']})" for _, r in tags_df.iterrows())
    return f"Size in Memory: {mem_mb:.2f} MB", f"Games: {rows}", f"Top Tags: {tags_str}"

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return game_page_layout()
    if pathname == "/genre":
        return genre_page_layout()
    if pathname == "/developer":
        return developer_page_layout()
    return html.H1("404: Page not found", className="text-danger text-center mt-5")


@app.callback(
    Output("game-plot1", "figure"),
    Input("df-store", "data"),
    Input("scatter-y-select", "value"),
    Input("scatter-x-select", "value"),
)
def update_game_scatter(df_meta, y, x):
    # Debugging metadata
    if not df_meta:
        print("[DEBUG] No metadata available.")
        return empty_fig("No data loaded")

    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        print("[DEBUG] Dataset not found!")
        return empty_fig("No data loaded")

    # Verify selected x and y data columns
    if (x not in df.columns) or (y not in df.columns):
        print(f"[DEBUG] Column '{x}' or '{y}' not found in dataset!")
        return empty_fig(f"Selected columns '{x}' or '{y}' are invalid.")





    # Call plot generation function
    try:
        scatter_plot = scatter_release_vs_fig(
            df,
            x,
            y,
            hide_zero=False,
            selected_genres=None,

        )
        return scatter_plot
    except Exception as e:
        print(f"[ERROR] Exception during scatter plot generation: {e}")
        return empty_fig("Failed to generate scatter plot")


@app.callback(
    Output("game-plot2", "figure"),
    Input("df-store", "data"),
    Input("game-hist-select", "value"),
)
def update_game_histogram(df_meta, col):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    if not col:
        return empty_fig("No column selected for game histogram")

    # Generate histogram based on default settings (no color swapping logic)
    return histogram_fig_for_column(df, col, bins=50)


@app.callback(Output("game-plot3", "figure"), Input("df-store", "data"))
def update_game_top_tags(df_meta):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")

    # Call the updated tags processing function
    tags_df = top_tags_from_df(df, top_n=10, tags_col="tags")

    # Handle case where no tags are found
    if tags_df.empty:
        return empty_fig("No tags found")

    # Plot using processed tags dataframe
    fig = px.bar(
        tags_df,
        x="tag",
        y="count",
        color="tag",
        title="Top Tags by Occurrence",
        color_discrete_sequence=STEAM_COLORS,
    )
    fig.update_layout(showlegend=False)
    return fig

@app.callback(
    Output("genre-plot1", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-y-select", "value"),

)
def update_genre_mean(df_meta, sel_genres, y_var):
    if not df_meta or y_var is None:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")

    palette = STEAM_COLORS
    top_n = 10
    sel = tuple(ensure_list(sel_genres))
    json_res = compute_mean_by_genre_json(df_meta["dataset_id"], y_var, top_n, sel)
    if not json_res:
        if y_var == "estimated_owners" and "estimated_owners" in df.columns:
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
                .head(top_n)
                .reset_index()
            )
            err_plus = (agg["estimated_owners_high_mean"] - agg["estimated_owners_mid_mean"]).clip(lower=0)
            err_minus = (agg["estimated_owners_mid_mean"] - agg["estimated_owners_low_mean"]).clip(lower=0)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=agg["main_genre"],
                        y=agg["estimated_owners_mid_mean"],
                        error_y=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True),
                        marker_color=palette[: len(agg)],
                    )
                ]
            )
            fig.update_layout(title=f"Mean estimated_owners (mid) by Genre (Top {len(agg)})", xaxis_tickangle=-45, showlegend=False)
            return fig
        df["_y_numeric"] = pd.to_numeric(df[y_var], errors="coerce")
        agg = (
            df.groupby("main_genre")["_y_numeric"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        if agg.empty:
            return empty_fig("No data for the selected genre(s)")
        fig = px.bar(agg, x="main_genre", y="_y_numeric", color="main_genre", color_discrete_sequence=palette, title=f"Mean {y_var} by Genre (Top {len(agg)})")
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        return fig
    agg_df = json_str_to_df(json_res, orient="split")
    if agg_df.empty:
        return empty_fig("No data for the selected genre(s)")
    if "estimated_owners_mid_mean" in agg_df.columns:
        err_plus = (agg_df["estimated_owners_high_mean"] - agg_df["estimated_owners_mid_mean"]).clip(lower=0)
        err_minus = (agg_df["estimated_owners_mid_mean"] - agg_df["estimated_owners_low_mean"]).clip(lower=0)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=agg_df["main_genre"],
                    y=agg_df["estimated_owners_mid_mean"],
                    error_y=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True),
                    marker_color=palette[: len(agg_df)],
                )
            ]
        )
        fig.update_layout(title=f"Mean {y_var} by Genre (Top {len(agg_df)})", xaxis_tickangle=-45, showlegend=False)
        return fig
    ycol = [c for c in agg_df.columns if c != "main_genre"][0]
    fig = px.bar(agg_df, x="main_genre", y=ycol, color="main_genre", color_discrete_sequence=palette, title=f"Chosen amount of Playtime by Genre")
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig

@app.callback(
    Output("game-hist-select", "options"),
    Output("game-hist-select", "value"),
    Input("df-store", "data"),
)
def populate_game_hist_options(df_meta):
    if not df_meta:
        return [], "release_date"  # Default to release_date
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return [], "release_date"  # Default fallback
    options = [{"label": c, "value": c} for c in df.columns]
    return options, "release_date"

@app.callback(
    Output("genre-plot2", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-bubble-y-select", "value"),

)
def update_genre_scatter_bubble(df_meta, sel_genres, y_metric):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    sel = ensure_list(sel_genres)
    if sel:
        df = df[df["main_genre"].isin(sel)]
    required = {"price", "average_playtime_forever", "main_genre", y_metric}
    missing = required - set(df.columns)
    if missing:
        return empty_fig(f"Missing columns: {', '.join(missing)}")
    agg = (
        df.groupby("main_genre", observed=True)
        .agg(
            price_mean=("price", "mean"),
            playtime_mean=("average_playtime_forever", "mean"),
            y_mean=(y_metric, "mean"),
        )
        .reset_index()
    )
    if agg.empty:
        return empty_fig("No data after aggregation")
    palette = STEAM_COLORS
    y_labels = {
        "user_score": "Average user score",
        "positive": "Average positive reviews",
        "negative": "Average negative reviews",
        "metacritic_score": "Average Metacritic score",
    }
    y_label = y_labels.get(y_metric, y_metric)
    fig = px.scatter(
        agg,
        x="price_mean",
        y="y_mean",
        size="playtime_mean",
        color="main_genre",
        color_discrete_sequence=palette,
        hover_data={"price_mean": ":.2f", "y_mean": ":.0f", "playtime_mean": ":.0f", "main_genre": True},
        title=f"Average Price vs. reviews by Genre",
    )
    fig.update_layout(xaxis_title="Average Price", yaxis_title=y_label)
    return fig


@app.callback(
    Output("genre-plot3", "figure"),
    Input("df-store", "data"),
    Input("year-metric-select", "value"),   
    Input("release-year-range", "value"),   
)
def update_genre_releases_subplot(df_meta, metric, year_range):

    if not df_meta:
        return empty_fig("No dataset loaded")

    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("Dataset could not be retrieved")

    df = df.copy()
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year", "main_genre"])


    min_year, max_year = year_range  
    df = df[(df["release_year"] >= min_year) & (df["release_year"] <= max_year)]

    if metric == "peak_ccu" and "peak_ccu" in df.columns:
        agg = (
            df.groupby(["main_genre", "release_year"], observed=True)["peak_ccu"]
            .sum()
            .reset_index(name="value")
        )
        title_metric = "Peak CCU (sum)"
    else:
        agg = (
            df.groupby(["main_genre", "release_year"], observed=True)
            .size()
            .reset_index(name="value")
        )
        title_metric = "Number of Releases"

    return genre_metric_subplots(
        agg_df=agg,
        metric_name=title_metric,
        max_genres=4,              
    )
