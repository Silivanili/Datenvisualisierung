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
    histogram_fig_for_column,
    games_per_year_by_genre_fig_from_counts_df,
    STEAM_COLORS,
    STEAM_COLORS_HIGH_CONTRAST,
)
from app.layout import developer_page_layout, game_page_layout, genre_page_layout
from app.utils import get_df_or_none, empty_fig, ensure_list, json_str_to_df

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
    Output("scatter-y-select", "options"),
    Output("scatter-y-select", "value"),
    Input("df-store", "data"),
)
def populate_scatter_y(df_meta):
    if not df_meta:
        return [], None
    return df_meta.get("scatter_y_opts", []), df_meta.get("scatter_y_default")

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
    Input("hide-zero-reviews", "value"),
    Input("y-filter-operator", "value"),
    Input("y-filter-value", "value"),
    Input("swap-colorscheme", "value"),
)
def update_game_scatter(df_meta, y, x, hide_zero, op, thr, swap):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    if not x or not y:
        return empty_fig("Select both X and Y variables for the scatter plot")
    color_swap = "swap_colors" in (swap or [])
    hide = bool(hide_zero and "hide" in hide_zero)
    return scatter_release_vs_fig(
        df,
        x,
        y,
        hide_zero=hide,
        operator=op,
        threshold=thr,
        selected_genres=None,
        color_by_genre=False,
        color_swap=color_swap,
    )

@app.callback(
    Output("game-plot2", "figure"),
    Input("df-store", "data"),
    Input("game-hist-select", "value"),
    Input("view-settings", "value"),
    Input("swap-colorscheme", "value"),
)
def update_game_histogram(df_meta, col, view_settings, swap):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    if not col:
        return empty_fig("No column selected for game histogram")
    swap_hist = "swap_hist" in (view_settings or [])
    color_swap = "swap_colors" in (swap or [])
    if swap_hist:
        ser = pd.to_numeric(df[col], errors="coerce").dropna()
        if ser.empty:
            return empty_fig(f"No numeric data in '{col}'")
        return px.box(pd.DataFrame({col: ser}), y=col, points="outliers", title=f"{col} distribution (Box‑plot)").update_layout(showlegend=False)
    return histogram_fig_for_column(df, col, bins=50, color_swap=color_swap)

@app.callback(Output("game-plot3", "figure"), Input("df-store", "data"), Input("swap-colorscheme", "value"))
def update_game_top_tags(df_meta, swap):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    tags_df = top_tags_from_df(df, top_n=10)
    if tags_df.empty:
        return empty_fig("No tags found")
    palette = STEAM_COLORS_HIGH_CONTRAST if "swap_colors" in (swap or []) else STEAM_COLORS
    fig = px.bar(tags_df, x="tag", y="count", color="tag", color_discrete_sequence=palette, title="Top Tags")
    fig.update_layout(showlegend=False)
    return fig

@app.callback(
    Output("genre-plot1", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-y-select", "value"),
    Input("swap-colorscheme", "value"),
)
def update_genre_mean(df_meta, sel_genres, y_var, swap):
    if not df_meta or y_var is None:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    color_swap = "swap_colors" in (swap or [])
    palette = STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS
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
    fig = px.bar(agg_df, x="main_genre", y=ycol, color="main_genre", color_discrete_sequence=palette, title=f"Mean {y_var} by Genre (Top {len(agg_df)})")
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig

@app.callback(
    Output("genre-plot3", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("release-year-range", "value"),
    Input("year-metric-select", "value"),
    Input("swap-colorscheme", "value"),
)
def update_genre_yearly(df_meta, sel_genres, yr_range, metric, swap):
    if not df_meta:
        return empty_fig("No data loaded")
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return empty_fig("No data loaded")
    color_swap = "swap_colors" in (swap or [])
    palette = STEAM_COLORS_HIGH_CONTRAST if color_swap else STEAM_COLORS
    sel = tuple(ensure_list(sel_genres))
    year_min, year_max = (yr_range or [None, None])
    if metric == "count":
        json_counts = compute_games_per_year_counts_json(df_meta["dataset_id"], sel, year_min, year_max)
        if not json_counts:
            return empty_fig("No data for selected metric")
        counts_df = json_str_to_df(json_counts, orient="split")
        return games_per_year_by_genre_fig_from_counts_df(
            counts_df,
            year_min,
            year_max,
            genres_order=sel,
            color_swap=color_swap,
        )
    if metric == "peak_ccu":
        json_ccu = compute_peak_ccu_by_year_json(df_meta["dataset_id"], sel, year_min, year_max)
        if not json_ccu:
            return empty_fig("No data for selected metric")
        ccu_df = json_str_to_df(json_ccu, orient="split")
        ccu_df["release_year"] = ccu_df["release_year"].astype(int)
        yr_min = year_min if year_min is not None else int(ccu_df["release_year"].min())
        yr_max = year_max if year_max is not None else int(ccu_df["release_year"].max())
        pivot = ccu_df.pivot(index="release_year", columns="main_genre", values="peak_ccu_sum")
        if sel:
            desired = [g for g in sel if g in pivot.columns]
            pivot = pivot.reindex(columns=desired, fill_value=0)
        pivot = pivot.reindex(range(yr_min, yr_max + 1), fill_value=0)
        melt = pivot.reset_index().melt(id_vars="release_year", var_name="main_genre", value_name="peak_ccu_sum")
        fig = px.line(
            melt,
            x="release_year",
            y="peak_ccu_sum",
            color="main_genre",
            markers=True,
            color_discrete_sequence=palette,
            title="Peak CCU per Year by Genre",
        )
        return fig
    return empty_fig("Unknown metric")

@app.callback(
    Output("game-hist-select", "options"),
    Output("game-hist-select", "value"),
    Input("df-store", "data"),
)
def populate_game_hist_options(df_meta):
    if not df_meta:
        return [], None
    df = get_dataset(df_meta["dataset_id"])
    if df is None:
        return [], None
    numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    opts = [{"label": c, "value": c} for c in numeric]
    return opts, (numeric[0] if numeric else None)

@app.callback(
    Output("genre-plot2", "figure"),
    Input("df-store", "data"),
    Input("genre-filter", "value"),
    Input("genre-bubble-y-select", "value"),
    Input("swap-colorscheme", "value"),
)
def update_genre_scatter_bubble(df_meta, sel_genres, y_metric, swap):
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
    palette = STEAM_COLORS_HIGH_CONTRAST if "swap_colors" in (swap or []) else STEAM_COLORS
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
        title=f"Average Price vs. {y_label} (Bubble size = Avg Playtime) by Genre",
    )
    fig.update_layout(xaxis_title="Average Price", yaxis_title=y_label)
    return fig
