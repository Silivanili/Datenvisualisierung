from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from flask_caching import Cache
import uuid

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Steam Dashboard"

# Initialize server-side cache (filesystem). Tune CACHE_DIR and timeout as needed.
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.cache',
    'CACHE_DEFAULT_TIMEOUT': 60 * 60  # 1 hour default
})

# =========================
# Utility functions & plotting helpers
#
# =========================
def parse_genre_list(x):
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

def mean_playtime_by_genre_fig_from_df(genre_playtime_df, y_variable="average_playtime_forever"):
    """Render a bar chart showing mean y_variable by genre (expects aggregated DF)."""
    if genre_playtime_df is None or genre_playtime_df.empty:
        return px.bar(title="No data for mean playtime by genre")
    fig = px.bar(
        genre_playtime_df,
        x="main_genre",
        y=y_variable,
        color="main_genre",
        labels={"main_genre": "Genre", y_variable: y_variable},
        title=f"Mean {y_variable} by Genre (Top {len(genre_playtime_df)})",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig

def games_per_year_by_genre_fig_from_df(counts_df):
    """Render a line chart showing number of games per year per genre (expects counts DF)."""
    if counts_df is None or counts_df.empty:
        return px.line(title="No data for games per year by genre")
    fig = px.line(
        counts_df,
        x="release_year",
        y="count",
        color="main_genre",
        markers=True,
        title="Number of Games per Year by Genre (selected)"
    )
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Games")
    return fig

def histogram_of_numeric_column_fig(df, series_col, title="Histogram", xlabel=None, bins=50, log_x=False, log_y=False, log_bins=False):
    """Render histogram for a numeric column (renamed for clarity)."""
    if series_col not in df.columns:
        return px.bar(title=f"Column '{series_col}' not found")

    data = df[series_col].dropna()
    data = pd.to_numeric(data, errors="coerce").dropna()
    if data.empty:
        return px.bar(title=f"No numeric data in column '{series_col}'")

    # Remove non-positive values if log scale is requested
    if (log_x or log_bins) and (data <= 0).any():
        data = data[data > 0]
        if data.empty:
            return px.bar(title=f"No positive values for log-scale histogram in '{series_col}'")

    if log_bins:
        minv = data.min()
        maxv = data.max()
        bins_edges = np.logspace(np.log10(minv), np.log10(maxv), bins + 1)
    else:
        bins_edges = np.linspace(data.min(), data.max(), bins + 1)

    counts, edges = np.histogram(data, bins=bins_edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist_df = pd.DataFrame({"bin_center": centers, "count": counts})

    if log_bins:
        fig = px.bar(hist_df, x="bin_center", y="count", title=title)
        fig.update_xaxes(type="log")
    else:
        fig = px.bar(hist_df, x="bin_center", y="count", title=title)

    fig.update_traces(marker_line_width=0)
    fig.update_layout(xaxis_title=(xlabel or series_col), yaxis_title="Count")

    if log_x and not log_bins:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    return fig

def scatter_release_vs_fig(df, y_col, hide_zero=False, operator=None, threshold=None, max_points=20000):
    """
    Scatter plot: X = release_date (datetime), Y = y_col.
    Filters:
      - hide_zero: if True, drop rows where y_col == 0
      - operator/threshold: apply comparator to the y_col (operator in {"eq","ge","le","gt","lt"})
    For performance sample to max_points if dataset larger.
    """
    if "release_date" not in df.columns:
        return px.scatter(title="Column 'release_date' not found")
    if y_col not in df.columns:
        return px.scatter(title=f"Column '{y_col}' not found")

    # Ensure release_date is datetime
    if not np.issubdtype(df["release_date"].dtype, np.datetime64):
        df = df.copy()
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    data = df.dropna(subset=["release_date", y_col]).copy()
    if data.empty:
        return px.scatter(title=f"No data for '{y_col}' vs release_date")

    # Ensure y is numeric if possible
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col])
    if data.empty:
        return px.scatter(title=f"No numeric data for '{y_col}'")

    # Apply hide-zero filter (applies to the selected y variable)
    if hide_zero:
        data = data[data[y_col] != 0]

    # Apply general comparator filter if provided
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
            # if threshold can't be parsed, ignore comparator
            pass

    if data.empty:
        return px.scatter(title="No points after applying filters")

    # Performance sampling
    if len(data) > max_points:
        data = data.sample(max_points, random_state=1)

    color = "main_genre" if "main_genre" in data.columns else None
    hover_cols = [c for c in ["name", "appid", "main_genre"] if c in data.columns]
    fig = px.scatter(
        data,
        x="release_date",
        y=y_col,
        color=color,
        hover_data=hover_cols,
        title=f"{y_col} over Time (scatter)"
    )
    fig.update_layout(xaxis_title="Release Date", yaxis_title=y_col)
    return fig

# =========================
# Cached computation helpers (renamed to match what they compute)
# =========================
@cache.memoize()  # cache keyed by arguments (dataset_id, y_var, top_n)
def compute_mean_playtime_by_genre(dataset_id, y_var, top_n):
    """
    Returns JSON (orient='split') of the aggregated mean y_var by main_genre (top_n).
    """
    df = cache.get(dataset_id)
    if df is None:
        return None
    if y_var not in df.columns:
        return None
    g = (
        df.groupby("main_genre")[y_var]
        .mean()
        .sort_values(ascending=False)
        .head(int(top_n))
        .reset_index()
    )
    return g.to_json(orient="split")

@cache.memoize()
def compute_games_per_year_counts(dataset_id, genres_tuple, year_min, year_max):
    """
    Returns JSON (orient='split') of counts per year for the specified genres/range.
    """
    df = cache.get(dataset_id)
    if df is None:
        return None
    df_local = df.copy()
    if "release_year" not in df_local.columns:
        return None
    if genres_tuple:
        genres = list(genres_tuple)
    else:
        if "main_genre" in df_local.columns:
            genres = df_local["main_genre"].value_counts().head(4).index.tolist()
        else:
            genres = []
    df_f = df_local[df_local["main_genre"].isin(genres)]
    if year_min is not None:
        df_f = df_f[df_f["release_year"] >= int(year_min)]
    if year_max is not None:
        df_f = df_f[df_f["release_year"] <= int(year_max)]
    if df_f.empty:
        return None
    counts = (
        df_f.groupby(["release_year", "main_genre"])
        .size()
        .reset_index(name="count")
    )
    return counts.to_json(orient="split")

# =========================
# Top Navigation Bar
# =========================
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            "Steam Dashboard",
                            className="fw-bold text-white",
                            style={"fontSize": "1.25rem"}
                        ),
                        width="auto"
                    ),
                    dbc.Col(
                        dbc.Nav(
                            [
                                dbc.NavLink("Game Page", href="/", active="exact", className="text-white fs-6"),
                                dbc.NavLink("Genre Page", href="/genre", active="exact", className="text-white fs-6"),
                                dbc.NavLink("Developer Page", href="/developer", active="exact", className="text-white fs-6"),
                            ],
                            pills=True,
                            justified=True,
                            style={"width": "100%", "display": "flex", "justifyContent": "space-evenly"}
                        ),
                        width=True
                    ),
                ],
                align="center",
                className="g-0 w-100",
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    sticky="top",
    className="shadow-sm py-2"
)

# =========================
# Sidebar
# =========================
sidebar = html.Div(
    [
        html.H5("Dataset", className="text-primary fw-bold mt-3 mb-2"),
        html.P("Size in Memory: ", className="mb-1"),
        html.P("Games: ", className="mb-1"),
        html.P("Top Tags: ", className="mb-2"),
        html.A("Link to dataset", href="https://www.kaggle.com/datasets/artermiloff/steam-games-dataset", className="small text-decoration-none text-secondary"),
        html.Hr(),

        dbc.Label("Dataset CSV Path or URL:", className="small text-muted"),
        dbc.Input(id="dataset-path", type="text", placeholder="Enter local path or URL to CSV", size="sm", className="mb-2"),
        dbc.Button("Load Dataset", id="load-dataset", color="primary", size="sm", className="mb-3"),

        html.H5("View Settings", className="text-primary fw-bold mb-2"),
        dbc.Checklist(
            options=[
                {"label": "Details on hover", "value": "hover"},
                {"label": "Top 15 / Top 30", "value": "top"},
                {"label": "Swap Colorscheme", "value": "color"},
                {"label": "Swap Histogram for Boxplot", "value": "swap"},
            ],
            value=[],
            id="view-settings",
            switch=True,
            className="mb-3"
        ),
        html.Hr(),

        html.H5("Lineplot Settings", className="text-primary fw-bold mb-2"),
        dbc.Label("Num. Genres:", className="small text-muted"),
        dbc.Input(id="num-genres", type="number", placeholder="Enter number", size="sm", className="mb-2"),

        # Replaced free-text y-axis input with a dynamic dropdown that will be populated after loading the dataset.
        dbc.Label("Y-Axis (select variable):", className="small text-muted"),
        dcc.Dropdown(id="y-axis-select", options=[], placeholder="Select Y variable", clearable=False, className="mb-3"),

        # New: hide zero toggle (applies to the selected Y variable)
        dbc.Checklist(
            options=[{"label": "Hide points where Y == 0", "value": "hide"}],
            value=[],
            id="hide-zero-reviews",
            switch=True,
            className="mb-3"
        ),

        # New: comparator and threshold for Y-filter
        dbc.Row([
            dbc.Col(dbc.Label("Filter operator", className="small text-muted"), width=6),
            dbc.Col(dbc.Label("Threshold", className="small text-muted"), width=6),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="y-filter-operator",
                    options=[
                        {"label": "=", "value": "eq"},
                        {"label": ">=", "value": "ge"},
                        {"label": "<=", "value": "le"},
                        {"label": ">", "value": "gt"},
                        {"label": "<", "value": "lt"},
                    ],
                    placeholder="Operator",
                    clearable=True,
                    className="mb-2"
                ), width=6
            ),
            dbc.Col(
                dbc.Input(id="y-filter-value", type="number", placeholder="Value", size="sm", className="mb-2"),
                width=6
            )
        ], className="mb-3"),

        html.Hr(),

        html.H5("Timeline Settings", className="text-primary fw-bold mb-2"),
        dbc.Label("Developer:", className="small text-muted"),
        dbc.Input(id="developer", type="text", placeholder="Enter developer", size="sm", className="mb-2"),
        dbc.Checklist(
            options=[{"label": "Toggle Zoom", "value": "zoom"}],
            value=[],
            id="toggle-zoom",
            switch=True
        )
    ],
    style={
        "backgroundColor": "#f8f9fa",
        "padding": "20px",
        "height": "100vh",
        "borderRight": "1px solid #ddd",
        "width": "18vw",
        "position": "fixed",
        "overflowY": "auto"
    },
)

# dcc.Store to keep only the dataset_id and light metadata (not the full DF)
store = dcc.Store(id="df-store")

# =========================
# Page Content Layouts
# =========================
def page_layout(title):
    return html.Div(
        [
            html.H4(title, className="fw-bold text-center mt-3 mb-4"),
            dbc.Row([
                dbc.Col(html.Div(id=f"{title.split()[0].lower()}-plot1",
                                 className="border rounded bg-light p-3 text-center text-muted",
                                 children="(Plot Area 1)"), width=6),
                dbc.Col(html.Div(id=f"{title.split()[0].lower()}-plot2",
                                 className="border rounded bg-light p-3 text-center text-muted",
                                 children="(Plot Area 2)"), width=6),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(html.Div(id=f"{title.split()[0].lower()}-plot3",
                                 className="border rounded bg-light p-3 text-center text-muted",
                                 children="(Plot Area 3)"), width=12)
            ])
        ],
        style={"marginLeft": "20vw", "padding": "20px"}
    )

game_page = page_layout("Game Page (whole Dataset)")
genre_page = page_layout("Genre Page (whole Dataset)")
developer_page = page_layout("Developer Page (Subset, for Timeline Graphic)")

# =========================
# App Layout
# =========================
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    sidebar,
    store,
    html.Div(id="page-content")
])

# =========================
# Navigation Callback
# =========================
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/":
        return game_page
    elif pathname == "/genre":
        return genre_page
    elif pathname == "/developer":
        return developer_page
    else:
        return html.H1("404: Page not found", className="text-danger text-center mt-5")

# =========================
# Load dataset callback (precompute expensive columns, cache server-side)
# =========================
@app.callback(
    Output("df-store", "data"),
    Input("load-dataset", "n_clicks"),
    State("dataset-path", "value"),
    prevent_initial_call=True
)
def load_dataset(n_clicks, path_or_url):
    """
    Loads CSV into server-side cache and stores only a dataset_id in dcc.Store.
    Also precomputes 'main_genre' and 'release_year' and casts common numeric columns.
    """
    if not path_or_url:
        return None
    try:
        df = pd.read_csv(path_or_url, low_memory=False)
    except Exception as e:
        print(f"Failed to load dataset from {path_or_url}: {e}")
        return None

    # Precompute columns that used to be computed per-callback (expensive)
    if "genres" in df.columns:
        df["main_genre"] = df["genres"].apply(parse_genre_list)
    else:
        df["main_genre"] = None

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype('Int64')
    else:
        df["release_year"] = pd.NA

    for col in ("average_playtime_forever", "price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save in server-side cache and return only small metadata to client
    dataset_id = str(uuid.uuid4())
    cache.set(dataset_id, df)

    # Optionally pre-warm cache for a common aggregation
    try:
        compute_mean_playtime_by_genre(dataset_id, "average_playtime_forever", 15)
    except Exception:
        pass

    meta = {"dataset_id": dataset_id, "nrows": int(len(df))}
    return meta

# =========================
# Populate Y-axis dropdown options after loading dataset
# =========================
@app.callback(
    Output("y-axis-select", "options"),
    Output("y-axis-select", "value"),
    Input("df-store", "data"),
)
def populate_y_options(df_meta):
    """
    When a dataset is loaded, populate the Y-axis select with available columns.
    Default selection prefers 'average_playtime_forever' or 'price' or first numeric column.
    """
    if not df_meta:
        return [], None
    dataset_id = df_meta.get("dataset_id")
    if not dataset_id:
        return [], None
    df = cache.get(dataset_id)
    if df is None:
        return [], None

    cols = df.columns.tolist()
    options = [{"label": c, "value": c} for c in cols]

    default = None
    for prefer in ("average_playtime_forever", "price"):
        if prefer in cols:
            default = prefer
            break
    if default is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        default = numeric[0] if numeric else (cols[0] if cols else None)

    return options, default

# =========================
# Callbacks to render interactive Plotly figures on the GENRE PAGE
# - NOTE: outputs were moved from game-page ids to genre-page ids
# =========================
@app.callback(
    Output("genre-plot1", "children"),
    Input("view-settings", "value"),
    Input("df-store", "data"),
    State("y-axis-select", "value"),
    State("num-genres", "value")
)
def update_genre_mean_playtime(view_settings, df_meta, y_axis_input, num_genres):
    # If dataset not loaded show helpful message
    if not df_meta:
        return html.Div(
            [
                html.P("No dataset loaded. Enter a local path or URL to a CSV in the sidebar and click 'Load Dataset'."),
                html.P("After loading, this area will display an interactive Plotly figure on the Genres page.")
            ],
            className="text-muted"
        )

    dataset_id = df_meta.get("dataset_id")
    if not dataset_id:
        return html.Div("No dataset id found.", className="text-danger")

    y_var = y_axis_input or "average_playtime_forever"
    top_n = int(num_genres) if num_genres else 15

    agg_json = compute_mean_playtime_by_genre(dataset_id, y_var, top_n)
    if not agg_json:
        return html.Div(f"No data available for column '{y_var}'.", className="text-muted")

    genre_playtime_df = pd.read_json(agg_json, orient="split")
    fig = mean_playtime_by_genre_fig_from_df(genre_playtime_df, y_variable=y_var)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

@app.callback(
    Output("genre-plot2", "children"),
    Input("df-store", "data"),
    State("num-genres", "value")
)
def update_genre_games_per_year(df_meta, num_genres):
    if not df_meta:
        return "(Plot Area 2)"
    dataset_id = df_meta.get("dataset_id")
    if not dataset_id:
        return "(Plot Area 2)"

    top_n = int(num_genres) if num_genres else None
    if top_n:
        df = cache.get(dataset_id)
        if df is None:
            return html.Div("Dataset not found in cache.", className="text-danger")
        if "main_genre" in df.columns:
            top_genres = tuple(df["main_genre"].value_counts().head(top_n).index.tolist())
        else:
            top_genres = None
    else:
        top_genres = tuple(["Action", "Indie", "RPG", "Strategy"])

    counts_json = compute_games_per_year_counts(dataset_id, top_genres, None, None)
    if not counts_json:
        return html.Div("No data for selected genres / years.", className="text-muted")
    counts_df = pd.read_json(counts_json, orient="split")
    fig = games_per_year_by_genre_fig_from_df(counts_df)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

@app.callback(
    Output("genre-plot3", "children"),
    Input("df-store", "data"),
    State("y-axis-select", "value")
)
def update_genre_histogram(df_meta, y_axis_input):
    if not df_meta:
        return "(Plot Area 3)"
    dataset_id = df_meta.get("dataset_id")
    if not dataset_id:
        return "(Plot Area 3)"
    df = cache.get(dataset_id)
    if df is None:
        return html.Div("Dataset not found in cache.", className="text-danger")

    target = y_axis_input or ("price" if "price" in df.columns else "average_playtime_forever")
    fig = histogram_of_numeric_column_fig(df, target, title=f"Histogram of {target}", xlabel=target, bins=50)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

# =========================
# Scatter on GAME PAGE (X = release_date, Y selectable from sidebar)
# - includes options: hide y==0 and general comparator filter
# =========================
@app.callback(
    Output("game-plot1", "children"),
    Input("df-store", "data"),
    Input("y-axis-select", "value"),
    Input("hide-zero-reviews", "value"),
    Input("y-filter-operator", "value"),
    Input("y-filter-value", "value")
)
def update_game_scatter(df_meta, selected_y, hide_zero_value, operator, threshold):
    if not df_meta:
        return html.Div(
            [
                html.P("No dataset loaded. Enter a local path or URL to a CSV in the sidebar and click 'Load Dataset'."),
                html.P("After loading, choose a Y variable and optional filters from the sidebar to see the scatter plot here.")
            ],
            className="text-muted"
        )
    dataset_id = df_meta.get("dataset_id")
    if not dataset_id:
        return html.Div("No dataset id found.", className="text-danger")
    df = cache.get(dataset_id)
    if df is None:
        return html.Div("Dataset not found in cache.", className="text-danger")

    y_col = selected_y or ("average_playtime_forever" if "average_playtime_forever" in df.columns else None)
    if y_col is None:
        return html.Div("No Y-axis column selected or available.", className="text-muted")

    hide_zero = bool(hide_zero_value and "hide" in hide_zero_value)
    fig = scatter_release_vs_fig(df, y_col, hide_zero=hide_zero, operator=operator, threshold=threshold)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)