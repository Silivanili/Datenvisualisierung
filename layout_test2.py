from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Steam Dashboard"

# =========================
# Utility plotting functions (Plotly replacements)
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

def genre_histogram_fig(df, y_variable="average_playtime_forever", top_n=15):
    # Compute main_genre
    df = df.copy()
    if "genres" not in df.columns:
        return px.bar(title="Column 'genres' not found in dataset")
    df["main_genre"] = df["genres"].apply(parse_genre_list)
    df = df.dropna(subset=["main_genre", y_variable])
    if df.empty:
        return px.bar(title="No data for genre histogram")

    genre_playtime = (
        df.groupby("main_genre")[y_variable]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name=y_variable)
    )

    fig = px.bar(
        genre_playtime,
        x="main_genre",
        y=y_variable,
        color="main_genre",
        labels={"main_genre": "Genre", y_variable: y_variable},
        title=f"Mean {y_variable} by Genre (Top {len(genre_playtime)})",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig

def genre_gamesPerYear_fig(df, genres_of_interest=None, year_min=None, year_max=None):
    df = df.copy()
    if "genres" not in df.columns or "release_date" not in df.columns:
        return px.line(title="Required columns ('genres' or 'release_date') not found")

    df["main_genre"] = df["genres"].apply(parse_genre_list)
    df = df.dropna(subset=["main_genre"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    if genres_of_interest is None:
        # pick some common genres if not specified
        genres_of_interest = ["Action", "Indie", "RPG", "Strategy"]

    df_filtered = df[df["main_genre"].isin(genres_of_interest)]
    if year_min:
        df_filtered = df_filtered[df_filtered["release_year"] >= int(year_min)]
    if year_max:
        df_filtered = df_filtered[df_filtered["release_year"] <= int(year_max)]

    if df_filtered.empty:
        return px.line(title="No data for selected genres / years")

    counts = (
        df_filtered.groupby(["release_year", "main_genre"])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        counts,
        x="release_year",
        y="count",
        color="main_genre",
        markers=True,
        title="Number of Games per Year by Genre (selected)"
    )
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Games")
    return fig

def price_histogram_fig(df, series_col, title="Histogram", xlabel=None, bins=50, log_x=False, log_y=False, log_bins=False):
    if series_col not in df.columns:
        return px.bar(title=f"Column '{series_col}' not found")

    data = df[series_col].dropna()
    # For many price-like fields zero might be meaningful; keep >0 filter only if all are positive or requested
    data = data[~pd.isnull(data)]
    # Keep only numeric
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

        # New: input for dataset path / URL and button to load it
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
        dbc.Label("Y-Axis:", className="small text-muted"),
        dbc.Input(id="y-axis", type="text", placeholder="Enter variable (e.g. average_playtime_forever)", size="sm", className="mb-4"),

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

# dcc.Store to keep the loaded dataframe (as JSON)
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
# Load dataset callback
# =========================
@app.callback(
    Output("df-store", "data"),
    Input("load-dataset", "n_clicks"),
    State("dataset-path", "value"),
    prevent_initial_call=True
)
def load_dataset(n_clicks, path_or_url):
    if not path_or_url:
        return None
    try:
        df = pd.read_csv(path_or_url)
        # store as split-orient JSON (good for round-trip)
        return df.to_json(date_format="iso", orient="split")
    except Exception as e:
        print(f"Failed to load dataset from {path_or_url}: {e}")
        return None

# =========================
# Callbacks to render interactive Plotly figures
# =========================
@app.callback(
    Output("game-plot1", "children"),
    Input("view-settings", "value"),
    Input("df-store", "data"),
    State("y-axis", "value"),
    State("num-genres", "value")
)
def update_game_plot(view_settings, df_json, y_axis_input, num_genres):
    # Show helpful placeholders if not loaded
    if not df_json:
        return html.Div(
            [
                html.P("No dataset loaded. Enter a local path or URL to a CSV in the sidebar and click 'Load Dataset'."),
                html.P("After loading, this area will display an interactive Plotly figure.")
            ],
            className="text-muted"
        )

    try:
        df = pd.read_json(df_json, orient="split")
    except Exception as e:
        print(f"Failed to parse stored dataframe: {e}")
        return html.Div("Failed to read dataset from store.", className="text-danger")

    y_var = y_axis_input or "average_playtime_forever"
    top_n = int(num_genres) if num_genres else 15

    fig = genre_histogram_fig(df, y_variable=y_var, top_n=top_n)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

@app.callback(
    Output("game-plot2", "children"),
    Input("df-store", "data"),
    State("num-genres", "value")
)
def update_game_plot2(df_json, num_genres):
    if not df_json:
        return "(Plot Area 2)"
    try:
        df = pd.read_json(df_json, orient="split")
    except Exception as e:
        print(f"Failed to parse stored dataframe: {e}")
        return html.Div("Failed to read dataset from store.", className="text-danger")

    # Show games per year for default genre set or top N if provided
    top_n = int(num_genres) if num_genres else None
    # Determine top genres automatically if top_n is provided
    if top_n:
        if "genres" in df.columns:
            df["main_genre"] = df["genres"].apply(parse_genre_list)
            top_genres = df["main_genre"].value_counts().head(top_n).index.tolist()
        else:
            top_genres = None
    else:
        top_genres = ["Action", "Indie", "RPG", "Strategy"]

    fig = genre_gamesPerYear_fig(df, genres_of_interest=top_genres)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

@app.callback(
    Output("game-plot3", "children"),
    Input("df-store", "data"),
    State("y-axis", "value")
)
def update_game_plot3(df_json, y_axis_input):
    if not df_json:
        return "(Plot Area 3)"
    try:
        df = pd.read_json(df_json, orient="split")
    except Exception as e:
        print(f"Failed to parse stored dataframe: {e}")
        return html.Div("Failed to read dataset from store.", className="text-danger")

    # Use provided y-axis as the column to histogram, fallback to 'price' or 'average_playtime_forever'
    target = y_axis_input or ("price" if "price" in df.columns else "average_playtime_forever")
    fig = price_histogram_fig(df, target, title=f"Histogram of {target}", xlabel=target, bins=50)
    return dcc.Graph(figure=fig, config={"displayModeBar": True})

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)