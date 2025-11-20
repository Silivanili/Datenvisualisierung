# \Datenvisualisierung\app\layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------
# Navbar 
# -----------------------------------------------------------------
def top_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Steam Dashboard",
                                className="fw-bold text-white",
                                style={"fontSize": "1.25rem"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavLink(
                                        "Game Page", href="/", active="exact", className="text-white fs-6"
                                    ),
                                    dbc.NavLink(
                                        "Genre Page", href="/genre", active="exact", className="text-white fs-6"
                                    ),
                                    dbc.NavLink(
                                        "Developer Page", href="/developer", active="exact", className="text-white fs-6"
                                    ),
                                ],
                                pills=True,
                                justified=True,
                                style={"width": "100%", "display": "flex", "justifyContent": "space-evenly"},
                            ),
                            width=True,
                        ),
                    ],
                    align="center",
                    className="g-0 w-100",
                )
            ],
            fluid=True,
        ),
        color="primary",
        dark=True,
        sticky="top",
        className="shadow-sm py-2",
    )


# -----------------------------------------------------------------
# Sidebar 
# -----------------------------------------------------------------
def sidebar():
    return html.Div(
        [
            html.H5("Dataset", className="text-primary fw-bold mt-3 mb-2"),
            html.P("Size in Memory: ", className="mb-1", id="dataset-size-text"),
            html.P("Games: ", className="mb-1", id="dataset-rows-text"),
            html.P("Top Tags: ", className="mb-2", id="dataset-top-tags"),
            html.A(
                "Link to dataset (Kaggle)",
                href="https://www.kaggle.com/datasets/artermiloff/steam-games-dataset",
                className="small text-decoration-none text-secondary",
            ),
            html.Hr(),

            dbc.Label("Select Dataset:", className="small text-muted"),
            dcc.Dropdown(
                id="dataset-path",
                options=[
                    {"label": "games_march2025_cleaned.csv", "value": "games_march2025_cleaned.csv"},
                    {"label": "games_march2025_full.csv", "value": "games_march2025_full.csv"},
                    {"label": "games_may2024_full.csv", "value": "games_may2024_full.csv"},
                    {"label": "games_may2024_cleaned.csv", "value": "games_may2024_cleaned.csv"},
                ],
                placeholder="Choose a dataset",
                clearable=False,
                className="mb-2",
            ),
            dbc.Button("Load Dataset", id="load-dataset", color="primary", size="sm", className="mb-3"),

            dbc.Label("Select Scatter X-Axis:", className="small text-muted"),
            dcc.Dropdown(
                id="scatter-x-select",
                options=[
                    {"label": c, "value": c} for c in [
                        "price",
                        "metacritic_score",
                        "user_score",
                        "positive",
                        "negative",
                        "pct_pos_total",
                        "average_playtime_forever",
                        "median_playtime_forever",
                        "peak_ccu",
                        "average_playtime_2weeks",
                        "median_playtime_2weeks",
                        "num_reviews_total"
                    ]
                ],
                value="metacritic_score",
                clearable=False,
                className="mb-3",
            ),

            dbc.Label("Select Scatter Y-Axis:", className="small text-muted"),
            dcc.Dropdown(
                id="scatter-y-select",
                options=[],    
                value=None,
                placeholder="Select Y variable for scatter",
                clearable=False,
                className="mb-3",
            ),

            dbc.Checklist(
                options=[{"label": "Hide points where Y == 0", "value": "hide"}],
                value=[],
                id="hide-zero-reviews",
                switch=True,
                className="mb-3",
            ),

            dbc.Row(
                [
                    dbc.Col(dbc.Label("Filter operator", className="small text-muted"), width=6),
                    dbc.Col(dbc.Label("Threshold", className="small text-muted"), width=6),
                ],
                className="g-0",
            ),
            dbc.Row(
                [
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
                            className="mb-2",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Input(id="y-filter-value", type="number", placeholder="Value", size="sm", className="mb-2"),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),

            html.Hr(),

            dbc.Label("Select Game histogram:", className="small text-muted"),
            dcc.Dropdown(
                id="game-hist-select",
                options=[],
                value=None,
                placeholder="Select numeric column for game histogram/boxplot",
                clearable=False,
                className="mb-3",
            ),

            dbc.Checklist(
                options=[{"label": "Swap Histogram for Boxplot", "value": "swap_hist"}],
                value=[],
                id="view-settings",
                switch=True,
                className="mb-3",
            ),

            dbc.Checklist(
                options=[{"label": "Swap Colorscheme (not implemented)", "value": "swap_colors"}],
                value=[],
                id="swap-colorscheme",
                switch=True,
                className="mb-2",
            ),
            dbc.Checklist(
                options=[{"label": "Details on hover (not implemented)", "value": "details_hover"}],
                value=[],
                id="details-on-hover",
                switch=True,
                className="mb-3",
            ),

            dbc.Label("Select metric for lineplot:", className="small text-muted"),
            dcc.Dropdown(
                id="year-metric-select",
                options=[
                    {"label": "Number of games", "value": "count"},
                    {"label": "Peak CCU", "value": "peak_ccu"},
                ],
                value="count",
                clearable=False,
                className="mb-3",
            ),

            html.Hr(),

            html.H5("Genre Filter", className="text-primary fw-bold mb-2"),
            dbc.Label("Select Genres:", className="small text-muted"),
            dcc.Dropdown(
                id="genre-filter",
                options=[],
                value=["Action"],
                multi=True,
                placeholder="Choose genres",
                className="mb-3",
            ),

            dbc.Label("Select Genre Y-Axis:", className="small text-muted"),
            dcc.Dropdown(
                id="genre-y-select",
                options=[],    
                value=None,
                placeholder="Select Y variable for genre plots",
                clearable=False,
                className="mb-3",
            ),

            html.Hr(),

            html.H5("Release Year Range", className="text-primary fw-bold mb-2"),
            dbc.Label("Filter by release year:", className="small text-muted"),
            dcc.RangeSlider(
                id="release-year-range",
                min=1970,
                max=2025,
                step=1,
                value=[2010, 2020],
                marks={1970: "1970", 1980: "1980", 1990: "1990", 2000: "2000", 2010: "2010", 2020: "2020"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
                className="mb-3",
            ),
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "20px",
            "height": "100vh",
            "borderRight": "1px solid #ddd",
            "width": "18vw",
            "position": "fixed",
            "overflowY": "auto",
        },
    )


# -----------------------------------------------------------------
# Developer page 
# -----------------------------------------------------------------
def developer_page_layout():
 
    return html.Div(
        [
            html.H4(
                "Developer Page (Subset, for Timeline Graphic)",
                className="fw-bold text-center mt-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot1")), width=6),
                    dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot2")), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot3")), width=12),
                ]
            ),
        ],
        style={"marginLeft": "20vw", "padding": "20px"},
    )


# -----------------------------------------------------------------
# Page‑specific small layouts
# -----------------------------------------------------------------
def game_page_layout():

    return html.Div(
        [
            html.H4(
                "Game Page",
                className="fw-bold text-center mt-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="game-plot1")), width=6),
                    dbc.Col(dcc.Loading(dcc.Graph(id="game-plot2")), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="game-plot3")), width=12),
                ]
            ),
        ],
        style={"marginLeft": "20vw", "padding": "20px"},
    )


def genre_page_layout():

    return html.Div(
        [
            html.H4(
                "Genre Page",
                className="fw-bold text-center mt-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot1")), width=6),
                    dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot2")), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot3")), width=12),
                ]
            ),
        ],
        style={"marginLeft": "20vw", "padding": "20px"},
    )


# -----------------------------------------------------------------
# Combined app layout
# -----------------------------------------------------------------
def app_layout():
    return html.Div(
        [
            dcc.Location(id="url"),
            top_navbar(),
            sidebar(),
            dcc.Store(id="df-store"),
            html.Div(id="page-content"),
        ]
    )
