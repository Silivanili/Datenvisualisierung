# \Datenvisualisierung\app\layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------
# Navbar 
# -----------------------------------------------------------------
def top_navbar():
    return dbc.Navbar(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            "Steam Dashboard",
                            className="fw-bold",
                            style={"fontSize": "1.25rem", "color": "#c6d4df"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Nav(
                            [
                                dbc.NavLink(
                                    "Game Page",
                                    href="/",
                                    active="exact",
                                    className="fs-6",
                                    style={"color": "#c6d4df"},
                                ),
                                dbc.NavLink(
                                    "Genre Page",
                                    href="/genre",
                                    active="exact",
                                    className="fs-6",
                                    style={"color": "#c6d4df"},
                                ),
                                dbc.NavLink(
                                    "Developer Page",
                                    href="/developer",
                                    active="exact",
                                    className="fs-6",
                                    style={"color": "#c6d4df"},
                                ),
                            ],
                            pills=True,
                            justified=True,
                            style={
                                "width": "100%",
                                "display": "flex",
                                "justifyContent": "space-evenly",
                            },
                        ),
                        width=True,
                    ),
                ],
                align="center",
                className="g-0 w-100",
            ),
            fluid=True,
        ),
        color="#171d25",
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
            html.H5("Dataset"),
            html.P("Size in Memory: ", className="mb-1", id="dataset-size-text"),
            html.P("Games: ", className="mb-1", id="dataset-rows-text"),
            html.P("Top Tags: ", className="mb-2", id="dataset-top-tags"),
            html.A(
                "Link to dataset (Kaggle)",
                href="https://www.kaggle.com/datasets/artermiloff/steam-games-dataset",
                className="small text-decoration-none text-secondary",
            ),
            html.Hr(),

            dbc.Label("Select Dataset:"),
            dcc.Dropdown(
                id="dataset-path",
                options=[
                    {"label": "games_march2025_cleaned.csv", "value": "games_march2025_cleaned.csv"},
                    {"label": "games_march2025_full.csv", "value": "games_march2025_full.csv"},
                    {"label": "games_may2024_full.csv", "value": "games_may2024_full.csv"},
                    {"label": "games_may2024_cleaned.csv", "value": "games_may2024_cleaned.csv"},
                ],
                value="games_march2025_cleaned.csv",
                clearable=False,
                className="mb-2",
            ),
            html.Div(
                dbc.Button(
                    "Load Dataset",
                    id="load-dataset",
                    color="primary",
                    size="sm",
                    className="w-100",
                    style={
                            "backgroundColor": "#c6d4df",
                            "color": "#000",
                            "border": "1px solid #9aa7b2",
                    },
                ),
                className="mb-3",
            ),

            dbc.Label("Select Scatter X-Axis:"),
            dcc.Dropdown(
                id="scatter-x-select",
                options=[
                    {"label": "Price (USD)", "value": "price"},
                    {"label": "Metacritic Score (0-100)", "value": "metacritic_score"},
                    {"label": "User Score (0-100)", "value": "user_score"},
                    {"label": "Number of positive reviews ", "value": "positive"},
                    {"label": "Number of negative reviews ", "value": "negative"},
                    {"label": "Percentage of positive reviews (total)", "value": "pct_pos_total"},
                    {"label": "Average Playtime (Minutes)", "value": "average_playtime_forever"},
                    {"label": "Median Playtime (Minutes)", "value": "median_playtime_forever"},
                    {"label": "Release Date (Date)", "value": "release_date"},
                ],
                value="release_date",
                clearable=False,
                className="mb-3",
            ),

            dbc.Label("Select Scatter Y-Axis:"),
            dcc.Dropdown(
                id="scatter-y-select",
                options=[
                    {"label": "Peak CCU (Count)", "value": "peak_ccu"},
                    {"label": "Number of Reviews (Count)", "value": "num_reviews_total"},
                    {"label": "Percentage of Positive Reviews (0-100)", "value": "pct_pos_total"},
                ],
                value="pct_pos_total",
                placeholder="Select Y variable for scatter",
                clearable=False,
                className="mb-3",
            ),



            

            html.Hr(),

            dbc.Label("Select Game histogram:"),
            dcc.Dropdown(
                id="game-hist-select",
                options=[],
                value=None,
                placeholder="Select numeric column for game histogram/boxplot",
                clearable=False,
                className="mb-3",
            ),

            dbc.Label(
                "Bubble‑plot Y‑axis:",
                className="small",
                style={"color": "#c6d4df"},
            ),
            dcc.Dropdown(
                id="genre-bubble-y-select",
                # Fixed options – the callback will use the column name directly
                options=[
                    {"label": "Average user score", "value": "user_score"},
                    {"label": "Number of positive reviews", "value": "positive"},
                    {"label": "Number of negative reviews", "value": "negative"},
                    {"label": "Metacritic score (0-100)", "value": "metacritic_score"},
                    {"label": "Number of recommendations","value": "recommendations"},
                    {"label": "Percentage of positive reviews (total)","value": "pct_pos_total"},


                ],
                # Show user_score by default (matches the original plot)
                value="user_score",
                clearable=False,
                className="mb-3",
            ),

            dbc.Label("Select metric for lineplot:"),
            dcc.Dropdown(
                id="year-metric-select",
                options=[
                    {"label": "Number of games", "value": "count"},
                    {"label": "Peak CCU (Concurrent Users)", "value": "peak_ccu"},
                ],
                value="count",
                clearable=False,
                className="mb-3",
            ),

            html.Hr(),

            html.H5("Genre Filter"),
            dbc.Label("Select Genres:"),
            dcc.Dropdown(
                id="genre-filter",
                options=[],
                value=["Action"],
                multi=True,
                placeholder="Choose genres",
                className="mb-3",
            ),

            dbc.Label("Select Genre Y-Axis:"),
            dcc.Dropdown(
                id="genre-y-select",
                options=[],
                value=None,
                placeholder="Select Y variable for genre plots",
                clearable=False,
                className="mb-3",
            ),

            html.Hr(),

            html.H5("Release Year Range"),
            dbc.Label("Filter by release year:"),
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
            "backgroundColor": "#171d25",
            "color":"#c6d4df",
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

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot1")), width=6),
                            dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot2")), width=6),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="developer-plot3")), width=12),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
            ),
        ],
        style={"marginLeft": "20vw", "padding": "20px"},
    )


# -----------------------------------------------------------------
# Game page 
# -----------------------------------------------------------------
def game_page_layout():
    return html.Div(
        [
            html.H4(
                "Game Page",
                className="fw-bold text-center mt-3 mb-4",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="game-plot1")), width=6),
                            dbc.Col(dcc.Loading(dcc.Graph(id="game-plot2")), width=6),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="game-plot3")), width=12),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
            ),
        ],
        style={"marginLeft": "20vw", "padding": "20px"},
    )


# -----------------------------------------------------------------
# Genre page 
# -----------------------------------------------------------------
def genre_page_layout():
    return html.Div(
        [
            html.H4(
                "Genre Page",
                className="fw-bold text-center mt-3 mb-4",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot1")), width=6),
                            dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot2")), width=6),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Loading(dcc.Graph(id="genre-plot3")), width=12),
                        ]
                    )
                ),
                className="mb-4 shadow-sm",
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
            html.Link(
                rel="stylesheet",
                href="data:text/css;charset=utf-8," + """
                .navlink-steam {
                    color: #c6d4df !important;
                }
                .navlink-steam-active {
                    background-color: #c6d4df !important;
                    color: #000 !important;
                    font-weight: 600;
                }
                """,
            ),
            dcc.Location(id="url"),
            top_navbar(),
            sidebar(),
            dcc.Store(id="df-store"),
            html.Div(id="page-content"),
        ]
    )
