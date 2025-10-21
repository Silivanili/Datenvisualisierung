from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Steam Dashboard"

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
        dbc.Input(id="y-axis", type="text", placeholder="Enter variable", size="sm", className="mb-4"),

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
# Placeholder Callback Example
# =========================
@app.callback(
    Output("game-plot1", "children"),
    Input("view-settings", "value")
)
def update_game_plot(view_settings):
    return f"Active View Settings: {', '.join(view_settings) if view_settings else 'None'}"

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
