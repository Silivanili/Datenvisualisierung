#Datenvisualisierung\app\__init__.py
from dash import Dash
import dash_bootstrap_components as dbc

from .cache import init_cache

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

cache = init_cache(server)
