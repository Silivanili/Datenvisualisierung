import dash
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, html  
import plotly.express as px 
import pandas as pd 

def load_data(): 
    data = pd.read_csv('games.csv')        
    return data

data = load_data()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Steam Dashboard"), width=15, className="text-center titel")  
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
