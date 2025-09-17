import dash
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, html  
import plotly.express as px 
import pandas as pd 



def load_data(): 
    data = pd.read_csv('games.csv')        
    return data

data = load_data()

avg_playtime_forever = data["Average playtime forever"]


price = data["Price"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Steam Dashboard"), width=15, className="text-center titel")  
    ]),

    dbc.Row([
        dbc.Col(html.Div(f"playtime forever: {avg_playtime_forever}", className="text-center iwas"), width = 7), 
        dbc.Col(html.Div("price: {price}", className = "text-center iwas"), width = 7)
    ], className = "some"), 



    dbc.Row([
        dbc.Col([
            dbc.CardBody([
                html.H4("something Demographics", className = "Cartman"), 
                dcc.Dropdown(
                    id = "idk"
                ), 
                dcc.Graph(id = "some-dist")
            ])
        ])
    ]),


])




if __name__ == '__main__':
    app.run(debug=True)
