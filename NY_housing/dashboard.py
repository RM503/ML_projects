from dash import Dash, html, dash_table, dcc 
import pandas as pd
import plotly.express as px

df = pd.read_csv('NY_housing_modified.csv')

app = Dash()

app.layout = [
    html.H1('New York City property prices', style={'textAlign' : 'center'}),
    html.Div(
        [
            html.P('This dashboard contains interactive analytics on different property prices in New York City across the five boroughs. '
                   'The data used in the creation of this dashboard can be found from Kaggle via the link https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market.'
                   , style={'fontSize' : 20})
        ]
    ),
    html.Hr(),
    dcc.Graph(
        figure = px.scatter_mapbox(
            df, lat='LATITUDE', lon='LONGITUDE', color='PRICE', size='PRICE', zoom = 10, mapbox_style="open-street-map", width=1500, height=650
        )
    )
]

if __name__ == '__main__':
    app.run(debug=True)