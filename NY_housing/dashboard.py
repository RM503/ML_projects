''' 
Dashboard for featuring the analytics of the NYC housing price dataset
'''

from dash import Dash, html, dcc, callback, Input, Output 
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

df = pd.read_csv('NY_housing_modified.csv')
df_cropped = df[ df.PRICE <= 60*10**6 ] # capped prices for easier visualization on a map

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    html.P('Here is a representation of the price distribution over the five boroughs. The price has been capped to 60 million USD '
           'for better visualization.'),
    dcc.Graph(
        figure = px.scatter_mapbox(
            df_cropped, lat='LATITUDE', lon='LONGITUDE', color='PRICE', size='PRICE',
            zoom = 10, mapbox_style="open-street-map", width=1500, height=650
        )
    ),
    html.Hr(),
    html.Div(
        [
            html.H2('Now let us look at some analytics', style={'textAlign' : 'center'}),
            html.H3('Property price histograms and charts')
        ]
    ),
    dcc.Dropdown(
        options = ['All', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
        value = 'All',
        id = 'borough_dropdown', style={'width' : '50%'}
    ),
    dcc.Graph(
        figure = {}, id = 'borough_dropdown_figure'
    )
]

@callback(
    Output(component_id = 'borough_dropdown_figure', component_property = 'figure'),
    Input(component_id = 'borough_dropdown', component_property = 'value')
)


# This function updates the property price histogram by interacting with a dropdown menu
# Checks options for borough information: if 'All', it uses the entire dataframe, else applies a filter

def update_price_histogram(borough):
    if borough == 'All':
        fig = px.histogram(
            df, x = 'LOG_PRICE', width=650, height=500,
            labels = {'LOG_PRICE' : 'log10(PRICE)'}
        )
    else:
        df_borough = df[df['BOROUGH'] == borough]
        fig = px.histogram(
            df_borough, x = 'LOG_PRICE', width=650, height=500,
            labels = {'LOG_PRICE' : 'log10(PRICE)'}
        )
    return fig

if __name__ == '__main__':
    app.run(debug=True)