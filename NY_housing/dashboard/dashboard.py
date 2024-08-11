''' 
Dashboard for featuring the analytics of the NYC housing price dataset
'''

from dash import Dash, html, dcc, callback, Input, Output, State 
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle 
import json 
import plotly.express as px

df = pd.read_csv('../NY_housing_modified.csv')
df_cropped = df[ df.PRICE <= 60*10**6 ] # capped prices for easier visualization on a map
df_sub = df.loc[:,['BEDS', 'BATH', 'LOG_SQFT', 'LOG_PRICE']]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = [
    html.H1('New York City property prices', style={'textAlign' : 'center'}),
    html.Div(
        [
            dcc.Markdown('This dashboard contains interactive analytics on different property prices in New York City across the five boroughs. '
                   'The data used in the creation of this dashboard can be found from Kaggle via the link https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market.'
            )
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
    
    html.H2('Now let us look at some analytics', style={'textAlign' : 'center'}),

    # Two columns of interactive histograms and boxplots: (i) property prices (left), (ii) property types (right) 

    dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.H3('Property price histograms and charts', style = {'fontSize' : 20}),
                        html.Label('Choose a borough'),
                        dcc.Dropdown(
                            options = ['All', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
                            value = 'All',
                            id = 'borough_dropdown', style={'width' : '50%'}
                        ),
                         dcc.Graph(
                            figure = {}, id = 'borough_dropdown_figure'
                        )
                    ]
                )
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H3('Property type histograms and charts', style = {'fontSize' : 20}),
                        html.Label('Choose a type of property'),
                        dcc.Dropdown(
                            options = df.TYPE.unique().tolist() + ['All'],
                            value = 'All',
                            id = 'type_dropdown', style = {'width' : '50%'}
                        ),
                        dcc.Graph(
                            figure = {}, id = 'type_dropdown_figure'
                        )
                    ]
                )
            )
        ]
    ),

    # Two columns for displaying the various correlations between attributes

    dbc.Row(
        [
            html.H3('Correlations', style = {'fontSize' : 20}),
            html.P('The correlations between different the property size and other attributes can be explored here.'),
            html.Label('Choose a variable'),
            dcc.RadioItems(
                options = [
                    {'label' : 'Beds', 'value' : 'BEDS'},
                    {'label' : 'Baths', 'value' : 'BATH'},
                    {'label' : 'log(Price)', 'value' : 'LOG_PRICE'}
                ],
                value = 'LOG_PRICE',
                id = 'corr_buttons'
            ),
            dbc.Col(
                html.Div(
                    [
                        dcc.Graph(
                            figure = {}, id = 'corr_figure'
                        )
                    ]
                )
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        figure = px.imshow(
                            np.round(df_sub.corr(),2), text_auto = True,
                            width=650, height=500, labels = dict(color='correlation'),
                            x = ['Beds', 'Baths', 'log(Sqft)', 'log(Price)'],
                            y = ['Beds', 'Baths', 'log(Sqft)', 'log(Price)']
                        )
                    )
                )
            )
        ]
    ),
    html.Hr(),

    # Prediction section 
    # Contains three dropdowns, two sliders and an input box
    # Outputs the predicted price

    html.H2('Prediction', style = {'textAlign' : 'center'}),
    dcc.Markdown(
                '''
                 The property prices are predicted by training the Kaggle dataset, after cleaning and error-correcting, and fitting it to a *regressor*. 
                 The regressor used here is a kind of Boosted Tree algorithm called **Extreme Gradient Boosting** (XGBoost). The model takes in the following parameters -

                 1. property type
                 2. no. of bed rooms
                 3. no. of bath rooms
                 4. property size (in square feet)
                 5. borough
                 6. neighborhood (from a predetermined list depending on ZIP code)

                 '''
        ),
    dbc.Row(
        html.Div(
            [
                html.H5(
                    'Choose a borough', style = {'textAlign' : 'center'}
                ),
                dcc.Dropdown(
                    options = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
                    id = 'predict_borough_dropdown',
                    style = {'margin' : 'auto', 'width' : '50%'}
                ),
                html.H5(
                    'Choose a neighborhood', style = {'textAlign' : 'center'}
                ),
                dcc.Dropdown(
                    options = [],
                    id = 'predict_nbhd_dropdown',
                    style = {'margin' : 'auto', 'width' : '50%'}
                ),
                html.H5('Choose a property type', style = {'textAlign' : 'center'}),
                dcc.Dropdown(
                    options = df['TYPE'].unique().tolist(),
                    id = 'predict_type_dropdown',
                    style = {'margin' : 'auto', 'width' : '50%'}
                ),
                html.H5('Choose a number of beds and baths (capped to 15)', style = {'textAlign' : 'center'}),
                html.Div(
                    [
                        dcc.Slider(
                            min = 0, max = 15, step = 1, value = 5,
                            id = 'predict_bed_slider'
                        ),
                        dcc.Slider(
                            min = 0, max = 15, step = 1, value = 5,
                            id = 'predict_bath_slider'
                        )
                    ], style = {'margin' : 'auto', 'width' : '50%'}
                )
            ]
        )
    ),
    html.H5('Enter property size in square feet', style = {'textAlign' : 'center'}),
    html.Div(
            [
                dcc.Input(
                    id = 'predict_sqft_input',
                    placeholder = 'Enter property size',
                    type = 'number',
                    debounce = True 
                )
            ],
            style={"display": "flex", "justifyContent": "center"},
        ),
    html.Hr(),
    html.H5('Prediction', style={'textAlign' : 'center'}),
    html.Output(id = 'predicted_price', style = {'textAlign' : 'center'}),
    html.Hr()
]

''' 
=== callbacks and update functions ===
'''

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
            labels = {'LOG_PRICE' : 'log(Price)'}, opacity = 0.75,
            marginal = 'box'
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    else:
        df_borough = df[df['BOROUGH'] == borough]
        fig = px.histogram(
            df_borough, x = 'LOG_PRICE', width = 650, height = 500,
            labels = {'LOG_PRICE' : 'log(Price)'}, opacity = 0.75,
            marginal = 'box'
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig

# This function controls the data for property type by interacting with a dropdown menu
# Checks options for type information: if 'All', it outputs boxplots for each type

@callback(
    Output(component_id = 'type_dropdown_figure', component_property = 'figure'),
    Input(component_id = 'type_dropdown', component_property = 'value')
)

def update_type_histogram(type):
    if type == 'All':
        fig = px.box(
            df, x = 'LOG_PRICE', y = 'TYPE', width = 650, height = 500,
            labels = {'LOG_PRICE' : 'log(Price)', 'TYPE' : 'Property type'}, color = 'TYPE'
        )
    else:
        df_type = df[ df['TYPE'] == type ]
        fig = px.histogram(
            df_type, x = 'LOG_PRICE', width = 650, height = 500,
            labels = {'LOG_PRICE' : 'log(Price)'}, opacity = 0.75,
            marginal = 'box'
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig

# This function controls the correlation scatter plots

@callback(
    Output(component_id = 'corr_figure', component_property = 'figure'),
    Input(component_id = 'corr_buttons', component_property = 'value')
)

def update_corr_plot(var):
    df_var = df.loc[:, [var, 'LOG_SQFT', 'BOROUGH']]
    fig = px.scatter(
        df_var, x = var, y = 'LOG_SQFT', color = 'BOROUGH',
        width=650, height=500
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    return fig

# This function function uses chained-callback for separating neighborhoods 
# for a chosen borough

@callback(
    Output(component_id = 'predict_nbhd_dropdown', component_property = 'options'),
    Input(component_id = 'predict_borough_dropdown', component_property = 'value')
)

def set_nbhd_dropdown(borough):
    df_nbhd = df[df['BOROUGH'] == borough]
    return df_nbhd['NEIGHBORHOOD'].unique().tolist()

# This function takes in the input for the final prediction
# The first three inputs are states such that the callback is not triggered until the final field is set

@callback(
    Output(component_id = 'predicted_price', component_property = 'children'),
    [
        State(component_id = 'predict_borough_dropdown', component_property = 'value'),
        State(component_id = 'predict_nbhd_dropdown', component_property = 'value'),
        State(component_id = 'predict_type_dropdown', component_property = 'value'),
        State(component_id = 'predict_bed_slider', component_property = 'value'),
        State(component_id = 'predict_bath_slider', component_property = 'value'),
        Input(component_id = 'predict_sqft_input', component_property = 'value')
    ]
)

def price_prediction(borough, nbhd, type, bed, bath, sqft):
    ''' 
    The if-else blocks are there to ensure that the callback only works if all the inputs
    are in place. This prevents subsequent callback errors from occuring.
    '''
    if any(arg is None for arg in (borough, nbhd, type, bed, bath, sqft)):
        raise PreventUpdate
    else:
        with open('./artifacts/columns.json', 'r') as file:
            cols = json.load(file)['data_columns']

        with open('./artifacts/NYC_property_price_regression.pickle', 'rb') as file:
            model = pickle.load(file)

            borough_idx = cols.index(borough.lower())
            nbhd_idx = cols.index(nbhd.lower())
            type_idx = cols.index(type.lower())

            x = np.zeros(len(cols))
            x[0] = bed 
            x[1] = bath
            x[2] = np.log10(sqft)
            x[type_idx] = 1.0
            x[borough_idx] = 1.0
            x[nbhd_idx] = 1.0

            logprice = model.predict([x])[0]
            price = np.round(10**logprice)

            return f'The predicted price is {price:,} USD.'


if __name__ == '__main__':
    app.run(debug=True)