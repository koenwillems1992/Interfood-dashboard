import datetime
from dash import Dash, html, dcc, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import numpy as np
import random
import plotly.graph_objs as go

app = Dash(__name__, suppress_callback_exceptions=True,
           extergnal_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

sidebar_style = {
    # "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "right": 0,
    "width": "18rem",
    "background-color": "#ffffff",
}

card_header_style = {
    "background-color": "#2bb353",
    "color": "white",
}


# Title
def title():
    return html.Div([
        dbc.Card([

            dbc.CardBody([
                html.Span([
                    html.I(className="bi bi-triangle-fill", style={'font-size': '35px'}),
                    html.Span(" interfood ", style={'font-size': '40px'}),

                ])

            ]),
        ], style={"textAlign": 'left', "background-color": "#2bb353", "color": 'white'})
    ])


def input_menu():
    return html.Div([dbc.Card([
        dbc.CardHeader([
            html.H4("Trade details")
        ], style=card_header_style),
        dbc.CardBody([
            html.Br(),

            dbc.Label('Client name'),
            dcc.Dropdown(
                id='client_name_dropdown',
                options=[
                    {'label': 'Customer A', 'value': 15412},
                    {'label': 'Customer B', 'value': 78954},
                    {'label': 'Customer C', 'value': 34532},
                    {'label': 'Customer D', 'value': 45562},
                    {'label': 'Customer E', 'value': 52155}
                ],
                value='Client Name',
                multi=False,
                searchable=True,
                clearable=False
            ),
            html.Br(),

            dbc.Label('Exchange'),
            dcc.Dropdown(
                id='exchange_dropdown',
                options=[
                    {'label': 'EEX', 'value': 1},
                    {'label': 'CME Group', 'value': 1},
                    {'label': 'NZD', 'value': 1},
                    {'label': 'SGX', 'value': 1},
                ],
                value='Exchange',
                multi=False,
                searchable=True,
                clearable=False
            ),
            html.Br(),

            dbc.Label('Product'),
            dcc.Dropdown(
                id='product_dropdown',
                options=[
                    {'label': 'Butter Futures', 'value': 1},
                    {'label': 'Skimmed Milk Powder Futures', 'value': 2},
                    {'label': 'Whey Powder Futures', 'value': 3},
                    {'label': 'Liquid Milk Futures', 'value': 4},
                ],
                value='Product',
                multi=False,
                searchable=True,
                clearable=False
            ),
            html.Br(),

            dbc.Label('Maturity Month'),
            dcc.Dropdown(
                id='maturity_month_dropdown',
                options=[
                    {'label': 'sep-22', 'value': 1},
                    {'label': 'okt-22', 'value': 2},
                    {'label': 'nov-22', 'value': 3},
                    {'label': 'dec-22', 'value': 4},
                    {'label': 'jan-23', 'value': 5},
                    {'label': 'feb-23', 'value': 6},
                    {'label': 'mrt-23', 'value': 7},
                    {'label': 'apr-23', 'value': 8},
                    {'label': 'mei-23', 'value': 9},
                    {'label': 'jun-23', 'value': 10},
                    {'label': 'jul-23', 'value': 11},
                    {'label': 'aug-23', 'value': 12},
                    {'label': 'sep-23', 'value': 13},
                    {'label': 'okt-23', 'value': 14},
                    {'label': 'nov-23', 'value': 15},
                    {'label': 'dec-23', 'value': 16},
                    {'label': 'jan-24', 'value': 17},
                    {'label': 'dec-24', 'value': 18},
                ],
                value='Maturity Month',
                multi=False,
                searchable=True,
                clearable=False
            ),
            html.Br(),

            dbc.Label('Number of contracts',
                      html_for="input_number_contracts", ),
            dbc.Input(
                type="number",
                id='input_number_contracts',
                value=1, min=0
            ),
            html.Br(),
        ]),
    ], style=sidebar_style, ), ])


def drawText(text):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(text),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])


def card_header_with_output(text_cardheader, id_output):
    return dbc.Card([
        dbc.CardHeader([
            html.H5(text_cardheader)
        ], style=card_header_style),
        dbc.CardBody([
            html.Div(id=id_output, className="text-center fs-3 text fw-bold")
        ])
    ])


app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                title()
            ])
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                input_menu()
            ], width=3),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        card_header_with_output("Trade PFE", id_output="pfe_trade")
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5("Current PFE of Customer"),
                                html.Br()
                            ], style=card_header_style),
                            dbc.CardBody([
                                html.Div(id="current_pfe_customer", className="text-center fs-3 text fw-bold")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        card_header_with_output("PFE of Customer including new trade",
                                                id_output="pfe_customer_with_trade")
                    ])
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        drawText("Explanation?")
                    ])
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        card_header_with_output("Simulated potential price path", id_output="output_random_walk_graph")
                    ]),
                    dbc.Col([
                        card_header_with_output("Terminal price distribution", id_output="output_distribution")
                    ])
                ]),
                html.Br(),

            ], width=9)
        ])
    ], style={"background-color": "#ffffff"})
])


# callback all inputs
@app.callback(
    Output('pfe_trade', 'children'),
    Input('client_name_dropdown', 'value'),
    Input('exchange_dropdown', 'value'),
    Input('product_dropdown', 'value'),
    Input('maturity_month_dropdown', 'value'),
    Input('input_number_contracts', 'value'),
)
def pfe_of_trade(client,
                 exchange,
                 product,
                 maturity_month,
                 number_contracts):
    if client != 'Client Name' and exchange != 'Exchange' and product != 'Product' and maturity_month != 'Maturity Month' and number_contracts != None:
        pfe_trade = int(client) * int(exchange) * int(product) * int(maturity_month) * number_contracts
    else:
        pfe_trade = '-'

    return pfe_trade


@app.callback(
    Output('current_pfe_customer', 'children'),
    Input('client_name_dropdown', 'value'),
)
def current_pfe_customer(client):
    if isinstance(client, int):
        current_pfe = client * 100
    else:
        current_pfe = '-'

    return current_pfe


@app.callback(
    Output('pfe_customer_with_trade', 'children'),
    Input('client_name_dropdown', 'value'),
    Input('exchange_dropdown', 'value'),
    Input('product_dropdown', 'value'),
    Input('maturity_month_dropdown', 'value'),
    Input('input_number_contracts', 'value'),
)
def pfe_customer_with_trade(client,
                            exchange,
                            product,
                            maturity_month,
                            number_contracts):
    if client != 'Client Name' and exchange != 'Exchange' and product != 'Product' and maturity_month != 'Maturity Month' and number_contracts != None:
        pfe_trade = int(client) * int(exchange) * int(product) * int(maturity_month) * number_contracts
        current_pfe = client
        new_pfe = pfe_trade + current_pfe
    else:
        new_pfe = '-'

    return new_pfe


@app.callback(
    Output('output_potential_future_exposure', 'children'),
    Input('client_name_dropdown', 'value'),
    Input('exchange_dropdown', 'value'),
    Input('product_dropdown', 'value'),
    Input('maturity_month_dropdown', 'value'),
    Input('input_number_contracts', 'value'),
)
def compute_potential_future_exposure(client_name,
                                      exchange,
                                      product,
                                      maturity_month,
                                      number_contracts):
    if client_name != 'Client Name' and exchange != 'Exchange' and product != 'Product' and maturity_month != 'Maturity Month':
        client_name = int(client_name)
        exchange = int(exchange)
        product = int(product)
        maturity_month = int(maturity_month)
    else:
        client_name = 0
        exchange = 0
        product = 0
        maturity_month = 0

    new_pfe = client_name * exchange * product * maturity_month * number_contracts
    old_pfe = 100
    delta = new_pfe - old_pfe
    delta_percentage = delta / old_pfe

    data = {'new_pfe': [new_pfe], 'old_pfe': [old_pfe], 'delta': [delta], 'delta_percentage': [delta_percentage]}
    data_dataframe = pd.DataFrame(data)
    data_to_dict = data_dataframe.to_dict("records")
    columns = [{"name": col, "id": col, "type": 'numeric', "format": Format(precision=2, scheme=Scheme.fixed), } for col
               in data_dataframe.columns]

    table = dash_table.DataTable(data=data_to_dict, columns=columns,
                                 style_cell={'textAlign': 'center'},
                                 style_header={'fontWeight': 'bold', 'color': 'rgb(108, 117, 125)'},
                                 fixed_columns={'headers': True, 'data': 1},
                                 style_table={'minWidth': '100%'})

    return table


@app.callback(
    Output('output_random_walk_graph', 'children'),
    Input('client_name_dropdown', 'value'),
    Input('exchange_dropdown', 'value'),
    Input('product_dropdown', 'value'),
    Input('maturity_month_dropdown', 'value'),
    Input('input_number_contracts', 'value'),
)
def compute_random_walk_graph(client_name,
                              exchange,
                              product,
                              maturity_month,
                              number_contracts):
    if maturity_month != 'Maturity Month':
        num_minutes = maturity_month * 30 * 12
    else:
        num_minutes = 0

    x, y, z = 0, 0, 0
    timepoints = np.arange(num_minutes + 1)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, num_minutes + 1):
        step = random.choice(directions)
        if step == "UP":
            y += 1
        elif step == "DOWN":
            y -= 1
        positions.append(y)

    date_list = [datetime.datetime.today() - datetime.timedelta(hours=x) for x in range(num_minutes + 1)]
    dates = []
    for i in range(1, len(date_list) + 1):
        dates.append(date_list[-i].strftime("%m/%d/%Y"))

    # Implement graph
    fig = go.Figure(data=[go.Scatter(x=dates, y=positions, line=dict(color="#2bb353"))], )
    return dcc.Graph(figure=fig)


@app.callback(
    Output('output_distribution', 'children'),
    Input('client_name_dropdown', 'value'),
    Input('exchange_dropdown', 'value'),
    Input('product_dropdown', 'value'),
    Input('maturity_month_dropdown', 'value'),
    Input('input_number_contracts', 'value'),
)
def compute_distribution(client_name,
                         exchange,
                         product,
                         maturity_month,
                         number_contracts):
    if client_name != 'Client Name' and exchange != 'Exchange' and product != 'Product' and maturity_month != 'Maturity Month' and number_contracts != None:
        client_name = int(client_name)
        exchange = int(exchange)
        product = int(product)
        maturity_month = int(maturity_month)
    else:
        client_name = 0
        exchange = 0
        product = 0
        maturity_month = 0

    n = client_name * product * maturity_month * exchange * number_contracts * 10000
    mean = 0
    std = 1
    data = np.random.normal(mean, std, size=n)
    fig = go.Figure(data=[go.Histogram(y=data, histnorm='probability')])
    fig.update_traces(marker_color="#2bb353")
    fig.update_yaxes(showticklabels=False)
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
