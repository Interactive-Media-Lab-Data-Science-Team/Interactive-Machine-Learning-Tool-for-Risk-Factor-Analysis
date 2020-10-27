import numpy as np
import ast
import plotly.express as px
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from .layouts.layout import html_layout
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dash.exceptions import PreventUpdate


FILE_PATH = 'download.csv'
# FILE_PATH = '/Users/wenchenliu/Desktop/dpt/cleaned_BRFSS.csv'
BACK_UP_PATH = 'data/original/311-calls-original.csv'

file = open('data/var_info.txt', 'r')
contents = file.read()
dictionary = ast.literal_eval(contents)
file. close()

file1 = open('data/section_name.txt', 'r')
contents1 = file1.read()
dictionary_name = ast.literal_eval(contents1)
file1.close()

file2 = open('data/categorized_type.txt', 'r')
contents2 = file2.read()
categories = ast.literal_eval(contents2)
file2.close()

global_df = pd.read_csv(FILE_PATH)



def server_layout():
    return html.Div(children =[

#Div that contains the title and description section
        html.Div(children=[
            html.H2('Exploratory Data Analysis'),
            html.H1('Data explore, data visualization'),
            dcc.Interval('interval-component', n_intervals=0, interval=1*1000)
        ],style={'textAlign': 'center'}),

#Div that contains the buttons
        html.Div([
            html.Br(),

#hidden div
            html.Div([

            ], id='hidden-div', style={'display': 'none'}),
#modal
            html.Div(
                [
                    dbc.Button("Page Description", id="open", style={'display': 'none'}),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Header"),
                            dbc.ModalBody("EDA page allow you to select a specific feature within a specific category to visualize and provide summarize for their characteristics. "
                                          "1. you must first select a section name in order to select its respective features. "
                                          "2. you must then select the feature you would like visualize and the results will appear on the left of the page."),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close", className="ml-auto")
                            ),
                        ],
                        id="modal", is_open = True, backdrop="static"
                    ),
                ]
            ),

            #buttons for page control
        ], className= 'row'),
        html.Br(),

#Div for the main dropdown and graph
        html.Div(children=[
           html.Div([
               html.Div([
                   html.H3('Data Visualization'),
                   dcc.Dropdown(
                       id='dropdown_section_name',
                       options=[
                           {'label': i, 'value': i} for i in dictionary_name

                       ],
                       placeholder="Select Section",
                       value='features'
                   ),
                   html.Div(id='dropdown_content'),
                    html.Div([
                        html.Br(),
                        html.A(html.Button('Back', style={'fontSize': '12px'}), href='/dashapp/'),
                        ], className='four columns'),
                    html.Div([
                        html.Br(),
                        html.A(html.Button('Next', style={'fontSize': '12px'}), href='/RFA/'),
                    ], className='four columns'),
               ], className='four columns'),

               html.Div([
                   # html.H3('Data Summary'),
                   html.Div([
                       html.Div([
                           html.Div(id='dd-notice'),
                       ], className='six columns'),

                       html.Div([
                           html.Div(id='dd-output-container'),
                       ], className='six columns'),
                   ], className='row'),

                   html.Br(),

                   # graph
                   html.Div(id = 'graph_plot', children=[
                    
                   ])
               ], className='eight columns'),
           ], className='row')
        ]),

    ], id='dash-container')


def create_EDA(server):
    dash_app = dash.Dash(name='EDA', server=server, url_base_pathname='/EDA/', external_stylesheets=[
                             dbc.themes.BOOTSTRAP,
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'
                             ]
                    )
    dash_app.index_string = html_layout

    # df = pd.read_csv(FILE_PATH)
    # df.to_csv(BACK_UP_PATH)

    left_margin = 200
    right_margin = 100
    
    dash_app.layout = server_layout


    @dash_app.callback(dash.dependencies.Output("hidden-div", "children"),
              [dash.dependencies.Input("interval-component", "n_intervals")])
    def update_df(n):
        global global_df 
        global_df= pd.read_csv(FILE_PATH)
        return dash.no_update


    @dash_app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_section_name', 'value')])

    def render_tab_preparation_multiple_dropdown(value):
        for key in dictionary_name:
            if key == value:
                return_div = html.Div([
                    html.Br(),
                    dcc.Dropdown(
                        id='dropdown',
                        options=[
                            {'label': i, 'value': i} for i in dictionary_name[key]
                        ],
                        placeholder="Select Feature",
                        value='features'
                    ),
                    html.Div(id='single_commands'),
                ])
                return return_div




    @dash_app.callback(
        dash.dependencies.Output('dd-notice', 'children'),
        [dash.dependencies.Input('dropdown', 'value'),])
    def update_selected_feature_div(value):
        result = []
        for key, values in dictionary[value].items():
            result.append('{}:{}'.format(key, values))
        div = html.Div([
            html.Div([
                html.H3('Feature Informatation')
            ]),
            html.Div([
                html.Ul([html.Li(x) for x in result])
            ]),
        ])

        return div

    @dash_app.callback(
        [dash.dependencies.Output('dd-output-container', 'children'),
         dash.dependencies.Output('graph_plot', 'children')],
        # [dash.dependencies.Input('dropdown', 'value'), dash.dependencies.Input("hidden-div", 'children')])
        [dash.dependencies.Input('dropdown', 'value')])

    def preparation_tab_information_report(value):
        str_value = str(value)
        global global_df
        R_dict = global_df[str_value].describe().to_dict()
        result = []
        for key in R_dict:
             result.append('{}: {}'.format(key, R_dict[key]))

        div = html.Div([
            html.Div([
                html.H3('Feature Statistics')
            ]),
            html.Div([
                html.Ul([html.Li(x) for x in result])
            ]),
        ])
        
        g = dcc.Loading(id='graph_loading', children=[
                            dcc.Graph(
                            figure={"layout": {
                                "xaxis": {"visible": False},
                                "yaxis": {"visible": False},
                                "annotations": [{
                                    "text": "Please Select the Feature you would like to Visualize",
                                    "xref": "paper",
                                    "yref": "paper",
                                    "showarrow": False,
                                    "font": {"size": 28}
                                }]
                            }}, id='dd-figure'),
                        ])
        return [div, g]

        # Define a function for drawing box plot for selected feature

    @dash_app.callback(
        dash.dependencies.Output('dd-figure', 'figure'),
        # [dash.dependencies.Input('dropdown', 'value'),dash.dependencies.Input("hidden-div", 'children')])
        [dash.dependencies.Input('dropdown', 'value')])

    def preparation_tab_visualize_features(value):
        global global_df
        integers = categories[0]
        floats = categories[1]
        str_value = str(value)
        if str_value in integers:
            fig = px.histogram(global_df[str_value], y=str_value)
        elif str_value in floats:
            fig = px.box(global_df[str_value], y=str_value)
        else:
            fig = px.histogram(global_df[str_value], y=str_value)
        return fig

    @dash_app.callback(
        # [dash.dependencies.Output("modal", "is_open"), dash.dependencies.Output('hidden-div', 'children')],
        dash.dependencies.Output("modal", "is_open"),
        [dash.dependencies.Input("open", "n_clicks"),dash.dependencies.Input("close", "n_clicks")],
        [dash.dependencies.State("modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open



    if __name__ == '__main__':
        dash_app.run_server(debug=True)



