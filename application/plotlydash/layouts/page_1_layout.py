import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import base64
import pandas as pd
import io
import dash_table
import os
from .layouts.layout import html_layout
from dash.exceptions import PreventUpdate
import requests


def make_page1(app):
    page_1_layout = html.Div([
            html.Div([], id='hidden-div', style={'display': 'none'}),
            dcc.Dropdown(
                id='demo-dropdown',
                options=[
                    {'label': 'BRFSS', 'value': 'BRFSS'}
                ],
                searchable=False,
                placeholder="Select A Medical Dataset",
                style = {
                    'width':'50%',
                    'margin':'2% 0%'
                }
            ),
            html.Div(id='dd-output-container'),
            # html.Div(id='output')
            dcc.Loading(
                children=[
                    html.Div(id='output')
                    ], type='cube')
            ],style={
                'width': '70%',
                'margin': '5% 15%',
                # 'text-align': 'center',
            })

    @app.callback(
        dash.dependencies.Output('dd-output-container', 'children'),
        [dash.dependencies.Input('demo-dropdown', 'value')])
    def update_output(value):
        if(not value):
            raise PreventUpdate
        return 'You have selected "{}"'.format(value)

    @app.callback(Output('output', 'children'),
                [Input('demo-dropdown', 'value')])
    def update_output(value):
        if value is not None:
            # file_id = allData[value]
            # destination = 'download.csv'
            # download_file_from_google_drive(file_id, destination)
            # return parse_contents('download.csv')
            if value == 'BRFSS':
                
        
    def parse_contents(filename):
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv("%s" % (filename))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
        size = convert_unit(os.stat(str(os.getcwd())+'/'+str(filename)).st_size, SIZE_UNIT.MB)

        return html.Div([
            # html.H5("Upload File: {}".format(filename)),
            html.H5("File size: {:.3f}MB".format(size)),
            dcc.Loading(children=[
                dash_table.DataTable(
                id='database-table',
                columns=[{'name': i, 'id': i} for i in df.columns],
                data=df[:1000].to_dict('records'),
                sort_action="native",
                sort_mode='native',
                page_size=300,
                fixed_rows = 100,
                style_table={
                    'maxHeight': '80ex',
                    'overflowY': 'scroll',
                    'overflowX': 'scroll',
                    'width': '100%',
                    # 'minWidth': '100%',
                    # 'margin': '5% 15%'
                },
            ),

            html.Hr(),  # horizontal line
            html.A(html.Button('Next', id='btn'), href='/EDA/')
            ], type='cube')
        ])
    return page_1_layout