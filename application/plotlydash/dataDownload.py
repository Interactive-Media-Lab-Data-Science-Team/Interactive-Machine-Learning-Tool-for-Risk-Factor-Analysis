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

## size convert
import enum
# Enum for size units
class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4
def convert_unit(size_in_bytes, unit):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if unit == SIZE_UNIT.KB:
       return size_in_bytes/1024
   elif unit == SIZE_UNIT.MB:
       return size_in_bytes/(1024*1024)
   elif unit == SIZE_UNIT.GB:
       return size_in_bytes/(1024*1024*1024)
   else:
       return size_in_bytes

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def dataDownload(server):
    app = dash.Dash(server=server,
                         routes_pathname_prefix='/dashapp/',
                         external_stylesheets=[
                             'https://codepen.io/dapraxis/pen/gOPGzPj.css',
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato'
                             ])

    allData = {'BRFSS':'1tNWPT9xW1jc3Qta_h4CGHp9lRbHM1540'}
    
    app.index_string = html_layout

    app.scripts.config.serve_locally = True  # Uploaded to npm, this can work online now too.

    app.layout = html.Div([
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
            file_id = allData[value]
            df = pd.DataFrame([])
            df.to_csv(r'uploads\download.csv')
            destination = 'uploads\download.csv'
            download_file_from_google_drive(file_id, destination)
            return parse_contents('download.csv')
        
    def parse_contents(filename):
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv("uploads/%s" % (filename))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
        size = convert_unit(os.stat(str(os.getcwd())+'/uploads/'+str(filename)).st_size, SIZE_UNIT.MB)

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
            html.A(html.Button('Next', id='btn'), href='/EDA')
            ], type='cube')
        ])



