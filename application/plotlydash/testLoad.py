import dash_resumable_upload
import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import base64
import pandas as pd
import io
import dash_table
import os
from .layouts.layout import html_layout
from dash.exceptions import PreventUpdate

df = pd.DataFrame([])

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


def test1(server):
    app = dash.Dash(server=server,
                         routes_pathname_prefix='/dashapp/',
                         external_stylesheets=[
                             'https://codepen.io/dapraxis/pen/gOPGzPj.css',
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato'
                             ])

    dash_resumable_upload.decorate_server(app.server, "uploads")
    
    app.index_string = html_layout

    app.scripts.config.serve_locally = True  # Uploaded to npm, this can work online now too.

    app.layout = html.Div([
        dash_resumable_upload.Upload(
            id='upload-data',
            maxFiles=1,
            maxFileSize=1024*1024*1000,  # 100 MB
            service="/upload_resumable",
            textLabel="Drag and Drop to upload",
            # startButton=True
        ),
        html.Div(id='output')
        ],style={
            'width': '70%',
            'margin': '5% 15%',
            'text-align': 'center',
        })


    @app.callback(Output('output', 'children'),
                [Input('upload-data', 'fileNames')])
    def update_output(list_of_names):
        if list_of_names is not None:
            children = [parse_contents(filename) for filename in list_of_names]
            return children
        
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
            html.H5("Upload File: {}".format(filename)),
            html.H5("File size: {:.3f}MB".format(size)),
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
        ])
