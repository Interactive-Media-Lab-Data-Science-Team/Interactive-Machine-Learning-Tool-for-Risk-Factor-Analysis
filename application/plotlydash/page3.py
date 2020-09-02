import numpy as np
import plotly.express as px
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
from .layouts.layout import html_layout

global_df = pd.read_csv("data/311-calls.csv")

def delete_threshold(df, value):
    delete_column_list = []
    for feature in df.columns.tolist():
        if df[feature].isnull().sum() / len([feature]) > value:
            delete_column_list.append(feature)
    return delete_column_list


def categorize_feature(df):
    integers = []
    categorical = []
    floaters = []
    for i in df.columns:
        if df[i].dtypes == np.int64:
            integers.append(i)
        elif df[i].dtypes == np.object:
            categorical.append(i)
        elif df[i].dtypes == np.float64:
            floaters.append(i)
    result = [integers, categorical, floaters]
    return result

def missing_percentage(df, threshold):
    missing_p = []
    for feature in df.columns.tolist():
        if df[feature].isnull().sum() / len(df[feature]) > threshold:
            missing_p.append(feature)
    return missing_p

def create_page3(server):
    dash_app = dash.Dash(name='cleaning', server=server, url_base_pathname='/data_cleaning/', external_stylesheets=[
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'
                             ]
                    )
    dash_app.index_string = html_layout

    dash_app.layout = html.Div(children =[
        html.H1('This is where our title for this page will be'),
        html.H2('This is the subtitle'),
        html.Div([
            html.Div([
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/EDA/'),
            ], className='two columns'),

            html.Div([html.A(html.Button('Save and Proceed', style={'fontSize': '12px'}), href='/non/', id='save_button'),

            ], className='two columns'),


        ], className= 'row'),

        html.Div([
            html.Div([
                html.Br(),
                dcc.Tabs(id="tabs", value='tab-1', vertical=True, children=[
                    dcc.Tab(label='Delete Features', value='tab-1'),
                    dcc.Tab(label='Delete Features given threshold', value='tab-2'),
                    dcc.Tab(label='Delete Row', value='tab-3'),
                ]),


            ], className='eight columns'),


            html.Div([

                html.Div(id='dd-notice'),
                html.Div(id='dd-output-container'),
                html.Div(id='tabs-content')

            ], className='four columns'),
        ], className='row')

    ], id='dash-container')

    @dash_app.callback(dash.dependencies.Output('tabs-content', 'children'),
                  [dash.dependencies.Input('tabs', 'value')])
    def render_content_tabs(tab):
        if tab == 'tab-1':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown_category',
                    options=[
                        {'label': 'Integer', 'value': 'int', },
                        {'label': 'Categorical', 'value': 'str'},
                        {'label': 'Float', 'value': 'float'}
                    ],
                    value='features'
                ),
                html.Div(id='dropdown_content'),
                html.Br(),
                html.Div([
                    html.A(html.Button('Submit', id = 'delete_col_submit')),
                ]),

            ])
        elif tab == 'tab-2':
            return html.Div([
                html.Br(),
                html.Div(dcc.Input(id='input-box', type='text')),
                html.Br(),
                html.Button('Submit', id='button_threshold'),
                html.Div(id='output-container-button-threshold',
                         children='Enter a value and press submit')
            ])
        elif tab == 'tab-3':
            return html.Div([
                html.Br(),
                html.Div(dcc.Input(id='input-box', type='text')),
                html.Br(),
                html.Button('Submit', id='button', n_clicks=0),
                html.Div(id='output-container-button-row',
                         children='Enter a value and press submit')
            ])

    @dash_app.callback(
        dash.dependencies.Output('output-container-button-threshold', 'children'),
        [dash.dependencies.Input('button_threshold', 'n_clicks')],
        [dash.dependencies.State('input-box', 'value')])

    def delete_features_threshold(n_clicks, value):

        # global_df.drop(delete_threshold(global_df, value))

        return u'''The Button has been pressed {} times, The features with a missing percentage higher than "{}%" has been deleted
        '''.format(n_clicks,
            value
        )

    @dash_app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_category', 'value')])

    def render_content_dropdown(value):
        features = categorize_feature(global_df)
        if value == 'int':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': i, 'value': i} for i in features[0]

                    ],
                    value='features'
                ),
                html.Br(),

            ])
        elif value == 'str':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': i, 'value': i} for i in features[1]

                    ],
                    value='features'
                ),
                html.Br(),
            ])
        elif value == 'float':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': i, 'value': i} for i in features[2]

                    ],
                    value='features'
                ),
                html.Br(),
            ])
    # @dash_app.callback(
    #     dash.dependencies.Output('output-container-button-row', 'children'),
    #     [dash.dependencies.Input('button', 'n_clicks')],
    #     [dash.dependencies.State('input-box', 'value')])
    #
    # def update_output_row(n_clicks, value, delete_column_list=None,df = global_df ):
    #     # for feature in df.columns.tolist():
    #     #     if df[feature].isnull().sum() / len([feature]) > value:
    #     #         delete_column_list.append(feature)
    #     # df = df.drop(delete_column_list, axis=1)
    #     return 'The rows with a missing percentage higher than "{}%" has been deleted'.format(
    #         value
    #     )


    if __name__ == '__main__':
        dash_app.run_server(debug=True)





