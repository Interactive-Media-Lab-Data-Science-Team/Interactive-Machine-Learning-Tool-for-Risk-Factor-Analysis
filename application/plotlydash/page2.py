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

def create_page2(server):
    dash_app = dash.Dash(name='EDA', server=server, url_base_pathname='/EDA/', external_stylesheets=[
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
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/dashapp/'),
            ], className='two columns'),

            html.Div([html.A(html.Button('Next page', style={'fontSize': '12px'}), href='/data_cleaning/'),

            ], className='two columns'),

        ], className= 'row'),

        html.Div([
            html.Div([
                html.H3('Feature Selections'),
                dcc.Dropdown(
                    id='dropdown_category',
                    options=[
                        {'label': 'Integer', 'value': 'int', },
                        {'label': 'Categorical', 'value': 'str'},
                        {'label': 'Float', 'value':'float'}
                    ],
                    value='features'
                ),

                html.Div(id='dropdown_content'),

            ], className='four columns'),


            html.Div([
                # html.H3('Data Summary'),
                html.Div(id='dd-notice'),
                html.Div(id='dd-output-container'),
                dcc.Graph(id='dd-figure'),


            ], className='eight columns'),
        ], className='row')

    ], id='dash-container')

    @dash_app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_category', 'value')])
    def render_content(value):
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

    @dash_app.callback(
        dash.dependencies.Output('dd-notice', 'children'),
        [dash.dependencies.Input('dropdown', 'value')])
    def update_output_div(value):
        return 'You have selected the feature: {}'.format(value)

    @dash_app.callback(
        dash.dependencies.Output('dd-output-container', 'children'),
        [dash.dependencies.Input('dropdown', 'value')])

    def report_information(value):
        str_value = str(value)
        R_dict = global_df[str_value].describe().to_dict()
        result = list()
        for key in R_dict:
             result.append('{}: {}'.format(key, R_dict[key]))
        return html.Ul([html.Li(x) for x in result])

        # Define a function for drawing box plot for selected feature

    @dash_app.callback(
        dash.dependencies.Output('dd-figure', 'figure'),
        [dash.dependencies.Input('dropdown', 'value')])

    def visualize_features(value):
        str_value = str(value)
        fig = px.box(global_df[str_value], y=str_value)
        return fig


        # basic_info_dict = dict(global_df[str_value].describe())
        # # Missing values
        # basic_info_dict['missing_ct'] = global_df[str_value].isnull().sum()
        # basic_info_dict['missing_pct'] = global_df[str_value].isnull().sum() / len(global_df[str_value])
        # return basic_info_dict


    if __name__ == '__main__':
        dash_app.run_server(debug=True)





