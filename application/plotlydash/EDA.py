import numpy as np
import plotly.express as px
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from .layouts.layout import html_layout
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output

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

def delete_threshold(df, value):
    delete_column_list = []
    for feature in df.columns.tolist():
        if df[feature].isnull().sum() / len([feature]) > int(value):
            delete_column_list.append(feature)
    return delete_column_list

def missing_percentage(df, threshold):
    missing_p = []
    for feature in df.columns.tolist():
        if df[feature].isnull().sum() / len(df[feature]) > int(threshold):
            missing_p.append(feature)
    return missing_p


def create_EDA(server):
    dash_app = dash.Dash(name='EDA', server=server, url_base_pathname='/EDA/', external_stylesheets=[
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'dbc.themes.BOOTSTRAP',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'

                             ]
                    )
    dash_app.index_string = html_layout
    dash_app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })

    left_margin = 200
    right_margin = 100
    dash_app.layout = dbc.Container(children =[
        html.H2('Exploratory Data Analysis'),
        html.H1('Data explore, data visualization'),
        html.Br(),
        html.Div([

            #buttons for page control
            html.Div([
                html.Br(),
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/dashapp/'),
            ], className='two columns'),

            html.Div([
                html.Br(),
                html.A(html.Button('Next page', style={'fontSize': '12px'}), href='/data_cleaning/'),

            ], className='two columns'),

        ], className= 'row'),
        html.Br(),

        #create 2 tabs
        dcc.Tabs(id='tabs-main', value='tab-1', children=[
            dcc.Tab(label='Box plot', value='tab-1'),
            dcc.Tab(label='Missing Visualization', value='tab-MV'),
        ]),
        html.Div(id='tabs-content-main'),


#--------------------------------------------------------------------------------------
        html.Div([
            html.Div([
                html.Br(),
                dcc.Tabs(id="tabs-Delete-Features", value='tab-1', vertical=True, children=[
                    dcc.Tab(label='Delete Features', value='tab-1'),
                    dcc.Tab(label='Delete Features given threshold', value='tab-2'),
                    dcc.Tab(label='Auto-fill', value='tab-3'),
                ]),
                html.Br(),
                html.Button('Submit', n_clicks=0, id='save-button'),
                html.Div(id="save-div")

            ], className='six columns'),

            html.Div([

                html.Div(id='dd-notice'),
                html.Div(id='dd-output-container'),
                html.Div(id='tabs-content1')

            ], className='six columns'),
        ], className='row')



    ], id='dash-container')

#Render tab contents
    @dash_app.callback(dash.dependencies.Output('tabs-content-main', 'children'),
                  [dash.dependencies.Input('tabs-main', 'value')])
    def render_content_main(tab):
        if tab == 'tab-1':
            return html.Br(),\
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

                # dbc.InputGroup(
                #     [
                #         dbc.InputGroupAddon("With textarea", addon_type="prepend"),
                #         dbc.Textarea(),
                #     ],
                #     className="mb-3",
                # ),

                html.Div(id='dropdown_content'),
            ], className='four columns'),

            html.Div([
                # html.H3('Data Summary'),
                html.Div(id='dd-notice'),
                html.Div(id='dd-output-container'),
                dcc.Graph(id='dd-figure',figure={ 'data': [], 'layout': []}),
            ], className='eight columns'),
        ], className='row')

        else:
            return html.Div([

                #left block of the page
                html.Div([
                    html.H3('Please Select the Lower and the Upper bound of the percentage missing'),
                    dcc.RangeSlider(
                        id='my-range-slider',
                        min=0,
                        max=100,
                        step=1,
                        value=[5, 15]
                    ),
                    html.Div(id='output-container-range-slider'),
                    html.Br(),
                    html.Button('Submit', id='submit-button-slider', n_clicks=0)

                ],className='six columns'),

                #right block of the page
                html.Div([
                    html.H3("test"),
                    html.Br(),
                    dcc.Graph(id='slider-output-container',figure={ 'data': [], 'layout': []}),

                ],className='six columns')

            ], className='row')

    @dash_app.callback(
        dash.dependencies.Output('output-container-range-slider', 'children'),
        [dash.dependencies.Input('my-range-slider', 'value')])
    def update_output_slider(value):
        return 'You have selected "{}" as your lower bound and {} as your upper bound'.format(value[0], value[1])

    @dash_app.callback(dash.dependencies.Output('slider-output-container', 'figure'),
                       [dash.dependencies.Input('submit-button-slider', 'n_clicks')],
                       [dash.dependencies.State('my-range-slider', 'value'),]
                       )
    def display_nans( n_clicks, slider_value):
        '''
        @df: data structure to look into
        @top: 0~1, % for data starts looking at
        @buttom: 0~1, % for data ends looking at
        return:
          ->(upper, lower, nan, selected),
              upper with highest nan% in selected region; \
              lower with lowest nan% in selected region;
              nan is all the nan params and nan% in descending order
              selected is the indexes in range interval
              cols_with_nans is all the indexes

        This function graphs the nan density of the whole dataset, better to use a (top-buttom) <= 0.03 to see the features names clearly
        '''
        if n_clicks > 0:
            nans = pd.concat([global_df.isnull().sum(), (global_df.isnull().sum() / global_df.shape[0]) * 100], axis=1,
                             keys=['Num_NaN', 'NaN_Percent'])
            nans = nans[nans.Num_NaN > 0]
            # print(nans.shape)
            nans = nans.sort_values(by=['NaN_Percent'], ascending=False)
            cols_with_nans = nans.index.tolist()
            selected = []
            for col in cols_with_nans:
                if (nans.loc[col, 'NaN_Percent'] <= slider_value[1] * 100) and (
                        nans.loc[col, 'NaN_Percent'] > slider_value[0] * 100):
                    selected.append(col)
            return msno.matrix(df=global_df[selected], figsize=(30, 15), color=(0.24, 0.77, 0.77))
            # return (nans['NaN_Percent'][selected[0]], nans['NaN_Percent'][selected[-1]], nans, selected, cols_with_nans)

            upper, lower, nans, selected, cols_with_nans = display_nans(df, top=lower_bound, buttom=upper_bound)
            # print(nans.loc[selected])
            # # print(nans)
            # print("with highest nan density at {}% and lowest at {}%".format(upper, lower))




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
        features = categorize_feature(global_df)

        if str_value in features[0]:
            fig = px.box(global_df[str_value], y=str_value)
        elif str_value in features[1]:
            fig = px.histogram(global_df[str_value], y=str_value)
        else:
            fig = px.box(global_df[str_value], y=str_value)
        return fig

















#----------------------------------------------------------------------------------------------
    @dash_app.callback(dash.dependencies.Output('tabs-content1', 'children'),
                       [dash.dependencies.Input('tabs-Delete-Features', 'value')])
    def render_content_tabs(tab):
        if tab == 'tab-1':
            return html.Div([
                html.Br(),
            dcc.Dropdown(
                    id='dropdown_category1',
                    options=[
                        {'label': 'Integer', 'value': 'int', },
                        {'label': 'Categorical', 'value': 'str'},
                        {'label': 'Float', 'value': 'float'}
                    ],
                    value='features'
                ),

                html.Div(id='dropdown_content1'),
                html.Br(),
                html.Div([
                    html.A(html.Button('Submit', n_clicks=0, id='delete_col_submit')),
                ]),
                html.Div(id='output-container-button-delete-column',
                         children='Select the feature that you would like to delete')

            ])
        elif tab == 'tab-2':
            return html.Div([
                html.Br(),
                html.Div(dcc.Input(id='input-box', type='text')),
                html.Br(),
                html.Button('Submit', n_clicks=0, id='button_threshold'),
                html.Div(id='output-container-button-threshold',
                         children='Enter a value and press submit')
            ])
        elif tab == 'tab-3':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown_category_auto',
                    options=[
                        {'label': 'Integer', 'value': 'int', },
                        {'label': 'Categorical', 'value': 'str'},
                        {'label': 'Float', 'value': 'float'}
                    ],
                    value='features'
                ),
                html.Div(id='dropdown_content_auto'),
                html.Br(),
                html.Div([
                    html.A(html.Button('Submit', n_clicks=0, id='auto_submit_button')),
                ]),
                html.Div(id='output-container-button-autofill',
                         children='Select the feature that you would like to autofill')

            ])





    @dash_app.callback(
        dash.dependencies.Output('output-container-button-threshold', 'children'),
        [dash.dependencies.Input('button_threshold', 'n_clicks')],
        [dash.dependencies.State('input-box', 'value')])
    def delete_features_threshold(n_clicks, value):
        if n_clicks > 0:
            x = delete_threshold(global_df, value)
            print(x)
            df2 = global_df.drop(delete_threshold(global_df, value), axis=1)
            print(df2.head())
            return u'''The features with a missing percentage higher than "{}%" has been deleted, they are {}
                '''.format(
                value, x
            )

    @dash_app.callback(dash.dependencies.Output('dropdown_content1', 'children'),
                       [dash.dependencies.Input('dropdown_category1', 'value')],
                       )
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

    @dash_app.callback(dash.dependencies.Output('dropdown_content_auto', 'children'),
                       [dash.dependencies.Input('dropdown_category_auto', 'value')],
                       )
    def render_content_dropdown_auto(value):
        features = categorize_feature(global_df)
        if value == 'int':
            return html.Div([
                html.Br(),
                dcc.Dropdown(
                    id='dropdown_auto',
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
                    id='dropdown_auto',
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
                    id='dropdown_auto',
                    options=[
                        {'label': i, 'value': i} for i in features[2]

                    ],
                    value='features'
                ),
                html.Br(),
            ])

    @dash_app.callback(
        dash.dependencies.Output('output-container-button-delete-column', 'children'),
        [dash.dependencies.Input('delete_col_submit', 'n_clicks')],
        [dash.dependencies.State('dropdown', 'value')])
    def delete_feature(n_clicks, value):
        if n_clicks > 0:
            print(value)
            df3 = global_df.drop(value, axis=1)
            print(df3.head())
            return u'''The feature "{}" has been deleted
                '''.format(
                value,
            )

    @dash_app.callback(
        dash.dependencies.Output('output-container-button-autofill', 'children'),
        [dash.dependencies.Input('auto_submit_button', 'n_clicks')],
        [dash.dependencies.State('dropdown_auto', 'value')]
        # [dash.dependencies.State('autofill_input', 'value')]
    )
    def autofill(n_clicks, value):
        if n_clicks > 0:
            print(value)

            if global_df[value].dtypes == np.int64:
                global_df[value].fillna(value=99999)
                print(global_df[value].isna().sum())
                return u'''The feature "{}" has been auto_filled with 99999
                                            '''.format(
                    value, )
            elif global_df[value].dtypes == np.object:
                global_df[value].fillna("missing")
                print(global_df[value].isna().sum())
                return u'''The feature "{}" has been auto_filled with missing
                                            '''.format(
                    value, )
            elif global_df[value].dtypes == np.float64:
                global_df[value].fillna(value=99999)
                print(global_df[value].isna().sum())
                return u'''The feature "{}" has been auto_filled with 99999
                                            '''.format(
                    value, )

    @dash_app.callback(dash.dependencies.Output('save_div', 'children'),
                       [dash.dependencies.Input('save-button', 'n_clicks')])
    def save(n_clicks):
        if n_clicks > 0:
            print("hello")
            global_df.to_csv(r'\Users\wenchenliu\Desktop\dpt\Temp_save\311-calls.csv')
        return "hi"

    if __name__ == '__main__':
        dash_app.run_server(debug=True)



#input-group, choose what type of input to type in (boot-strap)
#categorical data change to show histogram
#change the delete feature threshold to slide bars
#the default for the dropdown should not be a empty graph (keep it empty)
#move the autofill function to comply with the "feature selection"
