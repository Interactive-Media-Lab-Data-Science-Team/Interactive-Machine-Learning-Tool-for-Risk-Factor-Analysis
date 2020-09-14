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

    left_margin = 200
    right_margin = 100
    dash_app.layout = html.Div(children =[

#Div that contains the title and description section
        html.Div(children=[
            html.H2('Exploratory Data Analysis'),
            html.H1('Data explore, data visualization'),
        ],style={'textAlign': 'center'}),

#Div that contains the buttons
        html.Div([
            html.Br(),
            #buttons for page control
            html.Div([
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/dashapp/'),
            ], className='two columns'),
            html.Div([html.A(html.Button('Next page', style={'fontSize': '12px'}), href='/data_cleaning/'),
            ], className='two columns'),
        ], className= 'row'),
        html.Br(),

#Div for the tabs
        html.Div(children=[
            # create 2 tabs
            dcc.Tabs(id='tabs-main', value='main', children=[
                dcc.Tab(label='Box plot', value='tab-preparation'),
                dcc.Tab(label='Missing Visualization', value='tab-missing'),
            ]),

            #display for Tabs' content
            html.Div(children=[], id='tabs-content'),
        ]),

    ], id='dash-container')

#Render tab contents
    @dash_app.callback(dash.dependencies.Output('tabs-content', 'children'),
                  [dash.dependencies.Input('tabs-main', 'value')])
    def render_tab_main(tab):
        if tab == 'tab-preparation':
            return html.Br(),\
                   html.Div([
            html.Div([
                html.H3('Feature Preparation'),
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

                #graph
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
                }},id='dd-figure'),

            ], className='eight columns'),
        ], className='row')

        elif tab == 'tab-missing':
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
                    # html.Button('Submit', id='submit-button-slider', n_clicks=0)
                    html.Div(children=[
                    dbc.Button("Submit", color="primary", className="six columns", id='submit-button-slider'),
                    dbc.Button("Delete", color="secondary", className="six columns",id="missing_delete_threshold" ),]),
                    html.Div(id='outputbox_delete_threshold')

                ],className='six columns'),

                # right block of the page
                html.Div([
                    # dcc.Graph(
                    #     figure={"layout": {
                    #         "xaxis": {"visible": False},
                    #         "yaxis": {"visible": False},
                    #         "annotations": [{
                    #             "text": "Please Select the Range",
                    #             "xref": "paper",
                    #             "yref": "paper",
                    #             "showarrow": False,
                    #             "font": {"size": 28}
                    #         }]
                    #     }}, id='missing-figure'),

                ],className='six columns')

            ], className='row')



    @dash_app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_category', 'value')])
    def render_tab_preparation_dropdown(value):
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
                html.P("Type a number to substitute all the missing values"),
                dbc.Input(type="number", min=0, max=10, step=1, id='input_int'),
                html.Br(),
                html.Div(children=[], id='inputbox_int'),
                html.Br(),
                html.Div(children=[], id='inputbox_int_fill'),
                html.Br(),
                html.Div(children=[], id='delete_single_output')

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
                html.P("Type a text to Replace the missing values"),
                dbc.Input(type="text", min=0, max=10, step=1, pattern= '[a-zA-Z]*',id='input_str'),
                html.Br(),
                html.Div(children=[], id='inputbox_str'),
                html.Br(),
                html.Div(children=[], id='inputbox_str_fill'),
                html.Br(),
                html.Div(children=[], id='delete_single_output')
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
                html.P("Type a number to substitute all the missing values"),
                dbc.Input(type="number", min=0, max=10, step=1,id='input_float'),
                html.Br(),
                html.Div(children=[], id='inputbox_float'),
                html.Br(),
                html.Div(children=[], id='inputbox_float_fill'),
                html.Br(),
                html.Div(children=[], id='delete_single_output')
            ])

    @dash_app.callback(
        dash.dependencies.Output('dd-notice', 'children'),
        [dash.dependencies.Input('dropdown', 'value')])
    def update_selected_feature_div(value):
        return 'You have selected the feature: {}'.format(value)

    @dash_app.callback(
        dash.dependencies.Output('dd-output-container', 'children'),
        [dash.dependencies.Input('dropdown', 'value')])

    def preparation_tab_information_report(value):
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

    def preparation_tab_visualize_features(value):
        str_value = str(value)
        features = categorize_feature(global_df)

        if str_value in features[0]:
            fig = px.box(global_df[str_value], y=str_value)
        elif str_value in features[1]:
            fig = px.histogram(global_df[str_value], y=str_value)
        else:
            fig = px.box(global_df[str_value], y=str_value)
        return fig

#______________________________________________________________

    @dash_app.callback(
        dash.dependencies.Output('output-container-range-slider', 'children'),
        [dash.dependencies.Input('my-range-slider', 'value')])
    def update_output(value):
        return 'You have selected "{}" as your lower bound and {} as your upper bound'.format(value[0], value[1])


    #callback for the Missing Visualization
    @dash_app.callback(dash.dependencies.Output('missing-figure', 'figure'),
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

            # upper, lower, nans, selected, cols_with_nans = display_nans(df, top=lower_bound, buttom=upper_bound)
            # print(nans.loc[selected])
            # # print(nans)
            # print("with highest nan density at {}% and lowest at {}%".format(upper, lower))

    @dash_app.callback(dash.dependencies.Output('outputbox_delete_threshold','children'),
                       [dash.dependencies.Input('missing_delete_threshold', 'n_clicks')],
                       [dash.dependencies.State('my-range-slider', 'value')])
    def delete_threshold(n_clicks, value):
        delete_column_list = []
        if n_clicks is None:
            return "Click the Delete button if you would like to Delete all the features with in the missing range"
        else:
            for feature in global_df.columns.tolist():
                if int(value[0]) < (global_df[feature].isnull().sum() / len([feature])) > int(value[1]):
                    delete_column_list.append(feature)
            return "{} has been deleted " .format(delete_column_list)


#----------------------------------------------------

    #call backs that create buttons for each fill NAN sub categories
    @dash_app.callback(
        dash.dependencies.Output('inputbox_float', 'children'),
        [dash.dependencies.Input('input_float', 'value')])
    def render_button_float(value):
        return html.Div([
            dbc.Button("Submit", color="primary", className="six columns", id='input_submit_button_float'),
            dbc.Button("Delete", color="secondary", className="six columns",id="input_button_delete" ),

        ])

    @dash_app.callback(
        dash.dependencies.Output('inputbox_int', 'children'),
        [dash.dependencies.Input('input_int', 'value')])
    def render_button_int(value):
        return html.Div([
            dbc.Button("Submit", color="primary", className="six columns", id='input_submit_button_int'),
            dbc.Button("Delete", color="secondary", className="six columns",id="input_button_delete" ),
        ])

    @dash_app.callback(
        dash.dependencies.Output('inputbox_str', 'children'),
        [dash.dependencies.Input('input_str', 'value')])
    def render_button_str(value):
        return html.Div([
            dbc.Button("Submit", color="primary", className="six columns", id='input_submit_button_str'),
            dbc.Button("Delete", color="secondary", className="six columns",id="input_button_delete" ),
        ])


#Fill the column given numbers (need to change the data-frame to global for the real project, this is a demo)
    @dash_app.callback(
        dash.dependencies.Output('inputbox_float_fill', 'children'),
        [dash.dependencies.Input('input_submit_button_float', 'n_clicks')],
        [dash.dependencies.State('dropdown', 'value'),
         dash.dependencies.State('input_float', 'value')],)
    def fill_float(n_clicks, feature, value):
        print(feature)
        if n_clicks is None:
            return "Action not made"
        else:
            df1 = global_df
            df1.fillna(value= {feature:value},inplace=True)
            print(df1[feature].isna().sum())
            print(df1[feature])
            return u'''The feature "{}" has been auto_filled with {}
                                                    '''.format(
                feature, value )
    @dash_app.callback(
        dash.dependencies.Output('inputbox_str_fill', 'children'),
        [dash.dependencies.Input('input_submit_button_str', 'n_clicks')],
        [dash.dependencies.State('dropdown', 'value'),
         dash.dependencies.State('input_str', 'value')],)
    def fill_str(n_clicks, feature, value):
        print(feature)
        if n_clicks is None:
            return "Action not made"
        else:
            df1 = global_df
            df1.fillna(value= {feature:value},inplace=True)
            print(df1[feature].isna().sum())
            print(df1[feature])
            return u'''The feature "{}" has been auto_filled with {}
                                                    '''.format(
                feature, value )

    @dash_app.callback(
        dash.dependencies.Output('inputbox_int_fill', 'children'),
        [dash.dependencies.Input('input_submit_button_int', 'n_clicks')],
        [dash.dependencies.State('dropdown', 'value'),
         dash.dependencies.State('input_int', 'value')],)
    def fill_int(n_clicks, feature, value):
        print(feature)
        if n_clicks is None:
            return "Action not made"
        else:
            df1 = global_df
            df1.fillna(value= {feature:value},inplace=True)
            print(df1[feature].isna().sum())
            print(df1[feature])
            return u'''The feature "{}" has been auto_filled with {}
                                                    '''.format(
                feature, value )

#delete the selected column(feature)
    @dash_app.callback(
        dash.dependencies.Output('delete_single_output', 'children'),
        [dash.dependencies.Input('input_button_delete', 'n_clicks')],
        [dash.dependencies.State('dropdown', 'value')],)
    def delete_single(n_clicks, value):
        if n_clicks is None:
            return "Click the Delete Feature button if you would like to Delete this Feature"
        else:
            df3 = global_df.drop(value, axis=1)
            print(df3.head())
            return u'''The feature "{}" has been deleted
                        '''.format(
                value, )


    if __name__ == '__main__':
        dash_app.run_server(debug=True)




#input-group, choose what type of input to type in (boot-strap)
#categorical data change to show histogram
#change the delete feature threshold to slide bars
#the default for the dropdown should not be a empty graph (keep it empty)
#move the autofill function to comply with the "feature selection"


#incorp sub feature auto-fill under the feature preparation DIV
#incorp the sub features delete-threshold and delete column in the missing tab
