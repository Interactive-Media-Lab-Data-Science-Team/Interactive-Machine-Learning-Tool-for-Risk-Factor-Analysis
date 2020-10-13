import numpy as np
import plotly.express as px
import pandas as pd
import dash
import ast
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from .layouts.layout import html_layout
from .modelHelperFunctions import regression_models, classification_models, risk_factor_analysis


FILE_PATH = 'download.csv'
global_df = pd.read_csv(FILE_PATH)

REGRESSON_LIST = ["Linear", "Lasso", "Ridge","LassoLars", "Bayesian Ridge", "Elastic Net"]
REG_CRITERION = ['Label', 'Model', 'MAE', 'MSE', 'R2']
CLASSIFICATION_LIST = ["Logistic", "LDA"]
CLF_CRITERION = ["Label", "Model", "Accuracy", "ROC_AUC score","Precision", "Recall", "F1-Score"]

file = open('data/var_info.txt', 'r')
contents = file.read()
dictionary = ast.literal_eval(contents)
file. close()

file1 = open('data/section_name.txt', 'r')
contents1 = file1.read()
dictionary_name = ast.literal_eval(contents1)
file1.close()


def serve_layout():
    return html.Div(
        children=[
            html.H4("Model Fitting and Risk Factor Analysis"),
            html.Div([
                html.Div([
                    html.A(html.Button('Previous page', style={
                        'fontSize': '12px'}), href='/PCA/'),
                ]),

            ]),
            html.Div([
            ], id='hidden-div', style={'display': 'none'}),
            dcc.Interval('interval-component', n_intervals=0, interval=1*1000),

            html.Div([
                html.P(html.Label(
                    "Please select a feature for analysis: ")),
                html.Div([
                    # dcc.Dropdown(
                    #     id='feature_dropdown',
                    #     options=[
                    #         {'label': col, 'value': col} for col in global_df.columns
                    #     ],
                    #     placeholder="Please choose a feature as label",
                    #     style=dict(
                    #         width='80%',
                    #         verticalAlign="middle"
                    #     )
                    # )
                    dcc.Dropdown(
                        id='dropdown_section_name',
                        options=[
                            {'label': i, 'value': i} for i in dictionary_name

                        ],
                        placeholder="Select Section",
                    ),
                    html.Div(id='dropdown_content'),
                    ], style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                ),
                
                html.Div([
                        html.Div(id='dd-notice'),
                ]),

                html.Div([
                    html.Div(id='dd-output-container'),
                ]),

                html.Div(
                    html.P(html.Label(
                        "Please verify the type of selected feature: ")),
                ),
                html.Div([
                    dcc.Dropdown(
                        id='type_dropdown',
                        options=[
                            {'label': "Categorical Variable",
                             'value': "Classification"},
                            {'label': "Continuous Variable", 'value': "Regression"}

                        ],
                        placeholder="Please choose the type of label feature",
                        style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                    )
                ], style=dict(display='flex')),

                html.Div(
                    html.P(html.Label(
                        "Please select a model: ")),
                ),
                html.Div([
                    dcc.Dropdown(
                        id='model_dropdown',
                        options=[],
                        placeholder="Please choose a model for verified label type",
                        style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                    )
                ], style=dict(display='flex')),

                html.Div([
                    html.Label(
                        "Proceed risk factor analysis and see the results"),
                    html.Button('Run', id='run_button', n_clicks=0),
                    html.Div(id='RFA_output', children=[]),

                ]),

                html.Div([
                    html.Label(
                        "Performance Records for Regression Model"),
                    html.Div(
                        dash_table.DataTable(
                            id="reg_rec",
                            columns=[{'name': val, 'id': val}
                                     for val in REG_CRITERION],
                            data=[],
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            }
                        )
                    ),
                ]),

                html.Div([
                    html.Label(
                        "Performance Records for Classification Model"),
                    html.Div(
                        dash_table.DataTable(
                            id="clf_rec",
                            columns=[{'name': val, 'id': val}
                                     for val in CLF_CRITERION],
                            data=[],
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            }
                        )
                    ),
                ]),
            ])
        ], id='dash-container')

def create_RFA(server):
    global global_df
    dash_app = dash.Dash(name='RFA', server=server, url_base_pathname='/RFA/', external_stylesheets=[
        '/static/dist/css/styles.css',
        'https://fonts.googleapis.com/css?family=Lato',
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
    )

    dash_app.index_string = html_layout

    dash_app.layout = serve_layout
    
    @dash_app.callback(dash.dependencies.Output('dropdown_content', 'children'),
                       [dash.dependencies.Input('dropdown_section_name', 'value')])

    def render_tab_preparation_multiple_dropdown(value):
        global global_df
        print(global_df)
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
                    ),
                    html.Div(id='single_commands'),
                    html.P(
                        html.Label(
                            "Risk Factor Analysis: How many top risk factors you want to display?")
                    ),

                    dcc.Slider(
                        id='num-of-factors',
                        min=1,
                        max=min(round(0.5*len(global_df.columns)), 20),
                        marks={i: '{}'.format(i)
                               for i in range(min(round(0.5*len(global_df.columns)), 17))},
                        value=1
                    ),

                    html.Div(id='slider-output-container')
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
    
    @dash_app.callback(dash.dependencies.Output("hidden-div", "children"),
              [dash.dependencies.Input("interval-component", "n_intervals")])
    def update_df(n):
        global global_df 
        global_df= pd.read_csv(FILE_PATH)
        return dash.no_update
    
    # @dash_app.callback(dash.dependencies.Output("feature_dropdown", "options"),
    #           [dash.dependencies.Input("interval-component", "n_intervals")])
    # def update_df(n):
    #     global global_df 
    #     global_df= pd.read_csv(FILE_PATH)
    #     return [{'label': col, 'value': col} for col in global_df.columns]

    @dash_app.callback(Output(component_id='model_dropdown', component_property='options'),
                       [Input(component_id='type_dropdown', component_property='value')])
    def update_model_dropdown(task):
        if task == "Regression":
            return [{'label': i, 'value': i} for i in REGRESSON_LIST]
        elif task == "Classification":
            return [{'label': i, 'value': i} for i in CLASSIFICATION_LIST]
        else:
            return []

    @dash_app.callback(Output('slider-output-container', 'children'), [Input('num-of-factors', 'value')])
    def update_output(value):
        return 'You have selected {} top risk factors'.format(value)

    @dash_app.callback([Output('RFA_output', 'children'),
                        Output('reg_rec', 'data'),
                        Output('clf_rec', 'data')],
                       [Input('run_button', 'n_clicks')],
                       [State('dropdown', 'value'),
                        State('type_dropdown', 'value'),
                        State('model_dropdown', 'value'),
                        State('num-of-factors', 'value'),
                        State('reg_rec', 'data'),
                        State('clf_rec', 'data')])
    def perform_risk_factor_analysis(n_clicks, label, task_type, model_type, num_of_factor, reg_data, clf_data):
        global global_df
        if n_clicks > 0:
            y = global_df[label]
            X = global_df.drop([label], axis=1)
            col_names = X.columns

            if task_type == "Regression":
                model_res = regression_models(X, y, model_type, True)
                model = model_res[0]
                performance_layout = html.Div(
                    html.Div(
                        dash_table.DataTable(
                            id="reg_table",
                            columns=[{'name': val, 'id': val}
                                     for val in REG_CRITERION],
                            data=[{"Label": label, 'Model': model_type, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                   'R2':round(model_res[3], 5)}],
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            }
                        )
                    ),
                )
                info = "Perform Risk Factor Analysis with normalized data based on {} regression".format(
                    model_type)
            elif task_type == "Classification":
                model_res = classification_models(X, y, "Logistic", True)
                model = model_res[0]
                performance_layout = html.Div(
                    html.Div(
                        dash_table.DataTable(
                            id="clf_table",
                            columns=[{'name': val, 'id': val}
                                     for val in CLF_CRITERION],
                            data=[{"Label": label, 'Model': model_type, "Accuracy": round(model_res[1], 5), "ROC_AUC score":round(model_res[2], 5),
                                   "Precision":round(model_res[3], 5), "Recall":round(model_res[4], 5), "F1-Score":round(model_res[5], 5)}],
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            }
                        )
                    ),
                )
                info = "Perform Risk Factor Analysis with normalized data based on {} model".format(
                    model_type)
            else:
                return [], reg_data, clf_data

            res_tab_col = ["Rank", "Factor", "ABS weight", "sign"]
            res = risk_factor_analysis(model, col_names)

            layout = html.Div(children=[
                html.P(
                    html.Label(info)
                ),
                html.Div(
                    dash_table.DataTable(
                        id="RFA_table",
                        columns=[
                            {'name': i, 'id': i} for i in res_tab_col
                        ],
                        data=res[: num_of_factor],
                        style_cell={
                            'height': 'auto',
                            # all three widths are needed
                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        }
                    )
                ),

                html.P(
                    html.Label("{} model performance: ".format(model_type))
                ),

                performance_layout,


            ])

            if task_type == "Regression":
                return layout, reg_data + [{"Label": label, 'Model': model_type, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                            'R2':round(model_res[3], 5)}], clf_data
            elif task_type == "Classification":
                return layout, reg_data, clf_data + [{"Label": label, 'Model': model_type, "Accuracy": round(model_res[1], 5), "ROC_AUC score":round(model_res[2], 5),
                                                      "Precision":round(model_res[3], 5), "Recall":round(model_res[4], 5), "F1-Score":round(model_res[5], 5)}]
            else:
                return [], reg_data, clf_data
        else:
            return [], reg_data, clf_data

    return dash_app.server