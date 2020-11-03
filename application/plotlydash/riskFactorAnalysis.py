import ast
import numpy as np
import plotly.express as px
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from .layouts.layout import html_layout
from .modelHelperFunctions import regression_models, classification_models, risk_factor_analysis
import dash_bootstrap_components as dbc


FILE_PATH = 'download.csv'
#FILE_PATH = 'data/download.csv'
VAR_PATH = 'data/var_info.txt'
STATE_PATH = 'data/state_info.txt'
SECTION_PATH = 'data/section_name.txt'
REGRESSON_LIST = ["Linear", "Lasso", "Ridge",
                  "LassoLars", "Bayesian Ridge", "Elastic Net"]
REG_CRITERION = ['Index', 'Label', 'Model', 'Penalty', 'MAE', 'MSE']
CLASSIFICATION_LIST = ["Logistic", "LDA"]
#CLF_CRITERION = ["Index", "Label", "Model", "Penalty", "Accuracy", "ROC_AUC score", "Precision", "Recall", "F1-Score"]
CLF_CRITERION = ["Index", "Label", "Model", "Penalty",
                 "Accuracy", "Precision", "Recall", "F1-Score"]


df = pd.read_csv(FILE_PATH)[:5000]


def load_info_dict(file):
    f = open(file, 'r')
    cont = f.read()
    f.close()
    return ast.literal_eval(cont)


var_info = load_info_dict(VAR_PATH)
section_info = load_info_dict(SECTION_PATH)
state_info = load_info_dict(STATE_PATH)

SECTION = list(section_info.keys())
STATE = list(state_info.keys())


def serve_layout():
    return html.Div(
        children=[
            html.H4("Model Fitting and Risk Factor Analysis"),
            html.Div([
                html.Div([
                    html.A(html.Button('BACK', style={
                        'fontSize': '12px'}), href='/EDA/'),
                ]),

            ]),

            html.Div([
            ], id='hidden-div', style={'display': 'none'}),
            dcc.Interval('interval-component', n_intervals=0, interval=1*1000),

            html.Div([

                html.P(html.Label(
                    "Please select a state: ")),
                html.Div([
                    dcc.Dropdown(
                        id='state_dropdown',
                        options=[
                            {'label': col, 'value': state_info[col]} for col in STATE
                        ],
                        placeholder="Section",
                        style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                    )
                ], style=dict(display='flex')),


                html.P(html.Label(
                    "Please select a section: ")),
                html.Div([
                    dcc.Dropdown(
                        id='section_dropdown',
                        options=[
                            {'label': col, 'value': col} for col in SECTION
                        ],
                        placeholder="Section",
                        style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                    )
                ], style=dict(display='flex')),

                html.P(html.Label(
                    "Please select a variable as target for analysis: (Options are avaliable after selecting a section)")),
                html.Div([
                    dcc.Dropdown(
                        id='feature_dropdown',
                        options=[],
                        placeholder="Variable: Explanation",
                        style=dict(
                            width='80%',
                            verticalAlign="middle"
                        )
                    )
                ], style=dict(display='flex')),

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
                            {'label': "Continuous Variable",
                             'value': "Regression"}

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
                    html.P(
                        html.Label(
                            "Please select the penalty multiplier: ")
                    ),

                    dcc.Slider(
                        id='penalty',
                        min=0,
                        max=10,
                        step=0.05,
                        marks={i: '{}'.format(i)
                               for i in range(11)},
                        value=1
                    ),

                    html.Div(id='penalty-output-container'),


                ]),


                html.Div([
                    html.P(
                        html.Label(
                            "Risk Factor Analysis: How many top risk factors you want to display?")
                    ),

                    dcc.Slider(
                        id='num-of-factors',
                        min=1,
                        max=20,
                        marks={i: '{}'.format(i)
                               for i in range(21)},
                        value=1
                    ),

                    html.Div(id='slider-output-container')


                ]),

                html.Div([
                    html.Label(
                        "Proceed risk factor analysis and see the results"),
                    html.Hr(),
                    html.Button('Run', id='run_button', n_clicks=0),
                    # dbc.Popover(
                    #     [
                    #         dbc.PopoverHeader("Note:"),
                    #         dbc.PopoverBody("Make sure to fill out all values"),
                    #     ],
                    #     id="popover",
                    #     is_open=False,
                    #     target="run_button",
                    # ),
                    html.Div(id='error'),
                    dcc.Loading(
                        html.Div(id='RFA_output', children=[])
                    ),

                ]),
                html.Br(),
                html.Details(
                    [
                        html.Summary("Performance of History Models"),
                        html.Div(
                            [
                                html.Div([
                                    html.Details(
                                        [
                                            html.Summary(
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
                                                        'minWidth': '100px', 'width': '160px', 'maxWidth': '240px',
                                                        'marginLeft': '30px',
                                                        'whiteSpace': 'normal',
                                                        'textAlign': 'right'
                                                    }
                                                )
                                            ),
                                        ]
                                    )
                                ]),

                                html.Div([
                                    html.Details(
                                        [
                                            html.Summary(
                                                "Performance Records for Classification Model"),
                                            html.Div(
                                                dash_table.DataTable(
                                                    id="clf_rec",
                                                    columns=[{'name': val, 'id': val}
                                                             for val in CLF_CRITERION],
                                                    data=[],
                                                    style_cell={
                                                        'height': 'auto',
                                                        'textAlign': 'right'
                                                        # all three widths are needed
                                                        # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                                        # 'minWidth': '100px', 'width': '120px', 'maxWidth': '240px',
                                                        # 'whiteSpace': 'normal'
                                                    }
                                                )
                                            ),
                                        ]
                                    )
                                ]),
                            ], style={'marginLeft': 40})
                    ]
                )

            ])
        ], id='dash-container', style={'marginBottom': 40})


def create_RFA(server):
    global df
    dash_app = dash.Dash(name='RFA', server=server, url_base_pathname='/RFA/', external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        '/static/dist/css/styles.css',
        # 'https://fonts.googleapis.com/css?family=Lato',
        # 'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://codepen.io/dapraxis/pen/OJXNeMP.css'
    ]
    )

    dash_app.index_string = html_layout

    dash_app.layout = serve_layout

    @dash_app.callback(dash.dependencies.Output("hidden-div", "children"),
                       [dash.dependencies.Input("interval-component", "n_intervals")])
    def update_df(n):
        global df
        df = pd.read_csv(FILE_PATH)
        return dash.no_update

    @dash_app.callback(Output(component_id='feature_dropdown', component_property='options'),
                       [Input(component_id='section_dropdown', component_property='value')])
    def update_feature_dropdown(section):
        lst = section_info.get(section)
        return [{'label': '{}: {}'.format(i, var_info.get(i).get('Label')), 'value': i} for i in lst]

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

    @dash_app.callback(Output('penalty-output-container', 'children'), [Input('penalty', 'value')])
    def update_output(value):
        return 'You have selected {} as penalty multiplier'.format(value)

    @dash_app.callback([Output('RFA_output', 'children'),
                        Output('reg_rec', 'data'),
                        Output('clf_rec', 'data'),
                        Output('popover', 'is_open'), ],
                       [Input('run_button', 'n_clicks')],
                       [State('state_dropdown', 'value'),
                        State('feature_dropdown', 'value'),
                        State('type_dropdown', 'value'),
                        State('model_dropdown', 'value'),
                        State('penalty', 'value'),
                        State('num-of-factors', 'value'),
                        State('reg_rec', 'data'),
                        State('clf_rec', 'data')])
    def perform_risk_factor_analysis(n_clicks, state, label, task_type, model_type, penalty, num_of_factor, reg_data, clf_data):
        global df
        if n_clicks > 0:

            if((label == None) or (task_type == None) or (model_type == None)):
                return [], reg_data, clf_data, True

            state_df = df[df['_STATE'] == state]
            y = state_df[label]
            X = state_df.drop([label], axis=1)
            col_names = X.columns

            if task_type == "Regression":
                model_res = regression_models(
                    X, y, model_type, True, alpha=penalty)
                model = model_res[0]
                performance_layout = html.Div(
                    html.Div(
                        dash_table.DataTable(
                            id="reg_table",
                            columns=[{'name': val, 'id': val}
                                     for val in REG_CRITERION[1:]],
                            data=[{"Label": label, 'Model': model_type, 'Penalty': penalty, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                   'R2':round(model_res[3], 5)}],
                            style_cell={
                                'height': 'auto',
                                'textAlign': 'right'
                                # all three widths are needed
                                # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                # 'whiteSpace': 'normal'
                            },
                        )
                    ),
                )
                info = "Perform Risk Factor Analysis with normalized data based on {} regression".format(
                    model_type)
            elif task_type == "Classification":
                model_res = classification_models(
                    X, y, "Logistic", True, C=penalty)
                model = model_res[0]
                performance_layout = html.Div(
                    html.Div(
                        dash_table.DataTable(
                            id="clf_table",
                            columns=[{'name': val, 'id': val}
                                     for val in CLF_CRITERION[1:]],
                            data=[{"Label": label, 'Model': model_type, "Penalty": penalty, "Accuracy": round(model_res[1], 5),
                                   "Precision":round(model_res[2], 5), "Recall":round(model_res[3], 5), "F1-Score":round(model_res[4], 5)}],
                            style_cell={
                                'height': 'auto',
                                'textAlign': 'right'
                                # all three widths are needed
                                # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                # 'whiteSpace': 'normal'
                            }
                        )
                    ),
                )
                info = "Perform Risk Factor Analysis with normalized data based on {} model".format(
                    model_type)
            else:
                return [], reg_data, clf_data, False

            res_tab_col = ["Rank", "Factor", "Absolute Weight", "Sign"]
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
                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                            'whiteSpace': 'normal',
                            'textAlign': 'right'
                        },
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {
                                    'column_id': 'Sign',
                                    'filter_query': '{Sign} = "-"'
                                },
                                'backgroundColor': 'dodgerblue',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'column_id': 'Sign',
                                    'filter_query': '{Sign} = "+"'
                                },
                                'backgroundColor': '#85144b',
                                'color': 'white'
                            },
                        ],
                    )
                ),

                html.P(
                    html.Label("{} model performance: ".format(model_type))
                ),
                performance_layout,
            ])

            if task_type == "Regression":
                return layout, reg_data + [{"Index": len(reg_data)+1, "Label": label, 'Model': model_type, 'Penalty': penalty, 'MAE': round(model_res[1], 5), 'MSE':round(model_res[2], 5),
                                            }], clf_data, False
            elif task_type == "Classification":
                return layout, reg_data, clf_data + [{"Index": len(clf_data)+1, "Label": label, 'Model': model_type, "Penalty": penalty, "Accuracy": round(model_res[1], 5),
                                                      "Precision":round(model_res[2], 5), "Recall":round(model_res[3], 5), "F1-Score":round(model_res[4], 5)}], None
            else:
                return [], reg_data, clf_data, False
        else:
            return [], reg_data, clf_data, False

    return dash_app.server
