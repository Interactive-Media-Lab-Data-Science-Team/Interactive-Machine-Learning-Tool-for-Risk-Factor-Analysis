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
from ..modelHelperFunctions import regression_models, classification_models, reg_risk_factor_analysis, clf_risk_factor_analysis
import dash_bootstrap_components as dbc

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

layout3 = html.Div(
        children=[
            html.H4("Model Fitting and Risk Factor Analysis"),
            html.Div([
                html.Div([
                    html.A(html.Button('BACK', style={
                        'fontSize': '12px'}), href='/EDA/'),
                ]),

            ]),

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

