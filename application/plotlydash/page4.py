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
from .model_selection_func import logistic_regression, random_forest, knn
#sklearn 1.18.4
# To be delete: test data
from sklearn.datasets import load_breast_cancer

def create_page4(server):
    dash_app = dash.Dash(name='MODEL_SELECTION', server=server, url_base_pathname='/MODEL_SELECTION/', external_stylesheets=[
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'
                             ]
                    )
    
    dash_app.index_string = html_layout
    
    test_data = load_breast_cancer()
    # only use 5 features for quick training
    X = test_data.data[:,:5]
    y = test_data.target 

    MODEL_LIST = ['logistic regression', 'random forest', 'KNN']
    PERFORMANCE_MEASURE = ['Accuracy', 'ROC_AUC', 'Precision', 'F1_Score', 'Cross-entropy']
    all_res = []
    saved_res = []

    dash_app.layout = html.Div(children=[
        html.H4("Benchmark Machine Learning Model"),
        html.Div([
            html.Div([
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/data_cleaning/'),
            ], className='two columns'),

        ], className= 'row'),

        html.Div([
            html.Div([
                html.P(html.Label("Please select a model: ")),
                dcc.Dropdown(
                    id='model_dropdown',
                    options=[
                        {'label': i, 'value': i} for i in MODEL_LIST

                    ],
                    value = 'model'
                )
            ]),

            html.Div([

            ],
            id = 'parameters'),

        ]),
    ], id='dash-container')

    def logistic_regression_div():
        layout = html.Div([
            html.Div([
                html.H2("Please select Parameters for logistic regression model: ")
            ]),
            html.Div([
                html.Label("Please select C within range 0.001 to 20: "),
                dcc.Input(
                id="log_C",
                type="number",
                value=1.0,
                min=0.001,
                max=20
                ),

            ]),

            html.Div([
                html.Label("Please select penalty type: "),
                dcc.Dropdown(
                    id='log_panelty',
                    options = [
                        {'label':'L1','value':'l1'},
                        {'label':'L2','value':'l2'}
                    ],

                    value = 'l2'

                )

            ]),

            html.Div([
                html.Label("Please select solver: "),
                dcc.Dropdown(
                    id="log_solver",
                    options = [
                        {'label':'newton-cg','value':'newton-cg'},
                        {'label':'liblinear','value':'liblinear'},
                        {'label':'saga','value':'saga'}
                    ],
                    value = 'newton-cg'
                )

            ]),


            html.Div([
                html.Label("Proceed to model training with selected parameters"),
                html.Button('Run', id='logistic_button', n_clicks = 0),
                html.Div(id='logistic_output',children='Select parameters and press RUN')
            ]),

            
        ])

        return layout

    def random_forest_div():
        layout = html.Div([
            html.H2("Please select Parameters for random forest model: "),

            html.Div([
                html.Label("Please select number of estimators: "),
                dcc.Slider(
                    id = 'rf_n_est',
                    min = 10,
                    max = 400,
                    marks = {i: '{}'.format(i) for i in range(10,401,10)},
                    value = 100

                )
            ]),

            html.Div([
                html.Label("Please select minimum sample size: "),
                dcc.Slider(
                    id= 'rf_min_sample',
                    min = 1,
                    max = 10,
                    marks = {i: '{}'.format(i) for i in range(11)},
                    value = 1
                )

            ]),

            html.Div([
                html.Label("Please select maximum number of features : "),
                dcc.Dropdown(
                    id = 'rf_max_feature',
                    options = [
                        {'label':'Auto','value':'auto'},
                        {'label':'Square Root','value':'sqrt'},
                        {'label':'Log2','value':'log2'}
                    ],
                    value = 'auto'
                )

            ]),

            html.Div([
                html.Label("Please select a criterion : "),
                dcc.Dropdown(
                    id = 'rf_criterion',
                    options = [
                        {'label':'Gini','value':'gini'},
                        {'label':'Entropy','value':'entropy'}
                    ],
                    value = 'gini'
                )

            ]),

            html.Div([
                html.Label("Proceed to model training with selected parameters"),
                html.Button('Run', id='rf_button', n_clicks = 0),
                html.Div(id='rf_output',children='Select parameters and press RUN')
            ]),

        ])

        return layout

    def knn_div():
        layout = html.Div([
            html.H2("Please select Parameters for KNN model: "),
            html.Div([
                html.Label('Plaese select number of neighbours: '),
                dcc.Slider(
                    id = 'knn_neighbour',
                    min = 1,
                    max = 20,
                    marks = {i: '{}'.format(i) for i in range(21)},
                    value = 5
                )
                
            ]),

            html.Div([
                html.Label('Plaese select a weight measure: '),
                dcc.Dropdown(
                    id = 'knn_weight',
                    options = [
                        {'label':'Uniform','value':'uniform'},
                        {'label':'Distance','value':'distance'}
                    ],
                    value = 'uniform'
                )

            ]),

            html.Div([
                html.Label('Plaese select an algorithm to compute nearest neighbours: '),
                dcc.Dropdown(
                    id = 'knn_algo',
                    options = [
                        {'label':'Auto','value':'auto'},
                        {'label':'Ball_tree','value':'ball_tree'},
                        {'label':'KD_tree','value':'kd_tree'},
                        {'label':'Brute-Force','value':'brute'}
                    ],
                    value = 'auto'
                )
            ]),

            html.Div([
                html.Label("Proceed to model training with selected parameters"),
                html.Button('Run', id='knn_button', n_clicks = 0),
                html.Div(id='knn_output',children='Select parameters and press RUN')
            ]),
        ])

        return layout

    def empty_model_div():
        return html.Div([
            html.H2('No model has been selected yet.'),
        ])

    model_param_dict = {
        'model': empty_model_div(),
        'logistic regression': logistic_regression_div(),
        'random forest': random_forest_div(),
        'KNN': knn_div()
    }



    @dash_app.callback(Output(component_id='parameters',component_property= 'children'),
                    [Input(component_id='model_dropdown', component_property = 'value')])
    def update_parameter_section(model):
        return model_param_dict.get(model)

    @dash_app.callback(Output(component_id='log_solver', component_property='options'),
                    [Input(component_id='log_panelty',component_property='value')])
    def update_log_panelty(panelty):
        if panelty == 'l1':
            return [
                        {'label':'liblinear','value':'liblinear'},
                        {'label':'saga','value':'saga'}
                    ]
        else:
            return [
                        {'label':'newton-cg','value':'newton-cg'},
                        {'label':'liblinear','value':'liblinear'},
                        {'label':'saga','value':'saga'}
                    ]
    


    @dash_app.callback(Output(component_id='logistic_output',component_property='children'),
                        [Input('logistic_button','n_clicks')],
                        [State('log_C','value'),
                        State('log_solver','value'),
                        State('log_panelty','value')])

    def update_logistic_results(n_clicks, C, solver, panelty):
        res = logistic_regression(X, y, C, panelty, solver)
        #all_res += [res]
        fig = px.imshow(res[4],
        labels=dict(x='Predicted Label', y = 'True Label', color='# of cases'))# To be added: x and y
        layout = html.Div([
            html.Div(dash_table.DataTable(
                id = 'logistic_table',
                columns = [{"name": "Score","id":"Score"},{"name":"Performance","id":"Performance"}],
                data = [{'Score':'Accuracy','Performance':res[0]},
                {'Score':'ROC_AUC','Performance':res[1]},
                {'Score':'Precision','Performance':res[2]},
                {'Score':'F1-Score','Performance':res[3]},
                {'Score':'Cross-entropy','Performance':res[5]}]
                )
            ),
            html.Div([
                html.Label('Confusion Matrix:\n'),
                dcc.Graph(figure=fig)
            ]),

            html.Div([
                html.Button('Save the results', id='logistic_save'),
                html.Div(id='logistic_check_res',children='Click save to see the current results')
            ]),

        ])
        if n_clicks>0:
            return layout
        else:
            return html.Label("Click to see the results.")

    @dash_app.callback(Output(component_id='logistic_check_res',component_property='children'),
                        [Input('logistic_save', 'n_clicks')])
    def save_logistic_results(n_clicks):
        #SAVED_RES += [ALL_RES[-1]]
        return html.Div(html.H1('Result Saved'))

    
    

    @dash_app.callback(Output(component_id='rf_output',component_property='children'),
                        [Input('rf_button','n_clicks')],
                        [State('rf_n_est','value'),
                        State('rf_min_sample','value'),
                        State('rf_max_feature','value'),
                        State('rf_criterion','value')])

    def update_rf_results(n_clicks, n_est, min_sample, max_feature, criterion):
        res = random_forest(X, y, n_est, max_dep=10, min_sample=min_sample,max_features=max_feature,criterion=criterion )
        fig = px.imshow(res[4],
        labels=dict(x='Predicted Label', y = 'True Label', color='# of cases'))# To be added: x and y
        layout = html.Div([
            html.Div(dash_table.DataTable(
                id = 'rf_table',
                columns = [{"name": "Score","id":"Score"},{"name":"Performance","id":"Performance"}],
                data = [{'Score':'Accuracy','Performance':res[0]},
                {'Score':'ROC_AUC','Performance':res[1]},
                {'Score':'Precision','Performance':res[2]},
                {'Score':'F1-Score','Performance':res[3]},
                {'Score':'Cross-entropy','Performance':res[5]}]
                )
            ),
            html.Div([
                html.Label('Confusion Matrix:\n'),
                dcc.Graph(figure=fig)
            ]),

            html.Div([
                html.Button('Save the results', id='rf_save'),
                html.Div(id='rf_check_res',children='Click save to see the current results')
            ]),

        ])
        return layout
    
    @dash_app.callback(Output(component_id='knn_output',component_property='children'),
                        [Input('knn_button','n_clicks')],
                        [State('knn_neighbour','value'),
                        State('knn_weight','value'),
                        State('knn_algo','value')
                        ])

    def update_knn_results(n_clicks, n_neighbour, weight, algo):
        res = knn(X, y, n_neighbour, weight, algo)
        fig = px.imshow(res[4],
        labels=dict(x='Predicted Label', y = 'True Label', color='# of cases'))# To be added: x and y
        layout = html.Div([
           html.Div(dash_table.DataTable(
                id = 'knn',
                columns = [{"name": "Score","id":"Score"},{"name":"Performance","id":"Performance"}],
                data = [{'Score':'Accuracy','Performance':res[0]},
                {'Score':'ROC_AUC','Performance':res[1]},
                {'Score':'Precision','Performance':res[2]},
                {'Score':'F1-Score','Performance':res[3]},
                {'Score':'Cross-entropy','Performance':res[5]}]
                )
            ),
            html.Div([
                html.Label('Confusion Matrix:\n'),
                dcc.Graph(figure=fig)
            ]),

            html.Div([
                html.Button('Save the results', id='knn_save'),
                html.Div(id='knn_check_res',children='Click save to see the current results')
            ]),

        ])
        return layout

    return dash_app.server
