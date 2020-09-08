import numpy as np
import plotly.express as px
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
from .layouts.layout import html_layout
from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
from sklearn.decomposition import PCA

df = pd.read_csv('/Users/wenchenliu/Desktop/dpt/cleaned_BRFSS.csv')
df = df.drop(['Unnamed: 0'], axis=1)
cols = []
target = []


# Define a function to check if the dataset is compact
def check_compact(df):
    result = True
    for i in df.columns.tolist():
        if df[i].isnull().sum() != 0:
            result = False
    return result

# def scatter_cluster2D(df, feat_cols, target):
#     model = PCA(n_components=2)
#     result = model.fit_transform(df[feat_cols].values)
#     df['D1'] = result[:,0]
#     df['D2'] = result[:,1]
#     fig = px.scatter(df, x="D1", y="D2", color=target)
#     print('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
#     return fig
#
# def scatter_cluster3D(df, feat_cols, target):
#     model = PCA(n_components=3)
#     result = model.fit_transform(df[feat_cols].values)
#     df['D1'] = result[:,0]
#     df['D2'] = result[:,1]
#     df['D3'] = result[:,2]
#     fig = px.scatter_3d(df, x='D1', y='D2', z='D3', color=target)
#     print('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
#     return fig

def create_PCA(server):
    dash_app = dash.Dash(name='cleaning', server=server, url_base_pathname='/PCA/', external_stylesheets=[
                             '/static/dist/css/styles.css',
                             'https://fonts.googleapis.com/css?family=Lato',
                             'https://codepen.io/chriddyp/pen/bWLwgP.css'
                             ]
                    )
    dash_app.index_string = html_layout

    dash_app.layout = html.Div(children =[
        #title and subtitle
        html.H1('This is where our title for this page will be'),
        html.H2('This is the subtitle'),

        #the button div
        html.Div([
            html.Div([
                html.A(html.Button('Previous page', style={'fontSize': '12px'}), href='/EDA/'),
            ], className='two columns'),

            html.Div([html.A(html.Button('Save and Proceed', style={'fontSize': '12px'}), href='/non/', id='save_button'),

            ], className='two columns'),


        ], className= 'row'),

        html.Div([
            # the left Div
            html.Div([
                html.Br(),
                #dropdowns that user can select the features to explore/visualize

                dcc.Dropdown(
                    options=[
                        {'label': i, 'value': i} for i in df.columns
                    ],
                    multi=True,
                    id = 'multi-dd-visual'
                ),
                html.Br(),

                dcc.Dropdown(
                    options=[
                        {'label': i, 'value': i} for i in df.columns
                    ],
                    multi=True,
                    id='multi-dd-target'
                ),
                html.Br(),

                dcc.RadioItems(
                    id='visual-dimension',
                    options=[{'label': i, 'value': i} for i in ['2D', '3D']],
                    labelStyle={'display': 'inline-block'}
                ),
                html.Br(),

                html.Button(
                    id='submit-button-state', n_clicks=0, children='Submit'
                ),


            ], className='four columns'),

            #the right div
            html.Div([

                dcc.Loading(
                    id='loading',
                    children=[
                        dcc.Graph(id='fig')
                    ])


            ], className='eight columns'),
        ], className='row')

    ], id='dash-container')

    @dash_app.callback(dash.dependencies.Output('fig', 'figure'),
                    [dash.dependencies.Input('submit-button-state', 'n_clicks')],
                    [dash.dependencies.State('multi-dd-visual', 'value'),
                     dash.dependencies.State('multi-dd-target', 'value'),
                     dash.dependencies.State('visual-dimension', 'value'),]
                     )

    def pca_graph(n_click, multi_dd_visual, multi_dd_target, visual_dimension):

        cols.append(multi_dd_visual)
        target.append(multi_dd_target)


        if n_click > 0 and visual_dimension == '2D':
            model = PCA(n_components=2)
            result = model.fit_transform(df[cols[-1]].values)
            df['D1'] = result[:, 0]
            df['D2'] = result[:, 1]
            fig = px.scatter(df, x="D1", y="D2", color=target[-1][0])
            # fig.show()
            return fig
            # print('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))

        if n_click > 0 and visual_dimension == '3D':
            model = PCA(n_components=3)
            result = model.fit_transform(df[cols[-1]].values)
            df['D1'] = result[:, 0]
            df['D2'] = result[:, 1]
            df['D3'] = result[:, 2]
            fig = px.scatter_3d(df, x='D1', y='D2', z='D3', color=target[-1][0])
            # fig.show()
            return fig
            # print('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
        # layout = html.Div([
        #     html.Div([
        #         html.Label('Scatter plot\n'),
        #         dcc.Graph(figure = fig),
        #         html.Ul([html.Li(x) for x in cols])
        #     ]),
        #
        # ]),

        # if n_click > 0:
        #     print(cols)
        #     print(target)
        #
        else:
            return html.Label("Please Submit first.")


    if __name__ == '__main__':
        dash_app.run_server(debug=True)
