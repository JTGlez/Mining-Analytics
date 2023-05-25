# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Árboles de Decisión para regresión.

#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import base64
import datetime
import io
from io import BytesIO
import dash # Biblioteca principal de Dash.
from msilib.schema import Component
from dash import dcc, html, Input, Output, callback# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, State # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from modules import home, eda, pca, regtree, classtree, regforest, classforest
import pathlib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import uuid
import graphviz

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def regtree_card():
    """
    :retorna: Un div que contiene la explicación del módulo de Árboles de Decisión: Regresión.

    """

    return html.Div(

        # ID del div.
        id="regtree-card",

        # Elementos hijos del div "regtree-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Árbol de Decisión: Regresión"), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="Los árboles de decisión representan uno de los algoritmos de aprendizaje supervisado más utilizados, los cuales soportan tanto valores numéricos como nominales. Para esto, se construye una estructura jerárquica que divide los datos en función de condicionales."
                ,
            ),

            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="tree1",
                        src="/assets/tree1.png",
                        style = {'width': '25em', 'height': '15em'}
                    )
                ]
            ),
            # Texto secundario de explicacion.
            html.Div(
                style =  {'text-align': 'justify'},
                children = "Si bien los árboles son una opción muy noble para el modelado de datos, estos tienden a tener problemas de sobreajuste excesivo en los datos. Por ello, es necesario que se consideren cuidadosamente los hiperparámetros de elección para la generación del modelo. A continuación se muestran los parámetros que deben considerarse:",
            ),

            html.Br(),

            # Muestra una figura de parámetros del árbol
            dbc.Badge(
                    "Parámetros de generación",
                    pill=True,
                    color="primary",
                    style={"font-size":"15px"}
                ),
                html.Br(), html.Br(),
                html.P([
                    html.P("📏 Max Depth:", style={"font-family": "Acumin"}),
                    "Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting."
                ], className="ms-4"),

                html.P([
                    html.P("✂️ Min Samples Split:", style={"font-family": "Acumin"}),
                    "Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja."
                ], className="ms-4"),

                html.P([
                    html.P("🍃 Min Samples Leaf:", style={"font-family": "Acumin"}),
                    "Indica la cantidad mínima de datos que debe tener un nodo hoja."
                ], className="ms-4"),

        ],

    )

# Contenedor principal de la página en un Div.
regtree.layout = html.Div(
    id = "page-content",
    children=[

        # Contenido principal de la aplicación: se divide en 2 columnas: una con contenido explicativo y otra para elementos interactivos.
        html.Div(

            className="row",
            children=[

                # Columna a la izquierda: invoca a description_card para mostrar el texto explicativo de la izquierda.
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[regtree_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id = "right-column",
                    className="four columns",
                    children = html.Div([

                        html.H4("Carga o elige el dataset para iniciar la regresión con Árboles de Decisión", className= "text-upload"),

                        # Muestra el módulo de carga del dataset.
                        dcc.Upload(
                        id = 'upload-data-regtree',
                        children = html.Div([
                            'Arrastra aquí el archivo en formato CSV o selecciónalo'
                        ]),

                    # Por limitación de Dash estos elementos de carga no se pueden modificar desde la hoja de estilos y se debe hacer CSS inline.
                    style = {
                        'font-family': 'Acumin',
                        'width': '50%',
                        'height': '100%',
                        'lineHeight': '60px',
                        'borderWidth': '2px',
                        'borderStyle': 'solid',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2em auto',
                        'display': 'grid',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex-direction': 'column',
                        #'borderColor': '#2196F3',
                        'background-color': '#fEfEfE'
                    },
                    multiple = True,
                    accept = '.csv'
                ),

                html.Div(
                    children = "O utiliza como datos de entrada los históricos de algún activo (Stocks, Criptomonedas o Index)",
                        style = {
                        'font-family': 'Acumin',
                        'width' : '100%',
                        'text-align': 'center'
                        }
                ),


                html.Div(
                    children=[
                        dcc.Input(
                            id='ticker-input',
                            type='text',
                            placeholder='Ingrese el ticker aqui',
                            className='my-input',
                            style={
                                'background-color': '#fEfEfE',
                                'display': 'inline-block',
                                'margin': '0 auto'
                            }
                        ),
                        html.Button(
                            'Enviar', id='submit-ticker', n_clicks=0,
                            className='my-button',
                            style={
                                'background-color': '#fEfEfE'

                            }
                        )
                    ],
                    style={
                        'display': 'table',
                        'margin': '0 auto',
                    }
                ),

                html.Hr(),
                html.Div(id = 'output-data-upload-regtree'),                

                ]),
                ),
                #html.Div(id = 'output-dataset-upload'),
            ],
        ),
    ],
)


def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), index_col=None)
            return regtree(df, filename, df.columns)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return regtree(df, filename, df.columns)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

def regtree(df, filename, columns):
    """
    retorna: modelo de regresión usando un árbol de decisión regresor para la generación de pronósticos y valores siguientes en series de tiempo.

    """
    global global_df
    global_df = df


    # Preparación de variables para su despliegue.
    fig = create_yahoo_finance_chart(df, filename)

    # Div de visualización en el layout.
    return html.Div([

        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Solo mostramos las primeras 5 filas del dataframe, y le damos estilo para que las columnas se vean bien
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            cell_selectable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        
        html.Hr(),  # Línea horizontal

        dcc.Graph(
            id='yahoo-finance-chart',
            figure = fig  
        ),

        html.H3(
            "Elección de Variables Predictoras y Dependiente",
        ),

        html.Div(
            html.P("Selecciona de la siguiente lista las variables que deseas elegir como predictoras y tu variable target para realizar la regresión.")
        ),

        html.Div(
            children=[
                dcc.Store(id="original-options", data=[{'label': col, 'value': col} for col in df.columns]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Variables predictoras", color="primary",
                                                    id="tooltip-method", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center"},
                                                    ),
                                            dbc.Tooltip(
                                                "Selecciona aquí las variables predictoras de tu análisis.",
                                                target="tooltip-method"
                                            ),
                                        ],
                                        style={"height": "50px", "padding": "0"},
                                    ),
                                    style = {"height": "2.5em"}
                                ),
                                dbc.Row(
                                    dbc.Checklist(
                                        id='select-predictors',
                                        options = [{'label': col, 'value': col} for col in df.columns],
                                        style={"font-size": "small", "display": "grid", "justify-items": "start", "font-family": "Acumin, 'Helvetica Neue', sans-serif", "margin": "-1em 0 0 0"}
                                    ),
                                    style={"height": "auto"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Variable Regresora", color="primary",
                                                    id="tooltip-numpc", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center"},
                                                    ),
                                            dbc.Tooltip(
                                                "Selecciona la variable target de tu análisis.",
                                                target="tooltip-numpc"
                                            ),
                                        ],
                                        style={"height": "auto", "padding": "0"},
                                    ),
                                    style = {"height": "2.5em"}
                                ),
                                dbc.Row(
                                    dbc.Checklist(
                                        id='select-regressor',
                                        options = [{'label': col, 'value': col} for col in df.columns],
                                        style={"font-size": "small", "display": "grid", "justify-items": "start", "font-family": "Acumin, 'Helvetica Neue', sans-serif", "margin": "-1em 0 0 0"}
                                    ),
                                    style={"font-size":"small", "height": "2em", "font-family": "Acumin, 'Helvetica Neue', sans-serif"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                    ],
                    style={"justify-content": "between", "height": "100%"}
                ),

                        html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Profundad máxima del árbol", color="primary",
                                                id="tooltip-percent", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Profundidad máxima del árbol:**  
                                                        Coloca aquí el nivel máximo del árbol a generar.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-percent", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='input-max-depth',
                                        type='number',
                                        placeholder='None',
                                        min=1,
                                        step=1,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),

                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Muestras mínimas de división", color="primary",
                                                id="tooltip-div", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Muestras mínimas de división**  
                                                        Coloca aquí el mínimo de muestras para dividir nodos de decisión.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-div", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='input-min-samples-split',
                                        type='number',
                                        placeholder='None',
                                        min=1,
                                        step=1,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),

                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Muestras mínimas por hoja", color="primary",
                                                id="tooltip-leaf", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Muestras mínimas de división**  
                                                        Coloca aquí el mínimo de muestras en las hojas del árbol a generar.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-leaf", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='input-min-samples-leaf',
                                        type='number',
                                        placeholder='None',
                                        min=1,
                                        step=1,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),

                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Tamaño de la muestra", color="primary",
                                                id="tooltip-sample", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Muestras mínimas de división**  
                                                        Coloca aquí el mínimo de muestras en las hojas del árbol a generar.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-sample", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='input-test-size',
                                        type='number',
                                        placeholder='None',
                                        value=0.2,
                                        min=0.2,
                                        max = 0.5,
                                        step=0.1,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                    ],
                    style={"justify-content": "between", "height": "100%"}
                ),
            ],
            style={"font-size":"20px", "margin":"30px 0"}
        ),

                
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Generar árbol", id="submit-button", color="primary", className="mt-3", style={"display": "grid", "height": "80%", "align-items": "center", "margin": "0 auto"}),
                        width={"size": 2, "offset": 5},
                    ),
                    className="mt-3",
                ),
                html.Div(id="output-data", style = {"margin-top": "1em"}),
            ],
            className="mt-4",
        )

    ])

def get_yahoo_finance_data(ticker):
    """
    retorna: dataset con el histórico del ticker especificado.
    """
    df = yf.download(ticker, period="max", interval = "1d")
    return df

def create_yahoo_finance_chart(df, filename):
    # Crea el gráfico de Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines+markers', name='Open', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], mode='lines+markers', name='High', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='lines+markers', name='Low', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Close', line=dict(color='green')))

    fig.update_layout(
        title=f"Histórico de {filename}",
        xaxis_title="Fecha",
        yaxis_title="Precio de las acciones",
        legend_title="Precios",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
    )
    return fig

@callback(Output('output-data-upload-regtree', 'children'),
              [Input('upload-data-regtree', 'contents'),
               Input('submit-ticker', 'n_clicks')],
              [State('upload-data-regtree', 'filename'),
               State('upload-data-regtree', 'last_modified'),
               State('ticker-input', 'value')])
def update_output(list_of_contents, submit_ticker_clicks, list_of_names, list_of_dates, ticker):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-regtree.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'submit-ticker.n_clicks':
        if ticker:
            df = get_yahoo_finance_data(ticker)
            return regtree(df, ticker, df.columns)

def create_comparison_chart(Y_test, Y_Predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=Y_test.flatten(), mode='lines', name='Real', marker=dict(color='red', symbol='cross')))
    fig.add_trace(go.Scatter(y=Y_Predicted, mode='lines', name='Estimado', marker=dict(color='green', symbol='cross')))

    fig.update_layout(
        title="Pronóstico de las acciones",
        xaxis_title="Fecha",
        yaxis_title="Precio de las acciones",
        legend_title="Valores",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
    )
    return fig

def generate_decision_tree(X_train, X_test, Y_train, Y_test, max_depth=2, min_samples_split=4, min_samples_leaf=4):
    reg_tree = DecisionTreeRegressor(
        random_state=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    reg_tree.fit(X_train, Y_train)
    Y_Predicted = reg_tree.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    tree_parameters = {
        "criterion": reg_tree.criterion,
        "feature_importances": reg_tree.feature_importances_,
        "MAE": mean_absolute_error(Y_test, Y_Predicted),
        "MSE": mean_squared_error(Y_test, Y_Predicted),
        "RMSE": mean_squared_error(Y_test, Y_Predicted, squared=False),
        "score": r2_score(Y_test, Y_Predicted),
    }
    return comparison_df, reg_tree, tree_parameters, Y_Predicted

def generate_decision_treeS(X_train, X_test, Y_train, Y_test):
    reg_tree = DecisionTreeRegressor(
        random_state=0,
    )
    reg_tree.fit(X_train, Y_train)
    Y_Predicted = reg_tree.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    tree_parameters = {
        "criterion": reg_tree.criterion,
        "feature_importances": reg_tree.feature_importances_,
        "MAE": mean_absolute_error(Y_test, Y_Predicted),
        "MSE": mean_squared_error(Y_test, Y_Predicted),
        "RMSE": mean_squared_error(Y_test, Y_Predicted, squared=False),
        "score": r2_score(Y_test, Y_Predicted),
    }
    return comparison_df, reg_tree, tree_parameters, Y_Predicted

def create_input_form(predictors):
    input_form = []
    for predictor in predictors:
        input_form.append(
            html.Div(
                [
                    html.Label(predictor),
                    dcc.Input(
                        type="number",
                        id=f"input-{predictor}",  # Agrega el atributo id a la entrada
                    ),
                ],
                className="form-group",
            )
        )
    return input_form


@callback(
    [Output("select-predictors", "options"), Output("select-regressor", "options")],
    [Input("select-predictors", "value"), Input("select-regressor", "value")],
    [State("original-options", "data")]
)
def update_checklist_options(selected_predictors, selected_regressor, original_options):
    if selected_predictors is None:
        selected_predictors = []

    if selected_regressor is None:
        selected_regressor = ""

    updated_predictors_options = [
        {**option, "disabled": option["value"] == selected_regressor} for option in original_options
    ]

    updated_regressor_options = [
        {**option, "disabled": option["value"] in selected_predictors} for option in original_options
    ]


    return updated_predictors_options, updated_regressor_options


@callback(Output("input-form", "children"), Input("submit-button", "n_clicks"))
def update_input_form(n_clicks):
    if n_clicks is None:
        return ""
    return create_input_form(global_predictors)

def predict_new_values(reg_tree, predictors, input_values):
    input_data = pd.DataFrame(input_values, columns=predictors)
    prediction = reg_tree.predict(input_data)
    return prediction

@callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("input-form", "children"),
)
def show_prediction(n_clicks, input_form):
    if n_clicks is None or input_form is None:
        return ""

    input_values = {}
    all_states = dash.callback_context.states
    for child in input_form:
        label = child['props']['children'][0]['props']['children']
        if label in global_predictors:
            input_id = child['props']['children'][1]['props']['id']
            try:
                # Agrega el id del campo de entrada a all_states
                all_states[f"{input_id}.value"] = child['props']['children'][1]['props']['value']
                input_values[label] = float(all_states[f"{input_id}.value"])
            except KeyError:
                print(f"Error: No se encontró la clave '{input_id}.value' en dash.callback_context.states")
                print("Valores de entrada:", input_values)
                print("Claves presentes en dash.callback_context.states:", dash.callback_context.states.keys())

    prediction = predict_new_values(global_reg_tree, global_predictors, [input_values])
    return f"La predicción con base en los valores introducidos es: {prediction[0]:.2f}"

@callback(
    Output("download-ar", "data"),
    Input("btn-ar", "n_clicks"),
    prevent_initial_call=True,
)
def generar_arbol_svg(n_clicks):
    import graphviz
    from sklearn.tree import export_graphviz

    Elementos = export_graphviz(global_reg_tree,
                            feature_names = global_predictors,
                            filled = True,
                            rounded = True,
                            special_characters = True)
    Arbol = graphviz.Source(Elementos)
    Arbol.format = 'pdf'

    return dcc.send_file(Arbol.render(filename='TreeGraph', view=True))

@callback(
    Output("output-data", "children"),
    Input("submit-button", "n_clicks"),
    State("select-predictors", "value"),
    State("select-regressor", "value"),
    State("input-max-depth", "value"),
    State("input-min-samples-split", "value"),
    State("input-min-samples-leaf", "value"),
    State("input-test-size", "value")
)
def split_data(n_clicks, predictors, regressor, max_depth, min_samples_split, min_samples_leaf, test_size=0.2):
    global global_df
    global global_predictors
    global global_regressor

    if n_clicks is None:
        return ""

    if predictors is None or regressor is None:
        return "Por favor, seleccione las variables predictoras y la variable regresora."

    if global_df is None:
        return "No se ha cargado ningún dataset."

    global_predictors = predictors
    global_regressor = regressor
    print(global_predictors)
    print(global_regressor)
    print(global_df)
    # Resto del código

    X = np.array(global_df[global_predictors])
    global global_X 
    global global_Y
    global_X = X
    print(X)
    print(global_df[global_regressor])
    Y = np.array(global_df[global_regressor])
    global_Y = Y 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = 0, shuffle = True)
    
    if max_depth is None and min_samples_split is None and min_samples_leaf is None:
        comparison_df, reg_tree, tree_parameters, Y_Predicted = generate_decision_treeS(
            X_train, X_test, Y_train, Y_test
        )
    else:
        comparison_df, reg_tree, tree_parameters, Y_Predicted = generate_decision_tree(
            X_train, X_test, Y_train, Y_test, max_depth, min_samples_split, min_samples_leaf
        )

    global global_reg_tree 
    global_reg_tree = reg_tree
    comparison_chart = create_comparison_chart(Y_test, Y_Predicted)

    
    comparison_table = dash_table.DataTable(
        data=comparison_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in comparison_df.columns],
        style_table={'height': '300px', 'overflowX': 'auto'},
    )

    # Crea una tabla con los parámetros del árbol
    parameters_list = [
        {"parameter": key, "value": value}
        for key, values in tree_parameters.items()
        for value in (values if isinstance(values, (list, np.ndarray)) else [values])
    ]
    parameters_df = pd.DataFrame(parameters_list)
    parameters_table = dash_table.DataTable(
        data=parameters_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in parameters_df.columns],
        style_table={'overflowX': 'auto', "border": "none"},
    )

    importance_df = pd.DataFrame({'Variable': predictors, 'Importancia': tree_parameters['feature_importances']}).sort_values('Importancia', ascending=False)
    importance_table = dash_table.DataTable(
        data=importance_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in importance_df.columns],
        style_table={'overflowX': 'auto'},
    )

    tree_rules = export_text(reg_tree, feature_names=predictors)
    tree_rules_container = html.Div(
        children=[html.Pre(tree_rules)],
        style={'height': '20em', 'overflowY': 'scroll', 'border': '1px solid', 'padding': '10px'},
    )

    new_forecasts_section = html.Div(
        [
            html.H3("Generar nuevos pronósticos"),
            html.P("Introduce los valores de las variables predictoras:"),
            html.Div(id="input-form"),
            html.Button("Predecir", id="predict-button", className="mt-3"),
            html.Div(id="prediction-result", className="mt-4"),
        ],
        className="mt-4",
    )


    return html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                    html.H5("Los parámetros del árbol generado son los siguientes:"),
                                    parameters_table,
                                    html.Br(),
                                    html.H5("Se han obtenido los siguiente valores de pronóstico en el set de entrenamiento, los cuales se comparan con los valores reales:"),
                                    comparison_table,
                                    html.Br(),
                                    html.H5("A continuación se especifica la importancia numérica [0-1] de las variables predictoras en el modelo construido:"),
                                    importance_table,

                            ],
                            label="Parámetros del Árbol de Decisión", tab_id="tab-1", tab_style={"width": "auto"}),

                        dbc.Tab(
                            children=[
                                html.H5("El árbol fue construido de con las siguientes reglas:"),
                                tree_rules_container,
                                html.Br(),
                                html.H5("A continuación, puede descargar el árbol generado con el fin de identificar si es necesario llevar a cabo un proceso de podado. Para esto, puede modificar los parámetros de generación según sea necesario."),
                                html.Br(),
                                html.Div([
                                    dbc.Row(
                                        dbc.Col(
                                            html.Div([
                                                dbc.Button("Descargar Árbol", id="btn-ar", color="primary", className="mt-3", style={"display": "grid", "height": "80%", "align-items": "center", "margin": "0 auto"}),
                                                dcc.Download(id="download-ar")
                                            ]),
                                            width={"size": 2, "offset": 5},
                                        ),
                                        className="mt-3",
                                    ),
                                ]),

                            ],
                            label="Reglas del árbol y Gráfica", tab_id="tab-2", tab_style={"width": "auto"}),


                        dbc.Tab(
                            children=[

                                    html.H5("El siguiente gráfico permite comparar los valores estimados por el árbol de decisión contra los valores reales de prueba:"),
                                    dcc.Graph(figure=comparison_chart),

                            ],
                            label="Comparación entre Valores reales y Predecidos", tab_id="tab-3", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[

                                    new_forecasts_section

                            ],
                            label="Nuevos Pronósticos", tab_id="tab-4", tab_style={"width": "auto"}

                        ),

                    ],
                    id="tabs",
                    active_tab="tab-1",
                    style={"margin-top": "45px"}
                ),
            ],
        )