# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Bosques Aleatorios para regresión.

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
import plotly.express as px
import plotly.graph_objects as go
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
# Bibliotecas adicionales para Bosques Aleatorios
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf # Para descargar un dataframe a partir de un ticker
from sklearn.tree import export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import uuid
import graphviz

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def regforest_card():
    """
    :retorna: Un div que contiene la explicación del módulo de Bosque Aleatorio: Regresión.

    """

    return html.Div(

        # ID del div.
        id="regforest-card",

        # Elementos hijos del div "regforest-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Bosque Aleatorio: Regresión"), # Subtítulo.
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
                        src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/50/f9/ICLH_Diagram_Batch_03_27-RandomForest.component.xl.ts=1679336476850.png/content/adobe-cms/us/en/topics/random-forest/jcr:content/root/table_of_contents/body/simple_narrative/image",
                        style = {'width': '25em', 'height': '15em'}
                    )
                ]
            ),

            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando tu propio dataset o cargando los históricos de algún activo (Stock, Criptos, etc.) recuperados directamente desde Yahoo Finance."
            ),

            # Muestra una GIF
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="https://miro.medium.com/v2/resize:fit:960/1*w-b0xHDoUsCcwx4nY3x5Og.gif",
                        style = {'width': '80%', 'height': '80%'}
                    )
                ]
            ),

        ],

    )


#Contenedor principal de la página en un Div
regforest.layout = html.Div(
    id="page-content",
    children=[
        # El contenido se divide en 2 columnas: descripción | resultados
        html.Div(
            className="row",
            children=[
                #Columna izquierda: para la descripción
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[regforest_card()],
                ),
                #Columa derecha: para los resultados
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div(
                        [
                            html.H4("Carga el dataset para iniciar la regresión con Bosques Aleatorios", className="text-upload"),
                            # Muestra el módulo de carga
                            dcc.Upload(
                                id='upload-data-regforest',
                                children=html.Div(
                                    [
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ],
                                ),
                            style={
                                'font-family':'Acumin',
                                'width': '50%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '2em auto',
                                'cursor': 'pointer',
                            },
                            multiple=True,
                            accept='.csv',
                            className="drag"
                            ),
                            # Cargar dataframe de yfinance por medio de un ticker
                            html.P(
                                "O utiliza como datos de entrada los históricos de algún activo (Stocks, Criptomonedas o Index)",
                                style = {
                                    'text-align': 'center',
                                    'font-size':'18px',
                                }
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id='ticker-input',
                                        placeholder='Ingrese el ticker aquí',
                                        style={
                                            'font-size':'16px',
                                        }
                                    ),
                                    dbc.Button(
                                        'Enviar',
                                        id='submit-ticker',
                                        n_clicks=0,
                                        color='primary',
                                        style={
                                            'text-transform':'none',
                                            'font-size':'16px',
                                        }
                                    ),
                                ],
                                style={
                                    'width':'25%',
                                    'margin': '20px auto',
                                }
                            ),
                            html.Div(id = 'output-data-upload-regforest'),
                        ],
                    ),
                ),
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
            return regforest(df, filename, df.columns)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return regforest(df, filename, df.columns)
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('There was an error processing this file.', color="danger")
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

def regforest(df, filename, columns):
    """
    retorna: modelo de regresión usando un bosque aleatorio regresor regresor para la generación de pronósticos y valores siguientes en series de tiempo.

    """
    # Se hace global el dataframe
    global global_df
    global_df = df

    # Preparación de variables para su despliegue.
    fig = create_yahoo_finance_chart(df, filename)

    # Div de visualización en el layout.
    return html.Div(
        [
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

            dcc.Graph(
                id='yahoo-finance-chart',
                figure = fig
            ),

            html.H3(
                "Elección de Variables Predictoras y Dependiente",
                style={'margin-top': '30px'}
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
                                                        id="tooltip-predictoras", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%", 'font-size':'16px'},
                                                        ),
                                                dbc.Tooltip(
                                                    "Características o atributos que se utilizan como entrada para predecir o estimar el valor de la variable objetivo o variable regresora.",
                                                    target="tooltip-predictoras", style={"font-size":"10px"},
                                                ),
                                            ],
                                            style={"height": "50px", "padding": "0"},
                                        ),
                                    ),
                                    dbc.Row(
                                        dbc.Checklist(
                                            id='select-predictors',
                                            options = [{'label': col, 'value': col} for col in df.columns],
                                            style={"font-size": "14px", "display": "grid", "justify-items": "start", 'border':'1px solid #e1e1e1', 'border-radius':'5px', 'background-color':'white'}
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
                                                dbc.Badge("ⓘ Variable regresora", color="primary",
                                                        id="tooltip-regresora", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%", 'font-size':'16px'},
                                                        ),
                                                dbc.Tooltip(
                                                    "Es la variable objetivo que se intenta predecir o estimar utilizando las variables predictoras como entrada.",
                                                    target="tooltip-regresora", style={"font-size":"10px"}
                                                ),
                                            ],
                                            style={"height": "50px", "padding": "0"},
                                        ),
                                    ),
                                    dbc.Row(
                                        dbc.Checklist(
                                            id='select-regressor',
                                            options = [{'label': col, 'value': col} for col in df.columns],
                                            style={"font-size": "14px", "display": "grid", "justify-items": "start", 'border':'1px solid #e1e1e1', 'border-radius':'5px', 'background-color':'white'}
                                        ),
                                    ),
                                ],
                                class_name="me-3"
                            ),
                        ],
                        style={"justify-content": "between", "height": "100%"}
                    ),

                     html.H3(
                        "Generación del Modelo"
                    ),
                    html.P(
                        "Una vez que hayas identificado las variables predictoras y la variable objetivo, el siguiente paso consiste en configurar los parámetros necesarios para que el modelo funcione correctamente."
                    ),
                    html.P(
                        "Al terminar, presiona sobre el botón rojo para observar los resultados."
                    ),
                    dbc.Alert(
                        "ⓘ Es posible dejar vacíos los campos que controlan los parámetros de los árboles de decisión que se utilizarán en el bosque. Sin embargo, es importante tener en cuenta que esto puede aumentar el consumo de recursos y potencialmente llevar a un modelo sobreajustado.", color="warning", style={"font-size": "10px", 'margin-bottom': '0px'}
                    ),

                    # Div para los parámetros del Bosque
                    html.Div(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Profundad máxima de los árboles", color="primary",
                                                            id="tooltip-depht", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **📏 Max Depth:**  
                                                                    Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-depht", placement="left", style={"font-size":"10px"},
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
                                        class_name="me-3", style={'flex':'1 0 25%'}
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
                                                                    **✂️ Min Samples Split:**  
                                                                    Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
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
                                        class_name="me-3", style={'flex':'1 0 25%'}
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
                                                                    **🍃 Min Samples Leaf:**  
                                                                    Indica la cantidad mínima de datos que debe tener un nodo hoja.
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
                                        class_name="me-3", style={'flex':'1 0 25%'}
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
                                                                    **Tamaño de la muestra**  
                                                                    Indica el tamaño del conjunto de datos original que se utilizará para verificar el rendimiento del modelo. Por defecto se utiliza una división '80/20' en la que el 80% de los datos originales se utilizan para entrenar el modelo y el 20% restante para validarlo.
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
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),

                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Número de Estimadores", color="primary",
                                                            id="tooltip-estimators", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **🌳🌳 Número de Estimadores:**  
                                                                    Indica el número de árboles que va a tener el bosque aleatorio. Normalmente,
                                                                    cuantos más árboles es mejor, pero a partir de cierto punto deja de mejorar y se vuelve más lento.
                                                                    El valor por defecto es 100 árboles.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-estimators", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-estimators',
                                                    type='number',
                                                    value=100,
                                                    min=100,
                                                    max=200,
                                                    step=10,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),
                                ],
                                style={"justify-content": "between", "height": "100%"}
                            ),
                        ],
                        style={"font-size":"20px", "margin":"20px 0"}
                    ),
                    html.Div(
                        children=
                        [
                            dbc.Button(
                                "Generar Bosque", id="submit-button", color="danger", style={"width":"40%"},
                            ),
                        ],
                        style={"display": "flex", "justify-content": "center"},
                    ),
                    html.Div(id="output-data", style = {"margin-top": "1em"}),
                ],
            ),
        ]
    )

@callback(Output('output-data-upload-regforest', 'children'),
          [Input('upload-data-regforest', 'contents'),
           Input('submit-ticker', 'n_clicks')],
          [State('upload-data-regforest', 'filename'),
           State('upload-data-regforest', 'last_modified'),
           State('ticker-input', 'value')])
def update_output(list_of_contents, submit_ticker_clicks, list_of_names, list_of_dates, ticker):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-regforest.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'submit-ticker.n_clicks':
        if ticker:
            df = get_yahoo_finance_data(ticker)
            return regforest(df, ticker, df.columns)
        else:
            return html.Div([
                dbc.Alert('ⓘ Primero escribe un Ticker, por ejemplo: "AAPL" (Apple), "MSFT" (Microsoft), "GOOGL" (Google), etc. ', color="danger")
            ])
