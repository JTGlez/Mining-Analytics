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
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando uno de los datasets de prueba, o bien, cargando tu propio dataset."
            ),

            # Muestra una GIF
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="https://editor.analyticsvidhya.com/uploads/15391random_forest_gif.gif",
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
                            html.H4("Carga o elige el dataset para iniciar la regresión con Bosques Aleatorios", className="text-upload"),
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
                                }
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id='ticker-input',
                                        placeholder='Ingrese el ticker aqui',
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
                                    'width':'50%',
                                    'margin': '0 auto',
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
                io.StringIO(decoded.decode('utf-8')))
            return regforest(df, filename)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return regforest(df, filename)
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
    ])

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
    if ctx.triggered[0]['prop_id'] == 'upload-data-regtree.contents':
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
                dbc.Alert('Primero escribe un Ticker, por ejemplo: "AAPL" (Apple), "MSFT" (Microsoft), "GOOGL" (Google), etc. ', color="danger")
            ])
