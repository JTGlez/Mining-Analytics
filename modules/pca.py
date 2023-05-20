# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Análisis de Componentes Principales

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
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'Black',
    'color': 'white',
    'padding': '6px'
}

#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def pca_card():
    """
    :retorna: Un div que contiene la explicación del módulo de EDA.

    """

    return html.Div(

        # ID del div.
        id="pca-card",

        # Elementos hijos del div 'pca-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Análisis de Componentes Principales."), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children=
                [
                    html.P("El análisis de componentes principales (ACP) es una técnica de reducción de la dimensionalidad que se utiliza para identificar patrones y estructuras en datos multivariados. Esto significa que nos permite resumir una gran cantidad de información en unas pocas dimensiones, manteniendo la mayor cantidad posible de la varianza original de los datos."),
                    html.P("Para calcular los componentes principales se sigue el siguiente procedimiento:"),
                ],
            ),
            # Imagen PCA
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                children=[
                    html.Img(
                        id="pca-img",
                        src="/assets/pca.jpg",
                        style = {'width': '100%', 'height': '100%'}
                    )
                ]
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando uno de los datasets de prueba, o bien, cargando tu propio dataset.",
                className="mb-4"
            ),

            # Muestra una figura de exploración (GIF de lupa)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                children=[
                    html.Img(
                        id="pca",
                        src="https://matthewdeakos.me/wp-content/uploads/2018/02/ezgif.com-crop-4.gif",
                        style = {'width': '50%', 'height': '50%'},
                        className="mb-4"
                    ),
                ],
            ),
        ],

    )

#Contenedor principal de la página en un Div
pca.layout = html.Div(
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
                    children=[pca_card()],
                ),
                #Columa derecha: para los resultados
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div(
                        [
                            html.H4("Carga o elige el dataset para iniciar el Análisis Exploratorio de Datos", className="text-upload"),
                            # Muestra el módulo de carga
                            dcc.Upload(
                                id="upload-data",
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
                        html.Div(id='output-data-upload-pca'),
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
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('There was an error processing this file.', color="danger")
        ])

    return html.Div([
        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Solo mostramos las primeras 5 filas del dataframe, y le damos estilo para que las columnas se vean bien
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=False,
            cell_selectable=True,
            editable=False,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":False} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        dbc.Alert('Variables numéricas: {}'.format(df.select_dtypes(include='number').shape[1]), color="info", class_name="my-3 mx-auto text-center w-25"),

        html.H3(
            "Evidencia de datos correlacionados"
        ),

        dcc.Graph(
            id='matriz',
            figure={
                'data': [
                    {'x': df.corr(numeric_only=True).columns, 'y': df.corr(numeric_only=True).columns, 'z': np.triu(df.corr(numeric_only=True).values, k=1), 'type': 'heatmap', 'colorscale': 'RdBu', 'symmetric': False}
                ],
                'layout': {
                    'title': 'Matriz de correlación',
                    'xaxis': {'side': 'down'},
                    'yaxis': {'side': 'left'},
                    # Agregamos el valor de correlación por en cada celda (text_auto = True)
                    'annotations': [
                        dict(
                            x=df.corr(numeric_only=True).columns[i],
                            y=df.corr(numeric_only=True).columns[j],
                            text=str(round(df.corr(numeric_only=True).values[i][j], 4)),
                            showarrow=False,
                            font=dict(
                                color='white' if abs(df.corr(numeric_only=True).values[i][j]) >= 0.67  else 'black'
                            ),
                        ) for i in range(len(df.corr(numeric_only=True).columns)) for j in range(i)
                    ],
                },
            },
        ),
        html.Div(
            children=[
                dbc.Badge(
                    "Identificación de correlaciones",
                    pill=True,
                    color="primary",
                    style={"font-size":"15px"}
                ),
                html.P("🔴 Correlación fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                html.P("🟡 Correlación moderada: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                html.P("🔵 Correlación débil: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                dbc.Alert("ⓘ Si no se identifica almenos una correlación fuerte, entonces PCA no aplica.", color="warning"),
            ],
            className="mt-3"
        ),
        html.H3(
            "Cálculo de Componentes Principales"
        ),
        html.P(
            "Una vez que haz identificado correlaciones entre pares de variables el siguiente paso es configurar los parámetros que se muestran a continuación, los cuales son necesarios para que el algoritmo funcione."
        ),
        html.P(
            "Al terminar, presiona sobre el botón rojo para observar los resultados."
        ),
        dbc.Alert(
            "ⓘ Si tienes alguna duda posicionate sobre cada parámetro para más información.", color="secondary", style={"font-size": "10px","width":"41%" }
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
                                            dbc.Badge("ⓘ Método de  Estandarización", color="primary",
                                            id="tooltip-method", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"},
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Estandarización:**  
                                                        Consiste en escalar o normalizar el rango de las variables iniciales, para que cada una de éstas contribuya por igual en el análisis.

                                                        Selecciona alguno de los dos métodos:
                                                        - Escalamiento usando **StandardScaler( )**: sigue la distribución normal estándar, por lo que hace la media = 0 y escala los datos a la varianza unitaria.
                                                        - Normalización usando **MinMaxScaler( )**:  transforma las características de un conjunto de datos para que estén en un rango específico entre un valor mínimo y máximo.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-method", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Select(
                                        id='select-escale',
                                        options=[
                                            {'label': 'StandardScaler', 'value': "StandardScaler()"},
                                            {'label': 'MinMaxScaler', 'value': "MinMaxScaler()"},
                                        ],
                                        value="StandardScaler()",
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
                                            dbc.Badge("ⓘ Núm. Componentes principales", color="primary",
                                                id="tooltip-numpc", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Número de Componentes Principales:**  
                                                        El objetivo de PCA es reducir la dimensionalidad de un conjunto de datos, manteniendo al mismo tiempo la mayor cantidad posible de información.

                                                        Así que, por regla general, se suelen considerar tantas componentes como número de variables numéricas existan en el dataset. Sin embargo, siéntete en libertad de escoger tantas como desees.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-numpc", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='n_components',
                                        type='number',
                                        placeholder='None',
                                        value=None,
                                        min=1,
                                        max=df.select_dtypes(include='number').shape[1],
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
                                            dbc.Badge("ⓘ Porcentaje de Relevancia", color="primary",
                                                id="tooltip-percent", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Porcentaje de Relevancia:**  
                                                        Generalmente suele elegirse un porcentaje de relevancia que esté entre 75% y 90% de varianza acumulada, lo que se busca es perder la menor cantidad posible de información.
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
                                        id='relevancia',
                                        type='number',
                                        placeholder='None',
                                        value=0.9,
                                        min=0.75,
                                        max=0.9,
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
        html.Div(
            children=
            [
                dbc.Button(
                    "Presiona para obtener los Componentes Principales", id="pca-btn", color="danger", style={"width":"40%"},
                ),
            ],
            style={"display": "flex", "justify-content": "center"},
        ),
    ],)

@callback(Output('output-data-upload-pca', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children
