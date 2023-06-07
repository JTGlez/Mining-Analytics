# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Bosques Aleatorios para regresión.

# ------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import base64
import datetime
import io
from io import BytesIO
import dash  # Biblioteca principal de Dash.
from msilib.schema import Component
# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash import dcc, html, Input, Output, callback
# Dependencias de Dash para la implementación de Callbacks.
from dash.dependencies import Input, Output, State
# Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
import dash_bootstrap_components as dbc
from modules import home, eda, pca, regtree, classtree, regforest, classforest, kmeans
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dash.exceptions as de
from sklearn.preprocessing import LabelEncoder # Label encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# ---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def kmeans_card():
    """
    :retorna: Un div que contiene la explicación del módulo de K-Means

    """

    return html.Div(

        # ID del div.
        id="kmeans-card",

        # Elementos hijos del div 'pca-card".
        children=[
            html.H5("Mining Analytics"),  # Título de página.
            html.H3("K-Means + Bosques Aleatorios"),  # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children=[
                    html.P("En este modelo se combinan dos técnicas de aprendizaje automático: la agrupación K-means y el algoritmo de clasificación Random Forests. Primero, se utiliza K-means para agrupar los datos en clústeres. Luego, se generan características adicionales basadas en la agrupación. Estas características se utilizan para entrenar un modelo de Random Forests, que es un conjunto de árboles de decisión. El modelo resultante se utiliza para hacer predicciones sobre nuevos datos. Este enfoque aprovecha la agrupación inicial proporcionada por K-means para mejorar la precisión y el rendimiento del modelo de Random Forests al capturar mejor la estructura y las relaciones de los datos.")
                ],
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children="En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando tu propio dataset.",
                className="mb-4"
            ),
            # Muestra una figura (GIF)
            html.Div(
                style={'display': 'flex', 'align-items': 'center',
                       'justify-content': 'center'},
                children=[
                    html.Img(
                        id="pca-gif",
                        src="https://miro.medium.com/v2/resize:fit:960/1*h2WdqGZD6WsNcUdwZDqsFA.gif",
                        style={'width': '80%', 'height': 'auto'},
                        className="mb-4"
                    ),
                ],
            ),
        ],

    )


# Contenedor principal de la página en un Div
kmeans.layout = html.Div(
    id="page-content",
    children=[
        # El contenido se divide en 2 columnas: descripción | resultados
        html.Div(
            className="row",
            children=[
                # Columna izquierda: para la descripción
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[kmeans_card()],
                ),
                # Columa derecha: para los resultados
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div(
                        [
                            html.H4(
                                "Carga o elige el dataset para iniciar el modelo híbrido", className="text-upload"),
                            # Muestra el módulo de carga
                            dcc.Upload(
                                id="upload-data-kmeans",
                                children=html.Div(
                                    [
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ],
                                ),
                                style={
                                    'font-family': 'Acumin',
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
                            html.Div(id='output-data-upload-kmeans'),
                        ],
                    ),
                ),
            ],
        ),
    ],
)

# Función para generar un heatmap a partir de un dataframe
def generar_figura_heatmap(df):
    corr = df.corr(numeric_only=True)
    lower_corr = np.triu(corr)

    heatmap_correlaciones = go.Figure(data=go.Heatmap(
        z=lower_corr,
        colorscale='RdBu_r',
        x=list(corr.columns),
        y=list(corr.columns),
        zmin=-1,
        zmax=1
    ))
    heatmap_correlaciones.update_layout(
        title='Matriz de correlación',
        xaxis_title='Variables',
        yaxis_title='Variables',
        height=500,
        width=800
    )
    heatmap_correlaciones.update_yaxes(tickmode='linear')

    num_variables = len(corr.columns)
    size_factor = 1.0

    if num_variables > 15:
        size_factor = 15.0 / num_variables

    for i in range(0, corr.shape[0]):
        for j in range(i+1, corr.shape[1]):
            if corr.iloc[i, j] >= 0.67 or corr.iloc[i, j] <= -0.67:
                color = 'white'
            else:
                color = 'black'
            heatmap_correlaciones.add_annotation(
                x=corr.columns[j],
                y=i,
                text=str(round(corr.iloc[i, j], 3)),
                showarrow=False,
                font=dict(color=color, size=10 * size_factor)
            )

    return heatmap_correlaciones

# ---- FUNCIÓN PARA HACER LABEL ENCODING ----
def label_encoding(dataframe):
    label_encoder = LabelEncoder()

    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    global df_encoding
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df_encoding = label_encoding(df.copy())
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df_encoding = label_encoding(df.copy())
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('There was an error processing this file.', color="danger")
        ])

    return html.Div(
        [
            dbc.Alert('El archivo cargado es: {}'.format(
                filename), color="success"),
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
                columns=[{'name': i, 'id': i, "deletable": False}
                         for i in df.columns],
                style_table={'height': '300px', 'overflowX': 'auto'},
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Alert(
                            "Número de registros: {}".format(df.shape[0]), color="secondary", style={"margin-bottom": "0"})
                    ], width=3),  # Ajusta el ancho de la columna

                    dbc.Col([
                        dbc.Alert(
                            "Número de variables: {}".format(df.shape[1]), color="secondary", style={"margin-bottom": "0"})
                    ], width=3),
                ],
                justify='center', style={"margin": "20px 0"}
            ),
            dbc.Alert("ⓘ Esta version de K-Means solo funciona con variables numéricas, se aplicará un label encoding a las siguientes variables categóricas: {}. Si despúes deseas eliminarlas, hazlo en la sección correpsondiente.".format(", ".join(df.select_dtypes(
                "object").columns.tolist())), color="danger", style={"margin": "20px 0"}) if df.select_dtypes(include='number').shape[1] != df.shape[1] else dbc.Alert("Perfecto, todas las variables son numéricas, no tendrás problemas al usar K-Means.", color="info", style={"margin": "20px 0"}),
            html.H3("Información sobre el conjunto de datos:", style={
                    "margin": "20px 0"}),
            html.P("La siguiente tabla muestra el tipo de dato de cada variable, así como los valores nulos identificados:", style={
                   "margin": "20px 0"}),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th('Variable', className="text-center"),
                                html.Th('Tipo de dato',
                                        className="text-center"),
                                html.Th('Count', className="text-center"),
                                html.Th('Valores nulos',
                                        className="text-center"),
                            ],
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(column, className="text-center"),
                                    html.Td(
                                        str(df.dtypes[column]),
                                        style={
                                            'color': 'green' if df.dtypes[column] == 'float64' else 'blue' if df.dtypes[column] == 'int64' else 'red' if df.dtypes[column] == 'object' else 'orange' if df.dtypes[column] == 'bool' else 'purple'
                                        },
                                        className="text-center"
                                    ),
                                    html.Td(
                                        [
                                            html.P("{}".format(
                                                df[column].count()))
                                        ],
                                        className="text-center"
                                    ),
                                    html.Td(
                                        df[column].isnull().sum(),
                                        style={
                                            'color': 'red' if df[column].isnull().sum() > 0 else 'green'
                                        },
                                        className="text-center"
                                    ),
                                ]
                            ) for column in df.dtypes.index
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                style={"width": "50%", "margin": "0 auto "}
            ),
            html.H3("Selección de Características", style={
                    "margin": "20px 0"}),
            html.P("Utiliza el siguiente heatmap de correlaciones entre pares de variables para identificar aquellas que presenten redundancia en la información del dataset. Estas variables redundantes se mostrarán en color blanco en el heatmap."),
            dcc.Graph(
                id='matriz',
                figure=generar_figura_heatmap(df_encoding),
                style={"display": "flex", "justify-content": "center"}
            ),
            html.Div(
                children=[
                    dbc.Badge(
                        "Identificación de correlaciones",
                        pill=True,
                        color="primary",
                        style={"font-size": "15px"}
                    ),
                    html.P(
                        "🔴 Correlación fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                    html.P(
                        "🟡 Correlación moderada: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                    html.P(
                        "🔵 Correlación débil: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                ],
                className="mt-3"
            ),
            html.H3("Eliminación de variables", style={
                    "margin": "20px 0"}),
            html.P("Según lo observado en el análisis anterior puedes eliminar las variables que consideres según tus requisitos. Además, aprovecha esta oportunidad para eliminar aquellas variables categóricas que se hayan transformado en variables numéricas y que no deseas conservar en el dataset. También puedes eliminar cualquier otra variable que consideres innecesaria para la construcción de los modelos."),
            html.Div(
                [
                    dbc.Badge(
                        "ⓘ Lista de variables", color="primary",
                        id="tooltip-var", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "50px", 'font-size': '16px'},
                    ),
                    dbc.Tooltip(
                        "Elimina variables redundantes, categóricas e innecesarias",
                        target="tooltip-var", style={"font-size": "10px"},
                    ),
                    dbc.Checklist(
                        id='delete',
                        options=[{'label': col, 'value': col}
                                 for col in df.columns],
                        value='', # Lista vacía
                        style={"font-size": "14px", "display": "grid", "justify-items": "start",
                               'border': '1px solid #e1e1e1', 'border-radius': '5px', 'background-color': 'white',
                               "padding": "10px 30px"}
                    ),
                ],
                style={"width": "40%", "margin": "0 auto"},
            ),
            dbc.Alert("ⓘ Recuerda que las variables que no elimines serán tomadas como variables predictoras en los árboles de decisión.", color="warning", style={"margin-top":"20px"}),
            html.H3("Configuración del modelo híbrido", style={"margin":"20px 0"}),
            html.P(
                "Una vez que has seleccionado las variables a eliminar, el siguiente paso es configurar los parámetros que se muestran a continuación, los cuales son necesarios para que el modelo híbrido funcione."
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
                                            id='estandarizar',
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
                                class_name="me-3", style={'flex':'1 0 47%'}
                            ),

                            dbc.Col(
                                [
                                    dbc.Row(
                                        html.Div(
                                            [
                                                dbc.Badge("ⓘ K-Clusters", color="primary",
                                                    id="tooltip-clusters", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                ),
                                                dbc.Tooltip(
                                                    [
                                                        dcc.Markdown('''
                                                            **⚙️ n_clusters:**  
                                                            En el algoritmo K-Means, el objetivo es minimizar la suma de los errores al cuadrado (SSE, por sus siglas en inglés), también conocida como inercia. La inercia representa la dispersión dentro de cada cluster.  
                                                            Cuando se ajusta el modelo K-Means con diferentes valores de "n_clusters", es común observar que la inercia disminuye a medida que aumenta el número de clusters.  
                                                            Sin embargo, existe un punto en el que añadir más clusters ya no proporciona una mejora significativa en la inercia. A partir de ese punto, la mejora es mínima o incluso nula, y puede haber un aumento en la complejidad del modelo sin un beneficio sustancial.  
                                                            Se recomienda una cantidad de clusters que esté en el rango de 10 a 20, dependiendo del tamaño del dataset.
                                                        ''', style={'text-align': 'left'}),
                                                    ],
                                                    target="tooltip-clusters", placement="left", style={"font-size":"10px"},
                                                ),
                                            ],
                                            style={"height":"50px", "padding": "0"},
                                        ),
                                    ),
                                    dbc.Row(
                                        dbc.Input(
                                            id='n_clusters',
                                            type='number',
                                            placeholder='None',
                                            value=15,
                                            min=10,
                                            step=1,
                                            max=20,
                                            style={"font-size": "medium"}
                                        ),
                                        style={"height":"50px"}
                                    ),
                                ],
                                class_name="me-3", style={'flex':'1 0 47%'}
                            ),

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
                                            id='profundidad',
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
                                            id='min-samples-separacion',
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
                                            id='min-samples-hoja',
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
                                            id='datos-prueba',
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
                                            id='estimadores',
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
                        "Obtener Resultados", id="kmeans-btn", color="danger", style={"width":"40%"},
                    ),
                ],
                style={"display": "flex", "justify-content": "center"},
            ),
            html.Div(id="output-data-kmeans")
        ]
    )


@callback(Output('output-data-upload-kmeans', 'children'),
          Input('upload-data-kmeans', 'contents'),
          State('upload-data-kmeans', 'filename'),
          State('upload-data-kmeans', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# ---- FUNCIÓN PARA ESTANDARIZAR LOS DATOS ----
def estandarizar_datos(dataf, scale):
    """
    Escala las columnas numéricas de un DataFrame según el tipo de escala indicado.

    Parámetros:
    df (pandas.DataFrame): DataFrame con las columnas a escalar.
    scale (str): Tipo de escala a aplicar. Puede ser "StandardScaler()" o "MinMaxScaler()".

    Retorna:
    pandas.DataFrame: Nuevo DataFrame con las columnas numéricas escaladas.
    """
    if scale == "StandardScaler()":
        matriz_estandarizada = StandardScaler().fit_transform(dataf)
    elif scale == "MinMaxScaler()":
        matriz_estandarizada = MinMaxScaler().fit_transform(dataf)
    return pd.DataFrame(matriz_estandarizada, columns=dataf.columns)

# ---- FUNCIÓN PARA GENERAR UNA GRÁFICA DE CODO ----
def graph_elbow(sse):
    elbow = go.Figure(data=go.Scatter(x=list(range(2, 10)), y=sse, mode='lines+markers'))
    elbow.update_layout(
        xaxis_title='Cantidad de clústeres "K"',
        yaxis_title='SSE',
        title='Gráfica de codo',
        height=500,
        width=800
    )
    return elbow

# ---- FUNCIÓN PARA GRAFICAR EL PUNTO DEL CODO ----
def knee_localization(sse, kaele, num_clusters):
    knee = go.Figure(data=go.Scatter(x=list(range(2, num_clusters)), y=sse, mode='lines+markers'))
    knee.add_vline(x=kaele.elbow, line_width=3, line_dash='dash', line_color='red')
    knee.add_annotation(x=kaele.elbow, y=kaele.knee_y, text='Knee Point', showarrow=True, arrowhead=1)
    knee.update_layout(
        xaxis_title='Cantidad de clústeres "K"',
        yaxis_title='SSE',
        title='Ubicación del codo',
        height=500,
        width=800
    )
    return knee

# ---- FUNCIÓN PARA GRAFICAR LOS CLUSTERS EN 3D ----
def graph_3d_clusters(mat_standarized, clusters, num_clusters, df_ori):
    mat_standarized_np = mat_standarized.values
    colores = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'indigo', 'lavender', 'turquoise', 'gold', 'silver', 'teal', 'coral', 'olive']
    asignar = []

    for row in clusters.labels_:
        asignar.append(colores[row])

    clusters_3d = go.Figure(data=[go.Scatter3d(
        x=mat_standarized_np[:, 0],
        y=mat_standarized_np[:, 1],
        z=mat_standarized_np[:, 2],
        mode='markers',
        marker=dict(color=asignar, size=6, line=dict(color='black', width=12)),
        text=df_ori['cluster_partition']
    )])

    # Genera las etiquetas de los centroides de los clusters
    centroid_labels = [f'Cluster {i}' for i in range(num_clusters)]

    clusters_3d.add_trace(go.Scatter3d(
        x=clusters.cluster_centers_[:, 0],
        y=clusters.cluster_centers_[:, 1],
        z=clusters.cluster_centers_[:, 2],
        mode='markers',
        marker=dict(color=colores[:num_clusters], size=12, line=dict(color='black', width=12)),
        text=centroid_labels
    ))

    clusters_3d.update_layout(title='Clusters generados',showlegend=False, width=800, height=500)

    return clusters_3d

def kmeans_algorithm(df_estandarizada, num_clusters, df_origin):
     # Definición de k clusters pra K-means
    SSE = []
    for i in range(2, num_clusters):
        km = KMeans(n_clusters=i, n_init=10, random_state=0)
        km.fit(df_estandarizada)
        SSE.append(km.inertia_)

    # Gráfica de codo
    elbow_graph = graph_elbow(SSE)
    
    # Localización del codo
    kl = KneeLocator(range(2, num_clusters), SSE, curve="convex", direction="decreasing")

    # Gráfica Elbow point
    elbow_point_graph = knee_localization(SSE, kl, num_clusters)

    # Se crean las etiquetas de los elementos en los clusters
    cluster_partitional = KMeans(n_clusters=kl.elbow, n_init=num_clusters, random_state=0).fit(df_estandarizada)
    cluster_partitional.predict(df_estandarizada)
    df_origin['cluster_partition'] = cluster_partitional.labels_

    # Obtencion de los centroides
    centroides = df_origin.groupby('cluster_partition').mean()
    centroides['Núm. Elementos'] = df_origin.groupby('cluster_partition').size()

    #Gráfica 3D
    graph_3d = graph_3d_clusters(df_estandarizada, cluster_partitional, num_clusters, df_origin)

    return elbow_graph, elbow_point_graph, centroides, graph_3d, kl

def importance_bar(vars, modelo):
    importance = pd.DataFrame({'Variable':vars, 'Importancia': modelo.feature_importances_}).sort_values('Importancia', ascending=False)
    
    graph_importance = go.Figure(data=[
        go.Bar(
            x=importance['Variable'],
            y=importance['Importancia'],
            marker=dict(
                color=importance['Importancia'],
                colorscale='Bluered'
            ),
            text=importance['Importancia'],
            texttemplate='%{text:.2}',
            textposition='outside'
        )
    ])

    graph_importance.update_layout(
        title='Importancia de las variables',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Importancia'),
        uniformtext=dict(minsize=8, mode='hide'),
        legend_title='Importancia de las variables',
        width=600,
        height=400
    )
    return graph_importance

def auroc_curve(mod, kal, xval, yval):
    clusters = np.arange(kal.elbow)
    Y_score = mod.predict_proba(xval)
    Y_test_bin = label_binarize(yval, classes=clusters)
    n_classes = Y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()

    auroc = go.Figure()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], Y_score[:, i])
        auroc.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name='Clase {}'.format(i+1)+', AUC: {}'.format(auc(fpr[i], tpr[i]).round(6))))
    auroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Referencia', line=dict(color='navy', dash='dash')))
    auroc.update_layout(title_text='Rendimiento', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=800, height=500)
    
    return auroc

# CALLBACK PARA MOSTRAR LOS RESULTADOS
@callback(
    Output("output-data-kmeans", "children"),
    Input("kmeans-btn", "n_clicks"),
    State("delete", "value"),
    State("estandarizar", "value"),
    State("n_clusters", "value"),
    State("profundidad", "value"),
    State("min-samples-separacion", "value"),
    State("min-samples-hoja", "value"),
    State("datos-prueba", "value"),
    State("estimadores", "value")
)
def create_model(n_clicks, eliminar, method_standarization, k_clusters, max_prof, samples_split, samples_leaf, tam_size, estimators):
    global df_original
    global classification_rforest
    if n_clicks is not None:
        # ---- ELIMINACIÓN DE VARIABLES Y LABEL ENCODING ----
        if len(eliminar) != 0:
            df_drop = df_encoding.drop(eliminar, axis=1)
            df_original = df.drop(eliminar, axis=1)
            df_original = label_encoding(df_original)
        else:
            df_drop = df_encoding
            df_original = label_encoding(df)
        
        # ---- ESTANDARIZACIÓN ----
        matriz_estandarizada = estandarizar_datos(df_drop, method_standarization)
        # ---- K-Means ALGORITHM ----
        grafico_codo, grafico_punto_codo, centroids, tridi_graph, kl = kmeans_algorithm(matriz_estandarizada, k_clusters, df_original)
        # ---- Clasificación por Bosques ALGORITHM ----
        # Predictoras:
        X_train = df_original.drop('cluster_partition', axis=1)
        X = np.array(X_train)
        # Regresora:
        Y_train = df_original[['cluster_partition']]
        Y = np.array(Y_train)
        # Conjuntos de entrenamiento y validación
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=tam_size, random_state=0, shuffle=True)
        # Creación del modelo dependiendo si se pasan valores o no
        if max_prof is None and samples_split is None and samples_leaf is None:
            classification_rforest = RandomForestClassifier(random_state=0)
        else:
            classification_rforest = RandomForestClassifier(n_estimators=estimators,
                                                    max_depth=max_prof,
                                                    min_samples_split=samples_split,
                                                    min_samples_leaf=samples_leaf,
                                                    random_state=0)
        classification_rforest.fit(X_train, Y_train)
        Y_predict = classification_rforest.predict(X_validation)
        # Comparación
        compare = pd.DataFrame({'Y_Real:':Y_validation.flatten(), 'Y_Pronosticado':Y_predict})
        compare.columns = compare.columns.astype(str)
        compare_table = dash_table.DataTable(
            data=compare.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in compare.columns.astype(str)],
            style_table={'height': '300px', 'overflowX': 'auto'},
        )
        # Matriz de clasificación
        matrix = pd.crosstab(Y_validation.ravel(), Y_predict, rownames=['Reales'], colnames=['Clasificación'])
        matrix.index.set_names('Reales', inplace=True)
        matrix.columns.set_names('Clasificación', inplace=True)
        matrix = matrix.reset_index()
        matrix_table = dash_table.DataTable(
        data=matrix.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in matrix.columns.astype(str)],
        style_table={'overflowX': 'auto'},
    )

        # Parametros del modelo de clasificacion
        report = classification_report(Y_validation, Y_predict, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        report_df = report_df.reset_index().rename(columns={'index':'Metric'})
        report_table = dash_table.DataTable(
            data=report_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in report_df.columns.astype(str)],
            style_table={'overflowX': 'auto', "border": "none"},
        )
        parameters = {
            'criterion': classification_rforest.criterion,
            'feature_importances': classification_rforest.feature_importances_,
            'Exactitud': accuracy_score(Y_validation, Y_predict)
        }
        parameters_list = [
            {"parameter": key, "value": value}
            for key, values in parameters.items()
            for value in (values if isinstance(values, (list, np.ndarray)) else [values])
        ]
        parameters_df = pd.DataFrame(parameters_list)
        parameters_table = dash_table.DataTable(
            data=parameters_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in parameters_df.columns.astype(str)],
            style_table={'overflowX': 'auto', "border": "none"},
        )
        # Gráfico de barras con la importancia de cada variable
        bar = importance_bar(df_original.drop('cluster_partition', axis=1).columns, classification_rforest)
        # AUROC
        auroc_graph = auroc_curve(classification_rforest, kl, X_validation, Y_validation)
        # New predictions
        new_predictions = html.Div(
            [
                html.H3("Generar nuevos pronósticos"),
                html.P("Introduce los valores de las variables predictoras:"),
                html.Div(id="input-form-class-kmeans"),
                html.Button("Clasificar", id="predict-button-kmeans", className="mt-3"),
                html.Div(id="classification-result-kmeans", className="mt-4"),
            ],
            className="mt-4",
        )
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate

    return html.Div(
            [
                html.H3("K-Means Clusters"),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                    dash_table.DataTable(
                                        data=matriz_estandarizada.to_dict('records'),
                                        page_size=8,
                                        sort_action='native',
                                        column_selectable='single',
                                        row_deletable=False,
                                        cell_selectable=True,
                                        editable=False,
                                        row_selectable='multi',
                                        columns=[{'name': i, 'id': i, "deletable": False} for i in matriz_estandarizada.columns],
                                        style_table={'height': '300px', 'overflowX': 'auto'},
                                    )
                            ],
                            label="Matriz Estandarizada", tab_id="tab-1", tab_style={"width": "auto"}
                        ),
                        
                        dbc.Tab(
                            children=[
                                    dcc.Graph(figure=grafico_codo),
                                    dbc.Alert("ⓘ Número óptimo de clusters: {}".format(kl.elbow), style={"font-size":"20px", "font-weight":"bold"}),
                                    html.P("Se utilizó la clase KneeLocator de la biblioteca kneed para encontrar el número óptimo de clusters, es decir, el punto a partir del cual añadir más clusters ya no proporciona una mejora significativa en la inercia."),
                                    dcc.Graph(figure=grafico_punto_codo),
                            ],
                            label="Elbow Method", tab_id="tab-2", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[
                                    html.H3("Resumen estadístico de cada clúster"),
                                    html.P("La siguiente tabla muestra un resumen de las características que identifican a los elementos que pertenecen a cada tipo de Cluster.", style={"margin":"20px 0"}),
                                    dbc.Alert("ⓘ Recomendamos que analize a fondo esta tabla para que pueda identificar nuevos campos de interés en sus datos."),
                                    dash_table.DataTable(
                                        data=centroids.reset_index().to_dict('records'),
                                        page_size=8,
                                        sort_action='native',
                                        column_selectable='single',
                                        row_deletable=False,
                                        cell_selectable=True,
                                        editable=False,
                                        row_selectable='multi',
                                        columns=[{'name': i, 'id': i, "deletable": False} for i in centroids.reset_index().columns],
                                        style_table={'height': '300px', 'overflowX': 'auto'},
                                    )
                            ],
                            label="Centroides", tab_id="tab-4", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[
                                    dcc.Graph(figure=tridi_graph)
                            ],
                            label="Gráfico 3D", tab_id="tab-5", tab_style={"width": "auto"}
                        ),

                    ],
                    id="tabs",
                    active_tab="tab-1",
                ),

                html.H3("Clasificación con Bosques Aleatorios", style={"padding-top":"20px", "border-top":"solid 2px black", "margin-top":"30px"}),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                    # Parámetros
                                    dbc.Alert('ⓘ Parámetros del árbol:', style={"font-size":"20px", "font-weight":"bold"}),
                                    html.P("Se muestra la Exactitud del modelo, así como una gráfica con la importancia de cada variable para el modelo."),
                                    parameters_table,
                                    # Gráfica de importancia
                                    dcc.Graph(figure=bar),
                                    # Reporte completo
                                    dbc.Alert('ⓘ Reporte de la clasificación:', style={"font-size":"20px", "font-weight":"bold"}),
                                    html.P("Se muestra la siguiente información: "),
                                    html.Div(
                                        [
                                            dbc.Badge("Precisión: ", pill=True, color="primary", style={"width":"10%", "height":"fit-content"}),
                                            html.P(" Proporción de instancias clasificadas correctamente como positivas entre todas las instancias clasificadas como positivas, es decir, mide la exactitud del modelo al predecir los positivos verdaderos.", style={"line-height":"initial", "width":"90%"}),
                                        ],
                                        style={"display":"flex", "column-gap":"10px", "margin-top":"20px"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Badge("Recall: ", pill=True, color="success", style={"width":"10%", "height":"fit-content"}),
                                            html.P("Mide la capacidad del modelo para identificar correctamente los casos positivos. Un valor alto de recall indica que hay pocos falsos negativos.", style={"line-height":"initial", "width":"90%"}),
                                        ],
                                        style={"display":"flex", "column-gap":"10px", "margin-top":"20px"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Badge("F1-score: ", pill=True, color="warning", style={"width":"10%", "height":"fit-content"}),
                                            html.P("Medida que combina la precisión y el recall en un solo valor. Es útil cuando hay un desequilibrio entre las clases y deseamos tener una métrica única para evaluar el rendimiento general del modelo.", style={"line-height":"initial", "width":"90%"}),
                                        ],
                                        style={"display":"flex", "column-gap":"10px", "margin-top":"20px"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Badge("Support: ", pill=True, color="danger", style={"width":"10%", "height":"fit-content"}),
                                            html.P("Indica el número de instancias en cada clase del conjunto de datos.", style={"line-height":"initial", "width":"90%"}),
                                        ],
                                        style={"display":"flex", "column-gap":"10px", "margin":"20px 0"},
                                    ),
                                    report_table,
                            ],
                            label="Calidad de la predicción", tab_id="tab-1", tab_style={"width": "auto"}
                        ),
                        
                        dbc.Tab(
                            children=[
                                    # Comparación
                                    dbc.Alert('ⓘ Comparción:', style={"font-size":"20px", "font-weight":"bold"}),
                                    html.P("Vistazo rápido a la precisión del modelo, se comparan los valores reales contra los valores pronosticados"),
                                    compare_table,
                                    # Matriz de clasificación
                                    dbc.Alert('ⓘ Matriz de clasificación:', style={"font-size":"20px", "font-weight":"bold"}),
                                    html.P("La matriz de clasificación obtenida permite identificar en qué observaciones el modelo acertó y en cuáles falló de manera más precisa."),
                                    matrix_table,
                            ],
                            label="Clasificación", tab_id="tab-2", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[
                                    dcc.Graph(figure=auroc_graph)
                            ],
                            label="Curva ROC", tab_id="tab-3", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[
                                new_predictions,
                            ],
                            label='Nuevos Pronósticos',
                            tab_id='tab-4', tab_style={'width':'auto'}
                        ),

                    ],
                    id="tabs",
                    active_tab="tab-1",
                ),
            ],
        )

# CREACIÓN DE INPUTS
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

# CALLBACK PARA CREAR INPUTS
@callback(Output("input-form-class-kmeans", "children"), Input("kmeans-btn", "n_clicks"))
def update_input_form(n_clicks):
    if n_clicks is None:
        return ""
    return create_input_form(df_original.drop('cluster_partition', axis=1).columns)

def predict_new_values(class_tree, predictors, input_values):
    input_data = pd.DataFrame(input_values, columns=predictors)
    prediction = class_tree.predict(input_data)
    return prediction

# CALLBACK PARA MOSTRAR LAS NUEVAS PREDICCIONES
@callback(
    Output("classification-result-kmeans", "children"),
    Input("predict-button-kmeans", "n_clicks"),
    State("input-form-class-kmeans", "children"),
)
def show_prediction(n_clicks, input_form):
    if n_clicks is None or input_form is None:
        return ""

    input_values = {}
    all_states = dash.callback_context.states
    for child in input_form:
        label = child['props']['children'][0]['props']['children']
        if label in df_original.drop('cluster_partition', axis=1).columns:
            input_id = child['props']['children'][1]['props']['id']
            try:
                # Agrega el id del campo de entrada a all_states
                all_states[f"{input_id}.value"] = child['props']['children'][1]['props']['value']
                input_values[label] = float(all_states[f"{input_id}.value"])
            except KeyError:
                print(f"Error: No se encontró la clave '{input_id}.value' en dash.callback_context.states")
                print("Valores de entrada:", input_values)
                print("Claves presentes en dash.callback_context.states:", dash.callback_context.states.keys())

    prediction = predict_new_values(classification_rforest, df_original.drop('cluster_partition', axis=1).columns, [input_values])
    return dbc.Alert(f"La clasificación con base en los valores introducidos es: {prediction[0]}", color='info')