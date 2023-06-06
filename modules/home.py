# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: página principal de la aplicación.


#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import dash  # Biblioteca principal de Dash.
# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash import dcc, html, Input, Output, callback
# Dependencias de Dash para la implementación de Callbacks.
from dash.dependencies import Input, Output, ClientsideFunction
# Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
import dash_bootstrap_components as dbc
from modules import home, eda, pca, regtree, classtree, regforest, classforest
import pathlib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def description_card():
    """
    :retorna: Un div que contiene el título de la GUI y descripciones de la página principal, posicionado a la izquierda.

    """

    return html.Div(
        # ID del div.
        id="description-card",

        # Elementos hijos del div "description-card".
        children=[
            html.H5("Mining Analytics"),  # Título de página.
            html.H3("Analítica de Datos inteligente"),  # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="La Minería de Datos es una disciplina que se enfoca en descubrir patrones y relaciones ocultas dentro de grandes conjuntos de datos." +
                """ A través de técnicas avanzadas de análisis y procesamiento de datos, la Minería de Datos permite extraer información valiosa y conocimientos útiles""" +
                """ que pueden ser utilizados para mejorar la toma de decisiones y optimizar el rendimiento en una amplia gama de sectores y aplicaciones.""",
            ),

            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children="En esta aplicación podrás visualizar distintos procedimientos y algoritmos involucrados en un proyecto de Minería de Datos.",
                style = {'text-align': 'justify'}
            ),

            # Muestra un gif en la página principal.
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '25em'},
                children=[
                    html.Img(
                        id="crisp",
                        src="/assets/webscrap.gif",
                        style={'width': '20em'}
                    )
                ]
            ),

        ],

    )


# Aquí se añaden cardboards informativos de cada módulo.
eda_card = dbc.Card(
    dbc.CardLink(
        [
            dbc.CardImg(
                top=True,
                style={
                    "background-image": "url('https://strategicfocus.com/wp-content/uploads/2019/12/Google_Cloud_DataAnalytics.gif')",
                    "background-repeat": "no-repeat",
                    "background-position": "50% 33%",
                    "background-size": "cover",
                    "height": "150px",
                    "width": "100%"
                },
            ),
            dbc.CardBody(
                [
                    html.H4("Análisis Exploratorio de Datos (EDA)",
                            className="card-title"),
                    html.P(
                        "El análisis exploratorio de datos es una técnica de análisis de datos que se utiliza para examinar y resumir las características principales de un conjunto de datos con el fin de obtener una comprensión inicial de los mismos.",
                        className="card-text",
                    ),
                ]
            ),
        ],
        href="\eda",
        style={"text-decoration": "none", "color": "black"}
    ),
    class_name="mb-4 card-anim",
)

pca_card = dbc.Card(
    dbc.CardLink(
        [
            dbc.CardImg(
                top=True,
                style={
                    "background-image": "url('https://miro.medium.com/v2/resize:fit:1400/1*37a_i1t1tDxDYT3ZI6Yn8w.gif')",
                    "background-repeat": "no-repeat",
                    "background-position": "50% 45%",
                    "background-size": "cover",
                    "height": "150px",
                    "width": "100%"
                },
            ),
            dbc.CardBody(
                [
                    html.H4("Análisis de Componentes Principales (PCA)",
                            className="card-title"),
                    html.P(
                        "El análisis de componentes principales es una técnica estadística utilizada para reducir la dimensionalidad de un conjunto de datos mediante la identificación de las variables más importantes y la creación de nuevas variables que expliquen la mayor variabilidad posible en los datos.",
                        className="card-text",
                    ),
                ]
            ),
        ],
        href="\pca",
        style={"text-decoration": "none", "color": "black"}
    ),
    class_name="mb-4 card-anim",
)

dtree_card = dbc.Card(
    [
        dbc.CardImg(
            top=True,
            style={
                "background-image": "url('https://cdn-images-1.medium.com/v2/resize:fill:1600:480/gravity:fp:0.5:0.4/1*7cyzrfuh9hKqz2lZxi_8ug.gif')",
                "background-repeat": "no-repeat",
                "background-position": "50% 74%",
                "background-size": "cover",
                "height": "150px",
                "width": "100%"
            },
        ),
        dbc.CardBody(
            [
                html.H4("Árboles de Decisión", className="card-title"),
                html.P(
                    "Un árbol de decisión es un algoritmo de aprendizaje automático utilizado tanto para problemas de regresión como de clasificación. Es una representación gráfica de un conjunto de reglas de decisión que se utilizan para tomar decisiones o predecir un valor objetivo en función de las características o atributos de los datos de entrada.",
                    className="card-text",
                ),
                html.Div(
                    [
                        dbc.Button("Regresión", color="primary", style={"width": "25%", "height": "40px", "display": "flex", "flex-wrap": "wrap", "align-content": "center", "justify-content": "center", "font-size": "16px"}, href="\\regtree"),
                        dbc.Button("Clasificación", color="danger", style={"width": "25%", "height": "40px", "display": "flex", "flex-wrap": "wrap", "align-content": "center", "justify-content": "center", "font-size": "16px"}, href="\classtree")
                    ],
                    style={"display": "flex", "justify-content": "space-evenly"}
                )
            ]
        )
    ],
    class_name="mb-4"
)

rforest_card = dbc.Card(
    [
        dbc.CardImg(
            top=True,
            style={
                "background-image": "url('https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif')",
                "background-repeat": "no-repeat",
                "background-position": "50% 40%",
                "background-size": "cover",
                "height": "150px",
                "width": "100%"
            },
        ),
        dbc.CardBody(
            [
                html.H4("Bosque Aleatorio", className="card-title"),
                html.P(
                    "Un bosque aleatorio es un algoritmo de aprendizaje automático que combina múltiples árboles de decisión para realizar predicciones más precisas. Cada árbol individual en el bosque se construye a partir de un subconjunto aleatorio de datos y utiliza un enfoque llamado 'ensamble' para combinar las predicciones de todos los árboles.",
                    className="card-text",
                ),
                html.Div(
                    [
                        dbc.Button("Regresión", color="primary", style={"width": "25%", "height": "40px", "display": "flex", "flex-wrap": "wrap", "align-content": "center", "justify-content": "center", "font-size": "16px"}, href="\\regforest"),
                        dbc.Button("Clasificación", color="danger", style={"width": "25%", "height": "40px", "display": "flex", "flex-wrap": "wrap", "align-content": "center", "justify-content": "center", "font-size": "16px"}, href="\classforest")
                    ],
                    style={"display": "flex", "justify-content": "space-evenly"}
                )
            ]
        )
    ],
    class_name="mb-4"
)

kmeans_card = dbc.Card(
    dbc.CardLink(
        [
            dbc.CardImg(
                top=True,
                style={
                    "background-image": "url('https://miro.medium.com/v2/resize:fit:1400/1*Ht06cKFv9S9XWCsnR2kSpw.gif')",
                    "background-repeat": "no-repeat",
                    "background-position": "50% 50%",
                    "background-size": "cover",
                    "height": "150px",
                    "width": "100%"
                },
            ),
            dbc.CardBody(
                [
                    html.H4("K-Means + Bosques Aleatorios",
                            className="card-title"),
                    html.P(
                        "Este modelo híbrido combina dos técnicas de aprendizaje automático: el algoritmo de agrupamiento K-means y el algoritmo de clasificación Random Forests. El objetivo de este enfoque es utilizar la agrupación inicial de los datos realizada por K-means para mejorar la precisión y el rendimiento del modelo de Random Forests.",
                        className="card-text",
                    ),
                ]
            ),
        ],
        href="\kmeans",
        style={"text-decoration": "none", "color": "black"}
    ),
    class_name="mb-4 card-anim",
)


# Contenedor principal de la página en un Div.
layout = html.Div(
    id="page-content",
    children=[

        # Contenido principal de la aplicación: se divide en 2 columnas: una con contenido explicativo y otra para elementos interactivos.
        html.Div(

            className="row",
            children=[

                # Columna a la izquierda: invoca a description_card para mostrar el texto explicativo de la izquierda.
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[description_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id="right-column",
                    className="eight columns",
                    children=[
                        # Muestra los módulos disponibles.
                        eda_card,
                        pca_card,
                        dtree_card,
                        rforest_card,
                        kmeans_card
                    ],
                ),
            ],
        ),
    ],
)
