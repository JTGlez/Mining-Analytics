# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: página principal de la aplicación.


#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import dash # Biblioteca principal de Dash.
from dash import dcc, html, Input, Output, callback # Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, ClientsideFunction # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from modules import home, eda, pca
import pathlib



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
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Analítica de Datos inteligente"), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="La Minería de Datos es una disciplina que se enfoca en descubrir patrones y relaciones ocultas dentro de grandes conjuntos de datos."+
                """ A través de técnicas avanzadas de análisis y procesamiento de datos, la Minería de Datos permite extraer información valiosa y conocimientos útiles"""+
                """ que pueden ser utilizados para mejorar la toma de decisiones y optimizar el rendimiento en una amplia gama de sectores y aplicaciones."""
                ,
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta aplicación podrás visualizar todo el proceso involucrado en un proyecto de Minería de Datos enfocado al sector tecnológico y financiero."
            ),

            # Muestra una figura de CRISP-DM.
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '25em'},
                children=[
                    html.Img(
                        id="crisp",
                        src="/assets/crispdm.PNG",
                        style = {'width': '20em'}
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
                src="https://strategicfocus.com/wp-content/uploads/2019/12/Google_Cloud_DataAnalytics.gif",
                top=True,
                style={"height": "100px", "object-fit": "cover", "object-position": "center"},
            ),
            dbc.CardBody(
                [
                    html.H4("Análisis Exploratorio de Datos (EDA)", className="card-title"),
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
    class_name="mb-4 card",
)

pca_card = dbc.Card(
    dbc.CardLink(
        [
            dbc.CardImg(
                src="https://miro.medium.com/v2/resize:fit:1400/1*37a_i1t1tDxDYT3ZI6Yn8w.gif",
                top=True,
                style={"width": "100%", "height": "100px", "object-fit": "cover", "object-position": "center"},
            ),
            dbc.CardBody(
                [
                    html.H4("Análisis de Componentes Principales (PCA)", className="card-title"),
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
    class_name="mb-4 card",
)

forest_card = dbc.Card(
    [
        dbc.CardImg(
            src = "",
            top = True,
            style = {}
        ),
        dbc.CardBody(
            [

            ]
        )

    ],
)

cluster_card = dbc.Card(
    [
        dbc.CardImg(
            src = "",
            top = True,
            style = {}
        ),
        dbc.CardBody(
            [

            ]
        )

    ],
)




# Contenedor principal de la página en un Div.
layout = html.Div(
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
                    children=[description_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id="right-column",
                    className="eight columns",
                    children=[
                        # Muestra los módulos disponibles.
                        eda_card,
                        pca_card
                    ],
                ),
            ],
        ),
    ],
)
