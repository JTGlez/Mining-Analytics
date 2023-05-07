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
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '35em'},
                children=[        
                    html.Img(            
                        id="crisp",            
                        src="/assets/crispdm.PNG",
                        style = {'width': '25em'}
                    )    
                ]
            ),

        ],

    )

# Aquí se añaden cardboards informativos de cada módulo.
eda_card = dbc.Card(
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

pca_card = dbc.Card(
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
                        html.Div(
                            id="exploration_modules",
                            children=[
                                html.B("Módulos de exploración y reducción de dimensionalidad"),
                                html.Hr(),
                                dbc.CardGroup([eda_card, pca_card])
                            ],
                        ),

                         html.Div(
                            id="ml_modules",
                            children=[
                                html.B("Módulos de Machine Learning"),
                                html.Hr(),
                                dbc.CardGroup([forest_card, cluster_card])
                            ],
                        ),

                    ],
                ),
            ],
        ),
    ],
)
