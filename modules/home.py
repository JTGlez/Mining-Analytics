# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores: 
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: página principal de la aplicación.


#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import dash # Biblioteca principal de Dash.
from dash import dcc, html, callback # Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
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

            # Muestra una figura de CRISP-DM: falta hacerla responsive.
            html.Div(
                children=[        
                    html.Img(            
                        id="crisp",            
                        src="/assets/crispdm.PNG",            
                        style={'position': 'relative', 'left': '80px', 'top': '20px', 'width': '70%', 'height': 'auto'}
                    )    
                ]
            ),

        ],

    )


# Contenedor principal de la página en un Div.
layout = html.Div(
    id="app-container",
    children=[

        # Contenido principal de la aplicación: se divide en 2 columnas: una con contenido explicativo y otra para elementos interactivos.
        html.Div(

            id="main-content",
            className="row",
            children=[

                # Columna a la izquierda: invoca a description_card y generate_control_card para mostrar el texto explicativo de la izquierda.
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
                        # Agregar un div con id "page-content" para que se muestre el contenido de la página seleccionada
                        html.Div(id="page-content"),
                        # Muestra una gráfica random jajaja
                        html.Div(
                            id="patient_volume_card",
                            children=[
                                html.B("Patient Volume"),
                                html.Hr(),
                                dcc.Graph(id="patient_volume_hm"),
                            ],
                        ),

                    ],
                ),
            ],
        ),
    ],
)

# Definición de callbacks: se ejecutan para mostrar el contenido de la página.
@callback(Output("page-content-home", "children"), [Input("url-home", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home.layout
    if pathname == "/eda":
        return eda.layout
    elif pathname == "/pca":
        return pca.layout
    else:
        return html.Div(
        [
            html.H1("Error 404: Página no encontrada. :(", className="text-danger"),
            html.Hr(),
            html.P(f"La ruta del archivo {pathname} no fue encontrada."),
        ],
        className="p-3 bg-light rounded-3",
    )