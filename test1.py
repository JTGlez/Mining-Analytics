import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url
import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title = "Mining Analytics"

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "absolute",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "3rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
}

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Navbar
navbar = dbc.NavbarSimple(
    className="bg-white",
    children=[
        dbc.NavItem(
            dcc.Link("Inicio", href="/", className="nav-link mx-3", style={"whiteSpace": "nowrap"})
        ),
        dbc.NavItem(
            dcc.Link(
                "Análisis Exploratorio de Datos (EDA)",
                href="/page-1",
                className="nav-link mx-3",
                style={"whiteSpace": "nowrap"},
            )
        ),
        dbc.NavItem(
            dcc.Link(
                "Análisis de Componentes Principales (PCA)",
                href="/page-2",
                className="nav-link mx-3",
                style={"whiteSpace": "nowrap"},
            )
        ),
        dbc.NavItem(className="ml-auto"),
    ],
    brand_href="/",
    color="white",
    dark=False,
    sticky="top",
)

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Mining Analytics"),
            html.H3("Analítica de Datos inteligente"),
            html.Div(
                id="intro",
                children="La Minería de Datos es una disciplina que se enfoca en descubrir patrones y relaciones ocultas dentro de grandes conjuntos de datos."+ 
                """ A través de técnicas avanzadas de análisis y procesamiento de datos, la Minería de Datos permite extraer información valiosa y conocimientos útiles"""+ 
                """ que pueden ser utilizados para mejorar la toma de decisiones y optimizar el rendimiento en una amplia gama de sectores y aplicaciones."""
                ,
            ),
            html.Div(
                id="intro2",
                children = "En esta aplicación podrás visualizar todo el proceso involucrado en un proyecto de Minería de Datos enfocado al sector tecnológico y financiero."
            ),
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


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return dbc.ListGroup(
        [
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H5("This item has a heading", className="mb-1"),
                            html.Small("Yay!", className="text-success"),
                        ],
                        className="d-flex w-100 justify-content-between",
                    ),
                    html.P("And some text underneath", className="mb-1"),
                    html.Small("Plus some small print.", className="text-muted"),
                ],
                href="https://google.com"  
            ),
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H5(
                                "This item also has a heading", className="mb-1"
                            ),
                            html.Small("Ok!", className="text-warning"),
                        ],
                        className="d-flex w-100 justify-content-between",
                    ),
                    html.P("And some more text underneath too", className="mb-1"),
                    html.Small(
                        "Plus even more small print.", className="text-muted"
                    ),
                ]
            ),
        ]
    )

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("PRMain_logo.png")), navbar],
        ),
        # Contenido principal.
        html.Div(
            id="main-content",
            className="row",
            children=[
                # Left column
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[description_card(), generate_control_card()],
                ),
                # Right column
                html.Div(
                    id="right-column",
                    className="eight columns",
                    children=[
                        # Agregar un div con id "page-content" para que se muestre el contenido de la página seleccionada
                        html.Div(id="page-content"),
                        # Patient Volume Heatmap
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



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/page-1":
        return html.P("Contenido de la página 1")
    elif pathname == "/page-2":
        return html.P("Contenido de la página 2")
    else:
        return html.P("Contenido de la página principal")

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
