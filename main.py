# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores: 
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said


#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import dash # Biblioteca principal de Dash.
from dash import dcc, html # Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, ClientsideFunction # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from modules import home, eda, pca
import pathlib


# Inicialización de la aplicación: crea una instancia de Dash y su servidor en Flask con la hoja de estilos básica de Bootstrap.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title = "Mining Analytics"
server = app.server # Asigna la instancia de Dash a Flask.
app.config.suppress_callback_exceptions = True # Evita excepciones de interrupción.

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Barra de navegación de la aplicación: se define a su hijo como el Logo de Mining Analytics y los elementos de navegación.
navbar = dbc.NavbarSimple(
    id="navbar",
    brand=html.Img(src=app.get_asset_url("PRMain_logo.png"), className="navbar-logo"),
    brand_href="/",
    children=[
        # Elemento de navegación nav-link para estilo y mx-3 para un margen de 0.75rem en ambos lados.
        dbc.NavItem(
            dcc.Link(
                "Inicio",
                href="/",
                className="nav-link mx-3",
                style={"whiteSpace": "nowrap"},
            )
        ),
        dbc.NavItem(
            dcc.Link(
                "Análisis Exploratorio de Datos (EDA)",
                href="/eda",
                className="nav-link mx-3",
                style={"whiteSpace": "nowrap"},
            )
        ),
        dbc.NavItem(
            dcc.Link(
                "Análisis de Componentes Principales (PCA)",
                href="/pca",
                className="nav-link mx-3",
                style={"whiteSpace": "nowrap"},
            )
        ),
        dbc.NavItem(className="ml-auto"),
    ],
    color="white",
    dark=False,
    sticky="top",
)

#---------------------------------------------------Definición del layout de la página--------------------------------------------------------#

# Se define el contenido de la página base: a partir de esta se cargan el resto.
content = html.Div(id="page-content")

# Contenedor principal de la página en un Div, incluye la barra de navegación y el contenido de cada página.
app.layout = html.Div(
    [dcc.Location(id = "url"), navbar, content]
)


# Definición de callbacks: se ejecutan para mostrar el contenido de la página.
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
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
            html.P(f"La página {pathname} no fue encontrada."),
        ],
        className="p-3 bg-light rounded-3",
    )

# Ejecuta el servidor de Flask al iniciar el script.
if __name__ == "__main__":
    app.run_server(debug=True)