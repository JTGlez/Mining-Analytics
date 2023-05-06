# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores: 
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Análisis Exploratorio de Datos

#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import dash # Biblioteca principal de Dash.
from dash import dcc, html, callback # Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, ClientsideFunction # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from modules import home, eda, pca
import pathlib