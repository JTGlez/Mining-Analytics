# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Análisis Exploratorio de Datos

#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import base64
import datetime
import numpy as np
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
from modules import home, eda, pca, regtree, classtree, regforest
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def eda_card():
    """
    :retorna: Un div que contiene la explicación del módulo de EDA.

    """

    return html.Div(

        # ID del div.
        id="eda-card",

        # Elementos hijos del div "eda-card".
        children=[
            html.H5("Mining Analytics"),  # Título de página.
            html.H3("Análisis Exploratorio de Datos"),  # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="text-roboto",
                children="El análisis exploratorio de datos (AED) es una técnica utilizada en estadística y ciencias de la computación para analizar y resumir conjuntos de datos. El objetivo del AED es descubrir patrones, identificar valores atípicos y comprender las relaciones entre las variables. En lugar de simplemente calcular estadísticas descriptivas básicas, el AED implica una exploración más profunda de los datos para descubrir información oculta.",
                style={"font-family": "Roboto", "font-size": "large", "text-align": "justify"}
            ),

            html.Br(),

            # Muestra una figura de exploración (GIF de lupa)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="/assets/eda2.gif",
                        style={'width': '35em', 'height': '27em'}
                    )
                ]
            ),

            html.Br(),

            # Texto secundario de explicacion.
            html.Div(
                id="text-roboto",
                children="En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando uno de los datasets de prueba, o bien, cargando tu propio dataset. ¡Iniciemos a explorar los datos!",
                style={"font-family": "Roboto", "font-size": "large", "text-align": "justify"}
            ),

            # Muestra una figura de exploración (GIF de lupa)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="/assets/eda.gif",
                        style={'width': '25em', 'height': '15em'}
                    )
                ]
            ),

        ],

    )


# Datasets predeterminados.
dropdown_options = [
    {'label': 'Salarios de Data Science', 'value': 'data/SalariosDS.csv'},
    {'label': 'Dataset 2', 'value': 'assets/dt2.csv'},
    {'label': 'Dataset 3', 'value': 'assets/dt3.csv'}
]


# Contenedor principal de la página en un Div.
eda.layout = html.Div(
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
                    children=[eda_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div([

                        html.H4(
                            "Carga o elige el dataset para iniciar el Análisis Exploratorio de Datos", className="text-upload"),

                        # Muestra el módulo de carga del dataset.
                        dcc.Upload(
                            id='upload-data-eda',
                            children=html.Div([
                                'Arrastra aquí el archivo en formato CSV o selecciónalo'
                            ]),

                            # Por limitación de Dash estos elementos de carga no se pueden modificar desde la hoja de estilos y se debe hacer CSS inline.
                            style={
                                'font-family': 'Acumin',
                                'width': '50%',
                                'height': '100%',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'solid',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '2em auto',
                                'display': 'grid',
                                'justify-content': 'center',
                                'align-items': 'center',
                                'flex-direction': 'column',
                                # 'borderColor': '#2196F3',
                                'background-color': '#fEfEfE'
                            },
                            multiple=True,
                            accept='.csv'
                        ),

                        html.Div(
                            children="O selecciona un dataset predeterminado aquí",
                            style={
                                'font-family': 'Acumin',
                                'width': '100%',
                                'text-align': 'center'
                            }
                        ),

                        # Muestra el módulo de carga del dataset.
                        dcc.Dropdown(
                            id='upload-data-eda-static',
                            options=dropdown_options,
                            value=dropdown_options[0]['value'],
                            className='my-dropdown'
                        ),

                        html.Hr(),
                        # Aquí se muestra toda la ejecución del módulo EDA tras cargarse un dataset.
                        html.Div(id='output-data-upload'),
                    ]),
                ),
            ],
        ),
    ],
)


def parse_contents(contents, filename, date):
    """
    Realiza la lectura del archivo en formato .csv, lo convierte a base64 y lo carga como un dataset de pandas para ejecutar la función eda.
    Retorna: eda(df, filename)
    """

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Carga de CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            print(df)
            return eda(df, filename)
        elif 'xls' in filename:
            # Carga de XLS: en pruebas.
            df = pd.read_excel(io.BytesIO(decoded))
            return eda(df, filename)
    except Exception as e:
        print(e)
        return html.Div([
            '¡Archivo inválido! Verifique que se encuentre cargando un archivo en formato .csv'
        ])


def eda(df, filename):
    """
    retorna: realiza un Análisis Exploratorio de Datos con el dataset que recibe de acuerdo al procedimiento del curso de Minería de Datos.
    1. Descripción de la estructura del dataset
    2. Identificación de datos faltantes
    3. Identificación de outliers
    4. Análisis correlativo entre pares variables
    """

    # Crear el DataFrame con los tipos de datos, serializando la salida para su visualización como JSON
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index().rename(
        columns={"index": "Column"})
    dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(
        str)  # Convertir los tipos de datos a strings

    # Creamos una visualización serializable para la impresión de la tabla de datos nulos
    nulls_df = pd.DataFrame(df.isnull().sum(), columns=[
                            "Null Count"]).reset_index().rename(columns={"index": "Column"})
    nulls_df['Null Count'] = nulls_df['Null Count'].astype(str)

    # Obtenemos las variables numéricas para saber cuántos histogramas y diagramas se deben generar y dar la opción de seleccionar el que se desea visualizar.
    dropdown_options = [{'label': column, 'value': column}
                        for column in df.select_dtypes(include=['int64', 'float64']).columns]

    dropdown = dcc.Dropdown(
        id='variable-dropdown',
        options=dropdown_options,
        value=dropdown_options[0]['value']
    )

    dropdown_boxplot = dcc.Dropdown(
        id='variable-dropdown-box',
        options=dropdown_options,
        value=dropdown_options[0]['value']
    )

    dataframe_store = dcc.Store(
        id='dataframe-store', data=df.to_dict('records'))
    histogram_graph = dcc.Graph(id='histogram-graph')

    # Obtener el resumen estadístico
    describe_df = df.describe().reset_index().rename(columns={"index": "Stat"})
    describe_df['Stat'] = describe_df['Stat'].astype(str)

    # Verificar si hay variables categóricas en el dataset
    has_categorical = any(df.select_dtypes(include='object').columns)

    if has_categorical:
        # Obtener el resumen descriptivo
        describe_categoric_df = df.describe(
            include='object').reset_index().rename(columns={"index": "Stat"})
        describe_categoric_df['Stat'] = describe_categoric_df['Stat'].astype(
            str)
        categorical_histogram = create_categorical_bar_charts(df)
    else:
        describe_categoric_df = pd.DataFrame()

    boxplot_graph = dcc.Graph(id='boxplot-graph')

    return html.Div([

        # ---- TABS ----
        html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                html.H3(
                                    "Paso 1. Descripción de la estructura de los datos"),

                                html.Div(
                                    children="1) Dimensión del DataFrame: verificamos la estructura general de los datos observando la cantidad de filas y columnas en el dataset a explorar.", className="text-description"
                                ),

                                html.Br(),

                                dbc.Row([
                                        dbc.Col([
                                            dbc.Alert(
                                                "Número de filas: {}".format(df.shape[0]))
                                        ], width=3),  # Ajusta el ancho de la columna

                                        dbc.Col([
                                            dbc.Alert(
                                                "Número de columnas: {}".format(df.shape[1]))
                                        ], width=3),
                                        ],
                                        justify='center'  # Añade la propiedad justify con el valor 'center'
                                        ),

                                html.Br(),

                                html.Div(
                                    children="2) Tipos de datos: a continuación se muestran los tipos de datos detectados para el dataset a analizar.", className="text-description"
                                ),

                                html.Br(),

                                html.Div(

                                    dash_table.DataTable(
                                        data=dtypes_df.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in dtypes_df.columns],
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '1em'
                                        },
                                        style_header={
                                            'fontWeight': 'bold',
                                            'backgroundColor': 'rgb(230, 230, 230)',
                                            'border': '1px solid black'
                                        },
                                        style_table={
                                            # 'height': '400px',
                                            'overflowY': 'auto',
                                            'backgroundColor': 'rgb(230, 230, 230)'
                                        },
                                        style_data_conditional=[
                                            {
                                                'if': {'column_id': 'Column'},
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(248, 248, 248)',
                                                'border': '1px solid black'
                                            }
                                        ]
                                    ),
                                    style={'margin': '0 auto'}
                                ),
                            ],
                            label="Paso 1", tab_id="tab-1", tab_style={"width": "auto"}),

                        dbc.Tab(
                            children=[
                                html.H3(
                                    "Paso 2. Identificación de datos faltantes"),

                                html.Div(
                                    children="A continuación se muestran los valores nulos detectados por cada variable en el dataset:", className="text-description"
                                ),

                                dbc.Alert("ⓘ Las columnas detectadas con valores nulos se marcarán en naranja.", color="warning"),

                                html.Br(),

                                html.Div(

                                    dash_table.DataTable(
                                        data=nulls_df.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in nulls_df.columns],
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '1em'
                                        },
                                        style_header={
                                            'fontWeight': 'bold',
                                            'backgroundColor': 'rgb(230, 230, 230)',
                                            'border': '1px solid black'
                                        },
                                        style_table={
                                            # 'height': '400px',
                                            'overflowY': 'auto',
                                            'backgroundColor': 'rgb(230, 230, 230)'
                                        },
                                        style_data_conditional=[
                                            {
                                                'if': {'column_id': 'Column'},
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(248, 248, 248)',
                                                'border': '1px solid black'
                                            },
                                            # Marca nulos con color naranja.
                                            {
                                                "if": {"filter_query": '{Null Count} > 0'},
                                                "backgroundColor": "rgb(255, 131, 0)"
                                            },
                                        ]
                                    ),
                                    style={'width': '50%', 'margin': '0 auto'}
                                )
                            ],
                            label="Paso 2", tab_id="tab-2", tab_style={"width": "auto"}),


                        dbc.Tab(
                            children=[
                                html.H3(
                                    "Paso 3. Detección de valores atípicos"),

                                html.Div(
                                    children="1) Distribución de variables numéricas: selecciona una variable numérica en el menú desplegable para ver su histograma.",
                                    className="text-description"
                                ),

                                html.Br(),

                                html.Div([dropdown, histogram_graph]),
                                dataframe_store,

                                html.Br(),

                                html.Div(
                                    children="2) Descripción estadística: a continuación se muestran los estadísticos obtenidos para el dataset analizado.",
                                    className="text-description"
                                ),

                                html.Br(),

                                html.Div(
                                    dash_table.DataTable(
                                        data=describe_df.to_dict(
                                            'records'),
                                        columns=[{'name': i, 'id': i} for i in describe_df.columns],
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '1em',
                                            'border': '1px solid black',
                                            'borderRadius': '5px'
                                        },
                                        style_header={
                                            'fontWeight': 'bold',
                                            'backgroundColor': 'rgb(230, 230, 230)',
                                            'border': '1px solid black',
                                            'borderRadius': '5px'
                                        },
                                        style_data_conditional=[
                                            {
                                                'if': {'column_id': 'Stat'},
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(248, 248, 248)',
                                                'border': '1px solid black',
                                                'borderRadius': '5px'
                                            },


                                        ],
                                        style_table={
                                            'height': 'auto',
                                            'overflowX': 'auto'
                                        },
                                    ),
                                    style={'margin': '0 auto'}
                                ),


                                html.Br(),

                                html.Div(
                                    children="3) Diagramas para detectar valores atípicos: selecciona una variable numérica en el menú desplegable para ver su diagrama de caja.",
                                    className="text-description"
                                ),

                                html.Br(),

                                html.Div([dropdown_boxplot, boxplot_graph]),
                                dataframe_store,

                                html.Div([

                                    html.Div(
                                        children="4) Diagramas de variables categóricas: Se refiere a la observación de las clases de cada columna (variable) y su frecuencia. ",
                                        className="text-description"
                                    ),

                                    html.Br(),

                                    html.Div(
                                        dash_table.DataTable(
                                            data=describe_categoric_df.to_dict(
                                                'records'),
                                            columns=[
                                                {'name': i, 'id': i} for i in describe_categoric_df.columns],
                                            style_cell={
                                                'textAlign': 'left',
                                                'padding': '1em',
                                                'border': '1px solid black',
                                                'borderRadius': '5px'
                                            },
                                            style_header={
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'border': '1px solid black',
                                                'borderRadius': '5px'
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'column_id': 'Stat'},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'rgb(248, 248, 248)',
                                                    'border': '1px solid black',
                                                    'borderRadius': '5px'
                                                }
                                            ],
                                            style_table={
                                                'height': 'auto',
                                                'overflowX': 'auto'
                                            },
                                        ),
                                        style={'margin': '0 auto'}
                                    ),

                                    html.Div(
                                        children="Histogramas de variables categóricas:",
                                        className="text-description"
                                    ),

                                    html.Br(),


                                    html.Div(
                                        dcc.Graph(
                                            figure=categorical_histogram),
                                        className="categorical-histogram"
                                    ),

                                    html.Br(),

                                    html.Div(
                                        children="5) Agrupar por tipos de clases, por ejemplo, con propiedades tipo h. Se muestran específicamente promedios de cada tipo.",
                                        className="text-description"
                                    ),

                                    html.Br(),

                                    html.Div([
                                        html.Div([
                                            dash_table.DataTable(
                                                id=f'categorical-table-{i}',
                                                columns=[{"name": col, "id": col} for col in df.columns],
                                                data=df.to_dict("records"),
                                                style_cell={
                                                    "textAlign": "left", "padding": "1em 1em 1em 1em"},
                                                style_header={
                                                    "backgroundColor": "royalblue",
                                                    "color": "white",
                                                    "textAlign": "left"
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "white"
                                                    },
                                                    {
                                                        "if": {"row_index": "even"},
                                                        "backgroundColor": "paleturquoise"
                                                    }
                                                ],
                                                css=[
                                                    {"selector": ".dash-spreadsheet", "rule": "table-layout: auto"}],
                                            )
                                        ], style={"width": "100%", "max-width": "100%", "margin": '0 auto', 'overflowX': 'scroll'})  # No agrega desplazamiento horizontal
                                        for i, df in enumerate(create_categorical_tables(df))
                                    ], style={"display": "flex", "flex-wrap": "wrap", "width": "100%", "max-width": "100%", "margin": "0 auto"}),


                                    html.Br(),

                                ]) if has_categorical is True else html.Div(),  # Si no hay variables categóricas, no se muestran ni se crean los histogramas ni descripciones estadísticas de esa clase de variables.
                            ],
                            label="Paso 3", tab_id="tab-3", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[
                                html.H3(
                                    "Paso 4. Identificación de relaciones entre pares variables"),

                                html.Div(
                                    children="1) Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas. Se emplea la función corr()",
                                    className="text-description"
                                ),

                                html.Br(),

                                dcc.Graph(
                                    id='matriz',
                                    figure={
                                        'data': [
                                            {'x': df.corr(numeric_only=True).columns, 'y': df.corr(numeric_only=True).columns, 'z': np.triu(
                                                df.corr(numeric_only=True).values, k=1), 'type': 'heatmap', 'colorscale': 'RdBu', 'symmetric': False}
                                        ],
                                        'layout': {
                                            'title': 'Matriz de correlación',
                                            'xaxis': {'side': 'down'},
                                            'yaxis': {'side': 'left'},
                                            # Agregamos el valor de correlación por en cada celda (text_auto = True)
                                            'annotations': [
                                                dict(
                                                    x=df.corr(
                                                        numeric_only=True).columns[i],
                                                    y=df.corr(
                                                        numeric_only=True).columns[j],
                                                    text=str(
                                                        round(df.corr(numeric_only=True).values[i][j], 4)),
                                                    showarrow=False,
                                                    font=dict(
                                                        color='white' if abs(
                                                            df.corr(numeric_only=True).values[i][j]) >= 0.67 else 'black'
                                                    ),
                                                ) for i in range(len(df.corr(numeric_only=True).columns)) for j in range(i)
                                            ],
                                        },
                                    },
                                )
                            ],
                            label="Paso 4", tab_id="tab-4", tab_style={"width": "auto"}

                        ),

                    ],
                    id="tabs",
                    active_tab="tab-1",
                    style={"margin-top": "45px"}
                ),
            ],
        ),

    ])


# Callback de carga: se ejecuta cuando se realiza la carga de un dataset predeterminado o cargado por el usuario

@callback(Output('output-data-upload', 'children'),
          [Input('upload-data-eda', 'contents'),
           Input('upload-data-eda-static', 'value')],
          [State('upload-data-eda', 'filename'),
           State('upload-data-eda', 'last_modified')])
def update_output(list_of_contents, selected_file, list_of_names, list_of_dates):
    """
    Función Callback de EDA: ecibe tres parámetros de entrada (list_of_contents, selected_file, list_of_names, list_of_dates) y tiene como salida un objeto de tipo children que se utiliza para actualizar la salida de la aplicación.

    Output('output-data-upload', 'children'): indica que el objeto de salida de esta función es un objeto de tipo 'children' que se utilizará para actualizar la salida en la aplicación.
    [Input('upload-data-eda', 'contents'), Input('upload-data-eda-static', 'value')]: indica que esta función se ejecutará cuando se produzca un evento en los objetos de entrada 'upload-data-eda' y 'upload-data-eda-static' (designados para la carga estática o dinámica de datasets).
    [State('upload-data-eda', 'filename'), State('upload-data-eda', 'last_modified')]: indica que esta función también necesita los valores actuales de 'filename' y 'last_modified' del objeto de entrada 'upload-data-eda'.
    """
    # Se obtiene el contexto del callback: permite identificar si se cargó un dataset predeterminado o uno cargado por el usuario.
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-eda.contents':
        # Procesa el archivo cargado por el usuario.
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'upload-data-eda-static.value':
        # Lee directamente el archivo en la ruta especificada para el mismo y ejecuta el análisis exploratorio.
        df = pd.read_csv(selected_file)
        return eda(df, selected_file)

# Callback usado para detectar la variables numéricas en el dataset y generar histogramas de distribución para cada una de ellas.


@callback(
    Output('histogram-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('dataframe-store', 'data')
)
def update_histogram(selected_variable, stored_data):
    """
    Función Callback:  se ejecuta cuando se selecciona una variable numérica en el menú desplegable 'variable-dropdown' y se actualiza el almacenamiento de datos en 'dataframe-store'. La función utiliza los datos almacenados para generar un histograma de la variable numérica seleccionada utilizando la biblioteca Plotly. La salida es un objeto 'figure' que se actualiza en el componente 'histogram-graph' de la aplicación.
    """
    df = pd.DataFrame(stored_data)

    hist = go.Histogram(
        x=df[selected_variable],
        name=selected_variable,
        opacity=0.5,
    )
    figure = {
        'data': [hist],
        'layout': go.Layout(
            title=f'Distribución de {selected_variable}',
            xaxis=dict(title='Valor'),
            yaxis=dict(title='Frecuencia'),
            barmode='overlay',
            hovermode='closest'
        )
    }
    return figure


def create_boxplot_figure(column, df):
    """
    Función auxiliar que recibe una columna y un DataFrame de pandas como argumentos. Crea un diagrama de caja (boxplot) para la columna utilizando la biblioteca Plotly y devuelve el objeto 'figure' correspondiente.
    """
    box = go.Box(
        x=df[column],
        name=column,
        marker=dict(color='rgb(0, 128, 128)'),
        boxmean=True
    )
    layout = go.Layout(
        title=f'Diagrama de caja para {column}',
        yaxis=dict(title=column),
        xaxis=dict(title='Distribución'),
        hovermode='closest'
    )
    return go.Figure(data=[box], layout=layout)

# Callback para identificar las variables numéricas en el dataset y generar diagramas de caja para analizar los rangos entre las observaciones.


@callback(
    Output('boxplot-graph', 'figure'),
    Input('variable-dropdown-box', 'value'),
    Input('dataframe-store', 'data')
)
def update_boxplot(selected_variable, stored_data):
    """
    Función de callback qye se ejecuta cuando se selecciona una variable numérica en el menú desplegable 'variable-dropdown-box' y se actualiza el almacenamiento de datos en 'dataframe-store'. La función utiliza los datos almacenados y la función 'create_boxplot_figure' para generar un diagrama de caja de la variable seleccionada. La salida es un objeto 'figure' que se muestra a la salida en el elemento 'boxplot-graph' de la función EDA.
    """
    df = pd.DataFrame(stored_data)
    figure = create_boxplot_figure(selected_variable, df)
    return figure


def create_categorical_bar_charts(df):
    """
    Esta función recibe un DataFrame de pandas 'df' como argumento y crea histogramas para las variables categóricas en el DataFrame.
    """

    categorical_columns = df.select_dtypes(include='object').columns
    bar_charts = []
    for col in categorical_columns:
        if df[col].nunique() < 10:
            counts = df[col].value_counts()
            bar_chart = go.Bar(x=counts.index, y=counts.values, name=col)
            bar_charts.append(bar_chart)

    # Crea un objeto go.Figure con las gráficas de barras y un diseño personalizado
    figure = go.Figure(data=bar_charts, layout=go.Layout(title='Distribución de variables categóricas', xaxis=dict(
        title='Categoría'), yaxis=dict(title='Frecuencia'), hovermode='closest'))
    return figure


def create_categorical_tables(df):
    """
    Recibe un DataFrame de pandas 'df' como argumento y crea tablas de resumen para las variables categóricas en el DataFrame.
    """
    data_frames = []

    # Itera sobre las columnas categóricas del DataFrame.
    for col in df.select_dtypes(include='object'):
        # Agrupa el DataFrame por la columna categórica y calcula la media de las otras columnas utilizando df.groupby(col).mean().reset_index().
        if df[col].nunique() < 10:
            table_df = df.groupby(col).mean().reset_index()
            # Copia los valores de la columna categórica y elimina la columna del DataFrame agrupado.
            col_values = table_df[col].copy()
            # Elimina la columna categórica
            table_df = table_df.drop(columns=[col])
            # Inserta la columna categórica al principio del DataFrame agrupado.
            table_df.insert(0, col, col_values)
            # Añade el DataFrame agrupado a la lista data_frames.
            data_frames.append(table_df)

    return data_frames
