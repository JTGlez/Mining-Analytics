# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores: 
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Análisis Exploratorio de Datos

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
from modules import home, eda, pca
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
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Análisis Exploratorio de Datos"), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div( 
                id="intro",
                children="El análisis exploratorio de datos (AED) es una técnica utilizada en estadística y ciencias de la computación para analizar y resumir conjuntos de datos. El objetivo del AED es descubrir patrones, identificar valores atípicos y comprender las relaciones entre las variables. En lugar de simplemente calcular estadísticas descriptivas básicas, el AED implica una exploración más profunda de los datos para descubrir información oculta."
                ,
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando uno de los datasets de prueba, o bien, cargando tu propio dataset."
            ),

            # Muestra una figura de exploración (GIF de lupa)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[        
                    html.Img(            
                        id="eda",            
                        src="/assets/eda.gif",
                        style = {'width': '25em', 'height': '15em'}
                    )    
                ]
            ),

        ],

    )

dropdown_options = [
    {'label': 'Dataset 1', 'value': 'assets/dt1.csv'},
    {'label': 'Dataset 2', 'value': 'assets/dt2.csv'},
    {'label': 'Dataset 3', 'value': 'assets/dt3.csv'}
]


# Contenedor principal de la página en un Div.
eda.layout = html.Div(
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
                    children=[eda_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id = "right-column",
                    className="four columns",
                    children = html.Div([

                        html.H4("Carga o elige el dataset para iniciar el Análisis Exploratorio de Datos", className= "text-upload"),

                        # Muestra el módulo de carga del dataset.
                        dcc.Upload(
                        id = 'upload-data',
                        children = html.Div([
                            'Arrastra aquí el archivo en formato CSV o selecciónalo'
                        ]),

                    # Por limitación de Dash estos elementos de carga no se pueden modificar desde la hoja de estilos y se debe hacer CSS inline.
                    style = {
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
                        #'borderColor': '#2196F3',
                        'background-color': '#fEfEfE'
                    },
                    multiple = True,
                    accept = '.csv'
                ),

                html.Div(
                        children = "O selecciona un dataset predeterminado aquí",
                        style = {
                        'font-family': 'Acumin',
                        'width' : '100%',
                        'text-align': 'center'
                    }
                    ),

                    # Muestra el módulo de carga del dataset.
                    dcc.Dropdown(
                    id='upload-data-static',
                    options = dropdown_options,
                    value = dropdown_options[0]['value'],
                    className='my-dropdown'
                    ),

                html.Hr(),
                html.Div(id = 'output-data-upload'),
                

                    

                ]),
                    
                ),
                #html.Div(id = 'output-dataset-upload'),
            ],
        ),
    ],
)




def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return eda(df, filename)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return eda(df, filename)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
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
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index().rename(columns={"index": "Column"})
    dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)  # Convertir los tipos de datos a strings

    # Creamos una visualización serializable para la impresión de la tabla de datos nulos
    nulls_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"]).reset_index().rename(columns={"index": "Column"})
    nulls_df['Null Count'] = nulls_df['Null Count'].astype(str)

    dropdown_options = [{'label': column, 'value': column} for column in df.select_dtypes(include=['int64', 'float64']).columns]
    

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

    dataframe_store = dcc.Store(id='dataframe-store', data=df.to_dict('records'))
    histogram_graph = dcc.Graph(id='histogram-graph')

    # Obtener el resumen estadístico
    describe_df = df.describe().reset_index().rename(columns={"index": "Stat"})
    describe_df['Stat'] = describe_df['Stat'].astype(str)

    # Obtener el resumen descriptivo
    describe_categoric_df = df.describe(include = 'object').reset_index().rename(columns={"index": "Stat"})
    describe_categoric_df['Stat'] = describe_categoric_df['Stat'].astype(str)

    boxplot_graph = dcc.Graph(id='boxplot-graph')

    categorical_histogram = create_categorical_bar_charts(df)

    categorical_tables = create_categorical_tables(df)



    return html.Div([

        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Solo mostramos las primeras 5 filas del dataframe, y le damos estilo para que las columnas se vean bien
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            cell_selectable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        
        html.Hr(),  # Línea horizontal  

        html.H3("Paso 1. Descripción de la estructura de los datos"),

        html.Div(
            children = "1) Dimensión del DataFrame: verificamos la estructura general de los datos observando la cantidad de filas y columnas en el dataset a explorar.", className = "text-description"
        ),

        dbc.Row([
            dbc.Col([
                dbc.Alert("Número de filas: {}".format(df.shape[0]))
            ], width=3),  # Ajusta el ancho de la columna

            dbc.Col([
                dbc.Alert("Número de columnas: {}".format(df.shape[1]))
            ], width=3),
        ],
            justify='center'  # Añade la propiedad justify con el valor 'center'
        ),

        html.Div(
            children = "2) Tipos de datos: a continuación se muestran los tipos de datos detectados para el dataset a analizar.", className = "text-description"
        ),

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
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Column'},
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgb(248, 248, 248)',
                        'border': '1px solid black'
                    }
                ]
            ),
            style={'width': '50%', 'margin': '0 auto'}
        ),

        html.H3("Paso 2. Identificación de datos faltantes"),

        html.Div(
            children = "A continuación se muestran los valores nulos detectados por cada variable en el dataset:", className = "text-description"
        ),

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
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Column'},
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgb(248, 248, 248)',
                        'border': '1px solid black'
                    }
                ]
            ),
            style={'width': '50%', 'margin': '0 auto'}
        ),

        html.H3("Paso 3. Detección de valores atípicos"),

        html.Div(
            children="1) Distribución de variables numéricas: selecciona una variable numérica en el menú desplegable para ver su histograma.",
            className="text-description"
        ),

        html.Div([dropdown, histogram_graph]),
        dataframe_store,

         html.Div(
            children="2) Distribución de variables numéricas: selecciona una variable numérica en el menú desplegable para ver su histograma.",
            className="text-description"
        ),


        html.Div(
            dash_table.DataTable(
                data=describe_df.to_dict('records'),
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
                    }
                ],
                style_table={
                    'height': 'auto',
                    'overflowX': 'auto'
                },
            ),
            style={'width': '50%', 'margin': '0 auto'}
        ),

        html.Div(
            children="3) Diagramas para detectar valores atípicos: selecciona una variable numérica en el menú desplegable para ver su diagrama de caja.",
            className="text-description"
        ),
        html.Div([dropdown_boxplot, boxplot_graph]),
        dataframe_store,

        html.Div(
            children="4) Diagramas de variables categóricas: Se refiere a la observación de las clases de cada columna (variable) y su frecuencia. ",
            className="text-description"
        ),

        html.Div(
            dash_table.DataTable(
                data=describe_categoric_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in describe_categoric_df.columns],
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
            style={'width': '50%', 'margin': '0 auto'}
        ),

        html.Div(
            children="Histogramas de variables categóricas:",
            className="text-description"
        ),

    ])

@callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data-static', 'value')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, selected_file, list_of_names, list_of_dates):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'upload-data-static.value':
        df = pd.read_csv(selected_file)
        return eda(df, selected_file)

@callback(
    Output('histogram-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('dataframe-store', 'data')
)
def update_histogram(selected_variable, stored_data):
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


@callback(
    Output('boxplot-graph', 'figure'),
    Input('variable-dropdown-box', 'value'),
    Input('dataframe-store', 'data')
)
def update_boxplot(selected_variable, stored_data):
    df = pd.DataFrame(stored_data)
    figure = create_boxplot_figure(selected_variable, df)
    return figure


def create_categorical_bar_charts(df):
    categorical_columns = df.select_dtypes(include='object').columns
    bar_charts = []
    for col in categorical_columns:
        if df[col].nunique() < 10:
            counts = df[col].value_counts()
            bar_chart = go.Bar(x=counts.index, y=counts.values, name=col)
            bar_charts.append(bar_chart)
    # Crear un objeto go.Figure con las gráficas de barras y un diseño personalizado
    figure = go.Figure(data=bar_charts, layout=go.Layout(title='Distribución de variables categóricas', xaxis=dict(title='Categoría'), yaxis=dict(title='Frecuencia'), hovermode='closest'))
    return figure

def create_categorical_tables(df):
    tables = []
    for col in df.select_dtypes(include='object'):
        if df[col].nunique() < 10:
            table_df = df.groupby(col).agg(['mean'])
            tables.append((col, table_df))
    return tables
