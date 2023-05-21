# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Análisis de Componentes Principales

#------------------------------------------------Importación de bibliotecas------------------------------------------------------------#
import base64
import datetime
import io
from io import BytesIO
import dash # Biblioteca principal de Dash.
from msilib.schema import Component
from dash import dcc, html, Input, Output, dash_table, callback# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, State # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from modules import home, eda, pca, regtree, classtree, regforest, classforest
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def pca_card():
    """
    :retorna: Un div que contiene la explicación del módulo de EDA.

    """

    return html.Div(

        # ID del div.
        id="pca-card",

        # Elementos hijos del div 'pca-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Análisis de Componentes Principales."), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children=
                [
                    html.P("El análisis de componentes principales (ACP) es una técnica de reducción de la dimensionalidad que se utiliza para identificar patrones y estructuras en datos multivariados. Esto significa que nos permite resumir una gran cantidad de información en unas pocas dimensiones, manteniendo la mayor cantidad posible de la varianza original de los datos."),
                    html.P("Para calcular los componentes principales se sigue el siguiente procedimiento:"),
                ],
            ),
            # Imagen PCA
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                children=[
                    html.Img(
                        id="pca-img",
                        src="/assets/pca.jpg",
                        style = {'width': '100%', 'height': '100%'}
                    )
                ]
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando tu propio dataset.",
                className="mb-4"
            ),

            # Muestra una figura (GIF)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
                children=[
                    html.Img(
                        id="pca-gif",
                        src="https://miro.medium.com/v2/resize:fit:1400/1*37a_i1t1tDxDYT3ZI6Yn8w.gif",
                        style = {'width': '100%', 'height': '50%'},
                        className="mb-4"
                    ),
                ],
            ),
        ],

    )

#Contenedor principal de la página en un Div
pca.layout = html.Div(
    id="page-content",
    children=[
        # El contenido se divide en 2 columnas: descripción | resultados
        html.Div(
            className="row",
            children=[
                #Columna izquierda: para la descripción
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[pca_card()],
                ),
                #Columa derecha: para los resultados
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div(
                        [
                            html.H4("Carga o elige el dataset para iniciar el Análisis de Componentes Principales", className="text-upload"),
                            # Muestra el módulo de carga
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    [
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ],
                                ),
                            style={
                                'font-family':'Acumin',
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
                        html.Div(id='output-data-upload-pca'),
                        ],
                    ),
                ),
            ],
        ),
    ],
)

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('There was an error processing this file.', color="danger")
        ])

    return html.Div([
        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
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
            columns=[{'name': i, 'id': i, "deletable":False} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        dbc.Alert('Variables numéricas: {}'.format(df.select_dtypes(include='number').shape[1]), color="info", class_name="my-3 mx-auto text-center w-25"),

        html.H3(
            "Evidencia de datos correlacionados"
        ),

        dcc.Graph(
            id='matriz',
            figure={
                'data': [
                    {'x': df.corr(numeric_only=True).columns, 'y': df.corr(numeric_only=True).columns, 'z': np.triu(df.corr(numeric_only=True).values, k=1), 'type': 'heatmap', 'colorscale': 'RdBu', 'symmetric': False}
                ],
                'layout': {
                    'title': 'Matriz de correlación',
                    'xaxis': {'side': 'down'},
                    'yaxis': {'side': 'left'},
                    # Agregamos el valor de correlación por en cada celda (text_auto = True)
                    'annotations': [
                        dict(
                            x=df.corr(numeric_only=True).columns[i],
                            y=df.corr(numeric_only=True).columns[j],
                            text=str(round(df.corr(numeric_only=True).values[i][j], 4)),
                            showarrow=False,
                            font=dict(
                                color='white' if abs(df.corr(numeric_only=True).values[i][j]) >= 0.67  else 'black'
                            ),
                        ) for i in range(len(df.corr(numeric_only=True).columns)) for j in range(i)
                    ],
                },
            },
        ),
        html.Div(
            children=[
                dbc.Badge(
                    "Identificación de correlaciones",
                    pill=True,
                    color="primary",
                    style={"font-size":"15px"}
                ),
                html.P("🔴 Correlación fuerte: De -1.0 a -0.67 y 0.67 a 1.0", className="ms-4"),
                html.P("🟡 Correlación moderada: De -0.66 a -0.34 y 0.34 a 0.66", className="ms-4"),
                html.P("🔵 Correlación débil: De -0.33 a 0.0 y 0.0 a 0.33", className="ms-4"),
                dbc.Alert("ⓘ Si no se identifica almenos una correlación fuerte, entonces PCA no aplica.", color="warning"),
            ],
            className="mt-3"
        ),
        html.H3(
            "Cálculo de Componentes Principales"
        ),
        html.P(
            "Una vez que haz identificado correlaciones entre pares de variables el siguiente paso es configurar los parámetros que se muestran a continuación, los cuales son necesarios para que el algoritmo funcione."
        ),
        html.P(
            "Al terminar, presiona sobre el botón rojo para observar los resultados."
        ),
        dbc.Alert(
            "ⓘ Si tienes alguna duda posicionate sobre cada parámetro para más información.", color="secondary", style={"font-size": "10px","width":"41%" }
        ),
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
                                        id='select-scale',
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
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Núm. Componentes principales", color="primary",
                                                id="tooltip-numpc", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Número de Componentes Principales:**  
                                                        El objetivo de PCA es reducir la dimensionalidad de un conjunto de datos, manteniendo al mismo tiempo la mayor cantidad posible de información.

                                                        Así que, por regla general, se suelen considerar tantas componentes como número de variables numéricas existan en el dataset. Sin embargo, siéntete en libertad de escoger tantas como desees.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-numpc", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='num_components',
                                        type='number',
                                        placeholder='None',
                                        value=df.select_dtypes(include='number').shape[1],
                                        min=1,
                                        max=df.select_dtypes(include='number').shape[1],
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Porcentaje de Relevancia", color="primary",
                                                id="tooltip-percent", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    dcc.Markdown('''
                                                        **Porcentaje de Relevancia:**  
                                                        Generalmente suele elegirse un porcentaje de relevancia que esté entre 75% y 90% de varianza acumulada, lo que se busca es perder la menor cantidad posible de información.
                                                    ''', style={'text-align': 'left'}),
                                                ],
                                                target="tooltip-percent", placement="left", style={"font-size":"10px"},
                                            ),
                                        ],
                                        style={"height":"50px", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Input(
                                        id='relevance',
                                        type='number',
                                        placeholder='None',
                                        value=0.9,
                                        min=0.75,
                                        max=0.9,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"height":"50px"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                    ],
                    style={"justify-content": "between", "height": "100%"}
                ),
            ],
            style={"font-size":"20px", "margin":"30px 0"}
        ),
        html.Div(
            children=
            [
                dbc.Button(
                    "Presiona para obtener los Componentes Principales", id="pca-btn", color="danger", style={"width":"40%"},
                ),
            ],
            style={"display": "flex", "justify-content": "center"},
        ),
        # ---- TABS ----
        html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                dash_table.DataTable(
                                    id="matriz-estandarizada",
                                    columns=[{"name": i, "id": i} for i in df.select_dtypes(include=['float64', 'int64']).columns],
                                    page_size=8,
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
                            ],
                            label="Matriz Estandarizada", tab_id="tab-1", tab_style={"width":"auto"}),
                        dbc.Tab(
                                children=[
                                    dcc.Graph(
                                        id="varianza-explicada"
                                    ),
                                ],
                                label="Varianza Explicada (%)", tab_id="tab-2", tab_style={"width":"auto"}),
                        dbc.Tab(
                                children=[
                                    dcc.Graph(
                                        id="varianza-acumulada"
                                    ),
                                ],
                                label="Número de Componentes Principales y Varianza Acumulada", tab_id="tab-3", tab_style={"width":"30%"}),
                        dbc.Tab(
                                children=[
                                    html.Div(
                                        [
                                            dbc.Alert('ⓘ Selecciona solo aquellas variables que superen un umbral del 50%, son las más importantes', color="primary"),
                                            # Mostramos la gráfica generada en el callback ID = FigComponentes
                                            dcc.Graph(
                                                id='cargas'
                                            ),
                                        ],
                                    ),
                                ],
                                label="Proporción de cargas y Selección de variables", tab_id="tab-4", tab_style={"width":"30%"})
                    ],
                    id="tabs",
                    active_tab="tab-1",
                    style={"margin-top":"45px"}
                ),
            ],
        ),
    ],)

@callback(Output('output-data-upload-pca', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

# ---- FUNCIÓN PARA ESTANDARIZAR LOS DATOS ----
def estandarizar_datos(df_numerica, scale):
    """
    Escala las columnas numéricas de un DataFrame según el tipo de escala indicado.

    Parámetros:
    df (pandas.DataFrame): DataFrame con las columnas a escalar.
    scale (str): Tipo de escala a aplicar. Puede ser "StandardScaler()" o "MinMaxScaler()".

    Retorna:
    pandas.DataFrame: Nuevo DataFrame con las columnas numéricas escaladas.
    """
    if scale == "StandardScaler()":
        matriz_estandarizada = StandardScaler().fit_transform(df_numerica)
    elif scale == "MinMaxScaler()":
        matriz_estandarizada = MinMaxScaler().fit_transform(df_numerica)
    return pd.DataFrame(matriz_estandarizada, columns=df_numerica.columns)

# ---- FUNCIÓN PARA OBTENER LA GRÁFICA DE VARIANZA EXPLICADA ----
def grafico_varianza_explicada(varianza):
    """
    Crea un gráfico de barras que muestra la varianza explicada por cada componente principal.

    Parámetros:
    varianza (numpy.ndarray): Array con la varianza explicada por cada componente principal.

    Retorna:
    plotly.graph_objs._figure.Figure: Gráfico de barras con la varianza explicada por cada componente principal.
    """
    # Lista de colores para las barras
    colors = px.colors.qualitative.Plotly

    # Crear el gráfico de barras con colores específicos para cada barra
    fig = go.Figure()
    for i in range(1, varianza.size + 1):
        fig.add_trace(go.Bar(x=[i], y=[varianza[i-1]*100],marker_color=colors[i % len(colors)], legendgroup=f'Componente {i}', name=f'Componente {i}'))

    fig.update_layout(
        title='Varianza explicada por cada componente',
        xaxis=dict(title="Componentes Principales"),
        yaxis=dict(title="Varianza explicada (%)")
    )

    # Se muestra el porcentaje de varianza de cada componente encima de su respectiva barra
    for i in range(1, varianza.size + 1):
        fig.add_annotation(x=i, y=varianza[i - 1] * 100, text=str(round(varianza[i - 1] * 100, 2)) + '%', yshift=10, showarrow=False, font_color='black')

    # Se agrega un scatter que pase por la varianza de cada componente
    fig.add_scatter(x=np.arange(1, varianza.size + 1, step=1), y=varianza * 100, mode='lines+markers', name='Varianza explicada', showlegend=False)

    # Eje X: valores
    fig.update_xaxes(tickmode='linear')
    return fig

def grafico_varianza_acumulada(varianza, varAcumulada, nComponentes, relevance):
    """
    Genera un gráfico que muestra la varianza acumulada en los componentes de un modelo.

    Parámetros:
    varianza (numpy.ndarray): Array que contiene la varianza de cada componente.
    varAcumulada (float): Varianza acumulada en los componentes.
    nComponentes (int): Número total de componentes.
    relevance (float): Valor de referencia para la varianza acumulada.

    Retorna:
    Una instancia de go.Figure con el gráfico generado.
    """
    fig = go.Figure()

    x_range = np.arange(1, varianza.size + 1, step=1)
    y_range = np.cumsum(varianza)

    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines+markers', marker=dict(size=10, color='blue'), name='Núm. Componente'))

    fig.update_layout(title='Varianza acumulada en los componentes',
                      xaxis_title='Número de componentes',
                      yaxis_title='Varianza acumulada')

    fig.add_shape(type="line", x0=1, y0=relevance, x1=nComponentes + 1, y1=relevance, line=dict(color="Red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=nComponentes + 1, y0=0, x1=nComponentes + 1, y1=varAcumulada, line=dict(color="Green", width=2, dash="dash"))

    fig.add_annotation(x=nComponentes + 1, y=varAcumulada, text=str(round(varAcumulada * 100, 1)) + f'%. {nComponentes + 1} Componentes', showarrow=True, arrowhead=1)

    fig.add_trace(go.Scatter(x=x_range, y=y_range, fill='tozeroy', mode='none', name='Área bajo la curva', fillcolor='rgba(0, 147, 255, 0.44)'))

    fig.update_xaxes(range=[1, varianza.size], tickmode='linear')
    fig.update_yaxes(range=[0, 1.1],
                     tickmode='array',
                     tickvals=np.arange(0, 1.1, step=0.1))
    return fig

def heatmap_cargas(df, pca, numComponentesPCA):
    """
    Genera un gráfico de mapa de calor de las cargas de los componentes principales utilizando Plotly.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de entrada.
    pca : sklearn.decomposition.PCA
        Objeto PCA ajustado previamente con los datos de entrada.
    numComponentesPCA : int
        Número de componentes principales a considerar en el gráfico.

    Retorno:
    --------
    fig : plotly.graph_objs.Figure
        Gráfico de mapa de calor de las cargas de los componentes principales.
    """
    CargasComponentes = pd.DataFrame(abs(pca.components_), columns=df.columns)
    CargasComponentess = CargasComponentes.head(numComponentesPCA + 1)

    # Crear una instancia de go.Figure y configurar el heatmap
    fig = go.Figure(go.Heatmap(z=CargasComponentess,
                                   x=df.columns,
                                   y=list(range(1, numComponentesPCA + 2)),
                                   colorscale='RdBu_r'))

    # Actualizar el diseño del gráfico
    fig.update_layout(title='Cargas de los componentes', xaxis_title='Variables', yaxis_title='Componentes')

    # 'Heatmap' para detectar cargas >= 50%
    fig.update_yaxes(tickmode='linear')
    for i in range(0, CargasComponentess.shape[0]):
        for j in range(0, CargasComponentess.shape[1]):
            if CargasComponentess.iloc[i, j] >= 0.5:
                color = 'white'
            else:
                color = 'black'
            fig.add_annotation(x=df.columns[j], y=i + 1, text=str(round(CargasComponentess.iloc[i, j], 4)), showarrow=False, font=dict(color=color))
    return fig


# ---- CALLBACK PARA CALCULAR LAS COMPONENTES PRINCIPALES ----
# Outputs: - Matriz Estandarizada
#          - Varianza Explicada
#          - Varianza Acumulada
#          - Proporción de Cargas
#
# Inputs: - Botón PCA: id='pca-btn'
#
# States: - Dropdown Escala: id='select-scale'
#        - Num Componentes: id='num_components'
#        - % Relevancia: id='relevance'
@callback(
    Output('matriz-estandarizada', 'data'), # (id del componente que actualizará, propiedad que se actualiza)
    Output('varianza-explicada', 'figure'),
    Output('varianza-acumulada', 'figure'),
    Output('cargas', 'figure'),
    Input('pca-btn', 'n_clicks'),
    State('select-scale', 'value'),
    State('num_components', 'value'),
    State('relevance', 'value'),
)
def calculo_pca(n_clicks, scale, components, relevancia):
    if n_clicks is not None:
        # ---- ESTANDARIZACIÓN ----
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
        mat_stand_dataframe = estandarizar_datos(df_numeric, scale)

        # ---- CÁLCULO DE COMPONENTES PRINCIPALES (PCA) ----
        pca = PCA(n_components=components).fit(mat_stand_dataframe) # components: elegidos por el usuario
        Varianza = pca.explained_variance_ratio_

        # ---- GRÁFICO DE BARRAS PARA REPRESENTAR LA VARIANZA EXPLICADA POR CADA COMPONENTE ----
        varianza_explicada = grafico_varianza_explicada(Varianza)

        # ---- GRÁFICO QUE MUESTRA LA VARIANZA ACUMULADA ----
        for i in range(0, Varianza.size):
            varAcumulada = sum(Varianza[0:i+1])
            if varAcumulada >= relevancia:
                varAcumuladaPCA = (varAcumulada - Varianza[i])
                numComponentesPCA = i - 1
                break

        varianza_acumulada = grafico_varianza_acumulada(Varianza, varAcumuladaPCA, numComponentesPCA, relevancia)

        # ---- HEATMAP PARA REPRESENTAR LA PROPORCIÓN DE CARGAS ----
        cargas = heatmap_cargas(df_numeric, pca, numComponentesPCA)
        

        return mat_stand_dataframe.to_dict('records'), varianza_explicada, varianza_acumulada, cargas
    
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate
