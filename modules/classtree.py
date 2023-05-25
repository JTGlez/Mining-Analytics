# Mining Analytics: Proyecto final de la asignatura de Minería de Datos
# Autores:
# Téllez González Jorge Luis
# Cruz Rangel Leonardo Said
# Módulo: Árboles de Decisión para clasificación.

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
from modules import home, eda, pca, regtree, classtree, regforest, classforest
import pathlib
import numpy as np
import plotly.matplotlylib as mpl_plotly
import plotly.express as px
import plotly.graph_objects as go
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import uuid
import graphviz
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def classtree_card():
    """
    :retorna: Un div que contiene la explicación del módulo de Árboles de Decisión: Regresión.

    """

    return html.Div(

        # ID del div.
        id="classtree-card",

        # Elementos hijos del div "eda-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Árbol de Decisión: Clasificación"), # Subtítulo.
            # Texto que explica la temática de la página web.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="Los árboles de decisión representan uno de los algoritmos de aprendizaje supervisado más utilizados, los cuales soportan tanto valores numéricos como nominales. Para esto, se construye una estructura jerárquica que divide los datos en función de condicionales."
                ,
            ),

            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="tree1",
                        src="/assets/tree1.png",
                        style = {'width': '25em', 'height': '15em'}
                    )
                ]
            ),
            # Texto secundario de explicacion.
            html.Div(
                style =  {'text-align': 'justify'},
                children = "Si bien los árboles son una opción muy noble para el modelado de datos, estos tienden a tener problemas de sobreajuste excesivo en los datos. Por ello, es necesario que se consideren cuidadosamente los hiperparámetros de elección para la generación del modelo. A continuación se muestran los parámetros que deben considerarse:",
            ),

            # Muestra una figura de parámetros del árbol
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '15em'},
                children=[
                    html.Img(
                        id="tree2",
                        src="/assets/tree2.png",
                        style = {'width': '100%', 'height': '50%'}
                    
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
classtree.layout = html.Div(
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
                    children=[classtree_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id = "right-column",
                    className="four columns",
                    children = html.Div([

                        html.H4("Carga o elige el dataset para iniciar la regresión con Árboles de Decisión", className= "text-upload"),

                        # Muestra el módulo de carga del dataset.
                        dcc.Upload(
                        id = 'upload-data-classtree',
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
                        id='upload-data-static-classtree',
                        options=dropdown_options,
                        value=None,  # Cambia el valor predeterminado a None
                        clearable=False,  # Deshabilita la opción de borrar el valor
                        placeholder="Selecciona una opción",  # Agrega el texto del placeholder
                        className='my-dropdown',
                    ),

                html.Hr(),
                html.Div(id = 'output-data-upload-classtree'),
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
            return classtree(df, filename, df.columns)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return classtree(df, filename, df.columns)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def classtree(df, filename, columns):
    """
    retorna: modelo de regresión usando un árbol de decisión regresor para la generación de clasificaciones y etiquetado de nuevos valores.

    """
    # Preparación de variables para su despliegue.
    global global_df
    global_df = df


    # Div de visualización en el layout.
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

        html.H3(
            "Elección de Variables Predictoras y Dependiente",
        ),

        html.Div(
            html.P("Selecciona de la siguiente lista las variables que deseas elegir como predictoras y tu variable target para realizar la regresión.")
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
                                            dbc.Badge("ⓘ Variables predictoras", color="primary",
                                                    id="tooltip-method", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center"},
                                                    ),
                                            dbc.Tooltip(
                                                "Selecciona aquí las variables predictoras de tu análisis.",
                                                target="tooltip-method"
                                            ),
                                        ],
                                        style={"height": "50px", "padding": "0"},
                                    ),
                                    style = {"height": "2.5em"}
                                ),
                                dbc.Row(
                                    dbc.Checklist(
                                        id='select-predictors',
                                        options=[{'label': col, 'value': col} for col in columns],
                                        style={"font-size": "small", "display": "grid", "justify-items": "start", "font-family": "Acumin, 'Helvetica Neue', sans-serif", "margin": "-1em 0 0 0"}
                                    ),
                                    style={"height": "auto"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    html.Div(
                                        [
                                            dbc.Badge("ⓘ Variable Regresora", color="primary",
                                                    id="tooltip-numpc", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"},
                                                    ),
                                            dbc.Tooltip(
                                                "Selecciona la variable target de tu análisis.",
                                                target="tooltip-numpc"
                                            ),
                                        ],
                                        style={"height": "auto", "padding": "0"},
                                    ),
                                ),
                                dbc.Row(
                                    dbc.Select(
                                        id='select-regressor',
                                        options=[{'label': col, 'value': col} for col in columns],
                                        value=None,
                                        style={"font-size": "medium"}
                                    ),
                                    style={"font-size":"small", "height": "2em", "font-family": "Acumin, 'Helvetica Neue', sans-serif"}
                                ),
                            ],
                            class_name="me-3"
                        ),
                    ],
                    style={"justify-content": "between", "height": "100%"}
                ),
                html.Div(
                    children=[
                        html.H3("Parámetros del árbol de decisión"),
                        html.Div(
                            children="A continuación selecciona los parámetros que desees modificar para la generación del árbol. Si lo deseas, también es posible generar el árbol sin parámetros adicionales.",
                            className="text-description"
                        ),
                        html.H5("Profundidad máxima del árbol: "),
                        html.P("Parámetro empleado para evitar overfitting en el árbol a generar, sin embargo, una profundidad muy baja puede llevar a un underfitting. "),
                        dcc.Input(type="number", id="input-max-depth", min=1, step=1),
                        html.H5("Número mínimo de muestras para dividir un nodo interno: "),
                        html.P("Permite indicar la cantidad mínima de datos necesarios para que un nodo de decisión pueda dividirse."),
                        dcc.Input(type="number", id="input-min-samples-split", min=2, step=1),
                        html.H5("Número mínimo de muestras para una hoja:"),
                        html.P("Permite indicar la cantidad mínima de datos que debe de tener un nodo hoja al interior del árbol."),
                        dcc.Input(type="number", id="input-min-samples-leaf", min=1, step=1),
                        html.H5("Tamaño de la muestra:"),
                        html.P("Permite indicar el ratio de valores en el dataset que se usarán para verificar el modelo. Generalmente 0.2 es un buen valor, pero es posible ajustarlo a 0.3 para obtener mayor representatividad."),
                        dcc.Input(type="number", id="input-test-size", min=0.1, max = 0.5, step=0.1),
                    ],
                    style={"font-size": "20px"},
                    className="mt-4",
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Generar árbol", id="submit-button", color="primary", className="mt-3", style={"display": "grid", "height": "80%", "align-items": "center", "margin": "0 auto"}),
                        width={"size": 2, "offset": 5},
                    ),
                    className="mt-3",
                ),
                html.Div(id="output-data-classtree", style = {"margin-top": "1em"}),
            ],
            style={"font-size": "20px"},
            className="mt-4",
        )

    ])

@callback(Output('output-data-upload-classtree', 'children'),
              [Input('upload-data-classtree', 'contents'),
               Input('upload-data-static-classtree', 'value')],
              [State('upload-data-classtree', 'filename'),
               State('upload-data-classtree', 'last_modified')])
def update_output(list_of_contents, selected_file, list_of_names, list_of_dates):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-classtree.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'upload-data-static-classtree.value':
        df = pd.read_csv(selected_file)
        return classtree(df, selected_file, df.columns)


def generate_decision_tree(X_train, X_test, Y_train, Y_test, max_depth=2, min_samples_split=4, min_samples_leaf=4):
    class_tree = DecisionTreeClassifier(
        random_state=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    class_tree.fit(X_train, Y_train)
    Y_Predicted = class_tree.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    tree_parameters = {
        "criterion": class_tree.criterion,
        "feature_importances": class_tree.feature_importances_,
        "Exactitud": accuracy_score(Y_test, Y_Predicted),
    }
    return comparison_df, class_tree, tree_parameters, Y_Predicted

def generate_decision_treeS(X_train, X_test, Y_train, Y_test):
    class_tree = DecisionTreeClassifier(
        random_state=0,
    )
    class_tree.fit(X_train, Y_train)
    Y_Predicted = class_tree.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    tree_parameters = {
        "criterion": class_tree.criterion,
        "feature_importances": class_tree.feature_importances_,
        "Exactitud": accuracy_score(Y_test, Y_Predicted),
    }
    return comparison_df, class_tree, tree_parameters, Y_Predicted

def create_input_form(predictors):
    input_form = []
    for predictor in predictors:
        input_form.append(
            html.Div(
                [
                    html.Label(predictor),
                    dcc.Input(
                        type="number",
                        id=f"input-{predictor}",  # Agrega el atributo id a la entrada
                    ),
                ],
                className="form-group",
            )
        )
    return input_form

@callback(Output("input-form-class", "children"), Input("submit-button", "n_clicks"))
def update_input_form(n_clicks):
    if n_clicks is None:
        return ""
    return create_input_form(global_predictors)

def predict_new_values(class_tree, predictors, input_values):
    input_data = pd.DataFrame(input_values, columns=predictors)
    prediction = class_tree.predict(input_data)
    return prediction

@callback(
    Output("classification-result", "children"),
    Input("predict-button", "n_clicks"),
    State("input-form-class", "children"),
)
def show_prediction(n_clicks, input_form):
    if n_clicks is None or input_form is None:
        return ""

    input_values = {}
    all_states = dash.callback_context.states
    for child in input_form:
        label = child['props']['children'][0]['props']['children']
        if label in global_predictors:
            input_id = child['props']['children'][1]['props']['id']
            try:
                # Agrega el id del campo de entrada a all_states
                all_states[f"{input_id}.value"] = child['props']['children'][1]['props']['value']
                input_values[label] = float(all_states[f"{input_id}.value"])
            except KeyError:
                print(f"Error: No se encontró la clave '{input_id}.value' en dash.callback_context.states")
                print("Valores de entrada:", input_values)
                print("Claves presentes en dash.callback_context.states:", dash.callback_context.states.keys())

    prediction = predict_new_values(global_class_tree, global_predictors, [input_values])
    return f"La clasificación con base en los valores introducidos es: {prediction[0]}"

@callback(
    Output("download-ar-class", "data"),
    Input("btn-ar", "n_clicks"),
    prevent_initial_call=True,
)
def generar_arbol_svg(n_clicks):
    import graphviz
    from sklearn.tree import export_graphviz

    Elementos = export_graphviz(global_class_tree,
                            feature_names = global_predictors,
                            filled = True,
                            rounded = True,
                            special_characters = True)
    Arbol = graphviz.Source(Elementos)
    Arbol.format = 'pdf'

    return dcc.send_file(Arbol.render(filename='ArbolAR', view=True))

@callback(
    Output("output-data-classtree", "children"),
    Input("submit-button", "n_clicks"),
    State("select-predictors", "value"),
    State("select-regressor", "value"),
    State("input-max-depth", "value"),
    State("input-min-samples-split", "value"),
    State("input-min-samples-leaf", "value"),
    State("input-test-size", "value")

)
def split_data(n_clicks, predictors, regressor, max_depth, min_samples_split, min_samples_leaf, test_size = 0.2):
    global global_df
    global global_predictors
    global global_regressor
    
    if n_clicks is None:
        return ""

    if predictors is None or regressor is None:
        return "Por favor, seleccione las variables predictoras y la variable regresora."

    if global_df is None:
        return "No se ha cargado ningún dataset."

    global_predictors = predictors
    global_regressor = regressor

    X = np.array(global_df[predictors])
    global global_X 
    global_X = X
    Y = np.array(global_df[[regressor]])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = 0, shuffle = True)
    
    if max_depth is None and min_samples_split is None and min_samples_leaf is None:
        comparison_df, class_tree, tree_parameters, Y_Predicted = generate_decision_treeS(
            X_train, X_test, Y_train, Y_test
        )
    else:
        comparison_df, class_tree, tree_parameters, Y_Predicted = generate_decision_tree(
            X_train, X_test, Y_train, Y_test, max_depth, min_samples_split, min_samples_leaf
        )

    global global_class_tree 
    global_class_tree = class_tree

    comparison_df.columns = comparison_df.columns.astype(str)
    comparison_table = dash_table.DataTable(
        data=comparison_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in comparison_df.columns.astype(str)],
        style_table={'height': '300px', 'overflowX': 'auto'},
    )

    # Crea una tabla con los parámetros del árbol
    parameters_list = [
        {"parameter": key, "value": value}
        for key, values in tree_parameters.items()
        for value in (values if isinstance(values, (list, np.ndarray)) else [values])
    ]
    parameters_df = pd.DataFrame(parameters_list)
    parameters_table = dash_table.DataTable(
        data=parameters_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in parameters_df.columns.astype(str)],
        style_table={'overflowX': 'auto', "border": "none"},
    )

    # Calcular el informe de clasificación
    report = classification_report(Y_test, Y_Predicted, output_dict=True)

    # Crear un DataFrame de Pandas con los datos del informe de clasificación
    report_df = pd.DataFrame(report).transpose().round(2)

    # Reiniciar el índice y renombrar la columna de índice
    report_df = report_df.reset_index().rename(columns={'index': 'Metric'})

    report_table = dash_table.DataTable(
        data=report_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in report_df.columns.astype(str)],
        style_table={'overflowX': 'auto', "border": "none"},
    )

    importance_df = pd.DataFrame({'Variable': predictors, 'Importancia': tree_parameters['feature_importances']}).sort_values('Importancia', ascending=False)
    importance_table = dash_table.DataTable(
        data=importance_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in importance_df.columns.astype(str)],
        style_table={'overflowX': 'auto'},
    )
    # Calcular la matriz de clasificación
    ModeloClasificacion1 = global_class_tree.predict(X_test)
    class_matrix = pd.crosstab(Y_test.ravel(), 
                            ModeloClasificacion1, 
                            rownames=['Reales'], 
                            colnames=['Clasificación']) 

    # Columna con los nombres de los valores de predicción posibles
    class_matrix[' '] = [''] * len(class_matrix)
    for column in class_matrix.columns[:-1]:
        class_matrix[' '][class_matrix.columns.get_loc(column)] = column

    # Reiniciar el índice
    class_matrix = class_matrix.reset_index()

    class_matrix_table = dash_table.DataTable(
        data=class_matrix.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in class_matrix.columns.astype(str)],
        style_table={'overflowX': 'auto'},
    )

    tree_rules = export_text(class_tree, feature_names=predictors)
    tree_rules_container = html.Div(
        children=[html.Pre(tree_rules)],
        style={'height': '20em', 'overflowY': 'scroll', 'border': '1px solid', 'padding': '10px'},
    )

    # Binarizar las etiquetas de la prueba
    unique_classes = np.unique(Y_test)
    print(unique_classes)
    y_test_bin = label_binarize(Y_test, classes=unique_classes)
    y_score = global_class_tree.predict_proba(X_test)
    n_classes = y_test_bin.shape[1]
    print(n_classes)
    class_size = len(unique_classes)

    # Verificar si la clasificación es binaria o multiclase
    if class_size == 2:
        # Caso binario
        fig, ax = plt.subplots()
        buf = io.BytesIO() # in-memory files
        RocCurveDisplay.from_estimator(global_class_tree,
                                    X_test,
                                    Y_test,
                                    ax=ax)
        plt.savefig(buf, format = "png")
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        buf.close()
        roc_graph = "data:image/png;base64,{}".format(data)

    else:
        # Caso multiclase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        buf = io.BytesIO()  # in-memory files
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr[i], tpr[i], lw=2, label='AUC para la clase {}: {:.2f}'.format(i + 1, auc(fpr[i], tpr[i])))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Rendimiento')
        plt.legend(loc="lower right")  # Añade la leyenda en la esquina inferior derecha
        plt.savefig(buf, format="png")
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
        buf.close()
        roc_graph = "data:image/png;base64,{}".format(data)



    new_forecasts_section = html.Div(
        [
            html.H3("Generar nuevos pronósticos"),
            html.P("Introduce los valores de las variables predictoras:"),
            html.Div(id="input-form-class"),
            html.Button("Clasificar", id="predict-button", className="mt-3"),
            html.Div(id="classification-result", className="mt-4"),
        ],
        className="mt-4",
    )


    return (
    html.H3("Generación del Árbol de Decisión:"),
    html.P("Los parámetros del árbol generado son los siguientes:"),
    parameters_table,
    html.Br(),
    report_table,
    html.Br(),
    html.P("Se han obtenido los siguiente valores de pronóstico en el set de entrenamiento, los cuales se comparan con los valores reales:"),
    comparison_table,
    html.Br(),
    html.P("La matriz de clasificación obtenida permite identificar en qué observaciones el modelo acertó y en cuáles falló."),
    class_matrix_table,
    html.Br(),
    html.P("A continuación se especifica la importancia numérica [0-1] de las variables predictoras en el modelo construido:"),
    importance_table,
    html.P('A continuación se muestra la curva ROC del árbol de decisión:'),
    html.Div(
        children = html.Img(id='roc-curve', src=roc_graph),
        style = {"display": "grid", "justify-content": "center"}
    ),
    html.Br(),
    html.P("El árbol fue construido de con las siguientes reglas:"),
    tree_rules_container,
    html.Br(),
    html.P("A continuación, puede descargar el árbol generado con el fin de identificar si es necesario llevar a cabo un proceso de podado. Para esto, puede modificar los parámetros de generación según sea necesario."),
    html.Br(),
    html.Div([
        dbc.Row(
            dbc.Col(
                html.Div([
                    dbc.Button("Descargar Árbol", id="btn-ar", color="primary", className="mt-3", style={"display": "grid", "height": "80%", "align-items": "center", "margin": "0 auto"}),
                    dcc.Download(id="download-ar-class")
                ]),
                width={"size": 2, "offset": 5},
            ),
            className="mt-3",
        ),
    ]),

    html.Br(),

    html.Br(),
    new_forecasts_section
)