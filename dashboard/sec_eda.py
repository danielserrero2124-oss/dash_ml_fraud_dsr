"""Sección «EDA» — criterio 2 de la rúbrica: fuente y estructura, estadística descriptiva, distribuciones, atípicos,
faltantes, relaciones entre variables, justificación de las transformaciones y prevención de la fuga de datos."""

from dash import Input, Output, callback, dcc, html  # Componentes y decorador global de callbacks

from dashboard import figuras as fg  # Figuras de Plotly
from dashboard.componentes import desplegable, grafico, kpi, miles, numero, opciones_vars, rubrica, tabla, tarjeta  # Piezas de la interfaz
from dashboard.datos import ART, DF, TASA  # Datos compartidos
from src import config as cf  # Columnas del proyecto

SESGADAS = ["transaction_amount", "amount_last_1h", "amount_last_24h", "distance_from_home_km", "distance_from_last_tx_km",  # Variables muy asimétricas
            "amount_vs_avg_ratio", "avg_transaction_amount_30d", "std_transaction_amount_30d", "credit_limit"]  # (se sugiere escala log)
ATIPICOS = fg.tabla_atipicos(DF, cf.NUM_CONTINUAS)  # Atípicos por la regla de Tukey en las variables continuas
CATEGORICAS = ["transaction_channel", "merchant_category", "merchant_country", "device_type", "browser", "email_domain",  # Variables para el gráfico de tasas
               "international_transaction", "high_risk_country", "card_present", "used_chip", "used_pin",
               "transaction_hour", "transaction_day_of_week", "customer_state", "customer_gender", "is_weekend"]


def tabla_descriptiva():  # Estadística descriptiva de las 26 variables numéricas
    d = DF[cf.NUM_FEATURES].describe().T  # Conteo, media, desviación, mínimo, cuartiles y máximo
    asim = DF[cf.NUM_FEATURES].skew()  # Asimetría de cada variable
    filas = [(html.Code(v), numero(r["mean"], 2), numero(r["std"], 2), numero(r["min"], 2), numero(r["25%"], 2), numero(r["50%"], 2),  # Una fila por variable
              numero(r["75%"], 2), numero(r["max"], 2), numero(asim[v], 2)) for v, r in d.iterrows()]
    return tabla(["Variable", "Media", "Desv.", "Mín.", "Q1", "Mediana", "Q3", "Máx.", "Asimetría"], filas, numericas=range(1, 9))  # Tabla


def tabla_categoricas():  # Resumen de las 9 variables categóricas
    filas = []  # Filas
    for v in cf.CAT_FEATURES:  # Recorre las categóricas
        moda = DF[v].mode().iloc[0]  # Categoría más frecuente
        tasas = DF.groupby(v)[cf.TARGET].mean()  # Tasa de fraude por categoría
        filas.append((html.Code(v), DF[v].nunique(), str(moda), fg.pct((DF[v] == moda).mean()), f"{tasas.idxmax()} ({fg.pct(tasas.max())})"))  # Resumen
    return tabla(["Variable", "Categorías", "Moda", "% moda", "Categoría más riesgosa"], filas, numericas=[1, 3])  # Tabla


def seccion_eda():  # Construye la sección
    fraude = DF[DF[cf.TARGET] == 1]  # Subconjunto de fraudes
    legit = DF[DF[cf.TARGET] == 0]  # Subconjunto de legítimas
    estructura = [  # Estructura de las 43 columnas
        ("Numéricas continuas", len(cf.NUM_CONTINUAS), "Monto, distancias, historial, cupo, puntaje…"),
        ("Numéricas discretas / binarias", len(cf.NUM_DISCRETAS), "Banderas 0/1, conteos pequeños y la hora"),
        ("Categóricas", len(cf.CAT_FEATURES), "Canal, comercio, país, dispositivo, navegador…"),
        ("Objetivo", 1, "is_fraud"),
        ("Con fuga (excluidas)", 2, "fraud_type, fraud_amount_lost"),
        ("Identificadores (excluidos)", 4, "transaction_id, customer_id, card_id, merchant_id"),
        ("Fecha (para dividir)", 1, "transaction_datetime"),
    ]
    transformaciones = [  # Transformación, a qué se aplica y por qué
        ("Imputación con la mediana", "Numéricas (todos los modelos)", "No hay nulos en el dataset, pero protege al modelo en producción; la mediana es robusta a atípicos."),
        ("Imputación constante «None»", "browser", "El nulo es estructural (tienda y cajero no tienen navegador): imputar la moda inventaría un navegador."),
        ("One-hot (categorías con < 30 casos agrupadas)", "Categóricas", "No tienen orden; agrupar las raras evita columnas casi vacías que se sobreajustan."),
        ("Estandarización (media 0, desviación 1)", "LR, KNN y RVM", "KNN usa distancias y la regularización (C en LR, prior ARD en la RVM) compara pesos: requieren la misma escala."),
        ("B-splines cúbicos con nodos en cuantiles", "15 numéricas continuas (RVM)", "Permiten curvas no lineales en un modelo lineal en sus parámetros; los nodos en cuantiles toleran la asimetría."),
        ("One-hot de discretas y de la hora", "11 numéricas discretas (RVM)", "Efectos no monótonos, como el pico de fraude de 1:00 a 4:00."),
        ("Sin transformación", "RandomForest y GradientBoosting", "Los árboles son invariantes a transformaciones monótonas y robustos a atípicos."),
        ("Sin remuestreo; class_weight comparado", "Desbalance", "Se compara class_weight=None vs. balanced dentro de la validación cruzada; el umbral se ajusta en validación."),
    ]
    fuga = ART["fuga"]  # Resultados de la demostración de fuga
    return html.Div([
        rubrica(2),  # Criterio de la rúbrica
        html.Div([  # Indicadores
            kpi("Legítimas", miles(len(legit)), "clase 0"),
            kpi("Fraudes", miles(len(fraude)), f"clase 1 · {fg.pct(TASA, 2)}"),
            kpi("Nulos", fg.pct(DF['browser'].eq('None').mean()), "solo en browser (estructurales)"),
            kpi("Monto mediano", f"US$ {numero(legit['transaction_amount'].median(), 0)} → {numero(fraude['transaction_amount'].median(), 0)}", "legítima → fraude"),
            kpi("Distancia mediana", f"{miles(legit['distance_from_home_km'].median())} → {miles(fraude['distance_from_home_km'].median())} km", "legítima → fraude"),
        ], className="rejilla cols-5"),
        html.Div([  # Fuente y estructura
            tarjeta(
                html.H3("Fuente y estructura de los datos"),
                html.P(["Archivo ", html.Code("credit_card_fraud_detection_2026.csv"), ", entregado en el enunciado del proyecto integrador: datos sintéticos de "
                        "100 000 transacciones de enero a diciembre de 2026, 8 000 clientes, 3 000 comercios y 43 columnas. Cada fila es una transacción; "
                        "las variables de historial se calculan con transacciones anteriores."]),
                tabla(["Tipo de columna", "N.º", "Ejemplos"], estructura, numericas=[1]),
            ),
            tarjeta(html.H3("Composición del fraude"),
                    html.P("fraud_type se usa solo para describir y analizar errores, nunca como predictor.", className="nota"),
                    grafico("eda-tipos", fg.fig_tipos_fraude(DF))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Estadística descriptiva de las variables numéricas"),
                html.P("Asimetría > 1: distribución con cola larga a la derecha (montos, distancias, historial). Motiva la escala log en los gráficos y los nodos en cuantiles.", className="nota"),
                html.Div(tabla_descriptiva(), className="caja-scroll")),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([  # Categóricas + mensual
            tarjeta(html.H3("Resumen de las variables categóricas"), tabla_categoricas()),
            tarjeta(html.H3("Tasa de fraude por mes y división temporal"),
                    html.P("La tasa es estable entre meses (≈ 2 %): no hay una deriva fuerte que invalide la división temporal.", className="nota"),
                    grafico("eda-mensual", fg.fig_mensual(DF))),
        ], className="rejilla cols-2"),
        html.Div([  # Tasa por categoría + distribución numérica
            tarjeta(
                html.H3("Tasa de fraude por variable"),
                html.P("Rojo: categorías con tasa mayor que la global. Pasa el cursor para ver el volumen de cada barra.", className="nota"),
                html.Label("Variable", className="control-etiqueta"),
                desplegable("eda-cat-var", opciones_vars(CATEGORICAS), "merchant_country"),
                grafico("eda-categorica"),
            ),
            tarjeta(
                html.H3("Distribución de una variable numérica: fraude vs. legítima"),
                html.P("Densidades normalizadas: las dos clases son comparables aunque haya 48 legítimas por cada fraude.", className="nota"),
                html.Div([
                    html.Div([html.Label("Variable", className="control-etiqueta"), desplegable("eda-num-var", opciones_vars(cf.NUM_FEATURES), "distance_from_home_km")]),
                    html.Div([html.Label("Tipo de gráfico", className="control-etiqueta"),
                              dcc.RadioItems(id="eda-num-tipo", className="opciones", inline=True, value="histograma",
                                             options=[{"label": "Histograma", "value": "histograma"}, {"label": "Violín", "value": "violin"}, {"label": "Caja", "value": "caja"}])]),
                    html.Div([html.Label("Escala", className="control-etiqueta"),
                              dcc.Checklist(id="eda-num-log", className="opciones", inline=True, value=["log"],
                                            options=[{"label": "log10(x + 1)", "value": "log"}])]),
                ], className="rejilla cols-3", style={"marginBottom": "0"}),
                grafico("eda-numerica"),
            ),
        ], className="rejilla cols-2"),
        html.Div([  # Atípicos + faltantes
            tarjeta(html.H3("Valores atípicos (regla de Tukey: 1,5 × IQR)"),
                    html.P(["Los atípicos ", html.B("no se eliminan"), ": en monto, distancia y velocidad concentran más fraude que el resto, es decir, son informativos. "
                            "Se manejan con modelos robustos (árboles) y con splines de nodos en cuantiles (RVM)."], className="nota"),
                    grafico("eda-atipicos", fg.fig_atipicos(ATIPICOS))),
            html.Div([
                tarjeta(html.H3("Datos faltantes: los nulos de browser no son errores"),
                        html.P("Todas las compras en tienda y en cajero carecen de navegador. Se imputan con la categoría propia «None», no con la moda.", className="nota"),
                        grafico("eda-nulos", fg.fig_nulos_browser(DF))),
                tarjeta(html.H3("Tasa de fraude por hora y día"),
                        html.P("La madrugada (1:00–4:00) concentra tasas cuatro veces mayores que el resto del día.", className="nota"),
                        grafico("eda-hora", fg.fig_hora_dia(DF))),
            ], className="columna"),  # Dos tarjetas apiladas en la misma celda
        ], className="rejilla cols-2"),
        html.Div([  # Relaciones entre variables
            tarjeta(html.H3("Correlación de cada variable numérica con el fraude"),
                    html.P("Rojo: más fraude cuando la variable crece. Todas las relaciones lineales son débiles (|r| < 0,15): la señal está repartida en muchas variables.", className="nota"),
                    grafico("eda-corr-obj", fg.fig_correlacion_objetivo(DF, cf.NUM_FEATURES))),
            tarjeta(html.H3("Correlación entre predictoras numéricas"),
                    html.P("Azul: relación inversa. Rojo: directa. Los bloques de historial reciente están correlacionados entre sí; la regularización lo controla.", className="nota"),
                    grafico("eda-corr", fg.fig_matriz_correlacion(DF, cf.NUM_FEATURES))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Justificación de las transformaciones"),
                tabla(["Transformación", "Se aplica a", "Justificación"], transformaciones)),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([  # Prevención de fuga
            tarjeta(
                html.H3("Prevención de la fuga de información (data leakage)"),
                tabla(["Columna", "Tipo de fuga", "Decisión"], [
                    (html.Code("fraud_type"), "Directa: es el objetivo con otro nombre", "Excluida; solo para analizar errores"),
                    (html.Code("fraud_amount_lost"), "Posterior al evento", "Excluida"),
                    (html.Code("transaction_id, customer_id, card_id, merchant_id"), "Identificadores: memorización", "Excluidos"),
                    (html.Code("transaction_datetime"), "Marca de tiempo cruda", "Solo para ordenar y dividir"),
                    ("Preprocesamiento", "Ajustar escalas o categorías con todos los datos", "Dentro del Pipeline: se ajusta solo con entrenamiento"),
                    ("Partición", "Mezclar pasado y futuro", "División temporal antes de transformar"),
                ]),
            ),
            tarjeta(html.H3("Demostración: con fuga vs. sin fuga"),
                    html.P(f"Con las columnas posteriores al fraude el AUC sube a {numero(fuga['con_fuga']['auc_roc'])}: un modelo «perfecto» en el laboratorio e inútil en tiempo real, porque esas columnas no existen al momento de pagar.", className="nota"),
                    grafico("eda-fuga", fg.fig_fuga(fuga))),
        ], className="rejilla cols-2"),
        tarjeta(
            html.H3("Hallazgos principales del EDA"),
            html.Ul([
                html.Li([html.B("Desbalance extremo: "), "2,05 % de fraude. La exactitud es engañosa; se usan AUC-ROC, AUC-PR, precisión, recall y F1 de la clase fraude."]),
                html.Li([html.B("Geografía: "), "las transacciones internacionales tienen 10 veces más fraude que las nacionales (3,4 % vs. 0,35 %), y los países de alto riesgo (Rumania, Rusia, Nigeria, China) superan el 10 %."]),
                html.Li([html.B("Canal: "), "el comercio en línea (3,2 %) y la app móvil (1,9 %) superan a la tienda física (0,8 %); con tarjeta presente el riesgo cae a menos de la mitad."]),
                html.Li([html.B("Distancia y hora: "), "la mediana de distancia al domicilio pasa de 1 763 km en legítimas a 6 808 km en fraudes; la madrugada concentra tasas de 7–11 %."]),
                html.Li([html.B("Señal débil y repartida: "), "ninguna variable supera |r| = 0,15 con el objetivo; los efectos son casi aditivos, lo que justifica un modelo aditivo como la RVM con splines."]),
                html.Li([html.B("Calidad de datos: "), "no hay nulos salvo browser (41,7 %), que es estructural; los atípicos son informativos y se conservan."]),
            ]),
        ),
    ])


@callback(Output("eda-categorica", "figure"), Input("eda-cat-var", "value"))  # Tasa por categoría
def actualizar_categorica(var):
    return fg.fig_tasa_categorica(DF, var)  # Figura de tasas


@callback(Output("eda-num-log", "value"), Input("eda-num-var", "value"))  # Sugiere escala log para variables sesgadas
def sugerir_log(var):
    return ["log"] if var in SESGADAS else []  # Activa log solo si la variable es muy asimétrica


@callback(Output("eda-numerica", "figure"), Input("eda-num-var", "value"), Input("eda-num-tipo", "value"), Input("eda-num-log", "value"))  # Distribución
def actualizar_numerica(var, tipo, log):
    return fg.fig_numerica(DF, var, tipo, bool(log))  # Figura de distribución
