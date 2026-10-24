"""Sección «Contexto» — criterio 1 de la rúbrica: problema, objetivos, variables, tipo de tarea y formulación matemática."""

import dash  # get_asset_url para la imagen
from dash import html  # Componentes HTML

from dashboard import figuras as fg  # Formato de porcentajes
from dashboard.componentes import formula, kpi, miles, numero, pasos, rubrica, tabla, tarjeta  # Piezas de la interfaz
from dashboard.datos import DF, RES, TASA  # Datos compartidos
from src import config as cf  # Columnas del proyecto

FORMULACION = r"""
**Datos.** $\mathcal{D}=\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$, con $n = 100\,000$ transacciones ordenadas en el tiempo,
$\mathbf{x}_i \in \mathbb{R}^{26}\times\mathcal{C}_1\times\dots\times\mathcal{C}_9$ (26 numéricas y 9 categóricas) e $y_i \in \{0, 1\}$.

**Objetivo de aprendizaje.** Estimar la probabilidad condicional de fraude

$$f(\mathbf{x}) \approx P(y = 1 \mid \mathbf{x}), \qquad f:\mathcal{X}\rightarrow[0,1].$$

**Entrenamiento** (minimización del riesgo empírico regularizado con pérdida logarítmica):

$$\hat{\theta} = \arg\min_{\theta}\; -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \ln f_\theta(\mathbf{x}_i) + (1-y_i)\ln\big(1-f_\theta(\mathbf{x}_i)\big)\Big] + \lambda\,\Omega(\theta)$$

**Decisión.** $\hat{y} = \mathbb{1}\left[f(\mathbf{x}) \ge \tau\right]$, con $\tau^{*} = \arg\max_{\tau} F_1(\tau)$ calculado en validación (nunca en prueba).

**Evaluación** con clases desbalanceadas ($\pi = P(y=1) \approx 0{,}0205$):

$$\text{AUC-ROC} = P\big(f(\mathbf{x}^{+}) > f(\mathbf{x}^{-})\big), \qquad
\text{AUC-PR} = \sum_k (R_k - R_{k-1})\,P_k \;\;(\text{azar} = \pi).$$

**Restricción temporal.** Con $t_c$ = 1 de noviembre de 2026, se entrena con $t < t_c$ y se evalúa con $t \ge t_c$: el modelo nunca ve el futuro.
"""


def seccion_contexto():  # Construye la sección
    familias = [  # Diccionario de datos agrupado por familia de variables
        ("Transacción", "transaction_amount, transaction_hour, transaction_day_of_week, is_weekend, transaction_channel, card_present, used_chip, used_pin", "Monto, momento, canal y forma de pago"),
        ("Comercio", "merchant_category, merchant_country, international_transaction, high_risk_country", "Rubro y ubicación del comercio"),
        ("Cliente", "customer_age, customer_gender, customer_state, account_age_days, credit_limit, credit_score, num_cards_held, email_domain", "Perfil y antigüedad"),
        ("Dispositivo", "device_type, browser", "Terminal o equipo; browser es nulo en tienda y cajero"),
        ("Historial reciente", "transactions_last_1h/24h/7d, amount_last_1h/24h, days_since_last_transaction, avg/std_transaction_amount_30d, amount_vs_avg_ratio", "Velocidad y comportamiento, calculado con transacciones anteriores"),
        ("Geografía y riesgo", "distance_from_home_km, distance_from_last_tx_km, num_declined_last_30d, num_chargebacks_ever", "Distancias, rechazos y contracargos"),
    ]
    perdida = DF["fraud_amount_lost"].sum()  # Pérdida total por fraude en el dataset
    return html.Div([
        rubrica(1),  # Criterio de la rúbrica que cubre esta sección
        html.Div([  # Portada: texto + imagen
            tarjeta(
                html.H2(["Detectar el ", html.Span("fraude con tarjeta", className="resaltado"), " en el instante en que ocurre"]),  # Titular
                html.P("El fraude con tarjetas de crédito causa pérdidas de decenas de miles de millones de dólares al año en el mundo y crece "
                       "al ritmo del comercio electrónico y de los pagos móviles. Para un banco o un procesador de pagos, la decisión se toma en "
                       "milisegundos: bloquear a tiempo una transacción fraudulenta evita la pérdida, pero bloquear una legítima molesta al cliente "
                       "y genera costos operativos."),  # Importancia del problema
                html.P(["El dataset ", html.B("credit_card_fraud_detection_2026"), " reúne 100 000 transacciones de 2026 hechas por 8 000 clientes "
                        "en 3 000 comercios. El reto central es el ", html.B("desbalance"), ": solo el 2,05 % son fraude, así que un modelo que "
                        "siempre diga «legítima» acierta el 98 % de las veces sin detectar un solo fraude. Por eso se evalúa con ",
                        html.B("AUC-ROC"), " y, sobre todo, con ", html.B("AUC-PR"), "."]),  # Reto del dataset
                html.Div([html.Span(t, className="chip") for t in ["Aprendizaje supervisado", "Clasificación binaria", "Clases desbalanceadas (98/2)",
                                                                    "Validación temporal", "Modelo original: RVM bayesiana", "MLOps local"]]),  # Etiquetas
                clase="heroe",
            ),
            tarjeta(html.Img(src=dash.get_asset_url("tarjeta.svg"), className="imagen-heroe", alt="Tarjeta de crédito protegida por un escudo"), clase="sin-relleno"),  # Imagen
        ], className="rejilla cols-heroe"),
        html.Div([  # Indicadores generales
            kpi("Transacciones", miles(len(DF)), "8.000 clientes · 3.000 comercios"),
            kpi("Fraudes", miles(DF[cf.TARGET].sum()), f"{fg.pct(TASA, 2)} del total"),
            kpi("Pérdida por fraude", f"US$ {miles(perdida)}", "suma de fraud_amount_lost"),
            kpi("AUC-ROC de la RVM", numero(RES["RVM"]["test"]["auc_roc"]), "prueba nov–dic, nunca vista al entrenar"),
        ], className="rejilla cols-4"),
        html.Div([  # Objetivos + tipo de tarea
            tarjeta(
                html.H3("Objetivos"),
                html.P([html.B("General: "), "construir un clasificador que estime en tiempo real la probabilidad de que una transacción sea fraude, "
                        "con un flujo libre de fuga de datos y con capacidad de generalizar a meses futuros."]),
                pasos(["Explorar el dataset: desbalance, distribuciones, atípicos, faltantes y relaciones con el fraude.",
                       "Entrenar y comparar los cuatro modelos recomendados con Pipeline, GridSearchCV y validación cruzada estratificada.",
                       "Implementar desde cero el modelo original (Relevance Vector Machine) y contrastarlo con la prueba de DeLong.",
                       "Elegir el umbral de decisión en validación y analizar errores por tipo de fraude.",
                       "Desplegar el dashboard con Docker y Kubernetes."]),
            ),
            tarjeta(
                html.H3("Tipo de tarea de Machine Learning"),
                html.Ul([
                    html.Li([html.B("Aprendizaje supervisado: "), "cada transacción tiene su etiqueta is_fraud."]),
                    html.Li([html.B("Clasificación binaria probabilística: "), "el modelo entrega P(fraude | x) y un umbral la convierte en decisión."]),
                    html.Li([html.B("Clases muy desbalanceadas: "), "2,05 % de positivos; la exactitud no es informativa."]),
                    html.Li([html.B("Datos tabulares mixtos: "), "26 variables numéricas y 9 categóricas."]),
                    html.Li([html.B("Orden temporal: "), "la validación respeta el tiempo (se entrena con el pasado y se predice el futuro)."]),
                ]),
                html.H3("Variables", style={"marginTop": "14px"}),
                html.P([html.B("Objetivo: "), html.Code("is_fraud"), " (1 = fraude, 0 = legítima). ",
                        html.B("Predictoras: "), f"{len(cf.FEATURES)} variables ({len(cf.NUM_FEATURES)} numéricas y {len(cf.CAT_FEATURES)} categóricas)."]),
            ),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Formulación matemática del problema"), formula(FORMULACION)),  # Ecuaciones del problema
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(
            html.H3("Diccionario de datos: variables predictoras por familia"),
            tabla(["Familia", "Variables", "Qué describe"], [(html.B(f), html.Code(v), d) for f, v, d in familias]),  # Tabla
        ),
    ])
