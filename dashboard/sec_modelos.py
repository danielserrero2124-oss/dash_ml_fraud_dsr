"""Sección «ML Models», organizada en sub-pestañas según la rúbrica:
  * Fundamentos teóricos (criterio 3): formulación de la RVM, función objetivo, supuestos, regularización y complejidad.
  * Diseño experimental (criterio 4): partición, Pipeline, hiperparámetros, validación y reproducibilidad.
  * Resultados (criterio 5): métricas, comparación con la línea base y alternativas, errores y sobreajuste.
  * Conclusiones y preguntas (criterios 6 y 7): conclusiones, limitaciones, trabajo futuro y preguntas de la sustentación.
"""

import inspect  # Extrae el código fuente real de las funciones para mostrarlo en el dashboard
import textwrap  # Quita la sangría común del código de los métodos

import pandas as pd  # Construcción de la transacción del simulador
from dash import Input, Output, callback, dcc, html  # Componentes y decorador global de callbacks

from dashboard import figuras as fg  # Figuras de Plotly
from dashboard.componentes import (cita, desplegable, formula, grafico, kpi, miles, numero, opciones_vars, pasos, pregunta,  # Piezas de la interfaz
                                   rubrica, tabla, tarjeta)
from dashboard import entrenar  # Flujo de entrenamiento (solo para mostrar su código)
from dashboard.datos import ART, DF, EXP, INFO_RVM, MEJOR_REC, PRUEBAS, RES, RVM, TASA, TRAIN_DF, Y_TEST  # Datos compartidos
from src import config as cf  # Columnas del proyecto
from src import modelado as mod  # Métricas y catálogo de modelos
from src.rvm import RVMClassifier  # Modelo original (solo para mostrar su código)

CODIGO = {  # Fragmentos de código fuente reales que se muestran en «Diseño experimental»
    "Preprocesamiento de los modelos recomendados (ColumnTransformer)": mod.make_preprocessor,
    "Preprocesamiento de la RVM (splines + one-hot)": mod.make_preprocessor_rvm,
    "Catálogo de modelos y mallas de hiperparámetros": mod.catalogo_modelos,
    "Pipeline + GridSearchCV + StratifiedKFold": mod.train_pipeline,
    "Umbral de decisión en validación": mod.umbral_max_f1,
    "Flujo completo: división temporal, entrenamiento y evaluación": entrenar.entrenar_todo,
    "RVM: ajuste (Laplace + máxima evidencia + poda)": RVMClassifier.fit,
    "Prueba de DeLong": mod.prueba_delong,
}

ETQ = fg.ETIQUETAS_MODELOS  # Nombres visibles de los modelos
N_FRAUDES_TEST = int(Y_TEST.sum())  # Fraudes en el periodo de prueba
DL_MEJOR = ART["delong"][MEJOR_REC]  # Comparación RVM vs. mejor modelo recomendado


def params_texto(params):  # Hiperparámetros elegidos en texto corto (sin prefijos del Pipeline)
    return ", ".join(f"{k.split('__')[-1]} = {v}" for k, v in params.items())  # p. ej. "n_knots = 6"


# ---------------------------------------------------------------------------
# Sub-pestaña 1: Fundamentos teóricos de la RVM (criterio 3)
# ---------------------------------------------------------------------------
MODELO_RVM = r"""
**Modelo.** Cada transacción se representa con $M$ funciones base $\phi(\mathbf{x}) = (1, \phi_1(\mathbf{x}), \dots, \phi_{M-1}(\mathbf{x}))^\top$:
B-splines cúbicos de las variables continuas y variables indicadoras (one-hot) de las discretas y categóricas. La RVM es **aditiva** en el logit:

$$P(y = 1 \mid \mathbf{x}, \mathbf{w}) = \sigma\big(\mathbf{w}^\top \phi(\mathbf{x})\big), \qquad \sigma(a) = \frac{1}{1 + e^{-a}}.$$

**Verosimilitud** (Bernoulli): $\;p(\mathbf{y} \mid \mathbf{w}) = \prod_{n=1}^{N} \sigma(a_n)^{y_n}\big(1 - \sigma(a_n)\big)^{1 - y_n}, \quad a_n = \mathbf{w}^\top\phi(\mathbf{x}_n).$

**Prior ARD** (*Automatic Relevance Determination*): una precisión propia para cada peso,

$$p(\mathbf{w} \mid \alpha) = \prod_{i=0}^{M-1} \mathcal{N}\big(w_i \mid 0, \alpha_i^{-1}\big).$$

**Paso 1 — Aproximación de Laplace.** Con $\alpha$ fijo, el máximo a posteriori $\mathbf{w}^{*}$ se halla con Newton (IRLS):

$$\mathbf{g} = \Phi^\top(\mathbf{y} - \mathbf{p}) - \mathbf{A}\mathbf{w}, \qquad
\mathbf{H} = \Phi^\top\mathbf{B}\Phi + \mathbf{A}, \qquad
\mathbf{w} \leftarrow \mathbf{w} + \mathbf{H}^{-1}\mathbf{g},$$

con $\mathbf{B} = \operatorname{diag}\big(p_n(1 - p_n)\big)$ y $\mathbf{A} = \operatorname{diag}(\alpha)$. La posterior se aproxima por $\mathcal{N}(\mathbf{w}^{*}, \Sigma)$ con $\Sigma = \mathbf{H}^{-1}$.

**Paso 2 — Máxima evidencia (verosimilitud tipo II).** Se maximiza $\ln p(\mathbf{y} \mid \alpha)$ con la regla de punto fijo de MacKay:

$$\gamma_i = 1 - \alpha_i \Sigma_{ii}, \qquad \alpha_i^{\text{nuevo}} = \frac{\gamma_i}{(w_i^{*})^2}.$$

$\gamma_i \in [0, 1]$ mide cuánto determinan los datos al peso $w_i$. Cuando $\alpha_i \to \infty$ la posterior de $w_i$ colapsa en 0 y la función base se **poda**:
el modelo final usa solo los «vectores relevantes».

**Predicción moderada** (incluye la incertidumbre de $\mathbf{w}$):

$$P(y = 1 \mid \mathbf{x}) \approx \sigma\!\left(\frac{\mu_a}{\sqrt{1 + \pi s^2/8}}\right), \qquad \mu_a = \mathbf{w}^{*\top}\phi(\mathbf{x}), \quad s^2 = \phi(\mathbf{x})^\top\Sigma\,\phi(\mathbf{x}).$$
"""

OBJETIVO_RVM = r"""
**Función objetivo de los pesos** (log-posterior = − entropía cruzada − penalización ARD):

$$\mathcal{L}(\mathbf{w}) = \sum_{n}\big[y_n \ln p_n + (1 - y_n)\ln(1 - p_n)\big] - \tfrac{1}{2}\sum_i \alpha_i w_i^2.$$

**Función objetivo de los hiperparámetros** (log-evidencia aproximada por Laplace):

$$\ln p(\mathbf{y} \mid \alpha) \approx \ln p(\mathbf{y} \mid \mathbf{w}^{*}) + \tfrac{1}{2}\sum_i \ln\alpha_i - \tfrac{1}{2}\sum_i \alpha_i (w_i^{*})^2 - \tfrac{1}{2}\ln|\mathbf{H}|.$$

El término $-\tfrac{1}{2}\ln|\mathbf{H}|$ es el **factor de Occam**: penaliza los modelos que necesitan muchos parámetros bien determinados.
"""


def sub_fundamentos():  # Construye la sub-pestaña de fundamentos teóricos
    rel = INFO_RVM["relevancia"]  # Relevancia por variable
    curvas = INFO_RVM["curvas"]  # Curvas de contribución
    orden = sorted(curvas, key=lambda v: max(curvas[v]["y"]) - min(curvas[v]["y"]), reverse=True)  # Variables por efecto total
    comparacion = [  # Comparación teórica de los cinco modelos
        ("LogisticRegression", "Entropía cruzada + λ‖w‖²", "C (inverso de λ)", "Logit lineal en las variables", "O(n·d) por iteración"),
        ("RandomForestClassifier", "Impureza de Gini en cada corte", "Profundidad, hojas mínimas", "Ninguno paramétrico; árboles poco correlacionados", "O(T·n·log n·√d)"),
        ("KNeighborsClassifier", "Ninguna (memoriza los datos)", "k, ponderación", "Vecinos cercanos tienen la misma clase", "O(n·d) por predicción"),
        ("GradientBoostingClassifier", "Entropía cruzada por descenso de gradiente funcional", "Árboles, profundidad, tasa de aprendizaje", "Aditivo en árboles", "O(T·n·d)"),
        ("RVM-Fraude (original)", "Log-posterior + log-evidencia (bayesiana)", "Nodos de splines; α se estima, no se busca", "Aditivo en el logit, posterior ≈ gaussiana", "O(n·M² + M³) por iteración"),
    ]
    return html.Div([
        rubrica(3),  # Criterio de la rúbrica
        html.Div([
            tarjeta(
                html.H3("Modelo original: RVM-Fraude (Relevance Vector Machine bayesiana, desde cero)"),
                html.P(["Elegida del listado del curso (grupo «Máquinas de vectores soporte y métodos de kernel»). Es la versión ", html.B("bayesiana y dispersa"),
                        " de una SVM: en lugar de buscar C y γ por validación cruzada, ", html.B("aprende cuánto regularizar cada función base"),
                        " maximizando la evidencia marginal, entrega ", html.B("probabilidades"), " (no solo una frontera) y da ", html.B("incertidumbre"), " para cada predicción."]),
                html.P(["¿Por qué encaja con este problema? El EDA mostró señales débiles, no lineales y casi aditivas repartidas en muchas variables. "
                        "Con splines y one-hot, la RVM modela cada efecto como una curva y la poda ARD elimina automáticamente lo irrelevante: de ",
                        html.B(str(INFO_RVM["n_bases"])), " funciones base conserva ", html.B(str(INFO_RVM["n_relevantes"])), " tras ",
                        html.B(str(INFO_RVM["n_iter"])), " iteraciones."]),
                html.P(["Implementación propia en ", html.Code("src/rvm.py"), " (solo numpy); se integra a Pipeline y GridSearchCV como cualquier estimador de scikit-learn."]),
                cita("Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of Machine Learning Research, 1, 211–244."),
                cita("MacKay, D. J. C. (1992). The evidence framework applied to classification networks. Neural Computation, 4(5), 720–736.", "https://doi.org/10.1162/neco.1992.4.5.720"),
                cita("Bishop, C. M. (2006). Pattern Recognition and Machine Learning, cap. 7.2. Springer."),
                clase="destacada",
            ),
            tarjeta(html.H3("Formulación matemática del algoritmo"), formula(MODELO_RVM), clase="destacada"),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Función objetivo y principio de aprendizaje"), formula(OBJETIVO_RVM),
                    html.P(["Principio: ", html.B("inferencia bayesiana con máxima evidencia"), ". Se alterna (1) Newton para los pesos y (2) punto fijo para α, "
                            "hasta que α deja de cambiar. No hay descenso de gradiente con tasa de aprendizaje que ajustar."], className="nota")),
            tarjeta(
                html.H3("Supuestos del modelo"),
                html.Ul([
                    html.Li([html.B("Independencia: "), "las transacciones son condicionalmente independientes dado x (verosimilitud como producto)."]),
                    html.Li([html.B("Enlace logístico y aditividad: "), "el logit es suma de efectos por variable; no modela interacciones salvo que se agreguen como funciones base."]),
                    html.Li([html.B("Posterior casi gaussiana: "), "la aproximación de Laplace es buena con muchos datos (n = 66 634), pero no exacta."]),
                    html.Li([html.B("Estacionariedad: "), "la relación x → fraude de ene–oct se mantiene en nov–dic."]),
                ]),
                html.H3("Regularización, complejidad y generalización", style={"marginTop": "14px"}),
                html.Ul([
                    html.Li([html.B("Regularización automática: "), "cada α_i es una penalización ridge propia, estimada con los datos; el factor de Occam de la evidencia evita el sobreajuste sin validación cruzada para α."]),
                    html.Li([html.B("Complejidad del modelo: "), f"la controla el número de nodos de los splines (elegido por CV: {params_texto(RVM['params'])}) y la poda ARD."]),
                    html.Li([html.B("Costo computacional: "), "O(n·M² + M³) por iteración (n filas, M funciones activas); M baja con la poda, así que cada iteración es más barata que la anterior."]),
                    html.Li([html.B("Generalización: "), f"AUC-ROC de {numero(RVM['cv_auc_roc'])} en validación cruzada, {numero(RVM['val_auc_roc'])} en validación y {numero(RVM['test']['auc_roc'])} en prueba: sin brecha de sobreajuste."]),
                ]),
            ),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Poda ARD: funciones base relevantes por iteración"),
                    html.P("Al maximizar la evidencia, cada vez más α_i crecen sin límite y sus funciones base se eliminan.", className="nota"),
                    grafico("rvm-relevantes", fg.fig_rvm_relevantes(INFO_RVM["historial"]))),
            tarjeta(html.H3("Convergencia: log-evidencia por iteración"),
                    html.P("La RVM maximiza ln p(y | α); la curva crece y se estabiliza cuando α converge.", className="nota"),
                    grafico("rvm-evidencia", fg.fig_rvm_evidencia(INFO_RVM["historial"]))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Precisión ARD de cada función base"),
                html.P("Magenta: funciones relevantes (α pequeño, el peso puede alejarse de 0). Gris: podadas (α ≥ 10⁶, peso exactamente 0).", className="nota"),
                grafico("rvm-alphas", fg.fig_rvm_alphas(INFO_RVM["alphas"]))),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Efecto total de cada variable en la RVM"),
                    html.P("Rango de la curva de contribución al logit (máx − mín). Pasa el cursor para ver cuántas funciones base sobrevivieron.", className="nota"),
                    grafico("rvm-relevancia", fg.fig_rvm_relevancia(curvas, rel))),
            tarjeta(html.H3("Curva de contribución con incertidumbre bayesiana"),
                    html.P("Contribución de la variable al logit de fraude y su intervalo del 95 % (de la covarianza posterior Σ). Rojo/positivo = más riesgo.", className="nota"),
                    html.Label("Variable", className="control-etiqueta"),
                    desplegable("rvm-var", opciones_vars(orden), orden[0]),
                    grafico("rvm-curva")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Comparación teórica de los cinco modelos"),
                tabla(["Modelo", "Función objetivo", "Control de complejidad", "Supuesto principal", "Costo de entrenamiento"], comparacion)),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 2: Implementación y diseño experimental (criterio 4)
# ---------------------------------------------------------------------------
def tarjeta_tiempos(tiempos_grid):  # Tiempos de cómputo: búsqueda, un ajuste y predicción
    if PRUEBAS is None:  # Sin el script de pruebas solo se conoce el tiempo de GridSearchCV
        return tarjeta(html.H3("Tiempos de cómputo"), tabla(["Modelo", "Filas de entrenamiento", "GridSearchCV + reajuste"], tiempos_grid, numericas=[1, 2]))  # Tabla simple
    t = PRUEBAS["tiempos"]  # Tiempos medidos
    filas = [(ETQ[n], miles(RES[n]["n_train"]), f"{numero(t[n]['grid_s'], 1)} s", miles(t[n]["n_fit"]), f"{numero(t[n]['fit_s'], 2)} s",  # Una fila por modelo
              f"{numero(t[n]['pred_ms_1000'], 2)} ms") for n in RES]
    return tarjeta(
        html.H3("Tiempos de cómputo"),
        html.P("Medidos en un equipo de 12 núcleos (Windows 11, Python 3.14). GridSearchCV incluye todos los pliegues y combinaciones en paralelo; "
               "«un ajuste» es la media de los 10 ajustes de la validación 5×2 con los mejores hiperparámetros.", className="nota"),
        html.Div([grafico("dis-tiempos", fg.fig_tiempos(t)),
                  html.Div([html.Label("Predicción: latencia por 1 000 transacciones", className="control-etiqueta"), grafico("dis-prediccion", fg.fig_prediccion(t))])],
                 className="rejilla cols-2", style={"marginBottom": "0"}),
        tabla(["Modelo", "Filas (GridSearchCV)", "GridSearchCV + reajuste", "Filas (un ajuste)", "Un ajuste", "Predicción / 1 000 tx"], filas, numericas=[1, 2, 3, 4, 5]),
        html.P(f"KNN entrena casi instantáneamente (solo guarda los datos) pero es el más lento al predecir ({numero(t['KNN']['pred_ms_1000'], 0)} ms por 1 000 transacciones). "
               f"La RVM tarda {numero(t['RVM']['fit_s'], 1)} s en un ajuste (invierte H en cada iteración) y predice en {numero(t['RVM']['pred_ms_1000'], 0)} ms por 1 000 transacciones, "
               f"del mismo orden que el boosting ({numero(t['GradientBoosting']['pred_ms_1000'], 0)} ms): suficiente para autorizar pagos en tiempo real.", className="nota", style={"marginTop": "10px"}),
    )


def sub_diseno():  # Construye la sub-pestaña de diseño experimental
    catalogo = mod.catalogo_modelos()  # Mallas de hiperparámetros realmente usadas
    justificacion = {  # Por qué cada malla
        "LogisticRegression": "C controla la regularización L2; class_weight compara el manejo del desbalance.",
        "RandomForest": "Profundidad y hojas mínimas limitan la memorización de fraudes aislados; balanced_subsample pondera por árbol.",
        "KNN": "k grande suaviza las probabilidades con 2 % de positivos; distance pondera vecinos cercanos.",
        "GradientBoosting": "Árboles poco profundos y tasa baja con submuestreo (0,8) regularizan el boosting.",
        "RVM": "Los nodos fijan la flexibilidad de cada curva; la poda ARD regulariza el resto sin búsqueda adicional.",
    }
    hiper = [(ETQ[n], "; ".join(f"{k.split('__')[-1]} ∈ {v}" for k, v in catalogo[n][1].items()), params_texto(RES[n]["params"]), justificacion[n]) for n in RES]  # Filas
    preprocesos = [  # Pipeline por modelo
        ("LogisticRegression, KNN, RandomForest, GradientBoosting", "Imputación (mediana / «None») → estandarización → one-hot", "Estimador de scikit-learn"),
        ("RVM-Fraude", "Imputación → B-splines cúbicos (continuas) + one-hot (discretas y categóricas) → quitar constantes → estandarización", "RVMClassifier (src/rvm.py)"),
    ]
    tiempos = [(ETQ[n], miles(RES[n]["n_train"]), f"{numero(RES[n]['segundos'], 1)} s") for n in RES]  # Tamaño y tiempo de entrenamiento
    n = ART["n"]  # Tamaños de los conjuntos
    return html.Div([
        rubrica(4),  # Criterio de la rúbrica
        tarjeta(html.H3("Separación de los datos: entrenamiento, validación y prueba"),
                html.P("La división temporal se hace antes de cualquier transformación. La prueba (nov–dic) se usa una sola vez, al final.", className="nota"),
                grafico("dis-particion", fg.fig_particion(n))),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Estrategia de validación"),
                    pasos([f"División temporal: ene–oct ({miles(n['train'] + n['val'])} filas) para entrenar y nov–dic ({miles(n['test'])}) para evaluar.",
                           f"Dentro de ene–oct, 20 % estratificado ({miles(n['val'])} filas) para elegir el umbral de decisión.",
                           "GridSearchCV con StratifiedKFold de 5 pliegues (barajado, semilla 42) sobre el 80 % restante; criterio: AUC-PR.",
                           "El Pipeline completo se reajusta en cada pliegue: imputación, escalas, splines y one-hot se aprenden solo con el pliegue de entrenamiento.",
                           "Comparación final pareada con la prueba de DeLong sobre las mismas transacciones de prueba."])),
            tarjeta(html.H3("Pipeline de cada modelo"),
                    tabla(["Modelos", "Preprocesamiento (dentro del Pipeline)", "Estimador"], preprocesos),
                    html.P("Los modelos con class_weight se evalúan con y sin ponderación dentro de la misma búsqueda.", className="nota", style={"marginTop": "10px"})),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Código en Python del Pipeline y de GridSearchCV"),
                html.P(["Código fuente real del proyecto (", html.Code("src/modelado.py"), ", ", html.Code("dashboard/entrenar.py"), " y ", html.Code("src/rvm.py"),
                        "), leído en vivo: lo que se muestra es exactamente lo que se ejecutó."], className="nota"),
                html.Label("Fragmento", className="control-etiqueta"),
                desplegable("dis-codigo", [{"label": k, "value": k} for k in CODIGO], "Pipeline + GridSearchCV + StratifiedKFold"),
                html.Div(id="dis-codigo-texto", className="caja-codigo")),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Hiperparámetros: malla, valor elegido y justificación"),
                tabla(["Modelo", "Malla de GridSearchCV", "Elegido por CV", "Justificación"], hiper)),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta_tiempos(tiempos),  # Tiempos de cómputo
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Decisiones computacionales"),
                    html.Ul([
                        html.Li(f"KNN se entrena con una muestra estratificada de {miles(n['knn'])} filas: predecir exige comparar con todo el conjunto (O(n·d) por transacción)."),
                        html.Li("GridSearchCV usa todos los núcleos (n_jobs = −1); el bosque y el boosting se paralelizan por combinación."),
                        html.Li("La RVM invierte matrices de tamaño M × M (M ≤ 250 funciones activas), no de n × n: es viable con 66 634 filas."),
                        html.Li("La importancia por permutación se calcula en una muestra estratificada de 6 000 transacciones de prueba."),
                    ], style={"marginTop": "10px"})),
            tarjeta(html.H3("Reproducibilidad"),
                    html.Ul([
                        html.Li([html.B("Semilla fija: "), "42 en particiones, pliegues, bosques, boosting y muestreos."]),
                        html.Li([html.B("Versiones fijadas: "), "Python 3.14, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, dash 4.4.1 (requirements.txt)."]),
                        html.Li([html.B("Código modular: "), html.Code("src/config.py"), ", ", html.Code("src/modelado.py"), ", ", html.Code("src/rvm.py"), ", ", html.Code("dashboard/entrenar.py"), "."]),
                        html.Li([html.B("Artefactos: "), "modelos y métricas guardados en dashboard/artefactos.joblib."]),
                        html.Li([html.B("Contenedor: "), "imagen Docker y despliegue en el Kubernetes de Docker Desktop (k8s/)."]),
                    ]),
                    html.Pre("python -m dashboard.entrenar      # entrena los 5 modelos\npython -m dashboard.experimentos   # compara con el listado\npython -m dashboard.app            # abre el dashboard", className="codigo")),
        ], className="rejilla cols-2"),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 3: Resultados y evaluación crítica (criterio 5)
# ---------------------------------------------------------------------------
PAISES = sorted(DF["merchant_country"].unique())  # Países disponibles para el simulador
RIESGO_PAIS = DF.groupby("merchant_country")["high_risk_country"].max().to_dict()  # País → bandera de alto riesgo
CANAL_DISPOSITIVO = {"Online": ("Desktop", "Chrome", 0), "Mobile App": ("Mobile", "App", 0),  # Canal → (dispositivo, navegador, tarjeta presente)
                     "In-store": ("POS Terminal", "None", 1), "ATM": ("ATM", "None", 1)}


def valores_tipicos(X):  # Transacción típica: mediana de numéricas y moda de categóricas del entrenamiento
    fila = {c: float(X[c].median()) for c in cf.NUM_FEATURES}  # Medianas numéricas
    fila.update({c: str(X[c].mode().iloc[0]) for c in cf.CAT_FEATURES})  # Modas categóricas
    return fila  # Diccionario columna → valor


def campo(etiqueta, componente):  # Campo del simulador: etiqueta + control
    return html.Div([html.Label(etiqueta, className="control-etiqueta"), componente])  # Agrupa ambos elementos


REFERENCIAS_METRICAS = [  # Literatura sobre métricas y pruebas para comparar clasificadores
    ("Saito, T. y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.", "https://doi.org/10.1371/journal.pone.0118432"),
    ("Davis, J. y Goadrich, M. (2006). The relationship between precision-recall and ROC curves. ICML 2006, 233–240.", "https://doi.org/10.1145/1143844.1143874"),
    ("Chicco, D. y Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics, 21, 6.", "https://doi.org/10.1186/s12864-019-6413-7"),
    ("Dal Pozzolo, A. et al. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), 4915–4928.", "https://doi.org/10.1016/j.eswa.2014.02.026"),
    ("DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. Biometrics, 44(3), 837–845.", "https://doi.org/10.2307/2531595"),
    ("Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895–1923.", "https://doi.org/10.1162/089976698300017197"),
    ("Alpaydin, E. (1999). Combined 5 × 2 cv F test for comparing supervised classification learning algorithms. Neural Computation, 11(8), 1885–1892.", "https://doi.org/10.1162/089976699300016007"),
    ("Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.", "https://jmlr.org/papers/v7/demsar06a.html"),
]

FORMULAS_PRUEBAS = r"""
**Hipótesis de cada prueba.** $H_0$: los dos modelos tienen el mismo desempeño. $H_1$: el desempeño es distinto (bilateral, $\alpha = 0{,}05$).

**DeLong (AUC-ROC, en prueba).** El AUC es un estadístico U de Mann-Whitney; con sus componentes estructurales se estima la covarianza $\mathbf{S}$ de los dos AUC:

$$z = \frac{\widehat{AUC}_A - \widehat{AUC}_B}{\sqrt{S_{AA} + S_{BB} - 2S_{AB}}} \sim \mathcal{N}(0, 1).$$

**Bootstrap pareado (AUC-PR, en prueba).** Se remuestrean con reemplazo, dentro de cada clase, las mismas transacciones para ambos modelos ($B$ = 2 000 réplicas) y se obtiene la distribución de $\Delta AP^{*}$; $p = 2\min\{P(\Delta AP^{*} \le 0), P(\Delta AP^{*} \ge 0)\}$.

**McNemar (errores con el umbral, en prueba).** Con $b$ = casos que solo acierta A y $c$ = casos que solo acierta B:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1.$$

**5 × 2 validación cruzada (en entrenamiento).** Cinco repeticiones de validación cruzada con 2 pliegues; $p_i^{(j)}$ es la diferencia en la repetición $i$ y el pliegue $j$, y $s_i^2$ su varianza en la repetición $i$:

$$t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5}\sum_{i=1}^{5}s_i^2}} \sim t_5 \;\;\text{(Dietterich)}, \qquad
F = \frac{\sum_{i=1}^{5}\sum_{j=1}^{2}\big(p_i^{(j)}\big)^2}{2\sum_{i=1}^{5}s_i^2} \sim F_{10,\,5} \;\;\text{(Alpaydin)}.$$

**Friedman y Nemenyi (los cinco modelos a la vez).** Con $R_j$ el rango medio del modelo $j$ en los $N = 10$ pliegues y $k = 5$ modelos:

$$\chi^2_F = \frac{12N}{k(k+1)}\Big[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\Big], \qquad \text{DC} = q_\alpha\sqrt{\frac{k(k+1)}{6N}}.$$

Dos modelos difieren si sus rangos medios se separan más que la diferencia crítica DC.

**Comparaciones múltiples.** La RVM se compara con 4 modelos: los valores p se ajustan con el método de Holm.
"""


def resumen_5x2cv():  # Frase con el resultado de la prueba F 5×2cv (Holm) de la RVM frente a cada modelo
    if PRUEBAS is None:  # Sin el script de pruebas
        return "pendiente (ejecutar python -m dashboard.pruebas)."  # Aviso
    cvp = PRUEBAS["cv_pruebas"]  # Resultados 5×2cv

    def lista(metrica):  # Modelos que la RVM supera significativamente en una métrica
        return [ETQ[n] for n, d in cvp.items() if d[metrica]["p_f_holm"] < 0.05 and d[metrica]["dif_media"] > 0]  # Filtra

    roc, pr = lista("auc_roc"), lista("auc_pr")  # Por métrica
    return (f"con la prueba F de Alpaydin y ajuste de Holm, la RVM supera significativamente en AUC-ROC a {', '.join(roc) or 'ningún modelo'} "
            f"y en AUC-PR a {', '.join(pr) or 'ningún modelo'}; en el ranking de Friedman ocupa el rango medio "
            f"{numero(PRUEBAS['friedman']['auc_pr']['rangos']['RVM'], 1)} en AUC-PR (1 = mejor).")  # Frase


def p_texto(p):  # Formato de un valor p
    return "< 0,0001" if p < 1e-4 else numero(p, 4)  # Valores muy pequeños se acotan


def veredicto(dif, p):  # Insignia con el resultado de una prueba al 5 %
    if p >= 0.05:  # No se rechaza H0
        return html.Span("Sin diferencia", className="insignia no")  # Empate estadístico
    return html.Span("RVM mejor" if dif > 0 else "RVM peor", className=f"insignia {'si' if dif > 0 else 'no'}")  # Dirección de la diferencia


def tarjeta_metricas_literatura():  # Tabla con las métricas de comparación recomendadas en la literatura
    if PRUEBAS is None:  # Sin el script de pruebas no hay estas métricas
        return html.Div()  # No muestra nada
    m = PRUEBAS["metricas"]  # Métricas por modelo
    columnas = ["auc_roc", "auc_pr", "mcc", "bal_acc", "prec_k", "rec_k", "brier", "logloss"]  # Orden de las columnas
    mayor_mejor = {"brier": False, "logloss": False}  # En Brier y log-loss, menor es mejor
    mejor = {c: (max if mayor_mejor.get(c, True) else min)(m, key=lambda n: m[n][c]) for c in columnas}  # Mejor modelo por métrica

    def celda(n, c, texto):  # Celda resaltada si el modelo es el mejor en esa métrica
        return html.B(texto) if mejor[c] == n else texto  # Negrita para el mejor

    filas = [(ETQ[n],  # Una fila por modelo
              celda(n, "auc_roc", f"{numero(m[n]['auc_roc'])} [{numero(m[n]['ic_auc_roc'][0])}–{numero(m[n]['ic_auc_roc'][1])}]"),
              celda(n, "auc_pr", f"{numero(m[n]['auc_pr'])} [{numero(m[n]['ic_auc_pr'][0])}–{numero(m[n]['ic_auc_pr'][1])}]"),
              celda(n, "mcc", numero(m[n]["mcc"])), celda(n, "bal_acc", numero(m[n]["bal_acc"])),
              celda(n, "prec_k", numero(m[n]["prec_k"])), celda(n, "rec_k", numero(m[n]["rec_k"])),
              celda(n, "brier", numero(m[n]["brier"], 4)), celda(n, "logloss", numero(m[n]["logloss"], 4))) for n in RES]
    return tarjeta(
        html.H3("Métricas de comparación de modelos recomendadas en la literatura (prueba nov–dic)"),
        html.P(f"Entre corchetes, intervalos del 95 %: DeLong para el AUC-ROC y bootstrap estratificado para el AUC-PR. Precision@1 % y recall@1 %: las "
               f"{miles(int(round(PRUEBAS['top_k'] * len(Y_TEST))))} transacciones de mayor riesgo. En negrita, el mejor de cada columna.", className="nota"),
        tabla(["Modelo", "AUC-ROC [IC 95 %]", "AUC-PR [IC 95 %]", "MCC", "Exactitud balanceada", "Precision@1 %", "Recall@1 %", "Brier ↓", "Log-loss ↓"],
              filas, numericas=range(1, 9)),
        html.Details([html.Summary("Referencias de las métricas y pruebas"), html.Div([cita(t, u) for t, u in REFERENCIAS_METRICAS])], className="pregunta", style={"marginTop": "12px"}),
    )


def bloque_pruebas():  # Pruebas de significancia estadística: en prueba, 5×2cv y Friedman-Nemenyi
    if PRUEBAS is None:  # Sin el script de pruebas solo se muestra DeLong
        return tarjeta(html.H3("Prueba de DeLong"), grafico("ml-delong", fg.fig_delong(ART["delong"])))  # Versión mínima
    pr, cvp, fr = PRUEBAS["prueba"], PRUEBAS["cv_pruebas"], PRUEBAS["friedman"]  # Resultados
    filas_prueba = [(ETQ[n],  # Pruebas pareadas en el conjunto de prueba
                     f"{pr[n]['delong']['dif']:+.4f}".replace(".", ","), numero(pr[n]["delong"]["z"], 2), p_texto(pr[n]["delong"]["p_holm"]),
                     veredicto(pr[n]["delong"]["dif"], pr[n]["delong"]["p_holm"]),
                     f"{pr[n]['bootstrap_pr']['dif']:+.4f}".replace(".", ","), p_texto(pr[n]["bootstrap_pr"]["p_holm"]),
                     veredicto(pr[n]["bootstrap_pr"]["dif"], pr[n]["bootstrap_pr"]["p_holm"]),
                     f"{pr[n]['mcnemar']['b']} / {pr[n]['mcnemar']['c']}", numero(pr[n]["mcnemar"]["chi2"], 2), p_texto(pr[n]["mcnemar"]["p_holm"]))
                    for n in pr]
    filas_cv = [(ETQ[n],  # Pruebas 5×2cv
                 f"{cvp[n]['auc_roc']['dif_media']:+.4f}".replace(".", ","), numero(cvp[n]["auc_roc"]["t"], 2), p_texto(cvp[n]["auc_roc"]["p_t"]),
                 numero(cvp[n]["auc_roc"]["f"], 2), p_texto(cvp[n]["auc_roc"]["p_f_holm"]), veredicto(cvp[n]["auc_roc"]["dif_media"], cvp[n]["auc_roc"]["p_f_holm"]),
                 f"{cvp[n]['auc_pr']['dif_media']:+.4f}".replace(".", ","), numero(cvp[n]["auc_pr"]["f"], 2), p_texto(cvp[n]["auc_pr"]["p_f_holm"]),
                 veredicto(cvp[n]["auc_pr"]["dif_media"], cvp[n]["auc_pr"]["p_f_holm"]))
                for n in cvp]
    return html.Div([
        html.Div([
            tarjeta(html.H3("Pruebas de significancia estadística: planteamiento"), formula(FORMULAS_PRUEBAS), clase="destacada"),
            tarjeta(html.H3("RVM frente a cada modelo: prueba de DeLong (AUC-ROC)"),
                    html.P(f"Diferencia de AUC-ROC con su IC del 95 % sobre las mismas {miles(len(Y_TEST))} transacciones de prueba ({N_FRAUDES_TEST} fraudes). "
                           "Magenta: diferencia significativa.", className="nota"),
                    grafico("ml-delong", fg.fig_delong(ART["delong"])), clase="destacada"),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Pruebas pareadas en el conjunto de prueba (nov–dic), RVM frente a cada modelo"),
                html.P("Valores p ajustados por Holm. McNemar: b = transacciones que solo la RVM clasifica bien; c = las que solo el otro modelo clasifica bien (con el umbral de cada uno).", className="nota"),
                tabla(["RVM vs.", "ΔAUC-ROC", "z (DeLong)", "p Holm", "Resultado", "ΔAUC-PR", "p Holm (bootstrap)", "Resultado", "b / c", "χ² McNemar", "p Holm"],
                      filas_prueba, numericas=[1, 2, 3, 5, 6, 8, 9, 10])),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Pruebas con 5 × 2 validación cruzada (entrenamiento ene–oct), RVM frente a cada modelo"),
                html.P("Diferencia media en los 10 pliegues. Se reportan la prueba t de Dietterich y la F combinada de Alpaydin (más estable); la decisión usa la F con ajuste de Holm. "
                       "Los pliegues son aleatorios dentro de ene–oct: miden la variabilidad del entrenamiento y complementan, no reemplazan, la evaluación temporal en nov–dic.", className="nota"),
                tabla(["RVM vs.", "ΔAUC-ROC", "t", "p (t)", "F", "p Holm (F)", "Resultado", "ΔAUC-PR", "F", "p Holm (F)", "Resultado"],
                      filas_cv, numericas=[1, 2, 3, 4, 5, 7, 8, 9])),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Friedman + Nemenyi: AUC-ROC"),
                    html.P(f"χ²_F = {numero(fr['auc_roc']['chi2'], 2)}, p = {p_texto(fr['auc_roc']['p'])}. Los modelos dentro de la franja sombreada no difieren del mejor.", className="nota"),
                    grafico("ml-nemenyi-roc", fg.fig_nemenyi(fr["auc_roc"], "AUC-ROC"))),
            tarjeta(html.H3("Friedman + Nemenyi: AUC-PR"),
                    html.P(f"χ²_F = {numero(fr['auc_pr']['chi2'], 2)}, p = {p_texto(fr['auc_pr']['p'])}. Los modelos dentro de la franja sombreada no difieren del mejor.", className="nota"),
                    grafico("ml-nemenyi-pr", fg.fig_nemenyi(fr["auc_pr"], "AUC-PR"))),
        ], className="rejilla cols-2"),
    ])


def tabla_desbalance():  # class_weight=None vs. balanced en la validación cruzada
    filas = []  # Filas
    for n in ["LogisticRegression", "RandomForest"]:  # Modelos que admiten class_weight
        mejores = {}  # Mejor combinación por estrategia
        for f in RES[n]["cv_tabla"]:  # Combinaciones ordenadas por ranking
            mejores.setdefault(f["params"].get("class_weight", "None"), f)  # La primera es la mejor de su estrategia
        filas += [(ETQ[n], html.Code(cw), numero(f["cv_auc_pr"]), numero(f["cv_auc_roc"])) for cw, f in mejores.items()]  # Una fila por estrategia
    return tabla(["Modelo", "class_weight", "CV AUC-PR", "CV AUC-ROC"], filas, numericas=[2, 3])  # Tabla


def tarjeta_listado():  # Comparación con los modelos del listado del curso
    if not EXP:  # Si no se ha corrido dashboard/experimentos.py
        return html.Div()  # No muestra nada
    filas = [(e["nombre"], e.get("origen", "—"), numero(e["val_auc"], 4) if e.get("val_auc") is not None else "CV", numero(e["test_auc"], 4),  # Datos
              f"{e['dif']:+.4f}".replace(".", ","), numero(e["p"], 3),
              html.Span("Supera" if (e["p"] < 0.05 and e["dif"] > 0) else ("Peor" if e["p"] < 0.05 else "Empata"),  # Veredicto
                        className=f"insignia {'si' if (e['p'] < 0.05 and e['dif'] > 0) else 'no'}")) for e in EXP]
    return tarjeta(html.H3("Modelos alternativos del listado frente a GradientBoosting"),
                   html.P("Cada candidato se ajusta con ene–ago, se configura con sep–oct y se evalúa una vez con nov–dic. Magenta: supera significativamente a GradientBoosting.", className="nota"),
                   grafico("res-listado", fg.fig_experimentos(EXP)),
                   tabla(["Modelo", "Del listado", "AUC val.", "AUC prueba", "Δ vs. GB", "p", "Resultado"], filas, numericas=[2, 3, 4, 5]),
                   clase="destacada")


def sub_resultados():  # Construye la sub-pestaña de resultados
    return html.Div([
        rubrica(5),  # Criterio de la rúbrica
        html.Div([
            tarjeta(
                html.H3("Modelo y umbral de decisión"),
                html.Label("Modelo", className="control-etiqueta"),
                desplegable("ml-modelo", [{"label": ETQ[n], "value": n} for n in RES], "RVM"),
                html.Label("Umbral de probabilidad para marcar «fraude»", className="control-etiqueta"),
                dcc.Slider(id="ml-umbral", min=0, max=1, step=0.01, value=round(RVM["umbral"], 2), marks={i / 10: f"{i / 10:.1f}".replace(".", ",") for i in range(0, 11, 2)},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.P(id="ml-umbral-nota", className="nota", style={"marginTop": "14px"}),
                html.Div(id="ml-parametros"),
            ),
            tarjeta(html.H3("Selección y justificación de las métricas (según la literatura)"),
                    html.Ul([
                        html.Li([html.B("AUC-PR (principal): "), f"con clases desbalanceadas la curva PR es más informativa que la ROC (Saito y Rehmsmeier, 2015); su referencia es la tasa de fraude ({numero(Y_TEST.mean())})."]),
                        html.Li([html.B("AUC-ROC: "), "probabilidad de ordenar un fraude por encima de una legítima; admite la prueba de DeLong para comparar modelos (DeLong et al., 1988)."]),
                        html.Li([html.B("MCC: "), "usa las cuatro celdas de la matriz de confusión y no se infla con el desbalance, a diferencia de F1 y la exactitud (Chicco y Jurman, 2020)."]),
                        html.Li([html.B("Precision@1 %: "), "fracción de fraudes entre las alertas que un equipo de analistas puede revisar (Dal Pozzolo et al., 2014)."]),
                        html.Li([html.B("Brier y log-loss: "), "calidad de las probabilidades, no solo del orden (Brier, 1950)."]),
                        html.Li([html.B("Exactitud: "), f"se reporta pero no se usa: decir siempre «legítima» da {numero(1 - Y_TEST.mean())}."]),
                    ])),
        ], className="rejilla cols-control"),
        html.Div([  # Indicadores del modelo seleccionado
            kpi("AUC-ROC prueba", "", "", id_valor="kpi-auc", id_detalle="kpi-auc-ic"),
            kpi("AUC-PR prueba", "", f"azar = {numero(Y_TEST.mean())}", id_valor="kpi-ap"),
            kpi("Precisión", "", "alertas que son fraude", id_valor="kpi-prec"),
            kpi("Recall", "", "fraudes detectados", id_valor="kpi-rec"),
            kpi("F1", "", "balance precisión-recall", id_valor="kpi-f1"),
            kpi("Exactitud", "", "engañosa con 2 % de fraude", id_valor="kpi-acc"),
        ], className="rejilla", style={"gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))"}),
        html.Div([
            tarjeta(html.H3("Curvas ROC en prueba (nov–dic)"), html.P("El modelo seleccionado se resalta. La leyenda muestra el AUC-ROC.", className="nota"), grafico("ml-roc")),
            tarjeta(html.H3("Curvas precisión-recall en prueba"), html.P("La referencia del azar es la tasa de fraude.", className="nota"), grafico("ml-pr")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Matriz de confusión con el umbral elegido"), html.P(id="ml-confusion-nota", className="nota"), grafico("ml-confusion")),
            tarjeta(html.H3("Ranking comparativo frente a la línea base"),
                    html.P("Ordenado por AUC-PR de validación cruzada (criterio de selección, sin mirar la prueba). Métricas de umbral con el umbral de validación de cada modelo.", className="nota"),
                    html.Div(id="ml-ranking")),
        ], className="rejilla cols-2"),
        tarjeta_metricas_literatura(),  # Métricas de comparación recomendadas en la literatura
        html.Div(style={"height": "18px"}),  # Separador
        bloque_pruebas(),  # Pruebas de significancia estadística
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Diagnóstico de sobreajuste"),
                html.P("AUC-ROC en train, validación cruzada, validación y prueba (eje truncado en 0,6). Una brecha grande entre train y validación indica memorización.", className="nota"),
                grafico("ml-sobreajuste", fg.fig_sobreajuste(RES))),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta_listado(),  # Comparación con los modelos del listado
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Análisis de errores: ¿qué tipo de fraude se detecta peor?"),
                    html.P("Recall por fraud_type en prueba con el umbral de cada modelo (fraud_type solo para analizar).", className="nota"),
                    grafico("ml-tipo")),
            tarjeta(html.H3("Manejo del desbalance: class_weight"),
                    html.P("Ponderar cambia la escala de las probabilidades, pero apenas el orden (AUC): por eso el umbral se elige en validación.", className="nota"),
                    tabla_desbalance()),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Variables más importantes del modelo seleccionado"),
                html.P("Importancia por permutación en una muestra de prueba: cuánto cae el AUC-ROC al desordenar cada variable.", className="nota"),
                grafico("ml-importancia")),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(  # Simulador
            html.H3("Simulador: probabilidad de fraude de una transacción (RVM)"),
            html.P("Las demás variables toman el valor típico del periodo de entrenamiento. La RVM incorpora la incertidumbre de sus pesos en la probabilidad.", className="nota"),
            html.Div([
                html.Div([
                    html.Div([
                        campo("Monto (USD)", dcc.Input(id="s-monto", type="number", value=65, min=0, className="entrada")),
                        campo("Hora", dcc.Input(id="s-hora", type="number", value=3, min=0, max=23, className="entrada")),
                        campo("Canal", desplegable("s-canal", [{"label": c, "value": c} for c in CANAL_DISPOSITIVO], "Online")),
                        campo("País del comercio", desplegable("s-pais", [{"label": p, "value": p} for p in PAISES], "Romania")),
                    ], className="rejilla cols-4", style={"marginBottom": "0"}),
                    html.Div([
                        campo("Distancia al domicilio (km)", dcc.Input(id="s-dist", type="number", value=6800, min=0, className="entrada")),
                        campo("Antigüedad de la cuenta (días)", dcc.Input(id="s-cuenta", type="number", value=60, min=0, className="entrada")),
                        campo("Transacciones última hora", dcc.Input(id="s-tx1h", type="number", value=3, min=0, className="entrada")),
                        campo("Contracargos históricos", dcc.Input(id="s-cb", type="number", value=1, min=0, className="entrada")),
                    ], className="rejilla cols-4", style={"marginBottom": "0"}),
                ]),
                html.Div([grafico("s-medidor"), html.Div(id="s-veredicto", className="veredicto")], className="resultado"),
            ], className="rejilla", style={"gridTemplateColumns": "minmax(0, 2fr) minmax(0, 1fr)"}),
            dcc.Store(id="s-base", data=valores_tipicos(TRAIN_DF[cf.FEATURES])),
        ),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 4: Conclusiones, limitaciones y preguntas (criterios 6 y 7)
# ---------------------------------------------------------------------------
def sub_conclusiones():  # Construye la sub-pestaña de conclusiones
    t = {n: RES[n]["test"] for n in RES}  # Métricas de prueba por modelo
    mejor_ap = max(RES, key=lambda n: t[n]["auc_pr"])  # Modelo con mayor AUC-PR en prueba
    dl = ART["delong"]  # Pruebas de DeLong
    significativas = [ETQ[n] for n, d in dl.items() if d["p"] < 0.05 and d["dif"] > 0]  # Modelos que la RVM supera
    empates = [ETQ[n] for n, d in dl.items() if d["p"] >= 0.05]  # Modelos con los que empata
    generalizacion = [(ETQ[n], numero(RES[n]["train_auc_roc"]), numero(RES[n]["cv_auc_roc"]), numero(RES[n]["test"]["auc_roc"]),  # Brechas por modelo
                       f"{RES[n]['train_auc_roc'] - RES[n]['test']['auc_roc']:+.3f}".replace(".", ",")) for n in RES]
    return html.Div([
        html.Div([rubrica(6), rubrica(7)], className="rejilla cols-2", style={"marginBottom": "0"}),  # Criterios de la rúbrica
        html.Div([
            tarjeta(
                html.H3("Conclusiones sustentadas en los resultados"),
                html.Ul([
                    html.Li([html.B("La RVM es competitiva: "), f"AUC-ROC de {numero(t['RVM']['auc_roc'])} y AUC-PR de {numero(t['RVM']['auc_pr'])} en prueba "
                             f"(azar = {numero(Y_TEST.mean())}), es decir, {numero(t['RVM']['auc_pr'] / Y_TEST.mean(), 1)} veces mejor que el azar en precisión promedio."]),
                    html.Li([html.B("Significancia en prueba (nov–dic): "), f"supera significativamente a {', '.join(significativas) or 'ningún modelo'} (DeLong, p < 0,05) "
                             f"y no muestra diferencia significativa con {', '.join(empates) or 'ningún modelo'}."]),
                    html.Li([html.B("Significancia con 5×2cv: "), resumen_5x2cv()]),
                    html.Li([html.B("Techo de información: "), "más de 20 variantes (boosting, CatBoost, LightGBM, XGBoost, ensambles, rasgos de historial y de red) quedan en AUC ≈ 0,85–0,86: "
                             "el límite lo pone la información de las 35 variables, no el algoritmo."]),
                    html.Li([html.B("Valor agregado de la RVM: "), f"iguala a los ensambles con un modelo disperso ({INFO_RVM['n_relevantes']} de {INFO_RVM['n_bases']} funciones base), "
                             "interpretable variable por variable y con incertidumbre en cada predicción."]),
                    html.Li([html.B("Señales de fraude: "), "transacciones internacionales y a gran distancia, países de alto riesgo, madrugada, cuentas nuevas, picos de velocidad y contracargos previos."]),
                    html.Li([html.B("Operación: "), f"con el umbral de validación la RVM detecta {numero(100 * t['RVM']['recall'], 0)} % de los fraudes con una precisión de {numero(100 * t['RVM']['precision'], 0)} %; "
                             f"el mayor AUC-PR lo obtiene {ETQ[mejor_ap]} ({numero(t[mejor_ap]['auc_pr'])})."]),
                ]),
            ),
            tarjeta(
                html.H3("Limitaciones"),
                html.P(html.B("Teóricas")),
                html.Ul([
                    html.Li("La RVM es aditiva: no captura interacciones entre variables salvo que se agreguen como funciones base."),
                    html.Li("La aproximación de Laplace y la regla de punto fijo pueden converger a un óptimo local de la evidencia."),
                    html.Li("El costo O(M³) por iteración limita el número de funciones base (no escala a kernels sobre n = 66 634 filas)."),
                ]),
                html.P(html.B("Empíricas")),
                html.Ul([
                    html.Li("Datos sintéticos: los patrones pueden ser más simples o más ruidosos que en un banco real."),
                    html.Li(f"La prueba tiene solo {N_FRAUDES_TEST} fraudes: los intervalos del AUC son de ± 0,02 y diferencias menores a ≈ 0,008 no son detectables."),
                    html.Li("Un solo periodo de prueba (dos meses): no mide la deriva de largo plazo."),
                    html.Li("Las etiquetas se suponen correctas e inmediatas; en la práctica el fraude se confirma con días de retraso."),
                ]),
            ),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Capacidad de generalización"),
                    html.P("AUC-ROC en train, validación cruzada y prueba. Una brecha train − prueba pequeña indica que el modelo generaliza a meses futuros.", className="nota"),
                    tabla(["Modelo", "Train", "CV", "Prueba", "Brecha"], generalizacion, numericas=[1, 2, 3, 4])),
            tarjeta(html.H3("Posibles mejoras y líneas futuras"),
                    pasos(["Agregar interacciones seleccionadas (p. ej. canal × país) como funciones base de la RVM.",
                           "Umbral sensible al costo: usar el monto en riesgo (fraud_amount_lost en entrenamiento) para minimizar la pérdida esperada.",
                           "Validación con varios cortes temporales (rolling origin) y monitoreo de deriva con Evidently.",
                           "Calibración de probabilidades y reglas de revisión manual para la zona de incertidumbre de la RVM.",
                           "Variables de grafo y de secuencia con datos reales, donde el historial del cliente sí aporte información."])),
        ], className="rejilla cols-2"),
        tarjeta(
            html.H3("Preguntas esperadas en la sustentación"),
            html.P("Haz clic en cada pregunta para ver la respuesta.", className="nota"),
            pregunta("¿Por qué una RVM y no una SVM clásica?",
                     html.P("La SVM da una frontera, no probabilidades (requiere el escalado de Platt), y necesita buscar C y γ por validación cruzada. La RVM es bayesiana: "
                            "estima por evidencia cuánto regularizar cada función base, produce probabilidades y su incertidumbre, y suele usar muchas menos funciones que vectores soporte tiene una SVM.")),
            pregunta("¿Cómo produce dispersión el prior ARD?",
                     html.P("Cada peso tiene su propia precisión α_i. Al maximizar la evidencia, para las funciones que no explican los datos α_i → ∞, su posterior colapsa en cero y se podan. "
                            "Integrando α, el prior marginal de w es una t de Student, de colas pesadas, que favorece soluciones dispersas.")),
            pregunta("¿Por qué se necesita la aproximación de Laplace?",
                     html.P("Con verosimilitud Bernoulli-logística la posterior de w no tiene forma cerrada (el prior gaussiano no es conjugado). Se aproxima con una gaussiana centrada en el modo w*, "
                            "con covarianza igual a la inversa del hessiano H = ΦᵀBΦ + A, que se obtiene con Newton (IRLS).")),
            pregunta("¿Qué significa γ_i?",
                     html.P("γ_i = 1 − α_i Σ_ii ∈ [0, 1] es el grado en que los datos determinan el peso w_i (1: lo fijan los datos; 0: lo fija el prior). Su suma son los grados de libertad efectivos del modelo.")),
            pregunta("¿La RVM supera significativamente a GradientBoosting?",
                     html.P(f"Depende de la evidencia. En la prueba (nov–dic) no: la diferencia de AUC-ROC es {numero(DL_MEJOR['dif'], 4)} (DeLong, p = {numero(DL_MEJOR['p'], 2)}); "
                            f"con {N_FRAUDES_TEST} fraudes, una diferencia menor a ≈ 0,008 no es detectable. Con 5×2 validación cruzada sobre ene–oct, que promedia 10 particiones, "
                            "la RVM sí es significativamente mejor en AUC-PR (F de Alpaydin con Holm) pero no en AUC-ROC. Todos los modelos bien regularizados chocan con un techo "
                            "de información de AUC ≈ 0,85–0,86.")),
            pregunta("¿Por qué usar varias pruebas de significancia y no solo una?",
                     html.P("Cada una responde a una pregunta distinta (Dietterich, 1998; Demšar, 2006): DeLong compara AUC-ROC en datos fijos; el bootstrap pareado hace lo mismo para el AUC-PR, "
                            "que no tiene fórmula de varianza; McNemar compara los errores con el umbral elegido; 5×2cv incluye la variabilidad del entrenamiento; y Friedman + Nemenyi "
                            "compara todos los modelos a la vez. Holm controla el error por hacer varias comparaciones.")),
            pregunta("¿Cómo se evitó la fuga de datos?",
                     html.P("Excluyendo fraud_type, fraud_amount_lost, identificadores y la fecha cruda; dividiendo por tiempo antes de transformar; y poniendo todo el preprocesamiento "
                            "dentro del Pipeline para que GridSearchCV lo reajuste en cada pliegue. El umbral se elige en validación, nunca en prueba.")),
            pregunta("¿Por qué una división temporal y no aleatoria?",
                     html.P("En producción el modelo predice transacciones futuras con lo aprendido del pasado. Mezclar meses al azar permitiría «ver el futuro» y sobreestimar el desempeño.")),
            pregunta("¿Por qué el umbral no es 0,5?",
                     html.P("Con 2 % de fraude las probabilidades son pequeñas y 0,5 casi nunca se alcanza: el recall sería muy bajo. Se elige el umbral que maximiza el F1 en validación; "
                            "en un banco se elegiría por costo (pérdida evitada frente a costo de revisar).")),
            pregunta("¿Por qué AUC-PR como criterio de GridSearchCV?",
                     html.P("El AUC-ROC lo dominan las 98 % legítimas y puede verse alto aunque las alertas sean poco precisas. El AUC-PR se concentra en la clase fraude, que es la que importa.")),
            pregunta("¿Cómo se controla el sobreajuste en la RVM?",
                     html.P("Con el factor de Occam de la evidencia (penaliza la complejidad), la poda ARD, un número moderado de nodos elegido por validación cruzada y la salida moderada, "
                            "que encoge las predicciones inciertas. La brecha train − prueba es mínima.")),
            pregunta("¿Cuál es el costo computacional de la RVM?",
                     html.P("Cada iteración cuesta O(n·M² + M³): formar ΦᵀBΦ y invertir H. Como M baja con la poda, las iteraciones se abaratan. Predecir cuesta O(M²) por transacción por la varianza φᵀΣφ.")),
            pregunta("¿Qué es la prueba de DeLong?",
                     html.P("Compara dos AUC calculados sobre los mismos datos. Usa que el AUC es un estadístico U de Mann-Whitney, estima su covarianza con componentes estructurales "
                            "y contrasta la diferencia con una normal (prueba z). Es pareada: tiene en cuenta que ambos modelos ven las mismas transacciones.")),
            pregunta("¿Cómo se interpreta una curva de contribución?",
                     html.P("Es el aporte de una variable al logit de fraude manteniendo las demás fijas (el modelo es aditivo). Valores positivos aumentan el riesgo; la banda es el intervalo del 95 % de la posterior bayesiana.")),
        ),
    ])


def seccion_modelos():  # Construye la sección con sus sub-pestañas
    return html.Div([
        dcc.Tabs(id="ml-sub", value="fundamentos", className="subpestanas", children=[  # Navegación interna
            dcc.Tab(label="Fundamentos teóricos", value="fundamentos", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Diseño experimental", value="diseno", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Resultados", value="resultados", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Conclusiones y preguntas", value="conclusiones", className="subtab", selected_className="subtab--selected"),
        ]),
        html.Div(id="ml-sub-contenido"),  # Contenido de la sub-pestaña activa
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(Output("ml-sub-contenido", "children"), Input("ml-sub", "value"))  # Cambia de sub-pestaña
def mostrar_sub(valor):
    return {"fundamentos": sub_fundamentos, "diseno": sub_diseno, "resultados": sub_resultados, "conclusiones": sub_conclusiones}[valor]()  # Construye la elegida


@callback(Output("dis-codigo-texto", "children"), Input("dis-codigo", "value"))  # Muestra el fragmento de código elegido
def mostrar_codigo(nombre):
    fuente = inspect.getsource(CODIGO[nombre])  # Código fuente real de la función
    archivo = inspect.getsourcefile(CODIGO[nombre]).replace("\\", "/").split("/")[-2:]  # Carpeta y archivo de origen
    return [html.Div(f"{'/'.join(archivo)}", className="nota"),  # Ruta del archivo
            dcc.Markdown(f"```python\n{textwrap.dedent(fuente)}\n```", highlight_config={"theme": "dark"})]  # Código con resaltado de sintaxis


@callback(Output("rvm-curva", "figure"), Input("rvm-var", "value"))  # Curva de contribución de la variable elegida
def actualizar_curva(var):
    return fg.fig_rvm_curva(INFO_RVM["curvas"][var], var)  # Curva con banda o barras con error


@callback(Output("ml-umbral", "value"), Input("ml-modelo", "value"))  # Al cambiar de modelo, el umbral vuelve al elegido en validación
def umbral_por_defecto(modelo):
    return round(RES[modelo]["umbral"], 2)  # Umbral de validación redondeado al paso del deslizador


def fila_ranking(n, activo):  # Fila de la tabla de ranking
    r, t = RES[n], RES[n]["test"]  # Resultados y métricas de prueba
    return html.Tr([html.Td(ETQ[n]), html.Td(numero(r["cv_auc_pr"]), className="num"), html.Td(numero(t["auc_roc"]), className="num"),  # Nombre y AUC
                    html.Td(numero(t["auc_pr"]), className="num"), html.Td(numero(t["precision"]), className="num"),  # AUC-PR y precisión
                    html.Td(numero(t["recall"]), className="num"), html.Td(numero(t["f1"]), className="num"), html.Td(numero(t["accuracy"]), className="num")],  # Recall, F1, exactitud
                   className="activa" if activo else "")  # Resalta el modelo seleccionado


@callback(
    Output("kpi-auc", "children"), Output("kpi-auc-ic", "children"), Output("kpi-ap", "children"), Output("kpi-prec", "children"),
    Output("kpi-rec", "children"), Output("kpi-f1", "children"), Output("kpi-acc", "children"),
    Output("ml-roc", "figure"), Output("ml-pr", "figure"), Output("ml-confusion", "figure"), Output("ml-confusion-nota", "children"),
    Output("ml-ranking", "children"), Output("ml-tipo", "figure"), Output("ml-importancia", "figure"),
    Output("ml-parametros", "children"), Output("ml-umbral-nota", "children"),
    Input("ml-modelo", "value"), Input("ml-umbral", "value"))  # Todo lo que depende del modelo y el umbral
def actualizar_modelo(modelo, umbral):
    r = RES[modelo]  # Resultados del modelo elegido
    m = mod.metricas(Y_TEST, r["p_test"], umbral)  # Métricas recalculadas con el umbral del deslizador
    (vn, fp), (fn, vp) = m["matriz"]  # Celdas de la matriz
    nota_conf = f"Detecta {vp} de {vp + fn} fraudes y genera {miles(fp)} falsas alarmas sobre {miles(vn + fp)} transacciones legítimas."  # Lectura de negocio
    orden = sorted(RES, key=lambda n: RES[n]["cv_auc_pr"], reverse=True)  # Ranking por AUC-PR de validación cruzada
    ranking = html.Div(html.Table([html.Thead(html.Tr([html.Th("Modelo"), html.Th("CV AUC-PR", className="num"), html.Th("AUC-ROC", className="num"),  # Encabezado
                                                       html.Th("AUC-PR", className="num"), html.Th("Prec.", className="num"), html.Th("Recall", className="num"),
                                                       html.Th("F1", className="num"), html.Th("Exact.", className="num")])),
                                   html.Tbody([fila_ranking(n, n == modelo) for n in orden])], className="tabla"), className="tabla-envoltura")  # Filas
    params = html.Div([html.Label("Mejores hiperparámetros (GridSearchCV)", className="control-etiqueta"),  # Hiperparámetros elegidos
                       html.Div([html.Span(f"{k.split('__')[-1]} = {v}", className="chip") for k, v in r["params"].items()]),  # Una etiqueta por parámetro
                       html.Div(f"Entrenado con {miles(r['n_train'])} transacciones · {numero(r['segundos'], 1)} s", className="nota", style={"marginTop": "6px"})])  # Tamaño y tiempo
    nota_umbral = f"Umbral elegido en validación (máximo F1): {numero(r['umbral'])}. Bajarlo detecta más fraudes a costa de más falsas alarmas."  # Explicación
    ic = r["ic_auc"]  # Intervalo de confianza del AUC
    return (numero(m["auc_roc"]), f"IC 95 %: {numero(ic[0])} – {numero(ic[1])}", numero(m["auc_pr"]), numero(m["precision"]),  # Indicadores
            numero(m["recall"]), numero(m["f1"]), numero(m["accuracy"]),
            fg.fig_roc(RES, modelo), fg.fig_pr(RES, modelo, float(Y_TEST.mean())), fg.fig_confusion(m["matriz"]), nota_conf,  # Figuras
            ranking, fg.fig_recall_tipo(RES, modelo), fg.fig_importancia(r["importancia"], fg.COLORES_MODELOS[modelo]),
            params, nota_umbral)


@callback(Output("s-medidor", "figure"), Output("s-veredicto", "children"),
          Input("s-monto", "value"), Input("s-hora", "value"), Input("s-canal", "value"), Input("s-pais", "value"),
          Input("s-dist", "value"), Input("s-cuenta", "value"), Input("s-tx1h", "value"), Input("s-cb", "value"), Input("s-base", "data"))  # Simulador
def simular(monto, hora, canal, pais, dist, cuenta, tx1h, cb, base):
    fila = dict(base)  # Parte de la transacción típica
    dispositivo, navegador, presente = CANAL_DISPOSITIVO[canal]  # Valores coherentes con el canal
    fila.update({"transaction_amount": monto or 0, "transaction_hour": hora or 0, "transaction_channel": canal, "merchant_country": pais,  # Valores del formulario
                 "distance_from_home_km": dist or 0, "account_age_days": cuenta or 0, "transactions_last_1h": tx1h or 0, "num_chargebacks_ever": cb or 0,
                 "device_type": dispositivo, "browser": navegador, "card_present": presente, "used_chip": presente,  # Coherencia con el canal
                 "international_transaction": int(pais != "USA"), "high_risk_country": int(RIESGO_PAIS.get(pais, 0)),  # Banderas derivadas del país
                 "amount_vs_avg_ratio": (monto or 0) / max(base["avg_transaction_amount_30d"], 1)})  # Monto relativo al gasto promedio
    X = pd.DataFrame([fila])[cf.FEATURES]  # Una fila con las 35 columnas en orden
    p = float(ART["simulador"].predict_proba(X)[0, 1])  # Probabilidad de fraude de la RVM
    u = ART["umbral_simulador"]  # Umbral elegido en validación
    texto = html.Span("🚨 Enviar a revisión: supera el umbral" if p >= u else "✅ Aprobar: bajo el umbral",  # Veredicto
                      style={"color": fg.FRAUDE if p >= u else "#6ff0c8"})  # Color del veredicto
    return fg.fig_medidor(p, u), [texto, html.Div(f"umbral = {numero(u)} · riesgo {numero(p / TASA, 1)} × la tasa base", className="nota")]  # Medidor y texto
