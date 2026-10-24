"""Funciones reutilizables de modelado: preprocesamiento, catálogo de modelos, entrenamiento,
selección de umbral, métricas y prueba de DeLong para comparar AUC.

Todo el preprocesamiento vive DENTRO de un Pipeline: GridSearchCV lo reajusta en cada pliegue
con la parte de entrenamiento del pliegue, así la validación cruzada no se contamina (sin fuga).
"""

import numpy as np  # Cálculo numérico
from scipy import stats  # Distribución normal para el valor p de la prueba de DeLong
from sklearn.compose import ColumnTransformer  # Aplica transformaciones distintas a columnas numéricas y categóricas
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Modelos de ensamble recomendados
from sklearn.impute import SimpleImputer  # Imputación de valores faltantes
from sklearn.linear_model import LogisticRegression  # Regresión logística
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,  # Métricas de clasificación
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Búsqueda de hiperparámetros con validación cruzada estratificada
from sklearn.neighbors import KNeighborsClassifier  # K vecinos más cercanos
from sklearn.pipeline import Pipeline  # Encadena preprocesamiento y modelo en un solo estimador
from sklearn.feature_selection import VarianceThreshold  # Elimina columnas constantes (funciones base sin información)
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler  # One-hot, splines cúbicos y estandarización

from src import config as cf  # Columnas, semilla y rutas del proyecto
from src.rvm import RVMClassifier  # Modelo original implementado desde cero (Relevance Vector Machine)

N_PLIEGUES = 5  # Pliegues de la validación cruzada estratificada (como pide el enunciado)


def make_preprocessor(num_cols=cf.NUM_FEATURES, cat_cols=cf.CAT_FEATURES):  # Preprocesador para modelos que necesitan matriz numérica
    """Imputa, estandariza las numéricas y codifica en one-hot las categóricas."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),  # Rellena nulos numéricos con la mediana del pliegue de entrenamiento
                         ("scaler", StandardScaler())])  # Media 0 y desviación 1: imprescindible para KNN y la regresión logística
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")),  # Nulos como categoría propia ("None"), no la moda
                         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False))])  # One-hot; categorías raras agrupadas
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])  # Une ambas ramas


def make_preprocessor_rvm():  # Funciones base φ(x) de la RVM: modelo aditivo con splines y one-hot
    """Splines cúbicos para las numéricas continuas, one-hot para las discretas y categóricas, y estandarización."""
    spl = Pipeline([("imputer", SimpleImputer(strategy="median")),  # Imputa con la mediana del pliegue
                    ("splines", SplineTransformer(n_knots=6, degree=3, knots="quantile", include_bias=False))])  # B-splines cúbicos con nodos en cuantiles
    dis = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")),  # Nulos como categoría propia
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30, drop="first", sparse_output=False))])  # One-hot sin la primera categoría (evita colinealidad con el sesgo)
    base = ColumnTransformer([("spl", spl, cf.NUM_CONTINUAS), ("dis", dis, cf.NUM_DISCRETAS + cf.CAT_FEATURES)])  # Une ambas familias de funciones base
    return Pipeline([("base", base),  # Expansión en funciones base
                     ("varianza", VarianceThreshold(1e-8)),  # Quita columnas constantes (no aportan y vuelven singular el hessiano)
                     ("escala", StandardScaler())])  # Estandariza: el prior ARD N(0, 1/α) compara pesos en la misma escala


def catalogo_modelos():  # Modelos con sus mallas de hiperparámetros (pensadas para controlar el sobreajuste)
    """Devuelve {nombre: (estimador, malla, preprocesador, usa_muestra)}."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000),  # Modelo lineal; max_iter alto para asegurar convergencia
            {"clf__C": [0.01, 0.1, 1.0],  # Inverso de la regularización: C pequeño = modelo más simple
             "clf__class_weight": [None, "balanced"]},  # Compara sin ponderar vs. ponderación balanceada del desbalance
            make_preprocessor, False),  # Preprocesador one-hot; usa todos los datos
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=cf.SEMILLA),  # Bosque de 300 árboles
            {"clf__max_depth": [8, 14],  # Profundidad máxima: limita la complejidad de cada árbol
             "clf__min_samples_leaf": [10, 40],  # Mínimo de transacciones por hoja: evita memorizar fraudes aislados
             "clf__class_weight": [None, "balanced_subsample"]},  # Desbalance: sin ponderar vs. balanceado por árbol
            make_preprocessor, False),  # Usa todos los datos
        "KNN": (
            KNeighborsClassifier(n_jobs=1),  # Clasificador por vecindad (no admite class_weight)
            {"clf__n_neighbors": [51, 151, 301],  # Muchos vecinos: con 2 % de fraude, k pequeño da probabilidades muy ruidosas
             "clf__weights": ["uniform", "distance"]},  # Voto igual o ponderado por la distancia
            make_preprocessor, True),  # KNN escala mal: se entrena con la muestra estratificada
        "GradientBoosting": (
            GradientBoostingClassifier(subsample=0.8, random_state=cf.SEMILLA),  # Boosting de árboles con submuestreo (regulariza)
            {"clf__n_estimators": [200, 400],  # Número de árboles
             "clf__learning_rate": [0.05],  # Tasa de aprendizaje baja: pasos pequeños, menos sobreajuste
             "clf__max_depth": [2, 3]},  # Árboles poco profundos
            make_preprocessor, False),  # Usa todos los datos
        "RVM": (
            RVMClassifier(),  # Modelo original (Relevance Vector Machine desde cero)
            {"prep__base__spl__splines__n_knots": [4, 6, 8]},  # Nodos de los splines: flexibilidad de cada curva (la poda ARD controla el resto)
            make_preprocessor_rvm, False),  # Funciones base aditivas; usa todos los datos
    }


def train_pipeline(X_train, y_train, model, param_grid, preprocesador=make_preprocessor):  # Función reutilizable del enunciado
    """Construye Pipeline(prep + modelo) y lo ajusta con GridSearchCV + StratifiedKFold."""
    pipe = Pipeline([("prep", preprocesador()), ("clf", model)])  # El preprocesamiento queda dentro del Pipeline (sin fuga)
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=cf.SEMILLA)  # Pliegues estratificados: misma tasa de fraude en cada uno
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1,  # Prueba todas las combinaciones en paralelo
                        scoring={"auc_pr": "average_precision", "auc_roc": "roc_auc"},  # Registra ambas áreas bajo la curva
                        refit="auc_pr", return_train_score=True)  # Elige por AUC-PR (métrica principal con desbalance) y guarda el AUC de train
    grid.fit(X_train, y_train)  # Ajusta: validación cruzada + reentrenamiento final con la mejor combinación
    return grid  # Objeto con best_estimator_, best_params_ y cv_results_


def umbral_max_f1(y_val, p_val):  # Elige el umbral de decisión con datos de VALIDACIÓN (nunca con test)
    """Devuelve el umbral que maximiza el F1 de la clase fraude en validación."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)  # Precisión y recall para cada umbral posible
    f1 = 2 * prec * rec / (prec + rec + 1e-9)  # F1 para cada umbral
    return float(thr[int(np.argmax(f1[:-1]))])  # El último punto de la curva no tiene umbral asociado


def metricas(y, p, umbral):  # Métricas completas de un modelo con un umbral dado
    """Calcula AUC-ROC, AUC-PR y las métricas de la clase fraude con el umbral indicado."""
    pred = (p >= umbral).astype(int)  # Predicción binaria con el umbral
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()  # Celdas de la matriz de confusión
    return {
        "auc_roc": float(roc_auc_score(y, p)),  # Capacidad de ordenar fraudes por encima de legítimas
        "auc_pr": float(average_precision_score(y, p)),  # Precisión promedio: la métrica principal con desbalance
        "precision": float(precision_score(y, pred, zero_division=0)),  # De las alertas, cuántas son fraude
        "recall": float(recall_score(y, pred, zero_division=0)),  # De los fraudes, cuántos se detectan
        "f1": float(f1_score(y, pred, zero_division=0)),  # Media armónica de precisión y recall
        "accuracy": float(accuracy_score(y, pred)),  # Exactitud global (engañosa con 2 % de fraude)
        "matriz": [[int(tn), int(fp)], [int(fn), int(tp)]],  # Matriz de confusión [[VN, FP], [FN, VP]]
    }


# ---------------------------------------------------------------------------
# Prueba de DeLong (DeLong, DeLong y Clarke-Pearson, 1988; algoritmo rápido de Sun y Xu, 2014)
# ---------------------------------------------------------------------------
def _rangos_medios(x):  # Rangos con empates promediados (necesarios para el estadístico de Mann-Whitney)
    orden = np.argsort(x)  # Posiciones que ordenan el vector
    xs = x[orden]  # Valores ordenados
    n = len(x)  # Tamaño del vector
    t = np.zeros(n)  # Rangos en el orden ordenado
    i = 0  # Puntero de inicio de un bloque de empates
    while i < n:  # Recorre los bloques de valores iguales
        j = i  # Puntero de fin del bloque
        while j < n and xs[j] == xs[i]:  # Avanza mientras haya empate
            j += 1  # Siguiente posición
        t[i:j] = 0.5 * (i + j - 1) + 1  # Rango medio del bloque (base 1)
        i = j  # Siguiente bloque
    rangos = np.empty(n)  # Rangos en el orden original
    rangos[orden] = t  # Deshace el ordenamiento
    return rangos  # Vector de rangos medios


def _delong_componentes(y, preds):  # AUC y matriz de covarianza de DeLong para varios modelos a la vez
    pos, neg = preds[:, y == 1], preds[:, y == 0]  # Puntajes de fraudes y de legítimas para cada modelo
    m, n = pos.shape[1], neg.shape[1]  # Número de positivos y negativos
    k = preds.shape[0]  # Número de modelos
    tx = np.array([_rangos_medios(pos[r]) for r in range(k)])  # Rangos dentro de los positivos
    ty = np.array([_rangos_medios(neg[r]) for r in range(k)])  # Rangos dentro de los negativos
    tz = np.array([_rangos_medios(np.r_[pos[r], neg[r]]) for r in range(k)])  # Rangos en el conjunto completo
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n  # AUC de cada modelo (estadístico de Mann-Whitney)
    v01 = (tz[:, :m] - tx) / n  # Componentes estructurales de los positivos
    v10 = 1 - (tz[:, m:] - ty) / m  # Componentes estructurales de los negativos
    cov = np.cov(v01) / m + np.cov(v10) / n  # Matriz de covarianza de los AUC
    return auc, np.atleast_2d(cov)  # Devuelve AUC y covarianza


def prueba_delong(y, p_a, p_b):  # Compara dos AUC calculados sobre las MISMAS transacciones (muestras pareadas)
    """Devuelve (diferencia de AUC a − b, IC 95 % de la diferencia, valor p bilateral)."""
    y = np.asarray(y)  # Objetivo como arreglo
    auc, cov = _delong_componentes(y, np.vstack([p_a, p_b]))  # AUC y covarianza conjunta de ambos modelos
    dif = float(auc[0] - auc[1])  # Diferencia de AUC
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])  # Varianza de la diferencia (tiene en cuenta la correlación)
    ee = np.sqrt(max(var, 1e-12))  # Error estándar
    z = dif / ee  # Estadístico z
    p = float(2 * stats.norm.sf(abs(z)))  # Valor p bilateral
    return dif, (dif - 1.96 * ee, dif + 1.96 * ee), p  # Diferencia, intervalo de confianza y valor p


def ic_auc_delong(y, p):  # Intervalo de confianza del 95 % para un AUC individual
    auc, cov = _delong_componentes(np.asarray(y), np.atleast_2d(p))  # AUC y varianza de DeLong
    ee = float(np.sqrt(cov[0, 0]))  # Error estándar
    return float(auc[0] - 1.96 * ee), float(auc[0] + 1.96 * ee)  # Límites inferior y superior
