"""Entrena y evalúa los cinco modelos del dashboard (cuatro recomendados + RVM original) y guarda todo en artefactos.joblib.

Flujo sin fuga de datos y con control del sobreajuste:
1. División TEMPORAL antes de cualquier transformación: ene–oct = entrenamiento, nov–dic = prueba.
2. Del periodo de entrenamiento se aparta un 20 % estratificado de validación para elegir el umbral.
3. Cada modelo es un Pipeline (preprocesamiento + clasificador) ajustado con GridSearchCV y
   StratifiedKFold de 5 pliegues; el preprocesamiento se reajusta dentro de cada pliegue.
4. El conjunto de prueba (nov–dic) se usa una sola vez, para reportar.

Ejecución (desde la raíz del proyecto):  python -m dashboard.entrenar
"""

import time  # Mide el tiempo de entrenamiento de cada modelo

import joblib  # Guarda los artefactos en disco
import numpy as np  # Cálculo numérico
import pandas as pd  # Construcción de transacciones de referencia
from sklearn.inspection import permutation_importance  # Importancia de variables por permutación
from sklearn.linear_model import LogisticRegression  # Modelo rápido para la demostración de fuga
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # Métricas y curvas
from sklearn.model_selection import train_test_split  # Particiones estratificadas
from sklearn.pipeline import Pipeline  # Pipeline para la demostración de fuga

from src import config as cf  # Configuración del proyecto
from src import modelado as mod  # Funciones reutilizables de modelado

TAM_MUESTRA_KNN = 20_000  # Filas de la muestra estratificada para KNN (escala mal con todo el dataset)
TAM_MUESTRA_IMPORTANCIA = 6_000  # Filas de prueba usadas para la importancia por permutación


def reducir_curva(x, y, max_puntos=400):  # Reduce el número de puntos de una curva para que el dashboard sea ligero
    if len(x) <= max_puntos:  # Si ya es pequeña
        return np.asarray(x).tolist(), np.asarray(y).tolist()  # Se devuelve tal cual
    idx = np.unique(np.linspace(0, len(x) - 1, max_puntos).astype(int))  # Índices equiespaciados
    return np.asarray(x)[idx].tolist(), np.asarray(y)[idx].tolist()  # Puntos seleccionados


def demo_fuga(df):  # Reproduce el ejemplo del enunciado: modelo con fuga vs. sin fuga
    """Compara el AUC de un modelo que incluye columnas posteriores al fraude con el flujo correcto."""
    muestra, _ = train_test_split(df, train_size=10_000, stratify=df[cf.TARGET], random_state=cf.SEMILLA)  # Muestra estratificada de 10 000 filas
    y = muestra[cf.TARGET]  # Objetivo de la muestra
    num_l, cat_l = cf.NUM_FEATURES + ["fraud_amount_lost"], cf.CAT_FEATURES + ["fraud_type"]  # Variables con las dos columnas con fuga
    prep_l = mod.make_preprocessor(num_l, cat_l)  # Preprocesador con fuga
    X_leak_t = prep_l.fit_transform(muestra[num_l + cat_l])  # ERROR deliberado: se ajusta con TODOS los datos antes de dividir
    a, b, ya, yb = train_test_split(X_leak_t, y, test_size=0.2, stratify=y, random_state=cf.SEMILLA)  # División posterior a la transformación
    p_l = LogisticRegression(max_iter=3000, class_weight="balanced").fit(a, ya).predict_proba(b)[:, 1]  # Modelo con fuga
    Xa, Xb, ya2, yb2 = train_test_split(muestra[cf.FEATURES], y, test_size=0.2, stratify=y, random_state=cf.SEMILLA)  # Flujo correcto: dividir primero
    pipe = Pipeline([("prep", mod.make_preprocessor()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])  # Preprocesamiento dentro del Pipeline
    p = pipe.fit(Xa, ya2).predict_proba(Xb)[:, 1]  # Modelo sin fuga
    return {"con_fuga": {"auc_roc": float(roc_auc_score(yb, p_l)), "auc_pr": float(average_precision_score(yb, p_l))},  # Métricas con fuga
            "sin_fuga": {"auc_roc": float(roc_auc_score(yb2, p)), "auc_pr": float(average_precision_score(yb2, p))},  # Métricas sin fuga
            "azar": float(yb2.mean())}  # AUC-PR de referencia de un modelo al azar = tasa de fraude


def variable_de_base(nombre):  # Traduce el nombre de una función base (p. ej. "spl__transaction_amount_sp_3") a su variable original
    limpio = nombre.split("__", 1)[1]  # Quita el prefijo del ColumnTransformer ("spl__" o "dis__")
    return max((c for c in cf.FEATURES if limpio.startswith(c)), key=len)  # La variable más larga que coincide con el inicio del nombre


def interpretar_rvm(pipe, X_ref):  # Extrae lo necesario para explicar la RVM en el dashboard
    """Relevancia por variable, curvas de contribución con banda de incertidumbre, poda ARD e historial."""
    prep, rvm = pipe.named_steps["prep"], pipe.named_steps["clf"]  # Preprocesamiento (funciones base) y modelo
    nombres = prep.named_steps["base"].get_feature_names_out()[prep.named_steps["varianza"].get_support()]  # Nombres de las columnas que recibe la RVM
    var_col = np.array([variable_de_base(n) for n in nombres])  # Variable original de cada función base
    rel = rvm.relevantes_  # Índices relevantes (0 = sesgo; la columna j de X es el índice j + 1)
    peso = dict(zip(rel.tolist(), rvm.mu_.tolist()))  # Índice → peso posterior
    tipica = X_ref.median(numeric_only=True).to_dict() | {c: X_ref[c].mode().iloc[0] for c in cf.CAT_FEATURES}  # Transacción típica de referencia
    curvas, relevancia = {}, {}  # Resultados por variable
    for v in cf.FEATURES:  # Recorre las 35 variables
        cols = [k for k in range(len(nombres)) if var_col[k] == v]  # Funciones base de la variable v
        idx = [k + 1 for k in cols if (k + 1) in peso]  # Las que sobrevivieron a la poda (índices en el espacio con sesgo)
        relevancia[v] = {"bases": len(cols), "relevantes": len(idx), "peso_abs": float(sum(abs(peso[i]) for i in idx))}  # Resumen
        if v in cf.NUM_CONTINUAS:  # Continua: rejilla de cuantiles del 1 % al 99 %
            valores = np.unique(np.quantile(X_ref[v], np.linspace(0.01, 0.99, 80)))  # Valores donde se evalúa la curva
        else:  # Discreta o categórica: cada valor observado (las categorías raras quedan agrupadas por el one-hot)
            valores = X_ref[v].value_counts().index[:25].tolist()  # Hasta 25 valores más frecuentes
            valores = sorted(valores) if v in cf.NUM_DISCRETAS else valores  # Las discretas en orden natural
        filas = pd.DataFrame([{**tipica, v: x} for x in valores])[cf.FEATURES]  # Transacción típica cambiando solo v
        Z = np.hstack([np.ones((len(filas), 1)), prep.transform(filas)])  # Funciones base con la columna de sesgo
        pos = [list(rel).index(i) for i in idx]  # Posición de cada índice relevante dentro de mu_ y Sigma_
        if pos:  # La variable conserva funciones relevantes
            Zv = Z[:, idx]  # Funciones base relevantes de la variable
            f = Zv @ rvm.mu_[pos]  # Contribución al logit
            var = np.einsum("ij,jk,ik->i", Zv, rvm.Sigma_[np.ix_(pos, pos)], Zv)  # Varianza posterior de la contribución
        else:  # Todas sus funciones fueron podadas: contribución nula
            f, var = np.zeros(len(valores)), np.zeros(len(valores))  # Curva plana
        f = f - f.mean()  # Centra la curva (el nivel lo absorbe el sesgo)
        curvas[v] = {"tipo": "num" if v in cf.NUM_CONTINUAS else "cat", "x": [x if isinstance(x, str) else float(x) for x in valores],  # Datos de la curva
                     "y": f.tolist(), "sd": np.sqrt(var).tolist()}
    alphas = np.log10(rvm.alpha_[1:])  # log10 α de cada función base (sin el sesgo)
    return {"curvas": curvas, "relevancia": dict(sorted(relevancia.items(), key=lambda kv: kv[1]["peso_abs"], reverse=True)),  # Variables por relevancia
            "alphas": {"log10_alpha": alphas.tolist(), "variable": var_col.tolist(), "relevante": [(k + 1) in peso for k in range(len(nombres))]},  # Poda ARD
            "historial": rvm.historial_, "n_bases": int(rvm.n_bases_), "n_relevantes": int(len(rel)), "n_iter": int(rvm.n_iter_)}  # Convergencia


def tabla_cv(grid):  # Resume los resultados de la validación cruzada de cada combinación de hiperparámetros
    r = grid.cv_results_  # Diccionario de resultados de GridSearchCV
    filas = []  # Una fila por combinación
    for i, params in enumerate(r["params"]):  # Recorre las combinaciones
        filas.append({"params": {k.replace("clf__", ""): str(v) for k, v in params.items()},  # Hiperparámetros sin el prefijo del Pipeline
                      "cv_auc_pr": float(r["mean_test_auc_pr"][i]), "cv_auc_roc": float(r["mean_test_auc_roc"][i]),  # Métricas de validación
                      "train_auc_roc": float(r["mean_train_auc_roc"][i]),  # AUC en los pliegues de entrenamiento (para ver sobreajuste)
                      "rank": int(r["rank_test_auc_pr"][i])})  # Posición en el ranking
    return sorted(filas, key=lambda f: f["rank"])  # Ordenadas de mejor a peor


def entrenar_todo():  # Ejecuta el flujo completo y devuelve el diccionario de artefactos
    df = cf.cargar_datos()  # Carga y ordena el dataset
    train_df, test_df = cf.division_temporal(df)  # División temporal ANTES de cualquier transformación
    X_train, y_train = train_df[cf.FEATURES], train_df[cf.TARGET]  # Variables y objetivo del periodo ene–oct
    X_test, y_test = test_df[cf.FEATURES], test_df[cf.TARGET]  # Variables y objetivo del periodo nov–dic
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=cf.SEMILLA)  # Validación para el umbral
    X_knn, _, y_knn, _ = train_test_split(X_tr, y_tr, train_size=TAM_MUESTRA_KNN, stratify=y_tr, random_state=cf.SEMILLA)  # Muestra para KNN
    X_imp, _, y_imp, _ = train_test_split(X_test, y_test, train_size=TAM_MUESTRA_IMPORTANCIA, stratify=y_test, random_state=cf.SEMILLA)  # Muestra para importancias
    resultados = {}  # Resultados por modelo
    for nombre, (modelo, malla, prep, usa_muestra) in mod.catalogo_modelos().items():  # Recorre el catálogo
        t0 = time.time()  # Marca de tiempo inicial
        Xa, ya = (X_knn, y_knn) if usa_muestra else (X_tr, y_tr)  # KNN usa la muestra; el resto, todo el entrenamiento
        grid = mod.train_pipeline(Xa, ya, modelo, malla, prep)  # Pipeline + GridSearchCV + StratifiedKFold
        mejor = grid.best_estimator_  # Pipeline ganador reentrenado con todos sus datos de entrenamiento
        p_tr = mejor.predict_proba(Xa)[:, 1]  # Probabilidades en entrenamiento (solo para medir el sobreajuste)
        p_val = mejor.predict_proba(X_val)[:, 1]  # Probabilidades en validación (para elegir el umbral)
        p_te = mejor.predict_proba(X_test)[:, 1]  # Probabilidades en prueba (evaluación final, una sola vez)
        umbral = mod.umbral_max_f1(y_val, p_val)  # Umbral que maximiza F1 en validación
        fpr, tpr, _ = roc_curve(y_test, p_te)  # Curva ROC en prueba
        prec, rec, _ = precision_recall_curve(y_test, p_te)  # Curva precisión-recall en prueba
        imp = permutation_importance(mejor, X_imp, y_imp, scoring="roc_auc", n_repeats=3, random_state=cf.SEMILLA, n_jobs=-1)  # Importancia por permutación
        pred = p_te >= umbral  # Predicción con el umbral elegido
        por_tipo = {t: float(pred[(test_df["fraud_type"] == t).to_numpy()].mean())  # Recall por tipo de fraude: fraud_type SOLO para analizar
                    for t in test_df["fraud_type"].unique() if t != "Legitimate"}  # Solo los tipos de fraude
        resultados[nombre] = {
            "params": {k.replace("clf__", ""): str(v) for k, v in grid.best_params_.items()},  # Mejores hiperparámetros
            "cv_auc_pr": float(grid.cv_results_["mean_test_auc_pr"][grid.best_index_]),  # AUC-PR de validación cruzada
            "cv_auc_roc": float(grid.cv_results_["mean_test_auc_roc"][grid.best_index_]),  # AUC-ROC de validación cruzada
            "train_auc_roc": float(roc_auc_score(ya, p_tr)),  # AUC en entrenamiento
            "val_auc_roc": float(roc_auc_score(y_val, p_val)),  # AUC en validación
            "umbral": umbral,  # Umbral de decisión elegido en validación
            "test": mod.metricas(y_test.to_numpy(), p_te, umbral),  # Métricas en prueba con el umbral elegido
            "ic_auc": mod.ic_auc_delong(y_test.to_numpy(), p_te),  # Intervalo de confianza del AUC (DeLong)
            "p_test": p_te.astype(np.float32),  # Probabilidades de prueba (para el umbral interactivo y DeLong)
            "roc": reducir_curva(fpr, tpr),  # Curva ROC reducida
            "pr": reducir_curva(rec[::-1], prec[::-1]),  # Curva PR reducida (recall creciente)
            "importancia": dict(sorted(zip(cf.FEATURES, imp.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True)),  # Importancias ordenadas
            "recall_por_tipo": por_tipo,  # Detección por tipo de fraude
            "cv_tabla": tabla_cv(grid),  # Todas las combinaciones probadas
            "n_train": int(len(ya)),  # Filas usadas para entrenar
            "segundos": round(time.time() - t0, 1),  # Tiempo total del modelo
        }
        if nombre == "RVM":  # Datos de interpretabilidad del modelo original
            resultados[nombre]["rvm"] = interpretar_rvm(mejor, X_tr)  # Relevancia, curvas, poda ARD y convergencia
            modelo_simulador = mejor  # Se guarda para el simulador de predicción del dashboard
        print(f"{nombre:<20} CV AUC-PR={resultados[nombre]['cv_auc_pr']:.4f}  test AUC-ROC={resultados[nombre]['test']['auc_roc']:.4f}  "
              f"AUC-PR={resultados[nombre]['test']['auc_pr']:.4f}  ({resultados[nombre]['segundos']} s)", flush=True)  # Progreso en consola
    delong = {}  # Comparación del modelo original contra cada modelo recomendado
    for nombre in resultados:  # Recorre los modelos
        if nombre != "RVM":  # Excepto contra sí mismo
            dif, ic, p = mod.prueba_delong(y_test.to_numpy(), resultados["RVM"]["p_test"], resultados[nombre]["p_test"])  # Prueba de DeLong pareada
            delong[nombre] = {"dif": dif, "ic": ic, "p": p}  # Guarda diferencia, intervalo y valor p
    return {
        "resultados": resultados,  # Métricas por modelo
        "delong": delong,  # Pruebas de significancia
        "fuga": demo_fuga(df),  # Demostración de fuga de datos
        "y_test": y_test.to_numpy().astype(np.int8),  # Etiquetas reales de prueba
        "tipo_test": test_df["fraud_type"].to_numpy(),  # Tipo de fraude en prueba (solo para análisis de errores)
        "n": {"train": len(X_tr), "val": len(X_val), "test": len(X_test), "knn": TAM_MUESTRA_KNN},  # Tamaños de cada conjunto
        "simulador": modelo_simulador,  # Pipeline RVM entrenado para predicciones individuales
        "umbral_simulador": resultados["RVM"]["umbral"],  # Umbral del simulador
    }


if __name__ == "__main__":  # Solo se ejecuta al lanzar el archivo como programa
    artefactos = entrenar_todo()  # Entrena todo
    joblib.dump(artefactos, cf.RUTA_ARTEFACTOS, compress=3)  # Guarda comprimido para que el dashboard abra al instante
    for k, v in artefactos["delong"].items():  # Imprime las pruebas de DeLong
        print(f"RVM vs {k:<20} ΔAUC={v['dif']:+.4f}  IC95%=({v['ic'][0]:+.4f}, {v['ic'][1]:+.4f})  p={v['p']:.4g}")  # Resultado de cada prueba
    print("Artefactos guardados en", cf.RUTA_ARTEFACTOS)  # Confirma la ruta
