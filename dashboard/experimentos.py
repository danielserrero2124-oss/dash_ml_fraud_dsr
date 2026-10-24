"""Comparación de modelos del listado del curso frente a GradientBoosting (línea base más fuerte).

Candidatos del listado aplicables a clasificación binaria con datos tabulares:
  * Hierarchical CatBoost         → CatBoost (boosting ordenado con estadísticas de objetivo para categóricas)
  * LGBR + optimizador            → LightGBM con búsqueda aleatoria de hiperparámetros
  * TCR (XGBoost)                 → XGBoost
  * RFXG (Random Forest + XGBoost) → promedio de rangos de ambos modelos
  * Multi-task logistic regression → regresión logística multitarea (una tarea por canal de la transacción)
  * Relevance Vector Machine      → modelo original (resultados del entrenamiento principal)

Protocolo sin "espiar" la prueba:
  * Cada candidato se ajusta con ene–ago y su configuración (incluido el número de árboles) se elige con sep–oct.
  * La configuración elegida se reentrena con ene–oct y se evalúa UNA vez con nov–dic.
  * La comparación con GradientBoosting es pareada (mismas transacciones) con la prueba de DeLong.

Ejecución (desde la raíz del proyecto, después de python -m dashboard.entrenar):  python -m dashboard.experimentos
"""

import time  # Mide el tiempo de cada candidato

import catboost as cb  # CatBoost
import joblib  # Carga de artefactos y guardado de resultados
import lightgbm as lgb  # LightGBM
import numpy as np  # Cálculo numérico
import xgboost as xgb  # XGBoost
from scipy.stats import rankdata  # Rangos para combinar modelos con escalas de probabilidad distintas
from sklearn.ensemble import RandomForestClassifier  # Bosque aleatorio (parte de RFXG)
from sklearn.linear_model import LogisticRegression  # Regresión logística multitarea
from sklearn.metrics import roc_auc_score  # AUC-ROC
from sklearn.pipeline import Pipeline  # Preprocesamiento + modelo

from src import config as cf  # Configuración del proyecto
from src import modelado as mod  # Preprocesador y prueba de DeLong

RUTA_EXPERIMENTOS = cf.RAIZ / "dashboard" / "experimentos.joblib"  # Archivo de resultados para el dashboard
FIN_AJUSTE = "2026-09-01"  # Ene–ago: ajuste; sep–oct: validación temporal para elegir la configuración
CANALES = ["Online", "Mobile App", "In-store", "ATM"]  # Tareas de la regresión logística multitarea


def con_categorias(df):  # Variables con las categóricas como tipo category (LightGBM y XGBoost las usan de forma nativa)
    X = df[cf.FEATURES].copy()  # Las 35 variables
    for c in cf.CAT_FEATURES:  # Categóricas
        X[c] = X[c].astype("category")  # Tipo category
    return X  # Tabla lista


def rango(p):  # Convierte probabilidades en rangos normalizados (el AUC solo depende del orden)
    return rankdata(p) / len(p)  # Rango entre 0 y 1


def catboost(d_aj, d_va, d_tr, d_te, y):  # Candidato: CatBoost
    mejor = None  # (AUC de validación, profundidad, árboles)
    for prof in [3, 4, 6]:  # Profundidades candidatas
        m = cb.CatBoostClassifier(iterations=2000, depth=prof, learning_rate=0.03, random_seed=cf.SEMILLA, verbose=0,  # Modelo
                                  cat_features=cf.CAT_FEATURES, eval_metric="AUC", thread_count=-1)
        m.fit(d_aj[cf.FEATURES], y[d_aj.index], eval_set=(d_va[cf.FEATURES], y[d_va.index]), early_stopping_rounds=200)  # Parada temprana en sep–oct
        auc = roc_auc_score(y[d_va.index], m.predict_proba(d_va[cf.FEATURES])[:, 1])  # AUC de validación
        if mejor is None or auc > mejor[0]:  # Guarda la mejor configuración
            mejor = (auc, prof, m.best_iteration_ + 1)  # AUC, profundidad y número de árboles
    final = cb.CatBoostClassifier(iterations=mejor[2], depth=mejor[1], learning_rate=0.03, random_seed=cf.SEMILLA, verbose=0,  # Reentrena con ene–oct
                                  cat_features=cf.CAT_FEATURES, thread_count=-1).fit(d_tr[cf.FEATURES], y[d_tr.index])
    return mejor[0], final.predict_proba(d_te[cf.FEATURES])[:, 1], f"depth={mejor[1]}, iterations={mejor[2]}"  # Validación, prueba y configuración


def lightgbm(d_aj, d_va, d_tr, d_te, y):  # Candidato: LightGBM con búsqueda aleatoria de 25 configuraciones
    rng = np.random.default_rng(cf.SEMILLA)  # Generador reproducible
    mejor = None  # (AUC, parámetros, árboles)
    for _ in range(25):  # Configuraciones aleatorias
        p = dict(num_leaves=int(rng.choice([3, 4, 7, 15])), learning_rate=float(rng.choice([0.01, 0.02, 0.05])),  # Espacio de búsqueda
                 min_child_samples=int(rng.choice([50, 100, 200, 400])), reg_lambda=float(rng.choice([0, 1, 5, 20])),
                 colsample_bytree=float(rng.choice([0.5, 0.7, 1.0])), subsample=float(rng.choice([0.7, 1.0])), subsample_freq=1,
                 random_state=cf.SEMILLA, verbose=-1)
        m = lgb.LGBMClassifier(n_estimators=3000, **p).fit(con_categorias(d_aj), y[d_aj.index], eval_set=[(con_categorias(d_va), y[d_va.index])],  # Ajuste
                                                          eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        auc = roc_auc_score(y[d_va.index], m.predict_proba(con_categorias(d_va))[:, 1])  # AUC de validación
        if mejor is None or auc > mejor[0]:  # Guarda la mejor
            mejor = (auc, p, m.best_iteration_)  # AUC, parámetros y árboles
    final = lgb.LGBMClassifier(n_estimators=mejor[2], **mejor[1]).fit(con_categorias(d_tr), y[d_tr.index])  # Reentrena con ene–oct
    cfg = ", ".join(f"{k}={v}" for k, v in mejor[1].items() if k in ("num_leaves", "learning_rate", "min_child_samples", "reg_lambda"))  # Resumen
    return mejor[0], final.predict_proba(con_categorias(d_te))[:, 1], f"{cfg}, n_estimators={mejor[2]}"  # Resultados


def xgboost_(d_aj, d_va, d_tr, d_te, y):  # Candidato: XGBoost
    mejor = None  # (AUC, profundidad, árboles)
    for prof in [2, 3, 4]:  # Profundidades candidatas
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.02, max_depth=prof, subsample=0.8, colsample_bytree=0.7,  # Modelo
                              enable_categorical=True, tree_method="hist", early_stopping_rounds=200, eval_metric="auc", random_state=cf.SEMILLA)
        m.fit(con_categorias(d_aj), y[d_aj.index], eval_set=[(con_categorias(d_va), y[d_va.index])], verbose=False)  # Parada temprana en sep–oct
        auc = roc_auc_score(y[d_va.index], m.predict_proba(con_categorias(d_va))[:, 1])  # AUC de validación
        if mejor is None or auc > mejor[0]:  # Guarda la mejor
            mejor = (auc, prof, m.best_iteration + 1)  # AUC, profundidad y árboles
    final = xgb.XGBClassifier(n_estimators=mejor[2], learning_rate=0.02, max_depth=mejor[1], subsample=0.8, colsample_bytree=0.7,  # Reentrena
                              enable_categorical=True, tree_method="hist", random_state=cf.SEMILLA).fit(con_categorias(d_tr), y[d_tr.index])
    return mejor[0], final.predict_proba(con_categorias(d_te))[:, 1], f"max_depth={mejor[1]}, n_estimators={mejor[2]}"  # Resultados


def rfxg(d_aj, d_va, d_tr, d_te, y, xgb_cfg):  # Candidato: RFXG = promedio de rangos de Random Forest y XGBoost
    def bosque():  # Bosque con la configuración ganadora del entrenamiento principal
        return Pipeline([("prep", mod.make_preprocessor()), ("clf", RandomForestClassifier(300, max_depth=14, min_samples_leaf=40, n_jobs=-1, random_state=cf.SEMILLA))])
    def xg():  # XGBoost con la configuración elegida en su propio experimento
        return xgb.XGBClassifier(**xgb_cfg, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, enable_categorical=True, tree_method="hist", random_state=cf.SEMILLA)
    p_va = (rango(bosque().fit(d_aj[cf.FEATURES], y[d_aj.index]).predict_proba(d_va[cf.FEATURES])[:, 1])  # Bosque en validación
            + rango(xg().fit(con_categorias(d_aj), y[d_aj.index]).predict_proba(con_categorias(d_va))[:, 1])) / 2  # + XGBoost en validación
    p_te = (rango(bosque().fit(d_tr[cf.FEATURES], y[d_tr.index]).predict_proba(d_te[cf.FEATURES])[:, 1])  # Bosque reentrenado en prueba
            + rango(xg().fit(con_categorias(d_tr), y[d_tr.index]).predict_proba(con_categorias(d_te))[:, 1])) / 2  # + XGBoost reentrenado en prueba
    return roc_auc_score(y[d_va.index], p_va), p_te, "promedio de rangos RF + XGBoost"  # Resultados


def multitarea(d_aj, d_va, d_tr, d_te, y):  # Candidato: regresión logística multitarea (pesos compartidos + específicos por canal)
    def aumentar(Z, canal):  # Truco de aumento de variables: [Z, Z·1(canal=1), ..., Z·1(canal=4)]
        return np.hstack([Z] + [Z * (canal == c)[:, None] for c in CANALES])  # Bloque compartido + un bloque por tarea

    def ajustar(d_fit, d_eval, C):  # Ajusta con una partición y predice otra
        prep = mod.make_preprocessor().fit(d_fit[cf.FEATURES])  # Preprocesador ajustado solo con la partición de ajuste
        m = LogisticRegression(C=C, max_iter=4000).fit(aumentar(prep.transform(d_fit[cf.FEATURES]), d_fit["transaction_channel"].to_numpy()), y[d_fit.index])  # Modelo
        return m.predict_proba(aumentar(prep.transform(d_eval[cf.FEATURES]), d_eval["transaction_channel"].to_numpy()))[:, 1]  # Predicción

    aucs = {C: roc_auc_score(y[d_va.index], ajustar(d_aj, d_va, C)) for C in [0.01, 0.05]}  # AUC de validación por regularización
    C = max(aucs, key=aucs.get)  # Mejor C
    return aucs[C], ajustar(d_tr, d_te, C), f"C={C}, 4 tareas (canales)"  # Resultados


def main():  # Ejecuta todos los candidatos y guarda los resultados
    df = cf.cargar_datos()  # Datos ordenados en el tiempo
    art = joblib.load(cf.RUTA_ARTEFACTOS)  # Artefactos del entrenamiento principal
    y = df[cf.TARGET]  # Objetivo
    es_aj, es_tr = df["transaction_datetime"] < FIN_AJUSTE, df["transaction_datetime"] < cf.FECHA_CORTE  # Máscaras temporales
    d_aj, d_va, d_tr, d_te = df[es_aj], df[~es_aj & es_tr], df[es_tr], df[~es_tr]  # Ene–ago, sep–oct, ene–oct, nov–dic
    y_te = y[d_te.index].to_numpy()  # Objetivo de prueba
    assert (y_te == art["y_test"]).all()  # Garantiza que la comparación con las predicciones guardadas es pareada
    p_gb = art["resultados"]["GradientBoosting"]["p_test"]  # Línea base: GradientBoosting
    resultados = []  # Lista de resultados

    def registrar(nombre, origen, val_auc, p_test, cfg, t0):  # Calcula la prueba de DeLong y guarda el resultado
        dif, ic, p = mod.prueba_delong(y_te, p_test, p_gb)  # Comparación pareada con GradientBoosting
        r = {"nombre": nombre, "origen": origen, "config": cfg, "val_auc": float(val_auc) if val_auc is not None else None,  # Resultado
             "test_auc": float(roc_auc_score(y_te, p_test)), "dif": dif, "ic": ic, "p": p, "segundos": round(time.time() - t0, 1)}
        resultados.append(r)  # Lo agrega a la lista
        print(f"{nombre:<34} test={r['test_auc']:.4f}  ΔvsGB={dif:+.4f}  p={p:.3g}  ({cfg})", flush=True)  # Progreso

    t0 = time.time(); v, p, c = catboost(d_aj, d_va, d_tr, d_te, y); registrar("CatBoost", "Hierarchical CatBoost", v, p, c, t0)  # CatBoost
    t0 = time.time(); v, p, c = lightgbm(d_aj, d_va, d_tr, d_te, y); registrar("LightGBM + búsqueda aleatoria", "LGBR + optimizador", v, p, c, t0)  # LightGBM
    t0 = time.time(); v, p, c = xgboost_(d_aj, d_va, d_tr, d_te, y); registrar("XGBoost", "TCR (XGBoost)", v, p, c, t0)  # XGBoost
    prof = int(c.split("max_depth=")[1].split(",")[0])  # Profundidad elegida para XGBoost
    arboles = int(c.split("n_estimators=")[1])  # Árboles elegidos para XGBoost
    t0 = time.time(); v, p, c = rfxg(d_aj, d_va, d_tr, d_te, y, {"max_depth": prof, "n_estimators": arboles}); registrar("RFXG", "RFXG (Random Forest + XGBoost)", v, p, c, t0)  # RFXG
    t0 = time.time(); v, p, c = multitarea(d_aj, d_va, d_tr, d_te, y); registrar("Regresión logística multitarea", "Multi-task logistic regression", v, p, c, t0)  # Multitarea
    rvm = art["resultados"]["RVM"]  # Modelo original ya evaluado en el entrenamiento principal
    registrar("RVM (original)", "Relevance Vector Machine", None, rvm["p_test"], ", ".join(f"{k}={v}" for k, v in rvm["params"].items()), time.time())  # RVM
    joblib.dump(resultados, RUTA_EXPERIMENTOS)  # Guarda para el dashboard
    print("Resultados guardados en", RUTA_EXPERIMENTOS)  # Confirmación


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    main()  # Corre la comparación
