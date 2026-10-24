"""Métricas de comparación, pruebas de significancia estadística y tiempos de cómputo de los cinco modelos.

Métricas recomendadas en la literatura para clasificadores con clases desbalanceadas:
  * AUC-PR / precisión promedio (Saito y Rehmsmeier, 2015; Davis y Goadrich, 2006).
  * AUC-ROC con intervalo de DeLong (DeLong et al., 1988).
  * Coeficiente de correlación de Matthews, MCC (Chicco y Jurman, 2020).
  * Puntuación de Brier y log-loss: calidad de las probabilidades (Brier, 1950).
  * Precisión y recall en el 1 % de transacciones con mayor riesgo, precision@k (Dal Pozzolo et al., 2014).

Pruebas de significancia:
  * En la prueba (nov–dic), pareadas sobre las mismas transacciones:
      - DeLong para AUC-ROC (DeLong et al., 1988).
      - Bootstrap pareado y estratificado para AUC-PR (Efron y Tibshirani, 1993).
      - McNemar para los errores con el umbral elegido (Dietterich, 1998).
  * Con 5 × 2 validación cruzada sobre el entrenamiento (Dietterich, 1998; Alpaydin, 1999):
      - Prueba t pareada 5×2cv y prueba F combinada 5×2cv, RVM frente a cada modelo.
      - Prueba de Friedman y diferencia crítica de Nemenyi sobre los rangos de los 10 pliegues (Demšar, 2006).
  * Corrección de Holm para comparaciones múltiples (RVM frente a cuatro modelos).

Ejecución (desde la raíz del proyecto, después de python -m dashboard.entrenar):  python -m dashboard.pruebas
"""

import ast  # Convierte los hiperparámetros guardados como texto en valores de Python
import time  # Medición de tiempos de cómputo

import joblib  # Carga de artefactos, paralelismo y guardado
import numpy as np  # Cálculo numérico
from scipy import stats  # Distribuciones t, F, chi-cuadrado y normal
from sklearn.base import clone  # Copia sin entrenar de un estimador
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss,  # Métricas
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split  # 5 × 2 validación cruzada y particiones
from sklearn.pipeline import Pipeline  # Preprocesamiento + modelo

from src import config as cf  # Configuración del proyecto
from src import modelado as mod  # Catálogo de modelos y DeLong

RUTA_PRUEBAS = cf.RAIZ / "dashboard" / "pruebas.joblib"  # Archivo de resultados para el dashboard
N_BOOT = 2000  # Réplicas bootstrap para el AUC-PR
TOP_K = 0.01  # Fracción de transacciones de mayor riesgo para precision@k y recall@k
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}  # Valores críticos q_α (α = 0,05) de Nemenyi según el número de modelos (Demšar, 2006)


def valor(texto):  # Convierte un hiperparámetro guardado como texto ("0.01", "None", "balanced") en su valor de Python
    try:  # Números, None, True/False
        return ast.literal_eval(texto)  # Evalúa literales de forma segura
    except (ValueError, SyntaxError):  # Cadenas como "balanced"
        return texto  # Se deja como texto


def construir(nombre, params):  # Pipeline del modelo con los hiperparámetros elegidos por GridSearchCV
    estimador, _, prep, usa_muestra = mod.catalogo_modelos()[nombre]  # Estimador base, preprocesador y si usa muestra
    pipe = Pipeline([("prep", prep()), ("clf", clone(estimador))])  # Mismo Pipeline que en el entrenamiento
    pipe.set_params(**{(k if k.startswith("prep__") else f"clf__{k}"): valor(v) for k, v in params.items()})  # Aplica los mejores hiperparámetros
    return pipe, usa_muestra  # Pipeline y bandera de muestreo


def metricas_literatura(y, p, umbral):  # Métricas de comparación recomendadas en la literatura
    pred = (p >= umbral).astype(int)  # Clase predicha con el umbral de validación
    k = int(np.ceil(TOP_K * len(y)))  # Número de alertas en el 1 % de mayor riesgo
    top = np.argsort(-p)[:k]  # Índices de las transacciones con mayor probabilidad
    return {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),  # Áreas bajo las curvas
            "mcc": float(matthews_corrcoef(y, pred)), "bal_acc": float(balanced_accuracy_score(y, pred)),  # MCC y exactitud balanceada
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),  # Calidad probabilística
            "prec_k": float(y[top].mean()), "rec_k": float(y[top].sum() / y.sum())}  # precision@1 % y recall@1 %


def bootstrap_auc_pr(y, p_a, p_b, rng):  # Bootstrap pareado y estratificado de la diferencia de AUC-PR
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)  # Índices de fraudes y legítimas
    difs = np.empty(N_BOOT)  # Diferencias bootstrap
    for b in range(N_BOOT):  # Réplicas
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]  # Remuestreo con reemplazo dentro de cada clase
        difs[b] = average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx])  # Diferencia en la réplica
    dif = average_precision_score(y, p_a) - average_precision_score(y, p_b)  # Diferencia observada
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())  # Valor p bilateral (método percentil)
    return float(dif), (float(np.quantile(difs, 0.025)), float(np.quantile(difs, 0.975))), float(min(max(p, 1 / N_BOOT), 1.0))  # Diferencia, IC 95 % y p


def mcnemar(y, pred_a, pred_b):  # Prueba de McNemar con corrección de continuidad sobre los aciertos de dos clasificadores
    ok_a, ok_b = pred_a == y, pred_b == y  # Aciertos de cada modelo
    b = int(np.sum(ok_a & ~ok_b))  # A acierta y B falla
    c = int(np.sum(~ok_a & ok_b))  # A falla y B acierta
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0  # Estadístico con corrección de continuidad
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(stats.chi2.sf(chi2, 1))}  # Discordancias, estadístico y p


def t_5x2(d):  # Prueba t pareada 5×2cv de Dietterich (1998); d: matriz 5 × 2 de diferencias por repetición y pliegue
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)  # Varianza de cada repetición
    t = d[0, 0] / np.sqrt(s2.mean())  # Estadístico t con 5 grados de libertad
    return float(t), float(2 * stats.t.sf(abs(t), 5))  # t y p bilateral


def f_5x2(d):  # Prueba F combinada 5×2cv de Alpaydin (1999)
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)  # Varianza de cada repetición
    f = (d ** 2).sum() / (2 * s2.sum())  # Estadístico F con (10, 5) grados de libertad
    return float(f), float(stats.f.sf(f, 10, 5))  # F y p


def holm(pvalores):  # Corrección de Holm-Bonferroni para comparaciones múltiples
    orden = np.argsort(pvalores)  # De menor a mayor p
    m = len(pvalores)  # Número de comparaciones
    ajust = np.empty(m)  # p ajustados
    acum = 0.0  # Máximo acumulado (monotonía)
    for r, i in enumerate(orden):  # Recorre en orden
        acum = max(acum, min(1.0, (m - r) * pvalores[i]))  # p_(r) × (m − r), monótono no decreciente
        ajust[i] = acum  # Guarda el ajustado
    return ajust.tolist()  # Lista de p ajustados


def un_pliegue(nombres, params, X, y, idx_tr, idx_te, semilla):  # Ajusta los cinco modelos en un pliegue del 5×2cv
    salida = {}  # Resultados por modelo
    for n in nombres:  # Recorre los modelos
        pipe, usa_muestra = construir(n, params[n])  # Pipeline con los mejores hiperparámetros
        tr = idx_tr  # Filas de entrenamiento del pliegue
        if usa_muestra:  # KNN: muestra estratificada de 20 000 filas, como en el entrenamiento principal
            tr, _ = train_test_split(idx_tr, train_size=20_000, stratify=y[idx_tr], random_state=semilla)  # Submuestra
        t0 = time.perf_counter()  # Inicio del ajuste
        pipe.fit(X.iloc[tr], y[tr])  # Entrena
        t_fit = time.perf_counter() - t0  # Tiempo de entrenamiento
        t0 = time.perf_counter()  # Inicio de la predicción
        p = pipe.predict_proba(X.iloc[idx_te])[:, 1]  # Predice la mitad de evaluación
        t_pred = time.perf_counter() - t0  # Tiempo de predicción
        salida[n] = {"auc_roc": float(roc_auc_score(y[idx_te], p)), "auc_pr": float(average_precision_score(y[idx_te], p)),  # Métricas del pliegue
                     "t_fit": t_fit, "t_pred_1000": 1e6 * t_pred / len(idx_te), "n_fit": len(tr)}  # Tiempos (predicción en ms por 1 000 transacciones)
    return salida  # Diccionario por modelo


def main():  # Ejecuta todo y guarda los resultados
    art = joblib.load(cf.RUTA_ARTEFACTOS)  # Artefactos del entrenamiento principal
    res = art["resultados"]  # Resultados por modelo
    nombres = list(res)  # Orden de los modelos
    y_te = art["y_test"].astype(int)  # Objetivo de prueba
    rng = np.random.default_rng(cf.SEMILLA)  # Generador reproducible

    # --- 1. Métricas de la literatura en la prueba (nov–dic) ---
    metricas = {n: metricas_literatura(y_te, res[n]["p_test"].astype(float), res[n]["umbral"]) for n in nombres}  # Por modelo
    for n in nombres:  # Intervalos de confianza
        metricas[n]["ic_auc_roc"] = res[n]["ic_auc"]  # IC de DeLong (calculado en el entrenamiento)
        idx = [np.r_[rng.choice(np.flatnonzero(y_te == 1), int(y_te.sum())), rng.choice(np.flatnonzero(y_te == 0), int((y_te == 0).sum()))] for _ in range(500)]  # Réplicas
        aps = [average_precision_score(y_te[i], res[n]["p_test"][i]) for i in idx]  # AUC-PR en cada réplica
        metricas[n]["ic_auc_pr"] = (float(np.quantile(aps, 0.025)), float(np.quantile(aps, 0.975)))  # IC bootstrap del 95 %
    print("Métricas de la literatura calculadas", flush=True)  # Progreso

    # --- 2. Pruebas pareadas en la prueba: RVM frente a cada modelo ---
    p_rvm = res["RVM"]["p_test"].astype(float)  # Probabilidades de la RVM
    pred_rvm = (p_rvm >= res["RVM"]["umbral"]).astype(int)  # Predicciones de la RVM con su umbral
    prueba = {}  # Resultados por modelo
    for n in [m for m in nombres if m != "RVM"]:  # Modelos recomendados
        p_n = res[n]["p_test"].astype(float)  # Probabilidades del modelo
        d_roc, ic_roc, p_roc = mod.prueba_delong(y_te, p_rvm, p_n)  # DeLong
        se = (ic_roc[1] - ic_roc[0]) / (2 * 1.96)  # Error estándar de la diferencia
        d_pr, ic_pr, p_pr = bootstrap_auc_pr(y_te, p_rvm, p_n, rng)  # Bootstrap del AUC-PR
        mc = mcnemar(y_te, pred_rvm, (p_n >= res[n]["umbral"]).astype(int))  # McNemar
        prueba[n] = {"delong": {"dif": d_roc, "ic": ic_roc, "z": d_roc / se if se > 0 else 0.0, "p": p_roc},  # Guarda las tres pruebas
                     "bootstrap_pr": {"dif": d_pr, "ic": ic_pr, "p": p_pr}, "mcnemar": mc}
    for clave in ["delong", "bootstrap_pr", "mcnemar"]:  # Corrección de Holm por familia de pruebas
        aj = holm([prueba[n][clave]["p"] for n in prueba])  # p ajustados
        for n, pa in zip(prueba, aj):  # Los asigna
            prueba[n][clave]["p_holm"] = pa  # p ajustado por Holm
    print("Pruebas en el conjunto de prueba calculadas", flush=True)  # Progreso

    # --- 3. 5 × 2 validación cruzada sobre el entrenamiento (mismas particiones para los cinco modelos) ---
    df = cf.cargar_datos()  # Datos ordenados
    train_df, _ = cf.division_temporal(df)  # Periodo ene–oct
    X_tr, _, y_tr, _ = train_test_split(train_df[cf.FEATURES], train_df[cf.TARGET], test_size=0.2, stratify=train_df[cf.TARGET], random_state=cf.SEMILLA)  # Mismo 80 % que en el entrenamiento
    X_tr, y_arr = X_tr.reset_index(drop=True), y_tr.to_numpy()  # Índices posicionales
    params = {n: res[n]["params"] for n in nombres}  # Hiperparámetros elegidos
    splits = list(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=cf.SEMILLA).split(X_tr, y_arr))  # 10 pliegues: 5 repeticiones × 2
    t0 = time.time()  # Tiempo total
    pliegues = joblib.Parallel(n_jobs=5)(joblib.delayed(un_pliegue)(nombres, params, X_tr, y_arr, a, b, cf.SEMILLA + i) for i, (a, b) in enumerate(splits))  # Ejecuta en paralelo
    print(f"5x2cv terminada en {time.time() - t0:.0f} s", flush=True)  # Progreso
    cv = {n: {m: np.array([pl[n][m] for pl in pliegues]).reshape(5, 2) for m in ["auc_roc", "auc_pr"]} for n in nombres}  # Matrices 5 × 2 por modelo
    cv_pruebas = {}  # Pruebas 5×2cv de la RVM frente a cada modelo
    for n in [m for m in nombres if m != "RVM"]:  # Modelos recomendados
        cv_pruebas[n] = {}  # Por métrica
        for m in ["auc_roc", "auc_pr"]:  # Dos métricas
            d = cv["RVM"][m] - cv[n][m]  # Diferencias por pliegue
            t, pt = t_5x2(d)  # Prueba t 5×2cv
            f, pf = f_5x2(d)  # Prueba F combinada
            cv_pruebas[n][m] = {"dif_media": float(d.mean()), "t": t, "p_t": pt, "f": f, "p_f": pf}  # Guarda
    for m in ["auc_roc", "auc_pr"]:  # Holm sobre la prueba F de cada métrica
        aj = holm([cv_pruebas[n][m]["p_f"] for n in cv_pruebas])  # p ajustados
        for n, pa in zip(cv_pruebas, aj):  # Los asigna
            cv_pruebas[n][m]["p_f_holm"] = pa  # p ajustado

    # --- 4. Friedman + Nemenyi sobre los rangos de los 10 pliegues ---
    friedman = {}  # Por métrica
    for m in ["auc_roc", "auc_pr"]:  # Dos métricas
        M = np.column_stack([cv[n][m].ravel() for n in nombres])  # Matriz 10 pliegues × 5 modelos
        rangos = np.array([stats.rankdata(-fila) for fila in M])  # Rango 1 = mejor en cada pliegue
        chi2, p = stats.friedmanchisquare(*M.T)  # Prueba de Friedman
        k, N = len(nombres), M.shape[0]  # Modelos y pliegues
        cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))  # Diferencia crítica de Nemenyi
        friedman[m] = {"rangos": dict(zip(nombres, rangos.mean(axis=0).tolist())), "chi2": float(chi2), "p": float(p), "cd": float(cd)}  # Guarda

    # --- 5. Tiempos de cómputo ---
    tiempos = {n: {"grid_s": float(res[n]["segundos"]),  # GridSearchCV completo + reajuste (entrenamiento principal)
                   "fit_s": float(np.mean([pl[n]["t_fit"] for pl in pliegues])),  # Un ajuste con los mejores hiperparámetros (media de 10)
                   "n_fit": int(np.mean([pl[n]["n_fit"] for pl in pliegues])),  # Filas de cada ajuste
                   "pred_ms_1000": float(np.mean([pl[n]["t_pred_1000"] for pl in pliegues]))} for n in nombres}  # Predicción por 1 000 transacciones

    joblib.dump({"metricas": metricas, "prueba": prueba, "cv": {n: {m: v.tolist() for m, v in d.items()} for n, d in cv.items()},  # Guarda todo
                 "cv_pruebas": cv_pruebas, "friedman": friedman, "tiempos": tiempos, "top_k": TOP_K, "n_boot": N_BOOT}, RUTA_PRUEBAS)
    for n, d in cv_pruebas.items():  # Resumen en consola
        print(f"RVM vs {n:<20} 5x2cv F (AUC-ROC) p={d['auc_roc']['p_f']:.3g} | F (AUC-PR) p={d['auc_pr']['p_f']:.3g}")  # p de la prueba F
    print("Friedman AUC-ROC p =", round(friedman["auc_roc"]["p"], 4), "| rangos:", {k: round(v, 2) for k, v in friedman["auc_roc"]["rangos"].items()})  # Friedman
    print("Resultados guardados en", RUTA_PRUEBAS)  # Confirmación


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    main()  # Corre todo
