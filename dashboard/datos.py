"""Datos y artefactos compartidos por todas las secciones del dashboard (se cargan una sola vez)."""

import joblib  # Carga de los artefactos ya entrenados

from src import config as cf  # Rutas, columnas y división temporal

DF = cf.cargar_datos()  # Dataset completo (100 000 transacciones) para el análisis exploratorio
TRAIN_DF, TEST_DF = cf.division_temporal(DF)  # Periodos de entrenamiento (ene–oct) y prueba (nov–dic)
ART = joblib.load(cf.RUTA_ARTEFACTOS)  # Modelos ya entrenados sin fuga de datos, con sus métricas
RES = ART["resultados"]  # Resultados por modelo
RVM = RES["RVM"]  # Resultados del modelo original
INFO_RVM = RVM["rvm"]  # Interpretación de la RVM: curvas, relevancia, poda ARD e historial
Y_TEST = ART["y_test"]  # Etiquetas reales del periodo de prueba
RECOMENDADOS = [n for n in RES if n != "RVM"]  # Los cuatro modelos recomendados por el enunciado
MEJOR_REC = max(RECOMENDADOS, key=lambda n: RES[n]["cv_auc_pr"])  # Mejor recomendado según la validación cruzada (sin mirar la prueba)
TASA = DF[cf.TARGET].mean()  # Tasa global de fraude
RUTA_EXP = cf.RAIZ / "dashboard" / "experimentos.joblib"  # Comparación con los modelos del listado (opcional)
EXP = joblib.load(RUTA_EXP) if RUTA_EXP.exists() else []  # Resultados de esa comparación, si existen
RUTA_PRUEBAS = cf.RAIZ / "dashboard" / "pruebas.joblib"  # Métricas de la literatura, pruebas de significancia y tiempos
PRUEBAS = joblib.load(RUTA_PRUEBAS) if RUTA_PRUEBAS.exists() else None  # Se carga si ya se corrió python -m dashboard.pruebas
