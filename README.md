# Detección de Fraude en Tarjetas de Crédito · Dashboard de ML

Dashboard interactivo (Dash + Plotly) organizado según la rúbrica de sustentación:

| Pestaña | Criterio de la rúbrica | Contenido |
| :--- | :--- | :--- |
| **Contexto** | 1. Problema y formulación matemática | Importancia del problema, objetivos, tipo de tarea, variables y formulación matemática. |
| **EDA** | 2. EDA, datos y preprocesamiento | Fuente y estructura, estadística descriptiva, distribuciones, atípicos, faltantes, correlaciones, transformaciones y fuga de datos. |
| **ML Models › Fundamentos teóricos** | 3. Fundamentos teóricos del modelo | Formulación de la RVM, función objetivo, supuestos, regularización ARD, complejidad, poda y curvas con incertidumbre. |
| **ML Models › Diseño experimental** | 4. Implementación y diseño experimental | Partición temporal, Pipeline, hiperparámetros, validación, reproducibilidad y decisiones computacionales. |
| **ML Models › Resultados** | 5. Resultados y evaluación crítica | Métricas, ROC/PR, matriz de confusión, ranking, DeLong, sobreajuste, modelos del listado, errores por tipo y simulador. |
| **ML Models › Conclusiones y preguntas** | 6 y 7. Conclusiones, limitaciones y preguntas | Conclusiones, limitaciones, generalización, trabajo futuro y preguntas esperadas. |

## Modelos
- Recomendados: `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier`.
- **Original: RVM-Fraude**, una *Relevance Vector Machine* bayesiana (Tipping, 2001) implementada desde cero en
  `src/rvm.py`: prior ARD, aproximación de Laplace (Newton/IRLS), máxima evidencia (MacKay) y predicción moderada,
  sobre funciones base de B-splines cúbicos y one-hot.
- Comparación con modelos del listado del curso (CatBoost, LightGBM, XGBoost, RFXG y regresión logística multitarea)
  en `dashboard/experimentos.py`.

## Cómo evita la fuga de datos y el sobreajuste
1. División temporal antes de transformar: enero–octubre para entrenar, noviembre–diciembre para evaluar.
2. Preprocesamiento dentro del `Pipeline` (se reajusta en cada pliegue de `GridSearchCV`).
3. `GridSearchCV` + `StratifiedKFold(5)` con AUC-PR como criterio y mallas que limitan la complejidad.
4. Umbral elegido en una validación separada; la prueba se usa una sola vez.
5. Sin `fraud_type`, `fraud_amount_lost`, identificadores ni la fecha cruda entre las predictoras.

## Estructura
```text
data/credit_card_fraud_detection_2026.csv   # dataset (no se versiona)
src/config.py           # rutas, columnas, carga y división temporal
src/rvm.py              # modelo original RVM-Fraude (desde cero)
src/modelado.py         # Pipeline, catálogo de modelos, umbral, métricas y prueba de DeLong
dashboard/entrenar.py   # entrena los 5 modelos y guarda dashboard/artefactos.joblib
dashboard/experimentos.py  # modelos del listado vs. GradientBoosting (DeLong)
dashboard/pruebas.py    # métricas de la literatura, pruebas de significancia (DeLong, bootstrap, McNemar, 5×2cv, Friedman-Nemenyi) y tiempos
dashboard/datos.py      # datos y artefactos compartidos
dashboard/componentes.py   # piezas de la interfaz
dashboard/sec_contexto.py, sec_eda.py, sec_modelos.py   # secciones del dashboard
dashboard/figuras.py    # figuras de Plotly
dashboard/app.py        # aplicación Dash
dashboard/assets/       # estilos e imagen
docker/, k8s/           # contenedor y despliegue en Kubernetes
```

## Cómo ejecutar (Windows, desde la raíz del proyecto)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m dashboard.entrenar       # ~15 min, una sola vez
.venv/Scripts/python -m dashboard.experimentos   # opcional (~10 min): modelos del listado vs. GradientBoosting
.venv/Scripts/python -m dashboard.pruebas        # opcional (~5 min): pruebas de significancia y tiempos de cómputo
.venv/Scripts/python -m dashboard.app            # http://localhost:8050
```

## Despliegue en el Kubernetes de Docker Desktop
Docker Desktop (modo *kind*) no comparte las imágenes locales con el clúster, así que se usa un
registro local dentro del clúster (`k8s/registry.yaml`), publicado en `localhost:5000`.
```bash
kubectl config use-context docker-desktop
kubectl apply -f k8s/registry.yaml
docker build -t localhost:5000/fraud-dashboard:3.0 -f docker/Dockerfile .
docker push localhost:5000/fraud-dashboard:3.0
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/fraud-dashboard   # luego abrir http://localhost:8051
```
Para retirar todo: `kubectl delete -f k8s/`.

## Despliegue en Render
El archivo `render.yaml` define un *Blueprint* que construye la imagen con `docker/Dockerfile` (plan gratuito;
el dashboard usa ≈ 350 MB de los 512 MB disponibles). El dataset se versiona en `data/` porque la sección EDA lo necesita.
1. Subir el proyecto a GitHub (`git push`).
2. En [Render](https://dashboard.render.com): **New → Blueprint** → elegir el repositorio `dash_ml_fraud_dsr` → **Apply**.
3. Esperar la construcción (≈ 5–10 min) y abrir `https://fraud-detection-dashboard.onrender.com`.

Con `autoDeploy: true`, cada `git push` a `main` vuelve a desplegar. En el plan gratuito el servicio se suspende tras
15 minutos sin visitas y la primera carga posterior tarda ≈ 1 minuto.
