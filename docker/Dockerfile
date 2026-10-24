# Imagen del dashboard de detección de fraude (Dash + Plotly), lista para Kubernetes
FROM python:3.14-slim

# Variables de entorno: sin archivos .pyc, salida de logs sin búfer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor (raíz del proyecto)
WORKDIR /app

# Instala primero las dependencias para aprovechar la caché de capas de Docker
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código, los artefactos ya entrenados y el dataset que usa la sección EDA
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY data/credit_card_fraud_detection_2026.csv ./data/

# Ejecuta el contenedor con un usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home appuser
USER appuser

# Puerto en el que escucha el dashboard
EXPOSE 8050

# Gunicorn sirve la app Flask interna de Dash (dashboard.app:server): 1 proceso con 4 hilos
# (un solo proceso evita cargar el dataset y los modelos varias veces en memoria).
# El puerto se toma de la variable PORT (Render la asigna); si no existe (Docker local, Kubernetes) se usa 8050.
CMD ["sh", "-c", "exec gunicorn dashboard.app:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 4 --timeout 120"]
