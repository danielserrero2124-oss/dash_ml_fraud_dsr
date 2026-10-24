"""Configuración central del proyecto: rutas, columnas, semilla y carga de datos.

Todas las demás piezas (entrenamiento, dashboard) importan de aquí, de modo que los nombres
de columnas y la división temporal se definen en un único lugar.
"""

from pathlib import Path  # Manejo de rutas de archivos independiente del sistema operativo

import pandas as pd  # Lectura y manipulación de tablas de datos

RAIZ = Path(__file__).resolve().parent.parent  # Carpeta raíz del proyecto (un nivel por encima de src/)
RUTA_DATOS = RAIZ / "data" / "credit_card_fraud_detection_2026.csv"  # Ubicación del dataset (no se versiona en git)
RUTA_ARTEFACTOS = RAIZ / "dashboard" / "artefactos.joblib"  # Archivo con modelos y métricas ya calculados para el dashboard
SEMILLA = 42  # Semilla aleatoria común para que todos los resultados sean reproducibles
FECHA_CORTE = "2026-11-01"  # Inicio del periodo de prueba: se entrena con ene–oct y se evalúa con nov–dic

NUM_FEATURES = [  # Variables numéricas predictoras (lista exacta del enunciado)
    "transaction_amount", "transaction_hour", "card_present", "customer_age",  # Monto, hora, tarjeta física y edad
    "account_age_days", "credit_limit", "credit_score", "num_cards_held",  # Antigüedad de la cuenta, cupo, puntaje y número de tarjetas
    "transactions_last_1h", "transactions_last_24h", "transactions_last_7d",  # Velocidad: transacciones recientes
    "amount_last_1h", "amount_last_24h", "days_since_last_transaction",  # Montos recientes y días desde la última compra
    "distance_from_home_km", "distance_from_last_tx_km",  # Distancias geográficas
    "international_transaction", "high_risk_country", "amount_vs_avg_ratio",  # Banderas de riesgo y monto relativo
    "used_chip", "used_pin", "is_weekend", "avg_transaction_amount_30d",  # Chip, PIN, fin de semana y gasto promedio
    "std_transaction_amount_30d", "num_declined_last_30d", "num_chargebacks_ever",  # Variabilidad, rechazos y contracargos
]
CAT_FEATURES = [  # Variables categóricas predictoras (lista exacta del enunciado)
    "merchant_category", "merchant_country", "transaction_day_of_week",  # Rubro y país del comercio, día de la semana
    "customer_gender", "customer_state", "transaction_channel",  # Género y estado del cliente, canal de la transacción
    "device_type", "browser", "email_domain",  # Dispositivo, navegador y dominio del correo
]
FEATURES = NUM_FEATURES + CAT_FEATURES  # Las 35 variables que reciben los modelos
NUM_DISCRETAS = [  # Numéricas con pocos valores distintos (binarias, conteos pequeños y la hora): se codifican en one-hot
    "card_present", "num_cards_held", "transactions_last_1h", "international_transaction", "high_risk_country",
    "used_chip", "used_pin", "is_weekend", "num_declined_last_30d", "num_chargebacks_ever", "transaction_hour",
]
NUM_CONTINUAS = [c for c in NUM_FEATURES if c not in NUM_DISCRETAS]  # Numéricas continuas: se expanden con splines cúbicos
TARGET = "is_fraud"  # Variable objetivo: 1 = fraude, 0 = legítima
COLUMNAS_FUGA = ["fraud_type", "fraud_amount_lost"]  # Columnas con fuga de datos: solo se conocen después del fraude
COLUMNAS_ID = ["transaction_id", "customer_id", "card_id", "merchant_id"]  # Identificadores: no generalizan, se excluyen


def cargar_datos():  # Lee el CSV y aplica la limpieza mínima que no depende de los datos de entrenamiento
    """Devuelve el DataFrame completo ordenado en el tiempo, con la fecha como datetime."""
    df = pd.read_csv(RUTA_DATOS)  # Carga las 100 000 transacciones con sus 43 columnas
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])  # Convierte el texto de fecha a tipo fecha-hora
    df = df.sort_values("transaction_datetime").reset_index(drop=True)  # Ordena cronológicamente y renumera las filas
    df["browser"] = df["browser"].fillna("None")  # Los nulos de browser son compras sin navegador (tienda/cajero): categoría propia, no moda
    return df  # Devuelve la tabla lista para el análisis y la partición


def division_temporal(df):  # Separa pasado (entrenamiento) y futuro (prueba) ANTES de cualquier transformación
    """Devuelve (train_df, test_df): enero–octubre para entrenar y noviembre–diciembre para evaluar."""
    es_train = df["transaction_datetime"] < FECHA_CORTE  # Máscara booleana: True si la transacción ocurrió antes del 1 de noviembre
    return df[es_train].copy(), df[~es_train].copy()  # Copias independientes para no modificar el DataFrame original
