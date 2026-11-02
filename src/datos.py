"""
Carga y preparación de datos (0.1 y 0.4).

Autora: Sofía Isabela Sandoval Vásquez

Declaración de uso de IA: se utilizó Claude (Anthropic) como herramienta de apoyo
para comprender lo que pedía cada inciso del enunciado y como asistencia al escribir
y depurar el código. El análisis, la verificación de resultados y su interpretación
fueron revisados por la autora.
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _partir(X, y, escalar):
    # 0.1: 80/20 con random_state=0 y scaler ajustado solo con entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)
    # 0.4: desviación estándar antes de estandarizar (ddof=0, la misma que usa el scaler)
    sd = pd.Series(scaler.scale_, index=X.columns, name="sd")
    if escalar:
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train.to_numpy(float), y_test.to_numpy(float), sd


def cargar_taiwan(ruta, escalar=True):
    """X_train, X_test, y_train, y_test, sd. Con escalar=False devuelve X crudas (Parte 4)."""
    datos = pd.read_csv(ruta)
    return _partir(datos.drop(columns="default"), datos["default"], escalar)


def cargar_california(escalar=True):
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return _partir(X, y, escalar)
