"""
Pesos w(x) del random forest, proximidad y verificación (1.4, 1.5 y 1.6).

pesos_bosque devuelve W de forma (n_consultas, n_train), igual que las funciones
de pesos.py, así que W @ y_train reproduce el .predict() del bosque.

Autor: Herman Echeverría

Declaración de uso de IA: se utilizó Claude (Anthropic) como herramienta de apoyo
para comprender lo que pedía cada inciso del enunciado y como asistencia al escribir
y depurar el código. El análisis, la verificación de resultados y su interpretación
fueron revisados por el autor.
"""
import numpy as np


def pesos_bosque(modelo_rf, X_train, X_q):
    # w_i(x) = (1/T) Σ_t 1{x_i en la hoja de x en el árbol t} / (filas de entrenamiento en esa hoja)
    # Solo reproduce el .predict() con bootstrap=False: con bootstrap cada árbol promedió su hoja
    # con su muestra bootstrap, no con todo el entrenamiento.
    hojas_train = modelo_rf.apply(X_train)          # (n_train, T)
    hojas_q = modelo_rf.apply(X_q)                  # (n_q, T)
    T = hojas_train.shape[1]
    W = np.zeros((len(hojas_q), len(hojas_train)))
    for t in range(T):
        tamano = np.bincount(hojas_train[:, t])     # filas de entrenamiento en cada hoja
        misma = hojas_q[:, [t]] == hojas_train[None, :, t]
        W += misma / tamano[hojas_q[:, t]][:, None]
    return W / T


def proximidad(modelo_rf, X_train, X_q):
    # fracción de árboles en los que x_i y x caen en la misma hoja (sin dividir entre el tamaño)
    hojas_train = modelo_rf.apply(X_train)
    hojas_q = modelo_rf.apply(X_q)
    P = np.zeros((len(hojas_q), len(hojas_train)))
    for t in range(hojas_train.shape[1]):
        P += hojas_q[:, [t]] == hojas_train[None, :, t]
    return P / hojas_train.shape[1]


def verificar(w, y_train, pred, tol=1e-8):
    """assert |Σ w_i y_i − pred| < tol. Acepta un vector de pesos o una matriz W (una fila por consulta)."""
    error = np.abs(np.asarray(w) @ y_train - pred)
    assert np.all(error < tol), f"error máximo {error.max():.2e} >= {tol}"
    return error.max()
