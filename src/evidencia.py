"""
Ficha de evidencia de una predicción (Parte 2): tamaño efectivo de muestra de Kish,
concentración y testigos.

Todas aceptan W de forma (n_consultas, n_train) o un solo vector de pesos.

Autor: Herman Echeverría

Declaración de uso de IA: se utilizó Claude (Anthropic) como herramienta de apoyo
para comprender lo que pedía cada inciso del enunciado y como asistencia al escribir
y depurar el código. El análisis, la verificación de resultados y su interpretación
fueron revisados por el autor.
"""
import numpy as np


def n_efectivo(W):
    # N_ef(x) = (Σ_i w_i)² / Σ_i w_i²; si todos los pesos son 1/m, N_ef = m
    W = np.atleast_2d(W)
    return W.sum(axis=1) ** 2 / (W ** 2).sum(axis=1)


def concentracion(W):
    # c(x) = max_i w_i² / Σ_j w_j²: si es grande, una sola fila manda
    W2 = np.atleast_2d(W) ** 2
    return W2.max(axis=1) / W2.sum(axis=1)


def testigos(w, k=5):
    """Índices (posiciones en el entrenamiento) de las k filas con mayor |w_i|, de mayor a menor."""
    w = np.asarray(w)
    idx = np.argpartition(-np.abs(w), k)[:k]
    return idx[np.argsort(-np.abs(w[idx]))]
