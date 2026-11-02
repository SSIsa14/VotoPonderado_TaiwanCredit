"""
Pesos w(x) de OLS, ridge, k-NN y árbol (1.1, 1.2 y 1.3).

Cada función devuelve W de forma (n_consultas, n_train): la fila j son los pesos
de la consulta j, así que W @ y_train reproduce el .predict() del modelo.

Autora: Sofía Isabela Sandoval Vásquez

Declaración de uso de IA: se utilizó Claude (Anthropic) como herramienta de apoyo
para comprender lo que pedía cada inciso del enunciado y como asistencia al escribir
y depurar el código. El análisis, la verificación de resultados y su interpretación
fueron revisados por la autora.
"""
import numpy as np


def _con_unos(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X])


def pesos_ols(X_train, X_q):
    # w(x)^T = x~^T (X~^T X~)^{-1} X~^T
    Xt, Xq = _con_unos(X_train), _con_unos(X_q)
    return np.linalg.solve(Xt.T @ Xt, Xq.T).T @ Xt.T


def pesos_ridge(X_train, X_q, lam):
    # w(x)^T = 1^T/n + (x - xbar)^T (Xc^T Xc + lam I)^{-1} Xc^T  (sin penalizar el intercepto)
    X_train, X_q = np.asarray(X_train, float), np.asarray(X_q, float)
    n, p = X_train.shape
    media = X_train.mean(axis=0)
    Xc = X_train - media
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), (X_q - media).T).T
    return 1 / n + B @ Xc.T


def pesos_ridge_penalizando_unos(X_train, X_q, lam):
    # 1.2, la versión "obvia": la columna de unos se penaliza junto con las demás
    Xt, Xq = _con_unos(X_train), _con_unos(X_q)
    return np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xq.T).T @ Xt.T


def pesos_knn(modelo_knn, X_q, n_train):
    # w_i = 1/k para los k vecinos de x, 0 para el resto
    vecinos = modelo_knn.kneighbors(X_q, return_distance=False)
    W = np.zeros((len(vecinos), n_train))
    np.put_along_axis(W, vecinos, 1 / vecinos.shape[1], axis=1)
    return W


def pesos_arbol(modelo_arbol, X_train, X_q):
    # w_i = 1/(filas en la hoja de x) si x_i cae en la misma hoja que x
    hoja_train = modelo_arbol.apply(X_train)
    hoja_q = modelo_arbol.apply(X_q)
    misma = hoja_q[:, None] == hoja_train[None, :]
    return misma / misma.sum(axis=1, keepdims=True)
