"""Trazas y procedencia de 2.3–2.4 y 3.1–3.3 (responsabilidad: Jorge).

Asistencia de IA: OpenAI Codex apoyó la implementación, las explicaciones y
la validación numérica. El integrante debe revisar y poder defender su aporte.
No modifica las funciones compartidas de datos, pesos ni evidencia.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


FAMILIAS = ("OLS", "Ridge", "k-NN", "Árbol", "Bosque")
SEMILLA = 0


def crear_modelo(familia, hp, parametro=None, valor=None):
    """Configuración del proyecto; solo se reemplaza el parámetro del barrido."""
    modelos = {
        "OLS": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=hp["lambda"]),
        "k-NN": lambda: KNeighborsRegressor(n_neighbors=hp["k"]),
        "Árbol": lambda: DecisionTreeRegressor(max_depth=hp["max_depth"], random_state=SEMILLA),
        "Bosque": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=hp["min_samples_leaf"],
            max_features=0.33, bootstrap=False, random_state=SEMILLA, n_jobs=-1),
    }
    modelo = modelos[familia]()
    return modelo if parametro is None else modelo.set_params(**{parametro: valor})


def diagnostico_diseno(X):
    """Rango con tolerancia SVD y condición del diseño con intercepto."""
    A = np.column_stack((np.ones(len(X)), np.asarray(X, dtype=float)))
    d = np.linalg.svd(A, compute_uv=False)
    tol = d[0] * max(A.shape) * np.finfo(float).eps
    return {"rango_aumentado": int(np.sum(d > tol)),
            "columnas_aumentadas": A.shape[1], "tolerancia_rango": float(tol),
            "condicion_diseno": float(d[0] / d[-1]),
            "condicion_gram": float(np.linalg.cond(A.T @ A))}


def espectro_centrado(X):
    X = np.asarray(X, dtype=float)
    return np.linalg.svd(X - X.mean(axis=0), compute_uv=False)


def traza_ridge(d, lam):
    if lam <= 0:
        raise ValueError("Esta fórmula se evalúa con lambda estrictamente positiva.")
    return float(1 + np.sum(d ** 2 / (d ** 2 + lam)))


def diagonal_knn(modelo, X, lote=500, diagnosticar_empates=False):
    """Diagonal realizada con consultas explícitas, sin truncar otra búsqueda.

    X debe ser exactamente la matriz de ajuste, en su orden original. La llamada
    sin X a kneighbors excluye el propio punto y NO representa este suavizador.
    Una búsqueda k+1 opcional solo diagnostica empates; no define la diagonal.
    """
    if modelo.weights != "uniform":
        raise ValueError("La traza implementada corresponde a pesos uniformes.")
    n, k = len(X), modelo.n_neighbors
    diagonal = np.empty(n)
    empates = 0
    for inicio in range(0, n, lote):
        q = X.iloc[inicio:inicio + lote] if hasattr(X, "iloc") else X[inicio:inicio + lote]
        vecinos = modelo.kneighbors(q, return_distance=False)
        propios = np.arange(inicio, inicio + len(q))[:, None]
        diagonal[inicio:inicio + len(q)] = (vecinos == propios).any(axis=1) / k
        if diagnosticar_empates and k < n:
            distancias, _ = modelo.kneighbors(q, n_neighbors=k + 1)
            empates += int(np.isclose(distancias[:, k - 1], distancias[:, k],
                                     rtol=0, atol=1e-12).sum())
    return diagonal, {"auto_incluidos": int(np.count_nonzero(diagonal)),
                      "auto_excluidos": int(np.count_nonzero(diagonal == 0)),
                      "traza_n_sobre_k": float(n / k),
                      "empates_frontera_atol_1e_12": empates if diagnosticar_empates else None}


def diagonal_arbol(modelo, X):
    """S_ii=1/n_hoja para un árbol sin pesos de muestra."""
    hojas = modelo.apply(X)
    tamanos = np.bincount(hojas)
    return 1.0 / tamanos[hojas]


def diagonal_bosque(modelo, X):
    """Promedio de diagonales por árbol; no construye una matriz n×n."""
    if modelo.bootstrap:
        raise ValueError("La fórmula requiere bootstrap=False y todas las filas de ajuste.")
    hojas = modelo.apply(X)
    diagonal = np.zeros(len(X))
    # Control independiente: metadatos de los estimadores, no la suma diagonal.
    conteos = [arbol.get_n_leaves() for arbol in modelo.estimators_]
    for t in range(hojas.shape[1]):
        tamanos = np.bincount(hojas[:, t])
        diagonal += 1.0 / tamanos[hojas[:, t]]
        assert np.count_nonzero(tamanos) == conteos[t], "X debe contener todas las filas de ajuste."
    diagonal /= hojas.shape[1]
    return diagonal, float(np.mean(conteos))


def hash_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _version(paquete):
    # el metapaquete jupyter no siempre está instalado aunque sí sus componentes
    try:
        return importlib.metadata.version(paquete)
    except importlib.metadata.PackageNotFoundError:
        return "no instalado"


def registrar_procedencia(seccion, detalles, artefactos, raiz=Path("..")):
    """Registro pequeño y específico del aporte; agrega una sección por notebook."""
    raiz = Path(raiz)
    ruta = raiz / "resultados/parte_jorge_procedencia.json"
    registro = json.loads(ruta.read_text()) if ruta.exists() else {}
    protegidos = ["data/taiwan_credit.csv", "resultados/hiperparametros.json",
                  "resultados/consultas.json", "resultados/sd_taiwan.csv"]
    registro.update({
        "responsabilidad": "Jorge: 2.3, 2.4, 3.1, 3.2, 3.3",
        "asistencia_IA": "OpenAI Codex; requiere revisión y comprensión del integrante",
        "python": platform.python_version(),
        "paquetes": {p: _version(p) for p in
                     ("numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "jupyter", "nbclient", "nbformat")},
        "entradas_sha256": {p: hash_archivo(raiz / p) for p in protegidos[:2]},
        "auxiliares_entorno_ejecucion_sha256": {p: hash_archivo(raiz / p) for p in protegidos[2:]},
        "hiperparametros": json.loads((raiz / protegidos[1]).read_text()),
        "semilla_particion_y_estimadores": SEMILLA,
        "afinidad_cpu": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "hilos": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    fuentes = ["src/datos.py", "src/pesos.py", "src/bosque.py", "src/evidencia.py", "src/flexibilidad.py"]
    registro["fuentes_sha256"] = {p: hash_archivo(raiz / p) for p in fuentes}
    registro.setdefault("secciones", {})[seccion] = {
        **detalles,
        "rss_maximo_proceso_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "artefactos_sha256": {p: hash_archivo(raiz / p) for p in artefactos},
    }
    for nombre in ("04_evidencia_2_1_2_2.ipynb", "05_flexibilidad_3_1_3_3.ipynb"):
        archivo = raiz / "notebooks" / nombre
        if archivo.exists():
            notebook = json.loads(archivo.read_text())
            celdas = [{"id": c["id"], "cell_type": c["cell_type"], "source": c["source"]}
                      for c in notebook["cells"]]
            registro.setdefault("notebooks_fuente_sha256", {})[nombre] = hashlib.sha256(
                json.dumps(celdas, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n")
    return registro["secciones"][seccion]
