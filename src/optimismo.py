"""
Simulación del optimismo del error de entrenamiento (3.4).

Autor: Gregorio Coronel

Declaración de uso de IA: se utilizó Claude (Anthropic) como herramienta de apoyo
para comprender lo que pedía cada inciso del enunciado y como asistencia al escribir
y depurar el código. El análisis, la verificación de resultados y su interpretación
fueron revisados por el autor.
"""
import numpy as np


def simular_optimismo(modelo, X, f, sigma, replicas=50):
    """Optimismo de cada réplica y número de hojas por réplica (vacío si no es un árbol).

    En la réplica b (semilla b): y = f + σε, y_nueva = f + σε'. Se ajusta con y y el optimismo es
    el error con y_nueva en los mismos x menos el error de entrenamiento.
    """
    n = len(f)
    optimismo = np.empty(replicas)
    hojas = []
    for b in range(replicas):
        rng = np.random.default_rng(b)
        y = f + sigma * rng.standard_normal(n)
        y_nueva = f + sigma * rng.standard_normal(n)
        pred = modelo.fit(X, y).predict(X)
        optimismo[b] = np.mean((y_nueva - pred) ** 2) - np.mean((y - pred) ** 2)
        if hasattr(modelo, "get_n_leaves"):
            # en un árbol S cambia con cada y: se guarda tr(S) = número de hojas de cada réplica
            hojas.append(modelo.get_n_leaves())
    return optimismo, np.array(hojas)
