# Todo modelo es un voto ponderado – Grupo Taiwan

Parte grupal del proyecto final de Análisis y Ciencia de Datos, Universidad del Istmo. Se sacan a mano los pesos
w<sub>i</sub>(x) de OLS, ridge, k-NN, árbol y random forest, se comprueba que reproducen su `.predict()`, se mide con
ellos cuánta evidencia respalda cada predicción y cuánto cuesta la flexibilidad, y se estudia cómo la escala cambia la geometría.

**Integrantes:** Sofía Isabela Sandoval Vásquez · Herman Echeverría · Jorge Ferguson · Gregorio Coronel

Datos: California Housing de scikit-learn (Parte 1) y `data/taiwan_credit.csv` (Partes 2 a 5).

## Quién hizo qué

| Integrante | Partes |
|---|---|
| Sofía Sandoval | Preparación (0.1, 0.2, 0.4), pesos de OLS, ridge, k-NN y árbol (1.1–1.3), memo (Parte 5), integración del repositorio |
| Herman Echeverría | Pesos del bosque, verificación y proximidad (1.4–1.6), validación cruzada, 2.1–2.2 |
| Jorge Ferguson | 2.3–2.4, grados de libertad efectivos (3.1–3.3), diapositivas |
| Gregorio Coronel | Optimismo y out-of-bag (3.4–3.6), la escala (Parte 4) |

## Estructura

```
data/
  taiwan_credit.csv                    datos de Taiwan Credit
src/
  datos.py                             partición 80/20, estandarización y desviaciones estándar
  pesos.py                             pesos w(x) de OLS, ridge, k-NN y árbol
  bosque.py                            pesos w(x) del bosque, proximidad y verificar
  evidencia.py                         N_ef, concentración c(x) y testigos
notebooks/
  00_preparacion.ipynb                 0.1, 0.2 y 0.4
  01_pesos_ols_ridge_knn_arbol.ipynb   1.1, 1.2 y 1.3
  02_bosque_verificacion_proximidad.ipynb  1.4, 1.5 y 1.6
  03_validacion_cruzada.ipynb          λ, k, profundidad y min_samples_leaf con 5 pliegues
  04_evidencia_2_1_2_2.ipynb           2.1 y 2.2
resultados/
  hiperparametros.json                 λ, k, profundidad y min_samples_leaf (los sobrescribe la validación cruzada)
  sd_taiwan.csv                        desviación estándar de cada variable antes de estandarizar (0.4)
  consultas.json                       las tres filas de prueba de 2.1 (las usa 4.2)
figs/                                  figuras parteN_descripcion.png
slides/                                presentación en PDF
```

## Cómo correrlo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Ejecutar los notebooks en orden desde la carpeta `notebooks/`. Todos parten de los archivos de `data/`; California
se descarga con `fetch_california_housing()`.

## Notas

- Los pesos se calculan a mano con numpy; cada notebook comprueba con `assert` que reproducen el `.predict()` de scikit-learn.
- Se utilizaron herramientas de IA como apoyo; el detalle se declara al inicio de cada archivo.
