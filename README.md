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
  taiwan_credit.csv                        datos de Taiwan Credit
src/
  datos.py                                 partición 80/20, estandarización y desviaciones estándar
  pesos.py                                 pesos w(x) de OLS, ridge, k-NN y árbol
  bosque.py                                pesos w(x) del bosque, proximidad y verificar
  evidencia.py                             N_ef, concentración c(x) y testigos
  flexibilidad.py                          trazas tr(S) de 3.1 y 3.2
  optimismo.py                             simulación del optimismo (3.4)
notebooks/
  00_preparacion.ipynb                     0.1, 0.2 y 0.4
  01_pesos_ols_ridge_knn_arbol.ipynb       1.1, 1.2 y 1.3
  02_bosque_verificacion_proximidad.ipynb  1.4, 1.5 y 1.6
  03_validacion_cruzada.ipynb              λ, k, profundidad y min_samples_leaf con 5 pliegues
  04_evidencia_2_1_2_2.ipynb               2.1 a 2.4
  05_flexibilidad_3_1_3_3.ipynb            3.1, 3.2 y 3.3
  06_optimismo_oob_3_4_3_6.ipynb           3.4, 3.5 y 3.6
  07_escala_4_1_4_4.ipynb                  Parte 4 (4.1 a 4.4)
memo_parte5.md                             memo para el comité de crédito (Parte 5)
resultados/                                hiperparámetros, consultas de 2.1, desviaciones estándar y tablas generadas
figs/                                      figuras generadas por los notebooks
slides/                                    diapositivas de la presentación (PDF)
```

## Cómo correrlo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Ejecutar los notebooks en orden, de `00` a `07`, desde la carpeta `notebooks/`. Todos parten de `data/`; California
se descarga con `fetch_california_housing()`. El `03` escoge los hiperparámetros y los guarda en
`resultados/hiperparametros.json`, y el `04` guarda las tres consultas de 2.1 en `resultados/consultas.json`; los
notebooks siguientes leen esos archivos. El `06` va después del `05` porque usa sus curvas de error. Todo corre en
unos 15 minutos; los más lentos son el `03`, el `04` y el `05`.

## Notas

- Los pesos se calculan a mano con numpy; cada notebook comprueba con `assert` que reproducen el `.predict()` de scikit-learn.
- Se utilizaron herramientas de IA como apoyo (Claude de Anthropic y OpenAI Codex); el detalle se declara al inicio de cada archivo.
