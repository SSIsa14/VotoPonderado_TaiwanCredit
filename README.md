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
  flexibilidad.py                      trazas tr(S) de 3.1 y procedencia del aporte de Jorge
  optimismo.py                         simulación del optimismo (3.4)
notebooks/
  00_preparacion.ipynb                 0.1, 0.2 y 0.4
  01_pesos_ols_ridge_knn_arbol.ipynb   1.1, 1.2 y 1.3
  02_bosque_verificacion_proximidad.ipynb  1.4, 1.5 y 1.6
  03_validacion_cruzada.ipynb          λ, k, profundidad y min_samples_leaf con 5 pliegues
  04_evidencia_2_1_2_2.ipynb           2.1 a 2.4
  05_flexibilidad_3_1_3_3.ipynb        3.1, 3.2 y 3.3
  06_optimismo_oob_3_4_3_6.ipynb       3.4, 3.5 y 3.6
  07_escala_4_1_4_4.ipynb              Parte 4 (4.1 a 4.4)
resultados/
  hiperparametros.json                 λ, k, profundidad y min_samples_leaf (los sobrescribe la validación cruzada)
  sd_taiwan.csv                        desviación estándar de cada variable antes de estandarizar (0.4)
  consultas.json                       las tres filas de prueba de 2.1 (las usa 4.2)
  parte3_optimismo_simulado.csv        optimismo simulado contra 2σ²tr(S)/n (3.4)
  parte3_oob.csv                       error de entrenamiento, out-of-bag y prueba del bosque (3.6)
  parte3_4_aporte_memo.md              párrafo de 3.4–3.6 y la Parte 4 para el memo
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

## Aporte de Gregorio: 3.4–3.6 y Parte 4

- `06_optimismo_oob_3_4_3_6.ipynb` va después del 05, porque 3.5 lee `resultados/parte3_error_flexibilidad.csv`.
  Corre en menos de un minuto.
- `07_escala_4_1_4_4.ipynb` solo depende de los datos, `sd_taiwan.csv`, `consultas.json` y `hiperparametros.json`.
  Es la única parte que usa las variables crudas (`cargar_taiwan(..., escalar=False)`). Tarda unos 30 segundos.
- Figuras para las diapositivas: `parte3_optimismo_simulado.png`, `parte3_oob.png`, `parte4_knn_escala.png`,
  `parte4_kmeans_k.png` y `parte4_kmeans_grupos.png`.
- Se utilizó Claude (Anthropic) como apoyo; la declaración está al inicio de cada archivo.

## Aporte de Jorge: 2.3–2.4 y 3.1–3.3

La etiqueta de responsabilidad «Jorge» identifica este bloque, sin atribuir una
identidad ni una autoría Git. Se amplía `04_evidencia_2_1_2_2.ipynb` después de
2.2 y se agrega `05_flexibilidad_3_1_3_3.ipynb`; `src/flexibilidad.py` contiene
las trazas reutilizables y un registro específico de procedencia. Las funciones
compartidas de datos, pesos y evidencia se conservan.

Después de la selección de parámetros de 03, ejecutar **04 → 05**, cada uno en
un kernel nuevo y desde `notebooks/`. No hace falta repetir CV para este aporte.
04 reutiliza su bosque de 2.1 para los grupos de evidencia; 05 verifica matrices
de 2.000 filas y calcula las curvas MSE sobre las 24.000/6.000 filas compartidas.
Se necesitan varios minutos, según el equipo. No se crean matrices completas de
24.000 × 24.000 ni se reducen las rejillas o los 200 árboles para ahorrar tiempo.

El entorno usado para esta entrega está en `.venv/env/` (ignorado por Git);
se activa desde la raíz con `source .venv/env/bin/activate`. Las versiones
exactas están en la procedencia, sin cambiar `requirements.txt`.

**Protección de entradas:** las celdas anteriores de 04 escriben `consultas.json`
y figuras de Herman. La fuente completa se ejecutó en un kernel nuevo en
staging, pero el **04 entregado combina las celdas originales completas del
baseline con las celdas Jorge y sus salidas frescas**. Copiar directamente el
04 ejecutado no conserva esa combinación. En la ejecución registrada, el caso
mediano recalculado fue posición 1511 / índice 22904; la explicación original
describe posición 3615 / índice 591. Este ensamblaje conserva ambos antecedentes
y su distinción en la procedencia; **no corrige la sensibilidad preexistente de
la selección del ejemplo ni reescribe su narrativa**.

**Procedimiento reproducible:** desde la raíz del repositorio, con el entorno
activado, el bloque siguiente prepara staging, ejecuta **solo 04 y 05** en
procesos separados (un kernel nuevo por notebook), verifica fuentes y ensambla
04 antes de transferir una lista explícita. Si detecta cambios inesperados,
se detiene antes de escribir en el repositorio; revisar el mensaje, sin forzar
el reemplazo. No usar este bloque para una comprobación puramente documental,
pues las dos llamadas a nbconvert sí ejecutan los análisis.

```bash
python - <<'PY'
import copy, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path

raiz = Path.cwd().resolve()
base = "c4bc2305edcabc804026e25891984d9d90566871"
n04 = "notebooks/04_evidencia_2_1_2_2.ipynb"
n05 = "notebooks/05_flexibilidad_3_1_3_3.ipynb"
procedencia = "resultados/parte_jorge_procedencia.json"
artefactos = [
    "resultados/parte2_evidencia_prueba.csv", "resultados/parte2_quintiles.csv",
    "resultados/parte2_aporte_memo.md", "resultados/parte3_verificacion_trazas.csv",
    "resultados/parte3_error_flexibilidad.csv", procedencia,
    "figs/parte2_mae_quintiles.png", "figs/parte3_error_flexibilidad.png",
]
protegidos = ["data/taiwan_credit.csv", "resultados/hiperparametros.json",
              "resultados/consultas.json", "resultados/sd_taiwan.csv"]
protegidos += [str(p.relative_to(raiz)) for p in (raiz / "src").glob("*.py")]

def exigir(condicion, mensaje):
    if not condicion:
        raise RuntimeError(mensaje)

def huella(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def firma(celdas):
    ids = [c.get("id") for c in celdas]
    exigir(all(ids) and len(ids) == len(set(ids)), "IDs ausentes o repetidos")
    return [(c["id"], c["cell_type"],
             "".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in celdas]

baseline = json.loads(subprocess.check_output(
    ["git", "--no-optional-locks", "show", f"{base}:{n04}"], cwd=raiz))
locales = {p: json.loads((raiz / p).read_text(encoding="utf-8")) for p in (n04, n05)}
originales = baseline["cells"]
ids_originales = [c["id"] for c in originales]
cantidad = len(ids_originales)  # derivada de las identidades del baseline, no un 21 fijo
exigir(firma(locales[n04]["cells"])[:cantidad] == firma(originales),
       "04 local: IDs, tipos, fuentes u orden difieren del baseline")
exigir(locales[n04]["cells"][:cantidad] == originales,
       "04 local: salidas, contadores o metadatos originales cambiaron; revisar")
exigir({k: v for k, v in locales[n04].items() if k != "cells"} ==
       {k: v for k, v in baseline.items() if k != "cells"},
       "04 local: metadatos superiores o versión de formato inesperados")
anexos = locales[n04]["cells"][cantidad:]
exigir(anexos and all(c["id"].startswith("jorge-") for c in anexos),
       "04 local: anexo Jorge ausente o inesperado")
vigilados = list(dict.fromkeys([n04, n05] + artefactos + protegidos))
antes = {p: huella(raiz / p) for p in vigilados}
historica = json.loads((raiz / procedencia).read_text(encoding="utf-8"))
staging = Path(tempfile.mkdtemp(prefix="jorge-reproduccion-"))
print("Staging:", staging, flush=True)
for carpeta in ("src", "notebooks", "data", "resultados", "figs"):
    shutil.copytree(raiz / carpeta, staging / carpeta,
                    ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
# Generar procedencia nueva sin atribuirle los tiempos de la ejecución anterior.
(staging / procedencia).unlink()
for nombre in (n04, n05):
    subprocess.run([sys.executable, "-m", "nbconvert", "--execute", "--to", "notebook",
                    "--inplace", "--ExecutePreprocessor.timeout=7200",
                    "--ExecutePreprocessor.kernel_name=python3", Path(nombre).name],
                   cwd=staging / "notebooks", check=True)
ejecutados = {p: json.loads((staging / p).read_text(encoding="utf-8")) for p in (n04, n05)}
for p in (n04, n05):
    exigir(firma(ejecutados[p]["cells"]) == firma(locales[p]["cells"]),
           f"{p}: IDs, tipos, fuentes u orden cambiaron en staging")
    for c in ejecutados[p]["cells"]:
        if c["cell_type"] == "code":
            exigir(c.get("execution_count") is not None and
                   not any(o.get("output_type") == "error" for o in c.get("outputs", [])),
                   f"{p}: celda sin ejecutar o con error: {c['id']}")
exigir(firma(ejecutados[n04]["cells"])[:cantidad] == firma(originales),
       "04 ejecutado: prefijo original incompatible con el baseline")
ensamblado = copy.deepcopy(baseline)  # conserva TODOS los campos superiores del baseline
ensamblado["cells"] = copy.deepcopy(originales + ejecutados[n04]["cells"][cantidad:])
exigir(ensamblado["cells"][:cantidad] == originales, "Restauración original incompleta")
exigir(ensamblado["cells"][cantidad:] == ejecutados[n04]["cells"][cantidad:],
       "Las celdas Jorge o sus salidas fueron alteradas")
registro = json.loads((staging / procedencia).read_text(encoding="utf-8"))
registro["registro_anterior_al_reproducir"] = historica  # evidencia histórica íntegra
registro["ensamblaje_04"] = {"baseline": base, "ids_originales": ids_originales,
    "metadatos_superiores": "baseline; versiones reales de ejecución en la procedencia",
    "salidas": "originales del baseline y anexos Jorge ejecutados en kernel nuevo"}
(staging / procedencia).write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
# Preparar todas las transferencias y validar cambios concurrentes antes de escribir.
entregas = {p: (staging / p).read_bytes() for p in [n05] + artefactos}
entregas[n04] = (json.dumps(ensamblado, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
exigir(all(huella(raiz / p) == h for p, h in antes.items()),
       "Cambios concurrentes en entradas o destinos: no se transfiere nada")
for p, contenido in entregas.items():
    (raiz / p).write_bytes(contenido)
exigir(all(huella(raiz / p) == antes[p] for p in protegidos), "Entrada protegida alterada")
print("Transferidos 04 ensamblado, 05 ejecutado y los ocho artefactos indicados.")
PY
```

04 conserva también los metadatos superiores y la versión de formato del
baseline; su `language_info.version` histórico (3.13.3) no describe el kernel
real de la ejecución registrada (Python 3.12.14, en procedencia). 05 conserva
los metadatos de su ejecución. Al repetir el procedimiento, el JSON anterior
queda íntegro en `registro_anterior_al_reproducir` y `ensamblaje_04` identifica
la política aplicada; los tiempos y hashes históricos no se presentan como
mediciones de la nueva ejecución. No se transfieren `consultas.json`, otros
archivos compartidos ni las figuras anteriores de Herman.

| Archivo | Contenido / destinatario |
|---|---|
| `resultados/parte2_evidencia_prueba.csv` | Cada posición de test, índice original, respuesta, predicción, evidencia, concentración, error y grupo |
| `resultados/parte2_quintiles.csv` | Límites, tamaños, MAE, mora y predicciones; descomposición por etiqueta |
| `resultados/parte2_aporte_memo.md` | Párrafo para Sofía, condicionado a los hallazgos de 2.1–2.3 |
| `resultados/parte3_verificacion_trazas.csv` | Cinco verificaciones completas, configuración, rango y discrepancias |
| `resultados/parte3_error_flexibilidad.csv` | 50 ajustes, configuraciones, tamaños, trazas y MSE; entrega para 3.5 de Gregorio |
| `resultados/parte_jorge_procedencia.json` | Hashes, versiones, semillas, filas ordenadas de la submuestra, métricas, tiempos y memoria |
| `figs/parte2_mae_quintiles.png` | MAE del bosque por grupo de evidencia |
| `figs/parte3_error_flexibilidad.png` | Cinco familias, entrenamiento/prueba y mínimos descriptivos |

Los mínimos de prueba no reemplazan los parámetros de CV. La derivación de 3.3
supone respuestas nuevas en las mismas covariables y un suavizador independiente
del ruido observado; no se aplica sin más a árboles adaptativos. La comparación
con la simulación de 3.000 filas de Gregorio es conceptual, no una igualdad de
trazas o gaps. No se implementan sus simulaciones, OOB ni escala, ni se integra
el conjunto de diapositivas del grupo.

**Asistencia de IA:** OpenAI Codex apoyó este aporte (código, explicaciones y
validación); el integrante debe revisarlo y poder defenderlo. Esta declaración
no modifica las atribuciones de los bloques anteriores ni declara terminado
todo el proyecto.
