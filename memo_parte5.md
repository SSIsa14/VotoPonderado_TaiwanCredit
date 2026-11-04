# Memo: modelo de probabilidad de mora

**Para:** Comité de crédito
**De:** Grupo Taiwan (Sofía Sandoval, Herman Echeverría, Jorge Ferguson, Gregorio Coronel)
**Asunto:** qué modelo usar para estimar la probabilidad de mora, cuándo no confiar en él y qué lo cambia más

Comparamos cinco modelos sobre 30 000 clientes de tarjeta de crédito (24 000 para ajustar y 6 000 para evaluar):
regresión lineal (OLS), ridge, k vecinos más cercanos (k-NN), un árbol de decisión y un bosque aleatorio. Los cinco
calculan la probabilidad de mora de un cliente como un promedio ponderado de lo que pasó con clientes anteriores, así
que para cada predicción podemos ver qué clientes la respaldan y con cuánto peso.

## 1. Recomendamos el bosque aleatorio

**Es el que menos se equivoca.** Su error cuadrático medio en los 6 000 clientes de evaluación es 0.130, contra 0.134
del árbol, 0.139 de k-NN y 0.148 de OLS y ridge. Asignarle a todos la tasa de mora promedio daría 0.169.

**Sus cifras siempre son probabilidades.** El bosque, el árbol y k-NN promedian clientes reales con pesos positivos
que suman 1, así que su resultado siempre queda entre 0 y 1 y se lee como "de los clientes parecidos a este, esta
fracción cayó en mora". OLS y ridge no: le dan peso negativo a más de un tercio de los clientes históricos y en 436 de
los 6 000 casos de evaluación entregan cifras menores que 0 o mayores que 1, que no pueden presentarse como probabilidades.

**Cada predicción tiene respaldo y se puede explicar.** Una predicción típica del bosque se apoya en el equivalente
de unos 750 clientes (entre 128 y 2 371 según el caso) y ningún cliente individual pesa más de una pequeña fracción
del total. Para justificar una decisión se pueden mostrar los clientes con más peso: en el caso típico que revisamos,
un cliente que paga a tiempo y con un límite de 160 000, sus clientes de referencia también pagan a tiempo y tienen
límites parecidos, y el bosque estima 6 % de mora. En cambio, el cliente con más peso en OLS para ese mismo caso tiene
peso negativo y no se le parece en nada.

El árbol es más fácil de explicar con una sola regla, pero es poco específico: al mismo cliente lo mete en un grupo de
casi 11 800 clientes, prácticamente todos los que pagan a tiempo. El bosque tiene además una ventaja práctica: su
error "out-of-bag" (0.135, contra 0.130 en evaluación) permite vigilar su desempeño sin apartar clientes.

## 2. Cuándo no confiar en una predicción

Recomendamos que cada probabilidad venga con una ficha de respaldo y que el caso pase a revisión manual cuando:

- **Se apoya en pocos clientes.** Entre la quinta parte de los casos con menos respaldo (menos de unos 430 clientes
  equivalentes), el error absoluto medio es 0.396; en la quinta parte con más respaldo es 0.163. No todo se debe a la
  falta de respaldo: los casos con poco respaldo también son los de mayor riesgo (42.6 % de mora, contra 8.6 %), que
  son más difíciles de predecir. Por eso tomamos el respaldo bajo como una señal de alerta y no como una regla automática.
- **Unos pocos clientes concentran el peso**, o los clientes con más peso no se parecen al solicitante en lo que más
  pesa para la mora: los meses de atraso en los pagos recientes.
- **Los modelos no se ponen de acuerdo.** En el caso más extremo que encontramos, las estimaciones de los cinco modelos
  difieren en 0.75. Una diferencia así indica que el cliente es atípico y ninguna cifra sola basta.

Estas señales se calculan junto con la predicción, así que se pueden mostrar en el expediente sin trabajo adicional.
Los umbrales deben fijarse con los datos del propio banco y revisarse con el tiempo; el respaldo no es un intervalo de
confianza, y que un cliente pague o caiga en mora no demuestra por sí solo que su probabilidad estaba mal.

## 3. Qué decisión de modelado cambia más la respuesta

**La escala de las variables.** En los métodos basados en "clientes parecidos", la unidad de cada variable decide quién
se parece a quién. En las unidades originales, el límite de crédito y los montos de las facturas, que se miden en miles,
dominan por completo la comparación, y los meses de atraso, que son lo que mejor predice la mora, no cuentan. Así,
k-NN queda peor que asignarle a todos la tasa promedio (0.1706 contra 0.1695), mientras que con las variables
estandarizadas baja a 0.147. Para el cliente típico, los 10 clientes más parecidos en una escala y en la otra no
tienen ninguno en común. Los árboles y el bosque no tienen este problema, porque solo dependen del orden de los valores.
Si se usa un método basado en distancias, la escala es parte del modelo y debe decidirse y documentarse.

**La flexibilidad y cómo se evalúa.** Un modelo muy flexible se ve mucho mejor en los clientes con los que se ajustó
que en clientes nuevos: un bosque con hojas de un solo cliente tiene un error de 0.019 en sus propios datos y de 0.135
en clientes nuevos. Por eso nunca hay que evaluar un modelo con los mismos datos con que se ajustó. Los parámetros que
recomendamos (hojas de al menos 50 clientes) se escogieron con validación cruzada sobre los datos de ajuste, y el
error en los clientes de evaluación o el out-of-bag son las cifras que deben reportarse.

**La partición.** Todas las cifras de este memo salen de una sola división de los datos y de clientes de un mismo
periodo. Antes de usar el modelo en producción conviene confirmarlas con clientes más recientes.

---

*Declaración de uso de IA: se utilizaron Claude (Anthropic) y OpenAI Codex como herramientas de apoyo para el código y
la redacción. El análisis y las conclusiones fueron revisados por el grupo.*
