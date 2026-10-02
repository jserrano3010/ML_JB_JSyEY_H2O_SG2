# Factores tratados como constantes

El EDA y el modelo de regresión se hicieron **solo con los datos del Excel inicial** (NASA POWER, celda de Sabanagrande, 1981–2025). El Machine Learning se usa únicamente para estimar el agua que aporta la naturaleza: la lluvia semanal.

Los demás factores necesarios para calcular las hectáreas disponibles **no están en el Excel**. No se analizan ni se modelan: se ingresan como **constantes** en `src/config.py` y se usan solo en el análisis final. Los valores marcados como *ejemplo* deben reemplazarse por los de la finca.

| Factor | Símbolo | Unidad | ¿Está en el Excel? | Tratamiento en el proyecto | Valor usado | Fuente o cómo obtenerlo |
|---|---|---|---|---|---|---|
| Lluvia | $P$ | mm/día | Sí | Variable objetivo: EDA y regresión con ML | Observada y pronosticada | NASA POWER |
| Temperatura, humedad, viento, radiación, presión, humedad del suelo | — | varias | Sí | EDA y predictoras del modelo | Observadas | NASA POWER |
| Evapotranspiración de referencia | $ET_0$ | mm/día | No (se deriva) | Construcción matemática (FAO-56) calculada solo en el análisis final | Calculada | Allen et al. (1998) |
| Hectáreas totales de la finca | $H$ | ha | No | Constante | 20 (*ejemplo*) | Registro de la finca |
| Capacidad del reservorio | $J$ | m³ | No | Constante | 20.000 (*ejemplo*) | Medición de jagüeyes y tanques |
| Nivel del reservorio al sembrar | $S_0$ | m³ | No | Constante: lleno ($S_0 = J$) | 20.000 | Supuesto |
| Área que escurre al reservorio | $A_c$ | ha | No | Constante | 0,5 (*ejemplo*) | Medición en la finca |
| Coeficiente de escorrentía | $C$ | 0–1 | No | Constante | 1,0 | Lluvia sobre la lámina del reservorio |
| Aporte de otras fuentes (río, pozo) | $Q_{ext}$ | m³/semana | No | Constante | 0 | Concesión o bombeo disponible |
| Eficiencia del riego | $\eta$ | 0–1 | No | Constante | 0,90 (goteo) | Sistema de riego de la finca |
| Lluvia efectiva | $\alpha$, $P_{min}$ | —, mm | No | Constantes | 0,8 y 5 mm | Supuesto de referencia |
| Coeficiente del cultivo de referencia | $K_c$ | — | No | Constante por etapa | 0,30 / 1,20 / 0,35 (maíz) | FAO-56, tabla 12 |
| Duración de las etapas del cultivo | $L$ | días | No | Constante | 20 / 35 / 40 / 30 | FAO-56, tabla 11 |
| Tipos de cultivo y hectáreas de cada uno | $x_i$ | ha | No | **Fuera del alcance**: no se determina | — | — |
| Beneficio por hectárea | $b$ | — | No | **Fuera del alcance**: no se optimiza la mezcla de cultivos | — | — |

**Resultado que se calcula con estas constantes:** el número de hectáreas disponibles para el cultivo de referencia, en un escenario esperado y en uno conservador (95 %), a partir del pronóstico de lluvia del modelo (capítulo siguiente).
