# Planteamiento del problema

## Problema

El departamento del Atlántico combina una lluvia anual moderada con una distribución muy desigual: dos temporadas húmedas (abril–junio y agosto–noviembre) separadas por una temporada seca prolongada entre diciembre y marzo y un periodo más seco a mitad de año (veranillo). Para una finca de Sabanagrande, sembrar más área de la que el agua disponible puede sostener expone al cultivo a estrés hídrico, sobre todo en floración, y al agricultor a perder la inversión.

La decisión se toma antes de sembrar y abarca el ciclo completo del cultivo: cuántas hectáreas sembrar de cada cultivo, en qué fecha y cuánta agua reservar. A esa escala la lluvia diaria no es predecible, pero sí lo es parcialmente la tendencia de la temporada, gracias a la influencia de El Niño–Oscilación del Sur (ENSO).

## Pregunta de investigación

¿Con qué habilidad, frente a la climatología, pueden los modelos de Machine Learning pronosticar la lluvia semanal y la categoría de lluvia de la temporada en Sabanagrande a partir de variables agroclimáticas e índices oceánicos, y cómo se traduce ese pronóstico, con su incertidumbre, en el área máxima sembrable y el volumen de agua que la finca debe reservar para asegurar un ciclo de cultivo?

## Objetivos

**General.** Pronosticar con Machine Learning la lluvia semanal en Sabanagrande y usar ese pronóstico para estimar el número de hectáreas que la finca puede sostener con el agua disponible.

**Específicos.**

1. Documentar la base de datos de NASA POWER de la finca (1981–2025) y su calidad.
2. Caracterizar mediante un EDA, solo con las variables del Excel, la lluvia semanal: distribución, estacionalidad, dependencia temporal, relación con las demás variables y deriva.
3. Implementar líneas base y un modelo base de regresión para pronosticar la lluvia semanal, validado de forma cronológica.
4. Determinar el horizonte *n* hasta el cual el pronóstico supera a la climatología con 95 % de confianza.
5. Con el pronóstico y las constantes de la finca, calcular el número de hectáreas disponibles.

## Estructura del análisis

| Parte | Pregunta | Datos | Método | Resultado |
|---|---|---|---|---|
| 1. Pronóstico (ML) | ¿Cuánta lluvia caerá en la semana *h*? | Solo variables del Excel de NASA POWER | EDA y regresión (SVR lineal frente a Dummy y climatología) | Lluvia semanal con intervalo del 95 % |
| 2. Análisis final | ¿Cuántas hectáreas puede sostener la finca? | Pronóstico de la parte 1 y constantes de la finca | Balance de agua del cultivo y del reservorio (FAO-56) | Número de hectáreas disponibles |

Los factores de la finca (reservorio, hectáreas, riego, coeficientes del cultivo) no forman parte del EDA ni del modelo; se tratan como constantes y se listan en el capítulo *Factores tratados como constantes*.

## Variable objetivo

Para cada sitio *s* (en este entregable, solo la finca) y fecha de emisión *t* (un pronóstico por semana, los lunes), el objetivo es la lluvia acumulada en la semana *h* posterior:

$$y_h(s,t) = \sum_{d=7(h-1)+1}^{7h} P_{s,\,t+d}\qquad [\text{mm}]$$

Como 1 mm sobre 1 ha equivale a 10 m³, el mismo valor por hectárea es $10\,y_h$ m³/ha.

## Horizonte confiable *n*

$$SS_h = 1-\frac{RMSE_{\text{modelo},h}}{RMSE_{\text{climatología},h}}$$

Se usan dos criterios y se adopta el más conservador:

- **Prueba:** $n_{\text{prueba}} = \max\{h:\ \text{IC}_{95}(SS_h)_{\text{inf}} > 0\}$, con intervalos por bootstrap de bloques en 2021–2025.
- **Validación:** $n_{\text{val}}$ = mayor horizonte en el que el SVR supera a la climatología en al menos 4 de los 5 bloques de la validación cruzada temporal (1981–2020).
- $n = \min(n_{\text{prueba}},\ n_{\text{val}})$.

Para las semanas posteriores a *n* se usan los cuantiles climatológicos de la lluvia semanal.

## Diseño de validación

**En este entregable (un sitio):** partición cronológica. Entrenamiento 1981–2020, con purga de las filas cuya semana objetivo cae en 2021, y prueba 2021–2025, usada una sola vez al final. La validación cruzada es de ventana creciente con una separación de 7h + 3 días.

**Diseño regional (segundo entregable):** entrenamiento con los sitios regionales hasta 2020 y evaluación en la finca como sitio no visto en 2021–2025. Los datos de 2021–2025 de los sitios regionales no se usan, porque comparten los mismos eventos climáticos que la finca y producirían fuga. Al construir el dataset se excluye además la celda regional más cercana a la finca.

## Justificación del dataset

NASA POWER ofrece series diarias continuas desde 1981 de todas las variables necesarias para calcular la evapotranspiración FAO-56 y para caracterizar el estado del suelo, con acceso abierto y reproducible por API. Una grilla regional multiplica los ejemplos de entrenamiento y permite evaluar la generalización espacial; el índice ONI aporta la señal de ENSO, la principal fuente de predictibilidad estacional en Colombia.
