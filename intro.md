# Pronóstico de agua y planificación de siembras bajo sequía en Sabanagrande, Atlántico

**Edwin Yunis y Jairo Serrano** — Maestría en Ingeniería Industrial
Curso de Machine Learning — Profesor: Dr. Lihki Rubio
Primer entregable: base de datos, análisis exploratorio y modelo base

## Resumen

En el Atlántico la lluvia se concentra en dos temporadas y el resto del año casi no llueve. Antes de sembrar, el agricultor necesita saber cuántas hectáreas puede sostener con el agua disponible.

El proyecto tiene dos partes separadas:

1. **Machine Learning: regresión de la lluvia semanal.** Con los datos del Excel de NASA POWER (1981–2025) se hace el análisis exploratorio y se entrena un modelo que pronostica la lluvia de las próximas semanas, es decir, el agua que aporta la naturaleza. El EDA y el modelo usan **solo** esas variables observadas.
2. **Análisis final: hectáreas disponibles.** Los factores de la finca que no están en el Excel (capacidad del reservorio, hectáreas totales, eficiencia del riego, coeficiente de agua del cultivo, entre otros) se tratan como **constantes** y se listan en un cuadro. Con ellas y con el pronóstico de lluvia se calcula el número de hectáreas disponibles. No se decide qué cultivos sembrar ni cuánto de cada uno.

## Datos y validación

Serie real de NASA POWER de la celda de Sabanagrande (1981–2025, 16.436 días), con partición cronológica: entrenamiento 1981–2020 y prueba 2021–2025, reservada antes de cualquier análisis.

## Resultados principales

- La lluvia semanal es muy asimétrica (media de 18,5 mm y mediana de 7 mm) y tiene un ciclo anual bimodal, con enero–marzo casi sin lluvia y el máximo en octubre.
- El SVR lineal supera a la línea base trivial (Dummy) y a la climatología: a una semana, el RMSE baja de 46,5 mm (Dummy) y 43,4 mm (climatología) a 40,3 mm (mejora de 7 % sobre la climatología, IC 95 % de 3,7 % a 10,7 %).
- Con el criterio conservador, el pronóstico es confiable hasta **8 semanas**; después se usan cuantiles climatológicos.
- El periodo de prueba (2021–2025) es 61 % más lluvioso que el de entrenamiento; debe verificarse con estaciones del IDEAM.
- Con las constantes de ejemplo (reservorio de 20.000 m³, riego por goteo) y siembra en enero, hay **2,83 ha disponibles con 95 % de seguridad** (3,35 ha en el escenario esperado).

## Estructura del libro

```{tableofcontents}
```

## Reproducibilidad

- Dependencias en `requirements.txt`; semilla global `SEED = 42` en `src/config.py`, donde están todos los parámetros.
- El notebook 00 descarga los datos desde las fuentes oficiales y guarda las respuestas en `data/raw/power/`; las ejecuciones siguientes usan esa caché.
- Los notebooks se ejecutan en orden (00b → 04). Las constantes de la finca están en `src/config.py`.
