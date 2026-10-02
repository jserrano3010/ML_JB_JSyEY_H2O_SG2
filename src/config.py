"""
Configuración central del proyecto.

Todo parámetro que todavía no conocemos queda en None y se completa a medida
que avance el proyecto. Las funciones del paquete reciben estos valores como
argumentos, así que el modelo queda expresado en función de las variables.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# --------------------------------------------------------------------------
# Descarga NASA POWER
# --------------------------------------------------------------------------
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = [
    "PRECTOTCORR",        # precipitación corregida [mm/día]
    "T2M", "T2M_MAX", "T2M_MIN",  # temperatura a 2 m [°C]
    "RH2M",               # humedad relativa a 2 m [%]
    "WS2M",               # viento a 2 m [m/s]
    "ALLSKY_SFC_SW_DWN",  # radiación solar en superficie [MJ/m²/día] (comunidad AG)
    "PS",                 # presión en superficie [kPa]
    "GWETROOT",           # humedad del suelo, zona de raíces [0-1]
    "GWETTOP",            # humedad del suelo, capa superficial [0-1]
]
POWER_COMMUNITY = "AG"
FECHA_INICIO = "19810101"
FECHA_FIN = "20251231"          # años completos
VALOR_FALTANTE = -999.0

# Grilla regional (centros de celda de POWER/MERRA-2: 0,5° x 0,625°)
LAT_FINCA, LON_FINCA = 10.79, -74.76
LAT_MIN, LAT_MAX = 8.0, 11.5
LON_MIN, LON_MAX = -76.9, -72.5
ELEVACION_MAX_M = 400

# Sitio de prueba: la finca. La celda regional más cercana se excluye en la
# construcción del dataset para que ningún vecino inmediato quede en entrenamiento.
PUNTO_OBJETIVO = "FINCA_SABANAGRANDE"
ARCHIVO_DATOS = "dataset_sabanagrande.csv.gz"

# --------------------------------------------------------------------------
# Partición (se define ANTES del EDA)
# --------------------------------------------------------------------------
INICIO_TEST = "2021-01-01"      # test cronológico: 2021-2025
HORIZONTES_SEMANAS = [1, 2, 4, 8, 13, 26, 52]
# Un pronóstico por semana (los lunes): reduce el solapamiento de objetivos
# y el tamaño de la tabla de modelado con ~40 sitios.
EMISION_SEMANAL = True
# Armónicos del ciclo anual como predictoras (el EDA muestra un ciclo bimodal)
ARMONICOS = 3
DIAS_POR_SEMANA = 7
N_SPLITS_CV = 5
# Días entre el último dato disponible y el momento de emitir el pronóstico
# (NASA POWER publica con algunos días de retraso). Se aplica a las predictoras.
LATENCIA_DIAS = 3
# Usar el índice ONI (NOAA) como predictora exógena
USAR_ONI = True

# --------------------------------------------------------------------------
# Parámetros hídricos y de la finca (se completan más adelante)
# --------------------------------------------------------------------------
FINCA = {
    "H": 20,        # hectáreas totales disponibles [ha] — VALOR DE EJEMPLO: reemplazar por el de la finca
    "J": 20000,     # capacidad de almacenamiento en condiciones normales [m³] — VALOR DE EJEMPLO
    "A_c": 0.5,     # área de captación que drena al almacenamiento [ha] — VALOR DE EJEMPLO
    "C": 1.0,       # coeficiente de escorrentía [0-1] (1,0 = lluvia sobre la lámina del reservorio)
    "Q_ext": 0.0,   # entrada externa semanal fija (bombeo, pozo) [m³/semana]
    "eta": 0.9,     # eficiencia del sistema de riego [0-1] (goteo 0,90; aspersión 0,75; gravedad 0,60)
}

# Lluvia efectiva sobre el cultivo: Pe = alpha * P si P >= P_min, si no 0
LLUVIA_EFECTIVA = {"alpha": 0.8, "P_min": 5.0}  # mismos valores que usa apps/planificador_siembra.py

# Cultivos: coeficientes FAO-56 por etapa y duración de etapas [días].
# Plantilla: se agrega un diccionario por cultivo.
CULTIVOS = {
    "maiz": {
        "Kc_ini": 0.30, "Kc_mid": 1.20, "Kc_end": 0.35,   # FAO-56, tabla 12 (maíz grano)
        "L_ini": 20, "L_dev": 35, "L_mid": 40, "L_late": 30,   # FAO-56, tabla 11 (125 días)
        "b": 1,      # beneficio neto por hectárea (1 si se maximiza área)
    },
}

# Nivel de confianza para la versión conservadora
CONFIANZA = 0.95


def faltantes(diccionario, nombre):
    """Lista los parámetros que siguen en None (para avisar en los notebooks)."""
    return [f"{nombre}['{k}']" for k, v in diccionario.items() if v is None]
