"""
Planificador de siembra y agua — ¿Cuánta agua necesita mi siembra?

El campesino indica cultivos, hectáreas, fecha de siembra y datos del reservorio.
El planificador simula el ciclo completo con el clima diario de cada año desde 1981
(NASA POWER) y calcula:
  * el agua que consumen los cultivos y el riego necesario (m³),
  * la reserva que debe tener al sembrar para asegurar el cultivo con 80, 90 o 95 %,
  * la probabilidad de éxito con el reservorio actual,
  * el área máxima sembrable y la fecha de siembra que menos agua necesita.

Cómo ejecutarlo (Visual Studio Code):
    python -m pip install "dash==2.18.2" "plotly>=5.24,<6" pandas numpy
    python apps/planificador_siembra.py   → abrir http://127.0.0.1:8050

Datos: busca data/processed/dataset_sabanagrande.csv.gz (notebook 00) junto a este
archivo o en las carpetas superiores; si no existe, usa datos sintéticos de ejemplo.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dash_table, dcc, html

SEED = 42
PUNTO_OBJETIVO = "FINCA_SABANAGRANDE"
LAT_FINCA, LON_FINCA = 10.79, -74.76
# Lluvia efectiva diaria: Pe = alpha·P si P >= P_min (la llovizna no alcanza la raíz)
LLUVIA_EF = {"alpha": 0.8, "P_min": 5.0}
FUENTES = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700"
           "&family=Figtree:wght@400;500;600;700&display=swap")

# ==========================================================================
# CLIMA: ET0 FAO-56 Y DATOS SINTÉTICOS DE RESPALDO
# ==========================================================================
PUNTOS = {
    "sabanagrande":      {"lat": 10.79, "lon": -74.76, "nombre": "Sabanagrande", "factor": 1.20},
    "calamar":           {"lat": 10.25, "lon": -74.92, "nombre": "Calamar", "factor": 1.35},
    "fundacion":         {"lat": 10.52, "lon": -74.19, "nombre": "Fundación", "factor": 1.50},
    "el_carmen_bolivar": {"lat": 9.72,  "lon": -75.12, "nombre": "El Carmen de Bolívar", "factor": 1.45},
    "sincelejo":         {"lat": 9.30,  "lon": -75.40, "nombre": "Sincelejo", "factor": 1.55},
}



def _es(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def et0_fao56(tmax, tmin, rh, u2, rs, lat, doy, z=20.0, p=None):
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    tm = (tmax + tmin) / 2
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26 if p is None else np.asarray(p, float)
    g = 0.000665 * p
    es = (_es(tmax) + _es(tmin)) / 2
    ea = np.asarray(rh, float) / 100 * es
    d = 4098 * _es(tm) / (tm + 237.3) ** 2
    phi = np.deg2rad(lat)
    J = np.asarray(doy, float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    de = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(de), -1, 1))
    ra = 37.586 * dr * (ws * np.sin(phi) * np.sin(de) + np.cos(phi) * np.cos(de) * np.sin(ws))
    rso = (0.75 + 2e-5 * z) * ra
    rs = np.asarray(rs, float)
    rnl = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) \
        * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = 0.77 * rs - rnl
    u2 = np.asarray(u2, float)
    et0 = (0.408 * d * rn + g * 900 / (tm + 273) * u2 * (es - ea)) / (d + g * (1 + 0.34 * u2))
    return np.maximum(et0, 0)




LLUVIA_MENSUAL_BASE = np.array([2, 2, 5, 30, 95, 85, 60, 95, 135, 160, 80, 18], float)  # mm/mes
PROB_DIA_LLUVIOSO = np.array([.03, .03, .05, .15, .35, .33, .27, .35, .45, .50, .33, .10])


def _oni_sintetico(fechas, rng):
    meses = pd.period_range(fechas.min(), fechas.max(), freq="M")
    x = np.zeros(len(meses))
    for i in range(1, len(meses)):
        x[i] = 0.93 * x[i - 1] + rng.normal(0, 0.28)
    serie = pd.Series(x, index=meses.to_timestamp()).reindex(fechas, method="ffill")
    return serie.rolling(45, min_periods=1, center=True).mean().values


def generar_sintetico():
    rng = np.random.default_rng(SEED)
    fechas = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    mes = fechas.month.values - 1
    doy = fechas.dayofyear.values
    oni = _oni_sintetico(fechas, rng)
    # estado regional compartido (los puntos vecinos llueven parecido)
    comun = rng.random(len(fechas))
    tablas = []
    for clave, c in PUNTOS.items():
        enso = np.clip(1 - 0.30 * oni, 0.4, 1.8)            # La Niña (ONI<0) → más lluvia
        prob = np.clip(PROB_DIA_LLUVIOSO[mes] * enso, 0, 0.9)
        u = 0.6 * comun + 0.4 * rng.random(len(fechas))
        u = (u - u.min()) / (u.max() - u.min())
        llueve = u < prob
        media_dia = LLUVIA_MENSUAL_BASE[mes] * c["factor"] * enso / (30.4 * PROB_DIA_LLUVIOSO[mes])
        P = np.where(llueve, rng.gamma(0.75, media_dia / 0.75), 0.0)
        humedo = pd.Series(P).rolling(20, min_periods=1).sum().values
        estac = np.sin(2 * np.pi * (doy - 30) / 365)
        tmax = 32.8 + 1.1 * estac - 0.03 * np.minimum(humedo, 80) + 0.02 * (fechas.year - 1981) + rng.normal(0, 0.8, len(fechas))
        tmin = 23.5 + 0.7 * estac + 0.01 * np.minimum(humedo, 80) + 0.015 * (fechas.year - 1981) + rng.normal(0, 0.6, len(fechas))
        rh = np.clip(70 + 0.25 * np.minimum(humedo, 60) + 4 * llueve + rng.normal(0, 3, len(fechas)), 40, 100)
        ws = np.clip(2.6 - 1.2 * (PROB_DIA_LLUVIOSO[mes] - .05) + rng.normal(0, 0.4, len(fechas)), 0.3, None)
        rs = np.clip(21 - 5 * llueve - 0.03 * np.minimum(humedo, 60) + rng.normal(0, 1.5, len(fechas)), 5, 30)
        gw_root = np.clip(0.35 + 0.004 * pd.Series(P).rolling(60, min_periods=1).sum().values / c["factor"], 0.2, 0.95)
        gw_top = np.clip(0.25 + 0.012 * pd.Series(P).rolling(10, min_periods=1).sum().values / c["factor"], 0.1, 0.98)
        d = pd.DataFrame({
            "fecha": fechas, "punto": clave, "lat": c["lat"], "lon": c["lon"], "elevacion_m": 20.0,
            "PRECTOTCORR": P.round(2), "T2M": ((tmax + tmin) / 2).round(2), "T2M_MAX": tmax.round(2),
            "T2M_MIN": tmin.round(2), "RH2M": rh.round(1), "WS2M": ws.round(2),
            "ALLSKY_SFC_SW_DWN": rs.round(2), "PS": 101.1, "GWETROOT": gw_root.round(3),
            "GWETTOP": gw_top.round(3), "ONI": oni.round(2),
        })
        d.loc[d.fecha < "1984-01-01", "ALLSKY_SFC_SW_DWN"] = np.nan     # hueco típico del registro
        tablas.append(d)
    return pd.concat(tablas, ignore_index=True)


# ==========================================================================
# MOTOR DE DECISIÓN
# ==========================================================================
# Idea: cada año histórico es un escenario posible del clima que viene.
# Para una fecha de siembra, se toma el clima diario de cada año desde esa
# fecha y se simula el ciclo completo. El pronóstico estacional (probabilidad
# de temporada seca / normal / húmeda) le da más peso a los años parecidos.
# ==========================================================================
M3_POR_MM_HA = 10.0

# Coeficientes de referencia FAO-56 (Allen et al., 1998). Deben ajustarse a la
# variedad y a la zona: duración de etapas en días.
CATALOGO = [
    {"cultivo": "Maíz",           "Kc_ini": 0.30, "Kc_mid": 1.20, "Kc_end": 0.35, "L_ini": 20, "L_dev": 35, "L_mid": 40, "L_late": 30, "c_fijo": None},
    {"cultivo": "Millo (sorgo)",  "Kc_ini": 0.30, "Kc_mid": 1.05, "Kc_end": 0.55, "L_ini": 20, "L_dev": 35, "L_mid": 40, "L_late": 30, "c_fijo": None},
    {"cultivo": "Frijol",         "Kc_ini": 0.40, "Kc_mid": 1.15, "Kc_end": 0.35, "L_ini": 20, "L_dev": 30, "L_mid": 40, "L_late": 20, "c_fijo": None},
    {"cultivo": "Yuca",           "Kc_ini": 0.30, "Kc_mid": 0.80, "Kc_end": 0.30, "L_ini": 20, "L_dev": 40, "L_mid": 90, "L_late": 60, "c_fijo": None},
    {"cultivo": "Patilla",        "Kc_ini": 0.40, "Kc_mid": 1.00, "Kc_end": 0.75, "L_ini": 20, "L_dev": 30, "L_mid": 30, "L_late": 30, "c_fijo": None},
]


def _num(v, defecto=None):
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return defecto
        return float(v)
    except (TypeError, ValueError):
        return defecto


def duracion(c):
    return int(sum(_num(c.get(k), 0) for k in ("L_ini", "L_dev", "L_mid", "L_late")))


def curva_kc(c):
    """Kc diario por etapas (FAO-56)."""
    Li, Ld, Lm, Ll = (int(_num(c.get(k), 0)) for k in ("L_ini", "L_dev", "L_mid", "L_late"))
    ki, km, ke = (_num(c.get(k), 0) for k in ("Kc_ini", "Kc_mid", "Kc_end"))
    return np.concatenate([np.full(Li, ki), np.linspace(ki, km, Ld + 1)[1:],
                           np.full(Lm, km), np.linspace(km, ke, Ll + 1)[1:]])


def demanda_diaria(c, et0):
    """
    Agua que consume el cultivo por hectárea y día [m³/ha/día].
    Si el campesino da un coeficiente fijo c_fijo (m³/ha/día), se usa ese;
    si no, 10 · Kc · ET0 (FAO-56).
    """
    n = duracion(c)
    fijo = _num(c.get("c_fijo"))
    if fijo is not None and fijo > 0:
        return np.full(n, fijo)
    return M3_POR_MM_HA * curva_kc(c)[:n] * et0[:n]


def lluvia_efectiva(P, alpha, P_min):
    return np.where(P >= P_min, alpha * P, 0.0)


def escenarios(serie, mes, dia, largo):
    """
    Clima diario de cada año histórico desde (mes, día) durante `largo` días.
    serie: DataFrame indexado por fecha con PRECTOTCORR y ET0 (un punto).
    Devuelve lista de (año, P[largo], ET0[largo]).
    """
    out = []
    for anio in range(serie.index.year.min(), serie.index.year.max() + 1):
        try:
            ini = pd.Timestamp(anio, mes, min(dia, 28 if mes == 2 else dia))
        except ValueError:
            continue
        tramo = serie.loc[ini: ini + pd.Timedelta(days=largo - 1)]
        if len(tramo) == largo and tramo[["PRECTOTCORR", "ET0"]].notna().all().all():
            out.append((anio, tramo["PRECTOTCORR"].values, tramo["ET0"].values))
    return out


def pesos_pronostico(lluvias_ciclo, p_seca, p_normal, p_humeda):
    """
    Pesos de los años según el pronóstico por terciles.
    Cada tercil de lluvia del ciclo recibe la probabilidad pronosticada,
    repartida por igual entre los años de ese tercil. 1/3-1/3-1/3 = climatología.
    """
    l = np.asarray(lluvias_ciclo)
    q1, q2 = np.quantile(l, [1 / 3, 2 / 3])
    tercil = np.where(l <= q1, 0, np.where(l <= q2, 1, 2))
    probs = np.array([p_seca, p_normal, p_humeda], float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(3, 1 / 3)
    w = np.array([probs[t] / max(1, (tercil == t).sum()) for t in tercil])
    return w / w.sum(), tercil


def cuantil_ponderado(valores, pesos, q):
    v, w = np.asarray(valores, float), np.asarray(pesos, float)
    o = np.argsort(v)
    acum = np.cumsum(w[o])
    return float(v[o][np.searchsorted(acum, q * acum[-1], side="left").clip(0, len(v) - 1)])


def semanal(x):
    n = int(np.ceil(len(x) / 7))
    x = np.pad(np.asarray(x, float), (0, n * 7 - len(x)))
    return x.reshape(n, 7).sum(axis=1)


def simular(cultivos, P, ET0, finca, lluvia_ef):
    """
    Simula un escenario semana a semana.
    cultivos: lista de dicts con 'hectareas' y coeficientes.
    Devuelve dict con demanda y riego por cultivo, entradas, nivel del reservorio
    y la reserva mínima necesaria (algoritmo del pico secuente).
    """
    eta = finca["eta"]
    largo = len(P)
    Pe = semanal(lluvia_efectiva(P, lluvia_ef["alpha"], lluvia_ef["P_min"]))
    Pw = semanal(P)
    T = len(Pw)
    demanda, riego = {}, {}
    for c in cultivos:
        d = np.zeros(largo)
        dd = demanda_diaria(c, ET0)
        d[:len(dd)] = dd
        dw = semanal(d)                                         # m³/ha por semana
        activo = semanal((np.arange(largo) < len(dd)).astype(float)) > 0
        nr = np.where(activo, np.maximum(0, dw - M3_POR_MM_HA * Pe) / eta, 0.0)
        demanda[c["cultivo"]] = dw
        riego[c["cultivo"]] = nr
    uso = sum(riego[c["cultivo"]] * c["hectareas"] for c in cultivos) if cultivos else np.zeros(T)
    entrada = M3_POR_MM_HA * Pw * finca["A_c"] * finca["C"] + finca["Q_ext"]
    # Nivel del reservorio (capacidad J, arranca en S0)
    S, nivel = finca["S0"], []
    for t in range(T):
        S = min(finca["J"], S + entrada[t] - uso[t])
        nivel.append(S)
    # Pico secuente: reserva inicial mínima para no quedarse sin agua (sin límite de capacidad)
    K, K_max = 0.0, 0.0
    for t in range(T):
        K = max(0.0, K + uso[t] - entrada[t])
        K_max = max(K_max, K)
    return {"demanda": demanda, "riego": riego, "uso": uso, "entrada": entrada,
            "nivel": np.array(nivel), "reserva_necesaria": K_max, "lluvia": Pw, "lluvia_ef": Pe}


def evaluar(serie, cultivos, fecha_siembra, finca, lluvia_ef, pronostico, confianza):
    """Corre todos los escenarios y resume la decisión."""
    cultivos = [c for c in cultivos if _num(c.get("hectareas"), 0) > 0 and duracion(c) > 0]
    for c in cultivos:
        c["hectareas"] = _num(c["hectareas"], 0)
    if not cultivos:
        return None
    largo = max(duracion(c) for c in cultivos)
    esc = escenarios(serie, fecha_siembra.month, fecha_siembra.day, largo)
    sims = [simular(cultivos, P, E, finca, lluvia_ef) for _, P, E in esc]
    lluvia_ciclo = [P.sum() for _, P, _ in esc]
    w, tercil = pesos_pronostico(lluvia_ciclo, *pronostico)
    reservas = np.array([s["reserva_necesaria"] for s in sims])
    exito = np.array([(s["nivel"] >= 0).all() for s in sims])
    agua_riego = np.array([s["uso"].sum() for s in sims])
    return {
        "anios": [a for a, _, _ in esc], "pesos": w, "tercil": tercil, "sims": sims, "cultivos": cultivos,
        "largo": largo, "lluvia_ciclo": np.array(lluvia_ciclo),
        "reserva_conf": cuantil_ponderado(reservas, w, confianza),
        "reserva_mediana": cuantil_ponderado(reservas, w, 0.5),
        "prob_exito": float((w * exito).sum()),
        "riego_conf": cuantil_ponderado(agua_riego, w, confianza),
        "riego_mediana": cuantil_ponderado(agua_riego, w, 0.5),
        "demanda_total": float(np.average([sum(s["demanda"][c["cultivo"]].sum() * c["hectareas"] for c in cultivos) for s in sims], weights=w)),
    }


def area_maxima(serie, cultivos, fecha_siembra, finca, lluvia_ef, pronostico, confianza, H):
    """
    Mayor área total, manteniendo la proporción entre cultivos que dio el campesino,
    con la que el reservorio no se seca en al menos `confianza` de los escenarios.
    """
    base = [dict(c) for c in cultivos if _num(c.get("hectareas"), 0) > 0 and duracion(c) > 0]
    total = sum(_num(c["hectareas"]) for c in base)
    if total == 0:
        return 0.0
    prop = [_num(c["hectareas"]) / total for c in base]
    largo = max(duracion(c) for c in base)
    esc = escenarios(serie, fecha_siembra.month, fecha_siembra.day, largo)
    w, _ = pesos_pronostico([P.sum() for _, P, _ in esc], *pronostico)

    def prob(A):
        cs = [dict(c, hectareas=A * p) for c, p in zip(base, prop)]
        ok = np.array([(simular(cs, P, E, finca, lluvia_ef)["nivel"] >= 0).all() for _, P, E in esc])
        return (w * ok).sum()

    if prob(H) >= confianza:
        return float(H)
    lo, hi = 0.0, float(H)
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if prob(mid) >= confianza else (lo, mid)
    return lo


def mejor_fecha(serie, cultivos, finca, lluvia_ef, pronostico, confianza, anio_ref=2027):
    """Reserva necesaria (al nivel de confianza) por hectárea sembrada, según la fecha de siembra."""
    filas = []
    base = [dict(c) for c in cultivos if _num(c.get("hectareas"), 0) > 0 and duracion(c) > 0]
    area = sum(_num(c["hectareas"]) for c in base)
    if area == 0:
        return pd.DataFrame(columns=["fecha", "reserva_por_ha", "riego_por_ha"])
    for mes in range(1, 13):
        for dia in (1, 15):
            f = pd.Timestamp(anio_ref, mes, dia)
            r = evaluar(serie, [dict(c) for c in base], f, finca, lluvia_ef, pronostico, confianza)
            filas.append({"fecha": f, "reserva_por_ha": r["reserva_conf"] / area, "riego_por_ha": r["riego_conf"] / area})
    return pd.DataFrame(filas)


# ==========================================================================
# DATOS DEL PUNTO DE LA FINCA
# ==========================================================================
def _buscar(nombre):
    aqui = Path(__file__).resolve().parent
    for base in (aqui, aqui.parent, aqui.parent.parent):
        r = base / "data" / "processed" / nombre
        if r.exists():
            return r
    return None


@lru_cache(maxsize=1)
def cargar_serie():
    """Serie diaria (lluvia y ET0) del punto de la finca."""
    ruta = _buscar("dataset_sabanagrande.csv.gz") or _buscar("power_region.csv.gz") or _buscar("power_diario.csv.gz")
    if ruta is not None:
        df = pd.read_csv(ruta, parse_dates=["fecha"])
        fuente = "NASA POWER"
    else:
        df = generar_sintetico()
        fuente = "sintéticos de ejemplo"
    if "punto" in df and PUNTO_OBJETIVO in set(df.punto):
        g = df[df.punto == PUNTO_OBJETIVO].copy()
    else:  # el punto más cercano a la finca
        d = (df.lat - LAT_FINCA) ** 2 + (df.lon - LON_FINCA) ** 2
        g = df[df.punto == df.loc[d.idxmin(), "punto"]].copy()
    if "ET0" not in g or g["ET0"].isna().all():
        g["ET0"] = et0_fao56(g.T2M_MAX, g.T2M_MIN, g.RH2M, g.WS2M, g.ALLSKY_SFC_SW_DWN,
                             g.lat.iloc[0], g.fecha.dt.dayofyear, g.elevacion_m.iloc[0], g.PS)
    serie = g.set_index("fecha").sort_index()[["PRECTOTCORR", "ET0"]]
    # ET0 faltante (años sin radiación): climatología diaria de ET0
    clim = serie.groupby(serie.index.dayofyear)["ET0"].transform("mean")
    serie["ET0"] = serie["ET0"].fillna(clim)
    return serie, fuente


# ==========================================================================
# INTERFAZ
# ==========================================================================
TINTA, RIO, LLUVIA, SOL, TIERRA, SALVIA = "#17332E", "#1F6F78", "#3A7CA5", "#D69E2E", "#A5673F", "#7F9C80"
COLORES_CULTIVO = [RIO, SOL, TIERRA, LLUVIA, SALVIA, "#5B4630", "#8E5B9A"]
pio.templates["finca"] = go.layout.Template(layout=dict(
    font=dict(family="Figtree, 'Segoe UI', system-ui, sans-serif", size=13, color=TINTA),
    title=dict(font=dict(family="'Bricolage Grotesque', Figtree, sans-serif", size=17), x=0.01, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F6F9F7",
    xaxis=dict(gridcolor="#DCE5E0", linecolor="#B9C9C1"), yaxis=dict(gridcolor="#DCE5E0", linecolor="#B9C9C1"),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=60, r=20, t=60, b=45), hoverlabel=dict(bgcolor="white", bordercolor=RIO)))
pio.templates.default = "plotly_white+finca"
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
            "septiembre", "octubre", "noviembre", "diciembre"]


def miles(x, dec=0):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fecha_es(f):
    return f"{f.day} de {MESES_ES[f.month - 1]}"


COLUMNAS = [
    {"name": ["", "Cultivo"], "id": "cultivo", "type": "text"},
    {"name": ["", "Hectáreas"], "id": "hectareas", "type": "numeric"},
    {"name": ["Coeficiente de cultivo (Kc)", "inicial"], "id": "Kc_ini", "type": "numeric"},
    {"name": ["Coeficiente de cultivo (Kc)", "medio"], "id": "Kc_mid", "type": "numeric"},
    {"name": ["Coeficiente de cultivo (Kc)", "final"], "id": "Kc_end", "type": "numeric"},
    {"name": ["Duración de etapas (días)", "inicial"], "id": "L_ini", "type": "numeric"},
    {"name": ["Duración de etapas (días)", "desarrollo"], "id": "L_dev", "type": "numeric"},
    {"name": ["Duración de etapas (días)", "media"], "id": "L_mid", "type": "numeric"},
    {"name": ["Duración de etapas (días)", "final"], "id": "L_late", "type": "numeric"},
    {"name": ["Opcional", "m³/ha/día fijo"], "id": "c_fijo", "type": "numeric"},
]
FILAS_INICIALES = [dict(c, hectareas=h) for c, h in zip(CATALOGO, [5, 3, 0, 0, 0])]


def num_input(id_, valor, minimo=0, paso="any"):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, step=paso, debounce=True, className="entrada")


def campo(etiqueta, comp, ayuda=None):
    hijos = [html.Label(etiqueta, className="campo__etiqueta"), comp]
    if ayuda:
        hijos.append(html.Small(ayuda, className="campo__ayuda"))
    return html.Div(hijos, className="campo")


SERIE, FUENTE = cargar_serie()
app = Dash(__name__, title="Planificador de siembra y agua", external_stylesheets=[FUENTES])
server = app.server


app.index_string = """<!DOCTYPE html>
<html lang="es"><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>
:root { --fondo:#E6EDE9; --panel:#FBFCFB; --tinta:#17332E; --suave:#4E6A63; --rio:#1F6F78; --sol:#D69E2E; --tierra:#A5673F; --borde:#CBD8D2;
  --titulo:"Bricolage Grotesque", Figtree, "Segoe UI", system-ui, sans-serif; --texto:Figtree, "Segoe UI", system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin:0; background:var(--fondo); color:var(--tinta); font-family:var(--texto); font-size:16px; line-height:1.5; }
.hero { aspect-ratio:1600/560; max-height:340px; min-height:240px; width:100%;
  background:url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAwIDU2MCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgc2xpY2UiIHJvbGU9ImltZyIgYXJpYS1sYWJlbD0iRmluY2EgZW4gdGVtcG9yYWRhIHNlY2EganVudG8gYWwgcsOtbywgY29uIHVuIHJlc2Vydm9yaW8gZGUgYWd1YSB5IHVuYSBudWJlIGRlIGxsdXZpYSBhIGxvIGxlam9zIj4KPGRlZnM+CjxsaW5lYXJHcmFkaWVudCBpZD0iY2llbG8iIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzlGQzdDRiIvPjxzdG9wIG9mZnNldD0iLjU1IiBzdG9wLWNvbG9yPSIjRDhFNERDIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjRUZFM0M0Ii8+PC9saW5lYXJHcmFkaWVudD4KPHJhZGlhbEdyYWRpZW50IGlkPSJzb2wiIGN4PSIuNSIgY3k9Ii41IiByPSIuNSI+CiA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiNGNkM0NTMiLz48c3RvcCBvZmZzZXQ9Ii41NSIgc3RvcC1jb2xvcj0iI0U4QTkzQSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0U4QTkzQSIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD4KPGxpbmVhckdyYWRpZW50IGlkPSJ0aWVycmEiIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI0Q3QjI3QSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0I5OEE1MiIvPjwvbGluZWFyR3JhZGllbnQ+CjxsaW5lYXJHcmFkaWVudCBpZD0iYWd1YSIgeDE9IjAiIHkxPSIwIiB4Mj0iMCIgeTI9IjEiPgogPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjM0Y4Qzk1Ii8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMUY1RTY2Ii8+PC9saW5lYXJHcmFkaWVudD4KPGxpbmVhckdyYWRpZW50IGlkPSJyaW8iIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIwIj4KIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzVEOUNBMyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzdGQjJCNSIvPjwvbGluZWFyR3JhZGllbnQ+CjxjbGlwUGF0aCBpZD0iamFndWV5Ij48ZWxsaXBzZSBjeD0iMTE4MCIgY3k9IjQ1NSIgcng9IjIzMCIgcnk9IjU4Ii8+PC9jbGlwUGF0aD4KPC9kZWZzPgo8cmVjdCB3aWR0aD0iMTYwMCIgaGVpZ2h0PSI1NjAiIGZpbGw9InVybCgjY2llbG8pIi8+CjxjaXJjbGUgY3g9IjEzMzAiIGN5PSIxMjAiIHI9IjE1MCIgZmlsbD0idXJsKCNzb2wpIiBvcGFjaXR5PSIuNTUiLz4KPGNpcmNsZSBjeD0iMTMzMCIgY3k9IjEyMCIgcj0iNTgiIGZpbGw9IiNGMkI1NDQiLz4KPGcgb3BhY2l0eT0iLjg1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2NDAgLTMwKSI+PHBhdGggZD0iTTE1MCAxNTAgcTEwLTQ4IDYyLTQ0IHEyMi00MCA3Mi0yMiBxNDQtMTYgNjYgMjQgcTQ0IDQgNDAgNDQgeiIgZmlsbD0iIzZGOEY5OSIvPgo8bGluZSB4MT0iMTc1IiB5MT0iMTYwIiB4Mj0iMTYxIiB5Mj0iMjQ1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMTkxIiB5MT0iMTYwIiB4Mj0iMTc3IiB5Mj0iMjY1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjA3IiB5MT0iMTYwIiB4Mj0iMTkzIiB5Mj0iMjM5IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjIzIiB5MT0iMTYwIiB4Mj0iMjA5IiB5Mj0iMjQ3IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjM5IiB5MT0iMTYwIiB4Mj0iMjI1IiB5Mj0iMjU1IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjU1IiB5MT0iMTYwIiB4Mj0iMjQxIiB5Mj0iMjM2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjcxIiB5MT0iMTYwIiB4Mj0iMjU3IiB5Mj0iMjM3IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMjg3IiB5MT0iMTYwIiB4Mj0iMjczIiB5Mj0iMjYxIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzAzIiB5MT0iMTYwIiB4Mj0iMjg5IiB5Mj0iMjUyIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzE5IiB5MT0iMTYwIiB4Mj0iMzA1IiB5Mj0iMjM4IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzM1IiB5MT0iMTYwIiB4Mj0iMzIxIiB5Mj0iMjQ2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzUxIiB5MT0iMTYwIiB4Mj0iMzM3IiB5Mj0iMjUzIiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzY3IiB5MT0iMTYwIiB4Mj0iMzUzIiB5Mj0iMjM2IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8bGluZSB4MT0iMzgzIiB5MT0iMTYwIiB4Mj0iMzY5IiB5Mj0iMjY0IiBzdHJva2U9IiM2RjhGOTkiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8L2c+CjxwYXRoIGQ9Ik0wIDI4NSBDIDE4MCAyNDAgMzMwIDI2MiA1MjAgMjUwIFMgODgwIDIyMCAxMDYwIDI0OCBTIDE0MDAgMjM2IDE2MDAgMjU4IEwxNjAwIDMzMCBMMCAzMzAgWiIgZmlsbD0iIzlEQjM5QSIgb3BhY2l0eT0iLjc1Ii8+CjxwYXRoIGQ9Ik0wIDMwNSBDIDIyMCAyODAgNDIwIDMwMCA2NDAgMjg4IFMgMTAyMCAyNzAgMTI0MCAyOTIgUyAxNDgwIDI4NCAxNjAwIDI5NiBMMTYwMCAzNDAgTDAgMzQwIFoiIGZpbGw9IiM3RjlDODAiIG9wYWNpdHk9Ii44Ii8+CjxwYXRoIGQ9Ik0wIDMzMCBDIDMwMCAzMTggNjIwIDM0NCA5MDAgMzMyIFMgMTQwMCAzMjIgMTYwMCAzMzQgTDE2MDAgMzU2IEMgMTM4MCAzNDggMTEwMCAzNjAgODYwIDM1NiBTIDI4MCAzNDYgMCAzNTYgWiIgZmlsbD0idXJsKCNyaW8pIi8+CjxwYXRoIGQ9Ik02MCAzNDAgaDEyMCBNNDIwIDM0NSBoOTAgTTc2MCAzNDIgaDE0MCBNMTE4MCAzMzYgaDExMCBNMTQ1MCAzNDAgaDgwIiBzdHJva2U9IiNFOEYyRjEiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNyIvPgo8cGF0aCBkPSJNMCAzNTIgQyA0MDAgMzYyIDkwMCAzNDggMTYwMCAzNTYgTDE2MDAgNTYwIEwwIDU2MCBaIiBmaWxsPSJ1cmwoI3RpZXJyYSkiLz4KPHBhdGggZD0iTTEwMzkgNDU0IEwxMDE2IDQ1NyBMMTAxNCA0NDkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xODUgNTQxIEwxNjAgNTQ5IEwxMzkgNTQ2IEwxNTEgNTU2IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEyNiA1NDcgTDEyMyA1MzggTDEwOSA1MjkgTDExNiA1MjMgTDEwNiA1MjYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xMTA3IDQzMCBMMTA5OCA0MzcgTDExMjIgNDQ4IEwxMTA1IDQ0MSBMMTExNCA0NDkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNMzg0IDQ5NSBMMzkxIDUwNyBMMzY3IDUxNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEyNjcgNDUyIEwxMjgyIDQ1OSBMMTI4MSA0NzMgTDEyNzMgNDc3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTkyOCA0OTIgTDkxNSA0ODcgTDkzMSA1MDEgTDkxOCA0OTMiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNjE0IDUzNCBMNjQyIDUzNCBMNjYwIDUzOCBMNjUwIDU0NyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTI0MSA1MzEgTDIyMyA1NDUgTDIxNiA1MzkgTDIxOSA1NDIiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xMzY4IDQxOSBMMTM3NiA0MTkgTDEzNjkgNDMxIEwxMzYzIDQ0MCBMMTM2NiA0NDgiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjYiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xNDAgNDIzIEwxNDIgNDM1IEwxNTYgNDI3IEwxMzEgNDQwIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTYzNCA1NDcgTDY1OCA1NTEgTDY0OCA1NjMgTDY0NCA1NzQgTDYzOCA1NjQiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjYiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik03MjcgNDQzIEw3MDYgNDQ4IEw2ODEgNDQ0IEw3MDIgNDQzIEw2ODIgNDU2IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNODE0IDUwMCBMNzkxIDQ5NSBMNzkxIDQ5NyBMNzk4IDQ5NSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTg4MSA1NDAgTDg5OCA1NDMgTDg5MiA1NTQgTDkyMCA1NTYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0zMDkgNDIxIEwyOTAgNDE4IEwzMDQgNDE1IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTkzIDU1MCBMOTgxIDU0OSBMOTUzIDU0MyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEwOTQgNDk0IEwxMTAyIDQ5NCBMMTA4MiA1MDYgTDExMDggNTEyIEwxMTE5IDUyMiIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0xNTE1IDQxMyBMMTU0MiA0MjcgTDE1NjkgNDM4IEwxNTkyIDQ0NSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTgxNSA1MDIgTDc5MyA1MDcgTDgwNSA1MDkgTDc4MCA1MDUiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik00MjcgNTEyIEw0MDYgNTEyIEw0MTYgNTAzIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNMCA1NDUgTDYgNTM4IEwxIDU0NyIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTE0NCA0NTMgTDE0MCA0NDcgTDE1MiA0NDUgTDE0NiA0NTQgTDE0MSA0NTkiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0yMzYgNTI0IEwyMzggNTI5IEwyMjkgNTIxIEwyMTAgNTE0IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTcwMSA0NjcgTDcyNiA0NzkgTDcwOCA0ODUgTDY4MSA0ODEiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNzQwIDQzNyBMNzQ2IDQyNyBMNzY2IDQzMyBMNzU3IDQ0MyBMNzg0IDQzNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik01MzQgNTMyIEw1MTYgNTMzIEw1MzcgNTMwIEw1NDMgNTM3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTY3NSA0NTcgTDY5OCA0NzEgTDcyNCA0NjcgTDc0NyA0NjQgTDc3MSA0NjYiIHN0cm9rZT0iIzhDNjIzOCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNDY0IDQ1MSBMNDY3IDQ1MiBMNDg1IDQ0MiBMNDU4IDQ0MCBMNDYwIDQzOCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTE0MTggNDg4IEwxNDQxIDUwMSBMMTQzNSA1MDIgTDE0MTIgNDk5IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNDY0IDUyMCBMNDU3IDUxNiBMNDU5IDUyNSIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik0zIDUyMiBMLTMgNTMyIEwtMjYgNTQzIEwtNDcgNTQ1IEwtMjUgNTU3IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTc5IDQ0NSBMMTAwMSA0NTUgTDk5NCA0NDcgTDEwMTcgNDYwIiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS42IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNOTQ4IDUwMiBMOTI1IDUxNSBMOTA3IDUxMCBMODg3IDUwMCBMODY4IDUwOCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTEzNDMgNDM3IEwxMzY3IDQ0NiBMMTM2OSA0NTcgTDEzNjMgNDUxIEwxMzcwIDQ1OCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjEuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTQzIDQwMyBMNTYgMzk2IEw2MSA0MDkgTDQxIDQxMiBMNjggNDA4IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMS4yIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9Ii41NSIvPgo8cGF0aCBkPSJNNTcgNDY0IEw0NyA0NzAgTDM0IDQ4NCIgc3Ryb2tlPSIjOEM2MjM4IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iLjU1Ii8+CjxwYXRoIGQ9Ik02NjcgNDY2IEw2NjUgNDYwIEw2NDAgNDczIEw2MzQgNDc3IEw2NDggNDg1IiBzdHJva2U9IiM4QzYyMzgiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBvcGFjaXR5PSIuNTUiLz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDAgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzhEOEEzRSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC00IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTgiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0xMiIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0xMCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMzUgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIzMCAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzI1IDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDIwIDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1MTUgMzgwKSBzY2FsZSgwLjQ1KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDYxMCAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcwNSAzODApIHNjYWxlKDAuNDUpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODAwIDM4MCkgc2NhbGUoMC40NSkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1OCA0MjApIHNjYWxlKDAuNikiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxODEgNDIwKSBzY2FsZSgwLjYpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzA0IDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzhEOEEzRSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC00IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTgiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0xMiIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0xMCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0MjcgNDIwKSBzY2FsZSgwLjYpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjNEY3RDQ1IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTE4IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTQ4IHEgMTYuMCAtMzAgMzIgLTIyIiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMjYiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMjQiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNTUwIDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY3MyA0MjApIHNjYWxlKDAuNikiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNzk2IDQyMCkgc2NhbGUoMC42KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDc2IDQ3MCkgc2NhbGUoMC44KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIyNyA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzc4IDQ3MCkgc2NhbGUoMC44KSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDUyOSA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2ODAgNDcwKSBzY2FsZSgwLjgpIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgzMSA0NzApIHNjYWxlKDAuOCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5NCA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM4RDhBM0UiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtNCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC04IiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTcwIHEgLTE1LjAgLTM0IC0zMCAtMTIiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtOTAgcSAxMy4wIC0zMiAyNiAtMTAiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMSAtMTE4IGwtNiAtMTQgTTEgLTExOCBsNiAtMTUgTTEgLTExOCBsMCAtMTciIHN0cm9rZT0iI0M3QTI0QSIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjwvZz4KPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjczIDUzMCkgc2NhbGUoMS4wKSI+PHBhdGggZD0iTTAgMCBDIDEgLTQwIC0yIC04MCAxIC0xMTgiIHN0cm9rZT0iIzRGN0Q0NSIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtMzAgcSAtMTcuMCAtMjYgLTM0IC0xOCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC00OCBxIDE2LjAgLTMwIDMyIC0yMiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTI2IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTI0IiBzdHJva2U9IiM2QTlBNTUiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ1MiA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MzEgNTMwKSBzY2FsZSgxLjApIj48cGF0aCBkPSJNMCAwIEMgMSAtNDAgLTIgLTgwIDEgLTExOCIgc3Ryb2tlPSIjOEQ4QTNFIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC0zMCBxIC0xNy4wIC0yNiAtMzQgLTQiIHN0cm9rZT0iI0I1OUE0QiIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtOCIgc3Ryb2tlPSIjQjU5QTRCIiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC03MCBxIC0xNS4wIC0zNCAtMzAgLTEyIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTkwIHEgMTMuMCAtMzIgMjYgLTEwIiBzdHJva2U9IiNCNTlBNEIiIHN0cm9rZS13aWR0aD0iNSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEgLTExOCBsLTYgLTE0IE0xIC0xMTggbDYgLTE1IE0xIC0xMTggbDAgLTE3IiBzdHJva2U9IiNDN0EyNEEiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L2c+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgxMCA1MzApIHNjYWxlKDEuMCkiPjxwYXRoIGQ9Ik0wIDAgQyAxIC00MCAtMiAtODAgMSAtMTE4IiBzdHJva2U9IiM0RjdENDUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTAgLTMwIHEgLTE3LjAgLTI2IC0zNCAtMTgiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNDggcSAxNi4wIC0zMCAzMiAtMjIiIHN0cm9rZT0iIzZBOUE1NSIgc3Ryb2tlLXdpZHRoPSI1IiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48cGF0aCBkPSJNMCAtNzAgcSAtMTUuMCAtMzQgLTMwIC0yNiIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0wIC05MCBxIDEzLjAgLTMyIDI2IC0yNCIgc3Ryb2tlPSIjNkE5QTU1IiBzdHJva2Utd2lkdGg9IjUiIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0xIC0xMTggbC02IC0xNCBNMSAtMTE4IGw2IC0xNSBNMSAtMTE4IGwwIC0xNyIgc3Ryb2tlPSIjQzdBMjRBIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9nPgo8ZWxsaXBzZSBjeD0iMTE4MCIgY3k9IjQ1NSIgcng9IjI0NiIgcnk9IjY2IiBmaWxsPSIjOUM3NDQzIi8+CjxlbGxpcHNlIGN4PSIxMTgwIiBjeT0iNDU1IiByeD0iMjMwIiByeT0iNTgiIGZpbGw9IiNDOUE2NkUiLz4KPGcgY2xpcC1wYXRoPSJ1cmwoI2phZ3VleSkiPjxyZWN0IHg9Ijk0MCIgeT0iNDYyIiB3aWR0aD0iNDgwIiBoZWlnaHQ9IjYwIiBmaWxsPSJ1cmwoI2FndWEpIi8+CjxwYXRoIGQ9Ik05NjAgNDcwIHEgMzAgLTUgNjAgMCB0IDYwIDAgTTExMTAgNDgwIHEgMzAgLTUgNjAgMCB0IDYwIDAgTTEyNjAgNDcyIHEgMzAgLTUgNjAgMCIgc3Ryb2tlPSIjQTlEMkQ0IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiIG9wYWNpdHk9Ii43Ii8+PC9nPgo8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMzMwIDM5MikiPjxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMTgiIHJ4PSIzIiBmaWxsPSIjRjRFRkU0IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMiIvPgo8bGluZSB4MT0iMCIgeTE9IjYiIHgyPSIxMiIgeTI9IjYiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSIyMCIgeDI9IjciIHkyPSIyMCIgc3Ryb2tlPSIjNUI0NjMwIiBzdHJva2Utd2lkdGg9IjEuNiIvPgo8bGluZSB4MT0iMCIgeTE9IjM0IiB4Mj0iMTIiIHkyPSIzNCIgc3Ryb2tlPSIjNUI0NjMwIiBzdHJva2Utd2lkdGg9IjEuNiIvPgo8bGluZSB4MT0iMCIgeTE9IjQ4IiB4Mj0iNyIgeTI9IjQ4IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iNjIiIHgyPSIxMiIgeTI9IjYyIiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iNzYiIHgyPSI3IiB5Mj0iNzYiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSI5MCIgeDI9IjEyIiB5Mj0iOTAiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPGxpbmUgeDE9IjAiIHkxPSIxMDQiIHgyPSI3IiB5Mj0iMTA0IiBzdHJva2U9IiM1QjQ2MzAiIHN0cm9rZS13aWR0aD0iMS42Ii8+CjxsaW5lIHgxPSIwIiB5MT0iMTE4IiB4Mj0iMTIiIHkyPSIxMTgiIHN0cm9rZT0iIzVCNDYzMCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz4KPHJlY3QgeD0iMiIgeT0iNzIiIHdpZHRoPSI4IiBoZWlnaHQ9IjQ0IiBmaWxsPSIjMkY3Qzg2Ii8+PC9nPgo8cGF0aCBkPSJNOTU1IDQ1NSBDIDkwMCA0NzAgODgwIDUwMCA4MjAgNTA1IiBzdHJva2U9IiMyRTNGM0EiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtZGFzaGFycmF5PSIxIDkiIG9wYWNpdHk9Ii44Ii8+CjxwYXRoIGQ9Ik04MDAgNDk4IHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+CjxwYXRoIGQ9Ik03ODUgNTEwIHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+CjxwYXRoIGQ9Ik03NzAgNTAwIHEgNSA4IDAgMTIgcSAtNSAtNCAwIC0xMiB6IiBmaWxsPSIjM0E3Q0E1Ii8+Cjwvc3ZnPg==") center bottom / cover no-repeat, #9FC7CF; }
.hero__texto { max-width:660px; padding:clamp(20px,2.6vw,40px) 4vw 0; }
.hero h1 { font-family:var(--titulo); font-size:clamp(1.8rem,3.4vw,2.9rem); line-height:1.05; letter-spacing:-.02em; margin:0 0 10px; }
.bajada { margin:0 0 12px; max-width:36em; color:#21443D; }
.etiqueta { display:inline-block; font-size:.85rem; padding:3px 12px; border-radius:999px; background:rgba(251,252,251,.8); border:1px solid rgba(23,51,46,.18); }
.contenedor { max-width:1380px; margin:0 auto; padding:24px; display:grid; grid-template-columns:300px 1fr; gap:22px; align-items:start; }
.rail { position:sticky; top:16px; background:var(--tinta); color:#EAF2EE; border-radius:14px; padding:18px 18px 8px; max-height:calc(100vh - 32px); overflow-y:auto; }
.rail__titulo { font-family:var(--titulo); font-size:1.05rem; margin:18px 0 10px; padding-top:14px; border-top:1px solid rgba(234,242,238,.18); }
.rail__titulo:first-child { margin-top:0; padding-top:0; border-top:none; }
.rail__nota { font-size:.82rem; color:#A9C3BA; margin:4px 0 10px; }
.campo { margin-bottom:12px; }
.campo__etiqueta { display:block; font-size:.9rem; font-weight:600; margin-bottom:4px; }
.campo__ayuda { display:block; font-size:.78rem; color:#A9C3BA; margin-top:3px; }
.entrada { width:100%; padding:8px 10px; border-radius:8px; border:1px solid transparent; background:#F3F7F5; font:inherit; color:var(--tinta); }
.entrada:focus { outline:none; border-color:var(--sol); box-shadow:0 0 0 3px rgba(214,158,46,.35); }
.tres { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.rail .Select-control { background:#F3F7F5; border:1px solid transparent; border-radius:8px; min-height:38px; }
.rail .is-focused:not(.is-open) > .Select-control { border-color:var(--sol); box-shadow:0 0 0 3px rgba(214,158,46,.35); }
.rail .Select-value-label { color:var(--tinta) !important; }
.Select-option.is-selected { background:var(--rio); color:#fff; }
.Select-option.is-focused { background:#E1ECE7; }
.rail .rc-slider-rail { background:rgba(234,242,238,.25); }
.rail .rc-slider-track { background:var(--sol); }
.rail .rc-slider-handle { border:3px solid var(--sol); background:var(--tinta); opacity:1; }
.rail .rc-slider-mark-text { color:#A9C3BA; }
.fecha .SingleDatePickerInput { border-radius:8px; overflow:hidden; border:none; }
.fecha .DateInput_input { font:inherit; font-size:15px; padding:7px 10px; color:var(--tinta); background:#F3F7F5; }
.fecha .DateInput_input__focused { border-bottom-color:var(--sol); }
.CalendarDay__selected { background:var(--rio) !important; border-color:var(--rio) !important; }
.segmento { display:flex; background:rgba(234,242,238,.12); border-radius:8px; padding:3px; gap:3px; }
.segmento__opcion { flex:1; text-align:center; padding:7px 4px; border-radius:6px; cursor:pointer; color:#CFE0D9; margin:0 !important; }
.segmento__opcion:has(input:checked) { background:var(--sol); color:var(--tinta); font-weight:700; }
.segmento__radio { position:absolute; opacity:0; width:1px; height:1px; }
.principal { min-width:0; }
.panel { background:var(--panel); border:1px solid var(--borde); border-radius:14px; padding:14px 16px; margin-bottom:18px; min-width:0; }
.titulo { font-family:var(--titulo); font-size:1.25rem; margin:0 0 6px; }
.fila-titulo { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.nota { font-size:.88rem; color:var(--suave); margin:0 0 10px; max-width:80ch; }
.boton { background:var(--rio); color:#fff; border:none; border-radius:8px; padding:8px 14px; font:inherit; font-weight:600; cursor:pointer; }
.boton:hover { background:#185A62; } .boton:focus-visible { outline:3px solid var(--sol); }
.veredicto > div { border-radius:14px; padding:16px 20px; margin-bottom:18px; font-size:1.05rem; border-left:6px solid; }
.veredicto p { margin:0 0 6px; } .veredicto p:last-child { margin:0; }
.veredicto--ok { background:#E3F0EC; border-color:var(--rio) !important; }
.veredicto--alerta { background:#F7E9DC; border-color:var(--tierra) !important; }
.indicadores { display:grid; grid-template-columns:repeat(4,1fr); gap:0; background:var(--panel); border:1px solid var(--borde); border-radius:14px; margin-bottom:18px; }
.ind { padding:14px 18px; border-right:1px solid #E3EAE6; display:flex; flex-direction:column; }
.ind:last-child { border-right:none; }
.ind__valor { font-family:var(--titulo); font-size:1.7rem; font-weight:700; line-height:1.1; }
.ind__texto { font-size:.9rem; } .ind__detalle { font-size:.8rem; color:var(--suave); margin-top:4px; }
.ind--clave .ind__valor { color:var(--rio); }
.grilla { display:grid; grid-template-columns:1fr 1fr; gap:18px; } .grilla > .panel { margin-bottom:18px; }
.tabla-scroll { overflow-x:auto; }
.tabla { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:.93rem; }
.tabla th { text-align:right; font-weight:600; color:var(--suave); padding:6px 8px; border-bottom:2px solid var(--borde); }
.tabla td { text-align:right; padding:7px 8px; border-bottom:1px solid #E3EAE6; }
.tabla th:first-child, .tabla td:first-child { text-align:left; }
.tabla td:first-child { font-weight:600; }
.explicacion summary { cursor:pointer; font-weight:600; } .explicacion p { font-size:.92rem; margin:8px 0 0; max-width:80ch; }
.pie { text-align:center; font-size:.82rem; color:var(--suave); padding:8px 24px 32px; }
.dash-spreadsheet-container .dash-spreadsheet-inner input { font-family:var(--texto) !important; }
@media (max-width:1050px) { .contenedor, .grilla { grid-template-columns:1fr; } .rail { position:static; max-height:none; }
  .indicadores { grid-template-columns:1fr 1fr; } .ind:nth-child(2) { border-right:none; } }

</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

rail = html.Aside([
    html.H2("Tu siembra", className="rail__titulo"),
    campo("Fecha de siembra", dcc.DatePickerSingle(id="fecha", date="2027-04-15", display_format="DD/MM/YYYY",
                                                   first_day_of_week=1, className="fecha")),
    html.H2("Tu finca", className="rail__titulo"),
    campo("Tierra disponible (ha)", num_input("H", 20)),
    campo("Capacidad del reservorio J (m³)", num_input("J", 20000, paso=100),
          "Agua que cabe en jagüeyes, tanques o pozos en condiciones normales."),
    campo("Nivel al sembrar (% de J)", dcc.Slider(0, 100, 5, value=100, id="S0", marks={0: "0", 50: "50", 100: "100"},
                                                  tooltip={"placement": "bottom"})),
    campo("Área que escurre al reservorio (ha)", num_input("Ac", 0.5)),
    campo("Tipo de captación", dcc.Dropdown(
        [{"label": "Lámina del propio reservorio (C = 1,0)", "value": 1.0},
         {"label": "Techos o superficies duras (C = 0,85)", "value": 0.85},
         {"label": "Terreno natural (C = 0,5)", "value": 0.5},
         {"label": "Sin captación de lluvia (C = 0)", "value": 0.0}], 1.0, id="C", clearable=False)),
    campo("Agua externa (m³ por semana)", num_input("Q", 0, paso=10), "Bombeo del río, pozo o carrotanque."),
    campo("Sistema de riego", dcc.Dropdown(
        [{"label": "Goteo (eficiencia 0,90)", "value": 0.90}, {"label": "Aspersión (0,75)", "value": 0.75},
         {"label": "Gravedad o surcos (0,60)", "value": 0.60}], 0.90, id="eta", clearable=False)),
    html.H2("Pronóstico de la temporada", className="rail__titulo"),
    html.P("Probabilidad (%) de que el ciclo sea más seco, normal o más lluvioso que lo habitual. "
           "Sin pronóstico, déjalo en 33 / 34 / 33.", className="rail__nota"),
    html.Div([campo("Seco", num_input("p_seco", 33)), campo("Normal", num_input("p_normal", 34)),
              campo("Lluvioso", num_input("p_lluvioso", 33))], className="tres"),
    html.H2("Seguridad que quieres", className="rail__titulo"),
    dcc.RadioItems([{"label": f"{p}%", "value": p / 100} for p in (80, 90, 95)], 0.95, id="conf",
                   className="segmento", labelClassName="segmento__opcion", inputClassName="segmento__radio"),
    html.P("Porcentaje de los años en que el cultivo debe tener el agua asegurada.", className="rail__nota"),
], className="rail")

tabla_cultivos = html.Div([
    html.Div([html.H2("Cultivos que quieres sembrar", className="titulo"),
              html.Button("+ Agregar cultivo", id="agregar", n_clicks=0, className="boton")], className="fila-titulo"),
    html.P("Escribe las hectáreas de cada cultivo (0 = no se siembra). Los coeficientes vienen de FAO-56 como referencia; "
           "ajústalos a tu variedad. Si conoces el consumo diario por hectárea, escríbelo en la última columna y se usa en "
           "lugar de Kc × ET0.", className="nota"),
    dash_table.DataTable(
        id="cultivos", columns=COLUMNAS, data=FILAS_INICIALES, editable=True, row_deletable=True, merge_duplicate_headers=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#E6EDE9", "fontWeight": 600, "border": "none", "color": TINTA,
                      "fontFamily": "Figtree, sans-serif", "textAlign": "center"},
        style_cell={"fontFamily": "Figtree, sans-serif", "fontSize": 14, "padding": "8px 10px", "border": "none",
                    "borderBottom": "1px solid #E3EAE6", "color": TINTA, "minWidth": 70},
        style_cell_conditional=[{"if": {"column_id": "cultivo"}, "textAlign": "left", "minWidth": 140, "fontWeight": 600}],
        style_data_conditional=[{"if": {"filter_query": "{hectareas} > 0"}, "backgroundColor": "#EEF5F1"},
                                {"if": {"column_id": "hectareas"}, "backgroundColor": "#FFF6E3", "fontWeight": 700}],
    ),
], className="panel")

app.layout = html.Div([
    html.Header(html.Div([
        html.H1("¿Cuánta agua necesita mi siembra?"),
        html.P("Elige cultivos, hectáreas y fecha. El planificador simula el ciclo con el clima de cada año desde 1981 "
               "y te dice cuánta agua reservar y cuánto puedes sembrar.", className="bajada"),
        html.Span(f"Clima: {FUENTE}", className="etiqueta"),
    ], className="hero__texto"), className="hero"),
    html.Main([
        rail,
        html.Div([
            tabla_cultivos,
            dcc.Loading(html.Div([
                html.Div(id="veredicto", className="veredicto"),
                html.Div(id="indicadores", className="indicadores"),
                html.Div(id="resumen", className="panel"),
                html.Div([html.Div(dcc.Graph(id="g-demanda", config=CONFIG), className="panel"),
                          html.Div(dcc.Graph(id="g-reservorio", config=CONFIG), className="panel")], className="grilla"),
                html.Div([html.Div(dcc.Graph(id="g-anios", config=CONFIG), className="panel"),
                          html.Div(dcc.Graph(id="g-fecha", config=CONFIG), className="panel")], className="grilla"),
                html.Details([html.Summary("¿Cómo se calcula?"), html.Div([
                    html.P("1. Consumo del cultivo por hectárea y día: 10 × Kc × ET0 (m³/ha/día), o el valor fijo que escribas."),
                    html.P("2. La lluvia efectiva que cae sobre el cultivo se descuenta; lo que falta, dividido por la eficiencia "
                           "del riego, es el agua que debe salir del reservorio."),
                    html.P("3. El reservorio (capacidad J) sube con la lluvia del área de captación y el agua externa, y baja con el riego."),
                    html.P("4. Esto se repite con el clima real de cada año desde 1981, empezando en tu fecha de siembra. "
                           "El pronóstico de temporada da más peso a los años secos, normales o lluviosos según lo que indiques."),
                    html.P("5. La reserva necesaria es el mayor déficit acumulado del ciclo (algoritmo del pico secuente); "
                           "se reporta el valor que cubre el porcentaje de años que elegiste."),
                ])], className="panel explicacion"),
            ]), type="circle", color=RIO),
        ], className="principal"),
    ], className="contenedor"),
    html.Footer("Coeficientes de referencia: FAO-56 (Allen et al., 1998). Clima: NASA POWER. "
                "Herramienta de apoyo a la decisión; valida los coeficientes con un agrónomo de la zona.", className="pie"),
])


@app.callback(Output("cultivos", "data"), Input("agregar", "n_clicks"), State("cultivos", "data"), prevent_initial_call=True)
def agregar_cultivo(_, filas):
    filas = list(filas or [])
    filas.append({"cultivo": f"Cultivo {len(filas) + 1}", "hectareas": 0, "Kc_ini": 0.4, "Kc_mid": 1.0, "Kc_end": 0.5,
                  "L_ini": 20, "L_dev": 30, "L_mid": 40, "L_late": 30, "c_fijo": None})
    return filas


def _indicador(valor, texto, detalle="", clase=""):
    return html.Div([html.Span(valor, className="ind__valor"), html.Span(texto, className="ind__texto"),
                     html.Span(detalle, className="ind__detalle")], className=f"ind {clase}".strip())


@app.callback(
    Output("veredicto", "children"), Output("indicadores", "children"), Output("resumen", "children"),
    Output("g-demanda", "figure"), Output("g-reservorio", "figure"), Output("g-anios", "figure"), Output("g-fecha", "figure"),
    Input("cultivos", "data"), Input("fecha", "date"), Input("H", "value"), Input("J", "value"), Input("S0", "value"),
    Input("Ac", "value"), Input("C", "value"), Input("Q", "value"), Input("eta", "value"),
    Input("p_seco", "value"), Input("p_normal", "value"), Input("p_lluvioso", "value"), Input("conf", "value"))
def calcular(filas, fecha, H, J, S0, Ac, C, Q, eta, ps, pn, pl, conf):
    vacio = go.Figure().update_layout(xaxis_visible=False, yaxis_visible=False)
    fecha = pd.Timestamp(fecha or "2027-04-15")
    J = _num(J, 0); H = _num(H, 0)
    finca = {"J": J, "S0": J * _num(S0, 100) / 100, "A_c": _num(Ac, 0), "C": _num(C, 0),
             "Q_ext": _num(Q, 0), "eta": _num(eta, 0.9)}
    pron = (_num(ps, 0), _num(pn, 0), _num(pl, 0))
    if sum(pron) <= 0:
        pron = (1, 1, 1)
    cultivos = [dict(f) for f in (filas or []) if _num(f.get("hectareas"), 0) > 0 and duracion(f) > 0]
    if not cultivos:
        return (html.P("Escribe al menos un cultivo con hectáreas mayores que cero."), [], [], vacio, vacio, vacio, vacio)
    res = evaluar(SERIE, [dict(c) for c in cultivos], fecha, finca, LLUVIA_EF, pron, conf)
    A_max = area_maxima(SERIE, cultivos, fecha, finca, LLUVIA_EF, pron, conf, H) if H > 0 else 0
    area = sum(c["hectareas"] for c in res["cultivos"])
    pct = f"{conf:.0%}".replace("%", " %")
    meses = res["largo"] / 30.4

    # ---------- Veredicto ----------
    lista = ", ".join(f"{miles(c['hectareas'], 1 if c['hectareas'] % 1 else 0)} ha de {c['cultivo']}" for c in res["cultivos"])
    alcanza = res["reserva_conf"] <= finca["S0"] + 1e-6
    veredicto = [
        html.P([f"Para sembrar {lista} el {fecha_es(fecha)}, durante unos {miles(meses, 1)} meses de ciclo, necesitas tener ",
                html.Strong(f"{miles(res['reserva_conf'])} m³ de reserva"),
                f" para asegurar el agua en el {pct} de los años."]),
        html.P([("Tu reservorio alcanza. " if alcanza else "Tu reservorio no alcanza para ese nivel de seguridad. "),
                f"Con {miles(finca['S0'])} m³ al sembrar, el cultivo tiene el agua asegurada en ",
                html.Strong(f"{res['prob_exito']:.0%}".replace("%", " %")), " de los años. Con esa misma proporción entre cultivos puedes sembrar hasta ",
                html.Strong(f"{miles(A_max, 1)} ha"), f" con {pct} de seguridad."]),
    ]
    clase = "veredicto--ok" if alcanza else "veredicto--alerta"

    # ---------- Indicadores ----------
    ind = [
        _indicador(f"{miles(res['demanda_total'])} m³", "consumen los cultivos en el ciclo", "promedio de los escenarios"),
        _indicador(f"{miles(res['riego_conf'])} m³", f"de riego en un año seco ({pct})", f"mediana: {miles(res['riego_mediana'])} m³"),
        _indicador(f"{miles(res['reserva_conf'])} m³", "de reserva necesaria al sembrar", f"tu reservorio: {miles(finca['S0'])} m³",
                   "ind--clave"),
        _indicador(f"{miles(A_max, 1)} ha", f"máximo sembrable al {pct}", f"de {miles(H)} ha disponibles"),
    ]

    # ---------- Resumen por cultivo ----------
    w = res["pesos"]
    filas_res = []
    for c in res["cultivos"]:
        nombre = c["cultivo"]
        dem = np.array([s["demanda"][nombre].sum() for s in res["sims"]])
        rie = np.array([s["riego"][nombre].sum() for s in res["sims"]])
        dias = duracion(c)
        filas_res.append(html.Tr([
            html.Td(nombre), html.Td(f"{dias} días"), html.Td(miles(np.average(dem, weights=w))),
            html.Td(miles(np.average(dem, weights=w) / dias, 1)),
            html.Td(miles(cuantil_ponderado(rie, w, 0.5))), html.Td(miles(cuantil_ponderado(rie, w, conf))),
            html.Td(miles(cuantil_ponderado(rie, w, conf) * c["hectareas"])),
        ]))
    resumen = [html.H3("Agua por cultivo", className="titulo"),
               html.Div(html.Table([html.Thead(html.Tr([html.Th(t) for t in [
                   "Cultivo", "Ciclo", "Consumo (m³/ha)", "Consumo diario (m³/ha/día)", "Riego año normal (m³/ha)",
                   f"Riego año seco {pct} (m³/ha)", f"Riego total {pct} (m³)"]])), html.Tbody(filas_res)],
                   className="tabla"), className="tabla-scroll")]

    # ---------- Figura: demanda semanal ----------
    T = len(res["sims"][0]["uso"])
    semanas = [fecha + pd.Timedelta(weeks=t) for t in range(T)]
    fig_d = go.Figure()
    for i, c in enumerate(res["cultivos"]):
        prom = np.average([s["demanda"][c["cultivo"]] * c["hectareas"] for s in res["sims"]], axis=0, weights=w)
        fig_d.add_trace(go.Scatter(x=semanas, y=prom, name=c["cultivo"], stackgroup="dem", mode="lines",
                                   line=dict(width=0.5, color=COLORES_CULTIVO[i % len(COLORES_CULTIVO)]),
                                   hovertemplate="%{x|%d/%m}: %{y:,.0f} m³<extra>" + c["cultivo"] + "</extra>"))
    lluvia_area = np.average([M3_POR_MM_HA * s["lluvia_ef"] * area for s in res["sims"]], axis=0, weights=w)
    fig_d.add_trace(go.Scatter(x=semanas, y=lluvia_area, name="Lluvia efectiva sobre lo sembrado", mode="lines",
                               line=dict(color=LLUVIA, width=2.5, dash="dot")))
    fig_d.update_layout(title="Cuándo necesita agua cada cultivo", yaxis_title="m³ por semana", height=400)

    # ---------- Figura: reservorio ----------
    niveles = np.array([s["nivel"] for s in res["sims"]])
    q = {k: np.array([cuantil_ponderado(niveles[:, t], w, k) for t in range(T)]) for k in (1 - conf, 0.5, conf)}
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=semanas, y=q[conf], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_r.add_trace(go.Scatter(x=semanas, y=q[1 - conf], fill="tonexty", fillcolor="rgba(31,111,120,0.18)",
                               line=dict(width=0), name=f"rango de años ({1 - conf:.0%}–{conf:.0%})"))
    fig_r.add_trace(go.Scatter(x=semanas, y=q[0.5], name="año típico", line=dict(color=RIO, width=3)))
    fig_r.add_hline(y=J, line_dash="dash", line_color=SALVIA, annotation_text="capacidad J")
    fig_r.add_hline(y=0, line_color=TIERRA, annotation_text="reservorio vacío", annotation_position="bottom right")
    fig_r.update_layout(title="Nivel del reservorio durante el ciclo", yaxis_title="m³", height=400)

    # ---------- Figura: años ----------
    reservas = np.array([s["reserva_necesaria"] for s in res["sims"]])
    col = np.where(reservas <= finca["S0"], "#9CC2B4", TIERRA)
    etiqueta_t = np.array(["seco", "normal", "lluvioso"])[res["tercil"]]
    fig_a = go.Figure(go.Bar(x=[str(a) for a in res["anios"]], y=reservas, marker_color=col,
                             customdata=np.c_[res["lluvia_ciclo"], etiqueta_t, w * 100],
                             hovertemplate="%{x}: %{y:,.0f} m³<br>lluvia del ciclo %{customdata[0]:.0f} mm (%{customdata[1]})"
                                           "<br>peso %{customdata[2]:.1f} %<extra></extra>"))
    fig_a.add_hline(y=finca["S0"], line_dash="dash", line_color=TINTA, annotation_text="agua al sembrar")
    fig_a.update_layout(title="Reserva que habría hecho falta cada año (rojo = no alcanza)",
                        yaxis_title="m³", height=400, bargap=0.15, xaxis=dict(tickangle=-60, tickfont=dict(size=10)))

    # ---------- Figura: mejor fecha ----------
    mf = mejor_fecha(SERIE, cultivos, finca, LLUVIA_EF, pron, conf, anio_ref=fecha.year)
    mejor = mf.loc[mf.reserva_por_ha.idxmin()]
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=mf.fecha, y=mf.reserva_por_ha, mode="lines+markers", line=dict(color=SOL, width=3),
                               marker=dict(size=7), name="reserva por hectárea",
                               hovertemplate="siembra %{x|%d/%m}: %{y:,.0f} m³/ha<extra></extra>"))
    fig_f.add_trace(go.Scatter(x=[fecha], y=[res["reserva_conf"] / area], mode="markers", name="tu fecha",
                               marker=dict(size=14, color=RIO, symbol="diamond")))
    fig_f.add_annotation(x=mejor.fecha, y=mejor.reserva_por_ha, text=f"menos agua: {fecha_es(mejor.fecha)}",
                         showarrow=True, arrowhead=0, ay=-40, bgcolor="white", bordercolor=SOL)
    fig_f.update_layout(title=f"¿Qué fecha de siembra necesita menos reserva? ({pct})", yaxis_title="m³ por hectárea",
                        height=400, xaxis=dict(tickformat="%d/%m"))
    return (html.Div(veredicto, className=clase), ind, resumen, fig_d, fig_r, fig_a, fig_f)


if __name__ == "__main__":
    print(f"Clima: {FUENTE}. Abre http://127.0.0.1:8050 en el navegador.")
    app.run(debug=False, port=8050)
