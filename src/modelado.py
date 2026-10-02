"""
Construcción del problema de pronóstico, partición, líneas base y métricas.

Unidad de observación: (punto, día de emisión t).
Objetivo para el horizonte h (semanas): lluvia acumulada en la semana h después de t,
    y_h(t) = sum_{d = 7(h-1)+1}^{7h} P_{t+d}   [mm]
Todas las predictoras usan información hasta t - LATENCIA_DIAS.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from . import config as cfg

OBJ = "PRECTOTCORR"


# --------------------------------------------------------------------------
# Objetivo y predictoras
# --------------------------------------------------------------------------
def objetivo_semana_h(P, h, dias=cfg.DIAS_POR_SEMANA):
    """Suma de P en (t + 7(h-1), t + 7h]. P: Serie diaria de un punto, índice fecha."""
    futuro = P[::-1].rolling(dias, min_periods=dias).sum()[::-1]  # suma de t..t+6
    return futuro.shift(-(dias * (h - 1) + 1))


def ingenuo_estacional(P, h, dias=cfg.DIAS_POR_SEMANA):
    """Misma ventana objetivo, un año antes (observable en t si 7h <= 365)."""
    return objetivo_semana_h(P, h, dias).shift(365)


def construir_predictoras(df_punto, latencia=cfg.LATENCIA_DIAS):
    """
    Predictoras de un punto, solo con pasado. df_punto: columnas POWER, índice fecha.
    """
    d = df_punto
    X = pd.DataFrame(index=d.index)
    P = d[OBJ]
    for v in [7, 30, 90, 180, 365]:
        X[f"P_suma_{v}d"] = P.rolling(v, min_periods=int(0.8 * v)).sum()
    X["dias_secos_consecutivos"] = (
        P.lt(1).groupby(P.ge(1).cumsum()).cumsum()
    )
    for var in ["T2M", "T2M_MAX", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN",
                "GWETROOT", "GWETTOP", "ET0"]:
        if var in d:
            X[f"{var}_media_7d"] = d[var].rolling(7, min_periods=5).mean()
            X[f"{var}_media_30d"] = d[var].rolling(30, min_periods=24).mean()
    # (el déficit de 30 días se descartó: es combinación lineal exacta de ET0 y lluvia de 30 días, VIF > 1000)
    if "ONI" in d:
        X["ONI"] = d["ONI"]
    # Rezago operativo: lo que se sabe en t es lo observado hasta t - latencia
    return X.shift(latencia)


def componentes_ciclicos(fechas, h, dias=cfg.DIAS_POR_SEMANA, armonicos=cfg.ARMONICOS):
    """
    Armónicos (seno/coseno) del día del año del centro de la ventana objetivo.
    Es calendario conocido de antemano. Con un solo armónico un modelo lineal no puede
    representar el ciclo bimodal; por eso se usan hasta `armonicos`.
    """
    centro = pd.DatetimeIndex(fechas) + pd.to_timedelta(dias * h - dias / 2, unit="D")
    ang = 2 * np.pi * centro.dayofyear / 365.25
    out = {"sin_doy": np.sin(ang), "cos_doy": np.cos(ang)}
    for k in range(2, armonicos + 1):
        out[f"sin{k}_doy"] = np.sin(k * ang)
        out[f"cos{k}_doy"] = np.cos(k * ang)
    return pd.DataFrame(out, index=pd.DatetimeIndex(fechas))


def tabla_modelado(df, h):
    """Una fila por (punto, fecha de emisión) con predictoras, objetivo y bases."""
    partes = []
    for punto, g in df.groupby("punto"):
        g = g.set_index("fecha").sort_index()
        X = construir_predictoras(g)
        X = X.join(componentes_ciclicos(X.index, h))
        X["lat"], X["lon"], X["elevacion_m"] = g["lat"], g["lon"], g["elevacion_m"]
        X["y"] = objetivo_semana_h(g[OBJ], h)
        X["base_ingenuo_estacional"] = ingenuo_estacional(g[OBJ], h)
        X["punto"] = punto
        if cfg.EMISION_SEMANAL:
            X = X[X.index.dayofweek == 0]
        partes.append(X.reset_index())
    return pd.concat(partes, ignore_index=True).dropna(subset=["y"])


# --------------------------------------------------------------------------
# Partición cronológica con purga
# --------------------------------------------------------------------------
def particion_cronologica(tabla, h, inicio_test=cfg.INICIO_TEST,
                          dias=cfg.DIAS_POR_SEMANA):
    """
    Train: filas cuya ventana objetivo termina antes de inicio_test (purga).
    Test : filas emitidas desde inicio_test.
    """
    corte = pd.Timestamp(inicio_test)
    fin_objetivo = tabla["fecha"] + pd.to_timedelta(dias * h, unit="D")
    train = tabla[fin_objetivo < corte]
    test = tabla[tabla["fecha"] >= corte]
    return train.copy(), test.copy()


def particion_sitio_futuro(tabla, h, sitio_prueba=cfg.PUNTO_OBJETIVO,
                           inicio_test=cfg.INICIO_TEST, dias=cfg.DIAS_POR_SEMANA):
    """
    Esquema principal: entrenamiento con los DEMÁS sitios antes de inicio_test
    (con purga) y prueba en el sitio de la finca desde inicio_test.
    Los datos de otros sitios en el periodo de prueba no se usan: comparten
    los mismos eventos climáticos y producirían fuga.
    """
    corte = pd.Timestamp(inicio_test)
    fin_objetivo = tabla["fecha"] + pd.to_timedelta(dias * h, unit="D")
    train = tabla[(tabla["punto"] != sitio_prueba) & (fin_objetivo < corte)]
    test = tabla[(tabla["punto"] == sitio_prueba) & (tabla["fecha"] >= corte)]
    return train.copy(), test.copy()


def cv_temporal(fechas, n_splits=cfg.N_SPLITS_CV, gap_dias=0):
    """
    Ventana creciente sobre fechas únicas (varios puntos comparten fecha).
    Deja gap_dias entre el final de train y el inicio de validación.
    """
    fechas = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    unicas = np.sort(fechas.unique())
    bloques = np.array_split(unicas, n_splits + 1)
    for k in range(1, n_splits + 1):
        fin_train = bloques[k - 1][-1]
        val = bloques[k]
        tr_idx = np.where(fechas <= fin_train - np.timedelta64(gap_dias, "D"))[0]
        va_idx = np.where(fechas.isin(val))[0]
        yield tr_idx, va_idx


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def columnas_predictoras(tabla):
    """Predictoras; se descartan columnas constantes (p. ej. lat/lon con un solo sitio)."""
    excluir = {"fecha", "punto", "y", "base_ingenuo_estacional"}
    return [c for c in tabla.columns if c not in excluir and tabla[c].nunique(dropna=True) > 1]


def pipeline_svr(C=1.0, epsilon=0.0, log_objetivo=False, loss="squared_epsilon_insensitive"):
    """
    SVR lineal en Pipeline. Por defecto usa pérdida cuadrática (estima la media
    condicional, coherente con RMSE); la pérdida epsilon-insensible estima algo
    cercano a la mediana y subestima una variable tan asimétrica como la lluvia.
    """
    modelo = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("svr", LinearSVR(C=C, epsilon=epsilon, loss=loss, max_iter=50000,
                          random_state=cfg.SEED, dual="auto")),
    ])
    if log_objetivo:
        modelo = TransformedTargetRegressor(regressor=modelo,
                                            func=np.log1p, inverse_func=np.expm1)
    return modelo


def ajustar_svr(train, cols, splits,
                Cs=(0.001, 0.01, 0.1, 1.0),
                losses=("squared_epsilon_insensitive", "epsilon_insensitive"),
                logs=(False, True)):
    """
    Búsqueda de hiperparámetros con validación cruzada temporal: C, tipo de pérdida
    y transformación log1p del objetivo. Devuelve (modelo ajustado con todo train,
    mejores parámetros, RMSE medio de validación, tabla de resultados).
    """
    from sklearn.metrics import mean_squared_error
    filas = []
    for log in logs:
        for loss in losses:
            for C in Cs:
                errs = []
                for tr_i, va_i in splits:
                    m = pipeline_svr(C=C, log_objetivo=log, loss=loss).fit(train.iloc[tr_i][cols], train.iloc[tr_i]["y"])
                    errs.append(np.sqrt(mean_squared_error(train.iloc[va_i]["y"], m.predict(train.iloc[va_i][cols]))))
                filas.append({"log1p": log, "loss": loss, "C": C, "RMSE_cv": float(np.mean(errs))})
    tabla = pd.DataFrame(filas).sort_values("RMSE_cv").reset_index(drop=True)
    mejor = tabla.iloc[0]
    modelo = pipeline_svr(C=mejor.C, log_objetivo=bool(mejor.log1p), loss=mejor.loss).fit(train[cols], train["y"])
    return modelo, mejor.to_dict(), float(mejor.RMSE_cv), tabla


def pipeline_dummy():
    return DummyRegressor(strategy="mean")


def climatologia(train, dias=cfg.DIAS_POR_SEMANA, h=1):
    """Media de train por semana del año del inicio de la ventana objetivo."""
    inicio = train["fecha"] + pd.to_timedelta(dias * (h - 1) + 1, unit="D")
    semana = inicio.dt.isocalendar().week.clip(upper=52)
    tabla = train.groupby(semana.values)["y"].mean()

    def predecir(df):
        ini = df["fecha"] + pd.to_timedelta(dias * (h - 1) + 1, unit="D")
        s = ini.dt.isocalendar().week.clip(upper=52)
        return s.map(tabla).astype(float).values
    return predecir


# --------------------------------------------------------------------------
# Métricas e incertidumbre
# --------------------------------------------------------------------------
def metricas(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    pos = y > 0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "MAE": float(mean_absolute_error(y, yhat)),
        "WAPE": float(np.abs(y - yhat).sum() / np.abs(y).sum()),
        "MAPE_y>0": float(np.mean(np.abs((y[pos] - yhat[pos]) / y[pos]))) if pos.any() else np.nan,
        "R2": float(r2_score(y, yhat)),
    }


def bootstrap_bloques(y, yhat, yref=None, largo_bloque=60, n_boot=1000,
                      confianza=cfg.CONFIANZA, seed=cfg.SEED):
    """
    Bootstrap por bloques móviles (respeta autocorrelación).
    Si se pasa yref (climatología), también calcula el skill score SS = 1 - RMSE/RMSE_ref.
    """
    rng = np.random.default_rng(seed)
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    n = len(y)
    n_bloques = int(np.ceil(n / largo_bloque))
    resultados = []
    for _ in range(n_boot):
        inicios = rng.integers(0, max(1, n - largo_bloque), n_bloques)
        idx = np.concatenate([np.arange(s, s + largo_bloque) for s in inicios])[:n]
        m = metricas(y[idx], yhat[idx])
        if yref is not None:
            rmse_ref = np.sqrt(mean_squared_error(y[idx], np.asarray(yref)[idx]))
            m["SS_clim"] = 1 - m["RMSE"] / rmse_ref
        resultados.append(m)
    r = pd.DataFrame(resultados)
    a = (1 - confianza) / 2
    return pd.DataFrame({"inf": r.quantile(a), "sup": r.quantile(1 - a)})


def horizonte_confiable(tabla_skill):
    """
    tabla_skill: DataFrame con columnas h, SS_inf (límite inferior IC del skill score).
    n = mayor h tal que SS_inf > 0 en todos los horizontes hasta h.
    """
    t = tabla_skill.sort_values("h")
    n = 0
    for _, fila in t.iterrows():
        if fila["SS_inf"] > 0:
            n = int(fila["h"])
        else:
            break
    return n


def cuantiles_conformales(y_cal, yhat_cal, niveles=(0.05, 0.95)):
    """Cuantiles empíricos de residuos en calibración (conformal split)."""
    r = np.asarray(y_cal) - np.asarray(yhat_cal)
    return {q: float(np.quantile(r, q)) for q in niveles}


def aplicar_intervalo(yhat, cuantiles):
    """Devuelve predicciones por cuantil, truncadas en 0 (lluvia no negativa)."""
    return {q: np.maximum(0, np.asarray(yhat) + c) for q, c in cuantiles.items()}


def semanal_por_punto(df, variables, sumar=(OBJ, "ET0")):
    """
    Agregado semanal (semanas lunes-domingo) por punto: suma para lluvia y ET0,
    media para el resto. Descarta semanas incompletas (bordes del periodo).
    """
    reglas = {v: ("sum" if v in sumar else "mean") for v in variables}
    partes = []
    for punto, g in df.groupby("punto"):
        g = g.set_index("fecha").sort_index()
        w = g[variables].resample("W").agg(reglas)
        dias = g[variables[0]].resample("W").size()
        w = w[dias == 7]
        w.insert(0, "punto", punto)
        partes.append(w.reset_index())
    return pd.concat(partes, ignore_index=True)
