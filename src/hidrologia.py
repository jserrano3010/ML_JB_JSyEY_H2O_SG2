"""
Balance hídrico del cultivo y del almacenamiento de la finca.

Unidades: lámina en mm, volumen en m³, área en ha. 1 mm sobre 1 ha = 10 m³.

Notación (ver capítulo "Marco hídrico y de decisión"):
    P_t      lluvia de la semana t [mm]
    Pe_t     lluvia efectiva sobre el cultivo [mm]
    ET0_t    evapotranspiración de referencia [mm]
    Kc_i,t   coeficiente del cultivo i en la semana t [-]
    NR_i,t   necesidad de riego bruta del cultivo i [m³/ha]
    J        capacidad de almacenamiento [m³]
    S_t      nivel del almacenamiento al final de la semana t [m³]
    A_c, C   área de captación [ha] y coeficiente de escorrentía [-]
    Q_ext    entrada externa semanal [m³]
    eta      eficiencia de riego [-]
    H        tierra disponible [ha]
    x_i      hectáreas sembradas del cultivo i [ha]
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

M3_POR_MM_HA = 10.0


def _requerir(**kwargs):
    faltan = [k for k, v in kwargs.items() if v is None]
    if faltan:
        raise ValueError(f"Faltan parámetros por definir: {', '.join(faltan)}")


# --------------------------------------------------------------------------
# Evapotranspiración de referencia FAO-56 Penman-Monteith (diaria)
# --------------------------------------------------------------------------
def _presion_saturacion(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def radiacion_extraterrestre(lat_grados, dia_juliano):
    """Ra [MJ/m²/día], FAO-56 ec. 21."""
    phi = np.deg2rad(lat_grados)
    J = np.asarray(dia_juliano, dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def et0_penman_monteith(tmax, tmin, rh, u2, rs, lat, dia_juliano,
                        elevacion_m=0.0, presion_kpa=None):
    """
    ET0 diaria [mm/día] con las variables de NASA POWER:
    T2M_MAX, T2M_MIN [°C], RH2M [%], WS2M [m/s], ALLSKY_SFC_SW_DWN [MJ/m²/día], PS [kPa].
    """
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    tmean = (tmax + tmin) / 2
    if presion_kpa is None:
        presion_kpa = 101.3 * ((293 - 0.0065 * elevacion_m) / 293) ** 5.26
    gamma = 0.000665 * np.asarray(presion_kpa, float)
    es = (_presion_saturacion(tmax) + _presion_saturacion(tmin)) / 2
    ea = np.asarray(rh, float) / 100 * es
    delta = 4098 * _presion_saturacion(tmean) / (tmean + 237.3) ** 2
    ra = radiacion_extraterrestre(lat, dia_juliano)
    rso = (0.75 + 2e-5 * elevacion_m) * ra
    rs = np.asarray(rs, float)
    rns = (1 - 0.23) * rs
    sigma = 4.903e-9
    rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) \
        * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = rns - rnl
    u2 = np.asarray(u2, float)
    num = 0.408 * delta * rn + gamma * (900 / (tmean + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return np.maximum(num / den, 0)


# --------------------------------------------------------------------------
# Lluvia efectiva y coeficiente de cultivo
# --------------------------------------------------------------------------
def lluvia_efectiva(P, alpha, P_min):
    """Pe = alpha * P si P >= P_min, si no 0. P en mm (diaria o semanal)."""
    _requerir(alpha=alpha, P_min=P_min)
    P = np.asarray(P, float)
    return np.where(P >= P_min, alpha * P, 0.0)


def curva_kc(Kc_ini, Kc_mid, Kc_end, L_ini, L_dev, L_mid, L_late, **_):
    """Kc diario durante el ciclo (FAO-56, curva por etapas)."""
    _requerir(Kc_ini=Kc_ini, Kc_mid=Kc_mid, Kc_end=Kc_end,
              L_ini=L_ini, L_dev=L_dev, L_mid=L_mid, L_late=L_late)
    ini = np.full(L_ini, Kc_ini)
    dev = np.linspace(Kc_ini, Kc_mid, L_dev + 1)[1:]
    mid = np.full(L_mid, Kc_mid)
    late = np.linspace(Kc_mid, Kc_end, L_late + 1)[1:]
    return np.concatenate([ini, dev, mid, late])


def kc_semanal(cultivo, dias_por_semana=7):
    """Kc promedio por semana del ciclo."""
    kc = curva_kc(**cultivo)
    n = int(np.ceil(len(kc) / dias_por_semana))
    kc = np.pad(kc, (0, n * dias_por_semana - len(kc)), constant_values=np.nan)
    return np.nanmean(kc.reshape(n, dias_por_semana), axis=1)


def necesidad_riego(kc_t, et0_t, pe_t, eta):
    """
    NR_i,t = 10 * max(0, Kc_i,t * ET0_t - Pe_t) / eta   [m³/ha]
    kc_t, et0_t, pe_t en la misma escala temporal (semanal) y mm.
    """
    _requerir(eta=eta)
    deficit = np.maximum(0, np.asarray(kc_t) * np.asarray(et0_t) - np.asarray(pe_t))
    return M3_POR_MM_HA * deficit / eta


# --------------------------------------------------------------------------
# Almacenamiento (modelo de tanque)
# --------------------------------------------------------------------------
def entrada_tanque(P_t, A_c, C, Q_ext=0.0):
    """Entrada semanal = 10 * P_t * A_c * C + Q_ext  [m³]."""
    _requerir(A_c=A_c, C=C)
    return M3_POR_MM_HA * np.asarray(P_t, float) * A_c * C + Q_ext


def simular_tanque(x, NR, P_t, J, A_c, C, Q_ext=0.0, S0=None):
    """
    S_t = min(J, S_{t-1} + 10*P_t*A_c*C + Q_ext - sum_i NR_i,t * x_i),  S_0 = J.

    x : array (m,) hectáreas por cultivo
    NR: array (m, T) necesidad de riego [m³/ha]
    Devuelve DataFrame con nivel, entrada, consumo y rebose por semana.
    """
    _requerir(J=J)
    x = np.atleast_1d(np.asarray(x, float))
    NR = np.atleast_2d(np.asarray(NR, float))
    entrada = entrada_tanque(P_t, A_c, C, Q_ext)
    consumo = x @ NR
    S = J if S0 is None else S0
    filas = []
    for t in range(NR.shape[1]):
        bruto = S + entrada[t] - consumo[t]
        S = min(J, bruto)
        filas.append({"semana": t + 1, "entrada_m3": entrada[t], "consumo_m3": consumo[t],
                      "rebose_m3": max(0.0, bruto - J), "nivel_m3": S})
    return pd.DataFrame(filas)


def area_maxima_un_cultivo(NR, P_t, H, J, A_c, C, Q_ext=0.0, tol=1e-3):
    """Máximo x en [0, H] tal que el tanque nunca queda en negativo (búsqueda binaria)."""
    _requerir(H=H)
    def factible(x):
        return (simular_tanque([x], NR, P_t, J, A_c, C, Q_ext)["nivel_m3"] >= 0).all()
    if factible(H):
        return float(H)
    lo, hi = 0.0, float(H)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if factible(mid) else (lo, mid)
    return lo


def optimizar_cultivos(NR, P_t, b, H, J, A_c, C, Q_ext=0.0):
    """
    Programa lineal para varios cultivos.

        max  sum_i b_i x_i
        s.a. sum_i x_i <= H
             S_t = S_{t-1} + E_t - sum_i NR_i,t x_i - V_t      (S_0 = J)
             0 <= S_t <= J,  V_t >= 0 (rebose),  x_i >= 0

    Variables: [x_1..x_m, S_1..S_T, V_1..V_T]
    """
    _requerir(H=H, J=J)
    NR = np.atleast_2d(np.asarray(NR, float))
    m, T = NR.shape
    E = entrada_tanque(P_t, A_c, C, Q_ext)
    n = m + 2 * T
    c = np.zeros(n)
    c[:m] = -np.asarray(b, float)                 # linprog minimiza

    A_eq = np.zeros((T, n))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, :m] = NR[:, t]                    # consumo
        A_eq[t, m + t] = 1.0                      # S_t
        if t > 0:
            A_eq[t, m + t - 1] = -1.0             # -S_{t-1}
        A_eq[t, m + T + t] = 1.0                  # V_t
        b_eq[t] = E[t] + (J if t == 0 else 0.0)

    A_ub = np.zeros((1, n))
    A_ub[0, :m] = 1.0
    bounds = [(0, None)] * m + [(0, J)] * T + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=[H], A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x[:m]
    reserva = float((x @ NR).sum())
    return {"x": x, "Z": -res.fun, "nivel": res.x[m:m + T],
            "rebose": res.x[m + T:], "agua_riego_total_m3": reserva}
