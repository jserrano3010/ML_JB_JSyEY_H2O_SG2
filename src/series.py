"""
Herramientas de series de tiempo con respaldo propio.

Si statsmodels está instalado se usan sus implementaciones (STL, ADF, KPSS, VIF,
Holm); si no, se usan implementaciones equivalentes de este módulo, para que los
notebooks corran en cualquier entorno.
"""
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels  # noqa: F401
    HAY_STATSMODELS = True
except ImportError:
    HAY_STATSMODELS = False


def descomponer(serie, periodo=52):
    """Tendencia, estacionalidad y residuo. STL robusto si hay statsmodels; si no, descomposición clásica."""
    if HAY_STATSMODELS:
        from statsmodels.tsa.seasonal import STL
        r = STL(serie, period=periodo, robust=True).fit()
        return pd.DataFrame({"observada": serie, "tendencia": r.trend, "estacional": r.seasonal, "residuo": r.resid}), "STL"
    tend = serie.rolling(periodo, center=True, min_periods=int(0.75 * periodo)).mean()
    fase = np.arange(len(serie)) % periodo
    det = serie - tend
    est = pd.Series(det.groupby(fase).transform("mean").values, index=serie.index)
    est = est - est.mean()
    return pd.DataFrame({"observada": serie, "tendencia": tend, "estacional": est,
                         "residuo": serie - tend - est}), "clásica (media móvil)"


def fuerza(desc):
    r = desc["residuo"].dropna()
    fe = max(0, 1 - r.var() / (desc["estacional"] + desc["residuo"]).dropna().var())
    ft = max(0, 1 - r.var() / (desc["tendencia"] + desc["residuo"]).dropna().var())
    return fe, ft


def kpss(x):
    """Estadístico KPSS de nivel y valor p aproximado (tabla de Kwiatkowski et al., 1992)."""
    x = np.asarray(pd.Series(x).dropna(), float)
    if HAY_STATSMODELS:
        from statsmodels.tsa.stattools import kpss as _k
        est, p, *_ = _k(x, regression="c", nlags="auto")
        return float(est), float(p)
    n = len(x); e = x - x.mean(); S = np.cumsum(e)
    L = int(np.ceil(12 * (n / 100) ** 0.25))
    s2 = (e @ e) / n + 2 * sum((1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n for l in range(1, L + 1))
    est = (S @ S) / (n ** 2 * s2)
    crit = [(0.347, 0.10), (0.463, 0.05), (0.574, 0.025), (0.739, 0.01)]
    p = 0.10 if est < 0.347 else next((a for c, a in reversed(crit) if est >= c), 0.10)
    return float(est), p


def adf(x, lags=12):
    """Estadístico ADF con constante y valor p (MacKinnon si hay statsmodels; si no, tabla de críticos)."""
    x = np.asarray(pd.Series(x).dropna(), float)
    if HAY_STATSMODELS:
        from statsmodels.tsa.stattools import adfuller
        r = adfuller(x, maxlag=lags, autolag=None)
        return float(r[0]), float(r[1])
    dx = np.diff(x); Y = dx[lags:]; X = [np.ones(len(Y)), x[lags:-1]]
    for i in range(1, lags + 1):
        X.append(dx[lags - i:-i])
    X = np.column_stack(X); b, *_ = np.linalg.lstsq(X, Y, rcond=None); r = Y - X @ b
    cov = (r @ r / (len(Y) - X.shape[1])) * np.linalg.inv(X.T @ X)
    t = b[1] / np.sqrt(cov[1, 1])
    p = 0.01 if t < -3.43 else 0.05 if t < -2.86 else 0.10 if t < -2.57 else 0.5
    return float(t), p


def acf(x, rezagos):
    x = np.asarray(x, float) - np.nanmean(x)
    d = x @ x
    return np.array([1.0] + [(x[:-k] @ x[k:]) / d for k in range(1, rezagos + 1)])


def pacf(x, rezagos):
    """PACF por Durbin-Levinson."""
    r = acf(x, rezagos); phi = np.zeros((rezagos + 1, rezagos + 1)); out = [1.0, r[1]]
    phi[1, 1] = r[1]
    for m in range(2, rezagos + 1):
        phi[m, m] = (r[m] - phi[m - 1, 1:m] @ r[1:m][::-1]) / (1 - phi[m - 1, 1:m] @ r[1:m])
        for j in range(1, m):
            phi[m, j] = phi[m - 1, j] - phi[m, m] * phi[m - 1, m - j]
        out.append(phi[m, m])
    return np.array(out)


def graficar_acf(ax, valores, n, titulo):
    b = 1.96 / np.sqrt(n)
    ax.bar(range(len(valores)), valores, color=np.where(np.abs(valores) > b, "#1F6F78", "#AFC3BA"))
    ax.axhspan(-b, b, color="#7F9C80", alpha=.2); ax.set_title(titulo); ax.set_xlabel("rezago")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return adj


def vif(X):
    """VIF de cada columna de un DataFrame estandarizado."""
    A = X.values; out = []
    for i in range(A.shape[1]):
        y = A[:, i]; Z = np.c_[np.ones(len(A)), np.delete(A, i, 1)]
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - ((y - Z @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return pd.Series(out, index=X.columns)
