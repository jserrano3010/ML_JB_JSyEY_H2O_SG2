"""Herramientas espaciales mínimas para pocos puntos (sin dependencias pesadas)."""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def matriz_distancias(lats, lons):
    lats, lons = np.asarray(lats), np.asarray(lons)
    return haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])


def pesos_inverso_distancia(lats, lons, potencia=1.0):
    """Pesos W_ij = 1/d_ij^potencia, estandarizados por fila, diagonal 0."""
    D = matriz_distancias(lats, lons)
    with np.errstate(divide="ignore"):
        W = np.where(D > 0, 1.0 / D ** potencia, 0.0)
    return W / W.sum(axis=1, keepdims=True)


def moran_i(valores, W, n_perm=999, seed=42):
    """
    I de Moran global con prueba de permutación.
    Con pocos puntos (4-5) la prueba tiene muy poca potencia: reportarlo como limitación.
    """
    z = np.asarray(valores, float)
    z = z - z.mean()
    n, s0 = len(z), W.sum()

    def calc(v):
        return (n / s0) * (v @ W @ v) / (v @ v)

    I = calc(z)
    rng = np.random.default_rng(seed)
    perm = np.array([calc(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm - perm.mean()) >= np.abs(I - perm.mean())) + 1) / (n_perm + 1)
    return {"I": float(I), "E[I]": -1 / (n - 1), "p_perm": float(p)}
