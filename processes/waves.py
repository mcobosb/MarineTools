import numpy as np
from scipy.optimize import fsolve


def lim_frec(Te):
    """Obtiene las frecuencias de corte para el metodo de Baquerizo"""

    fp = 1 / Te
    f1, f2 = 0.5 * fp, 1.5 * fp

    fp, f1, f2 = np.round(fp, 2), np.round(f1, 2), np.round(f2, 2)
    return f1, f2, fp


def clsquare_s(eta, x, h, dt, eps, f1, f2):
    """Calcula las amplitudes complejas de los trenes incidente y reflejado a
    partir de las series temporales medidas por tres sensores alineados en la
    direccion de propagacion del oleaje (Baquerizo, 1995).

    [f, Zi, Zr]= clsquare_s(eta, x, h, dt, eps, f1, f2)

    Args:
        eta: elevacion de la superficie libre (m) matriz(ndat x 3)
        x: posiciones de los sensores (m) vector (3)
        h: profundidad de agua (m)
        dt: intervalo de muestreo (s)
        eps: tolerancia, valor minimo del denominador
        f1, f2: frecuencias minima y maxima de analisis

    Returns:
        Tres vectores. Las frecuencias (hz), las amplitudes incidentes y
        reflejadas

    """

    c = np.conj(np.fft.fft(eta.T)).T
    ndat = len(eta)
    df = 1 / (ndat * dt)

    k, f = np.zeros(np.int(ndat / 2.0 + 1)), np.zeros(np.int(ndat / 2.0 + 1))
    S_mas = np.zeros([len(k)], dtype=np.complex)
    S_min = np.zeros([len(k)], dtype=np.complex)
    Bi = np.zeros([len(k)], dtype=np.complex)
    Br = np.zeros([len(k)], dtype=np.complex)

    for j in range(1, ndat / 2 + 1):
        f[j] = j * df
        k[j] = wavnum(1 / f[j], h)
        S_mas[j] = np.sum(np.exp(2.0 * 1j * k[j] * x))
        S_min[j] = np.sum(np.exp(-2.0 * 1j * k[j] * x))

        Bi[j] = np.sum(c[j, :] * np.exp(-1j * k[j] * x))
        Br[j] = np.sum(c[j, :] * np.exp(1j * k[j] * x))

    mask = (f > f1) & (f < f2) & (np.abs(9.0 - S_mas * S_min) > eps)

    Zi = np.conj(
        (3.0 * Br[mask] - Bi[mask] * S_mas[mask]) / (9.0 - S_mas[mask] * S_min[mask])
    )
    Zr = np.conj(
        (3.0 * Bi[mask] - Br[mask] * S_min[mask]) / (9.0 - S_mas[mask] * S_min[mask])
    )

    Zi, Zr = Zi / ndat, Zr / ndat
    return f[mask], Zi, Zr


def wavnum(t, h):
    """Devuelve el número de onda calculado de la ecuación de la dispersión
    para un periodo T y una profundidad h
    """

    if isinstance(t, (int, float)):
        t = np.array([t])

    gamma = (2 * np.pi / t) ** 2.0 * h / 9.81
    sigma2 = (2 * np.pi / t) ** 2

    def func(k, sigma2, h):
        g = 9.81
        return sigma2 / g - k * np.tanh(k * h)

    k = np.zeros(len(t))
    for ind_, period in enumerate(t):
        if gamma[ind_] > np.pi**2:
            ko = np.sqrt(gamma[ind_] / h)
            k[ind_] = fsolve(func, ko, args=(sigma2[ind_], h))
        else:
            ko = gamma[ind_] / h
            k[ind_] = fsolve(func, ko, args=(sigma2[ind_], h))

        k[ind_] = np.abs(k[ind_])
    return k


def ref(T, mdat, t, xsn, h, dt):

    eps = 1e-10
    [f1, f2, fp] = lim_frec(T)

    eta = mdat[:, :]

    # 2 - Separacion incidente-reflejada
    [f, Zi, Zr] = clsquare_s(eta, xsn, h, dt, eps, f1, f2)

    # 3 - R y fi
    ZiR = np.abs(2.0 * Zi) ** 2.0
    ZrR = np.abs(2.0 * Zr) ** 2.0

    # Espectro incidente
    df = f[1] - f[0]
    Si = ZiR / (2.0 * df)

    # Momento de primer orden
    moI = df * (0.5 * Si[0] + 0.5 * Si[-1] + np.sum(Si[1:-2]))
    # Altura de ola incidente
    HI = np.sqrt(8 * moI)

    # Coeficiente de reflexion
    R = np.sqrt(np.sum(ZrR) / np.sum(ZiR))
    index = np.argmax(Zi)
    # Fase de la reflexión
    fi = np.angle(Zi[index]) - np.angle(Zr[index])

    return HI, R, fi, f, Zi, Zr


def closure_depth(data, Hallermeier=True):
    """Compute the closure depth according to Hallermeier or Birkemeier empirical formula

    Args:
        data (pd.DataFrame): wave characteristics
        Hallermeier (bool, optional): return the Hallermeier/Birkemeier formula if True/False. Defaults to True.

    Returns:
        _type_: _description_
    """
    G = 9.81
    list_H12, list_T12 = [], []
    for year in data.index.year.unique():
        # For every year, obtain the location of 12 hours exceedance
        data_year = data.loc[str(year), :].dropna()  # removing nans
        isort = np.argsort(data_year["Hm0"]).values  # index of 12 hrs

        # to ensure there is enough hourly data in a year (24*365.25 = 8766)
        if len(data_year) < 8766 / 2:
            list_H12.append(data_year.loc[str(year), "Hm0"].values[isort[-12]])
            list_T12.append(data_year.loc[str(year), "Tp"].values[isort[-12]])

    # Compute the average for all years
    H12, T12 = np.mean(np.asarray(list_H12)), np.mean(np.asarray(list_T12))
    if Hallermeier:
        dc = 2.28 * H12 - 68.5 * (H12**2 / (G * T12**2))
    else:
        dc = 1.75 * H12 - 57.9 * (H12**2 / (G * T12**2))
    return dc
