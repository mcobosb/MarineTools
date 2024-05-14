import importlib.util

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as scs
from marinetools.utils import auxiliar, read, save
from scipy.stats import chi2, mode


def lombscargle(data, variable, fname=None, max_period=None, nperiods=5, freq="H"):
    """Computes the LombScargle Periodogram for uneven sampling

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * fname (string, optional): name of the output file with the power spectral density and frequencies. Defaults to None.
        * max_period (float, optional): maximum value to analyze the diferent time periods. Defaults to None.
        * ref (float, optional): maximum value to look for the main time periods. Defaults to 1.1.
        * nperiods (int, optional): number of main periods. Defaults to 5.
        * freq (string, optional): 'D' for daily data and 'H' for hourly data. Defaults to H.

    Returns:
        - The periodogram/power spectral density and the significant frequencies
    """
    if freq == "D":
        time = (data.index - data.index[0]).days / 365.25
        periods = np.linspace(7 / 365.25, 2, 1000)
    else:
        time = (data.index - data.index[0]).total_seconds() / (365.25 * 24 * 3600)
        periods = np.linspace(24 * 7 / (24 * 365.25), 2, 1000)

    if max_period is not None:
        periods = np.hstack([periods, np.arange(2.01, max_period, 0.02)])

    freqs = 1.0 / periods
    angular_freq = 2 * np.pi * freqs

    psd = scs.lombscargle(
        time.values, data[variable].values, angular_freq, normalize=True
    )
    psd = pd.DataFrame(psd, index=periods, columns=["psd"])
    signf = auxiliar.moving(psd, 100)
    # signf.columns, signf.index.name = ['PSD'], 'periods'
    signf = signf.nlargest(nperiods, "psd")
    psd["significant"] = False
    psd.loc[signf.index, "significant"] = True

    return psd


def fft(data, variable, fname=None, freq="H", alpha=0.05):
    """Computes the Fast-Fourier Transform for regular sampling

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * fname (string, optional): name of the output file with the power spectral density and frequencies. Defaults to None.
        * freq (string, optional): 'D' for daily data and 'H' for hourly data. Defaults to H.

    Returns:
        - The periodogram/power spectral density and the significant frequencies
    """

    N = len(data)
    if freq == "H":
        fmin = 1 / (
            (data.index[-1] - data.index[0]).total_seconds() / 3600
        )  # resolución espectral, cuanto más pequeño, más frecuencias podré resolver
    elif freq == "D":
        fmin = 1 / (
            (data.index[-1] - data.index[0]).days
        )  # resolución espectral, cuanto más pequeño, más frecuencias podré resolver

    coefs = np.fft.fft(data)  # - np.mean(data))

    # Choose one side of the spectra
    cn = np.ravel(coefs[0 : N // 2] / N)

    N_ = len(cn)
    f = np.arange(0, N_) * fmin
    S = 0.5 * np.abs(cn) ** 2 / fmin

    psd = pd.DataFrame(S, index=f, columns=["psd"])

    var = data.var().values
    M = N / 2
    phi = (2 * (N - 1) - M / 2.0) / M
    chi_val = chi2.isf(q=1 - alpha / 2, df=phi)  # /2 for two-sided test
    psd["significant"] = S > (var / N) * (chi_val / phi) / (S * f**2)

    return psd


def harmonic(data: pd.DataFrame, lat: float, file_name: str = None):
    """Computes the u-tide (Codiga, 2011) for time series with gaps

    Args:
        - data (pd.DataFrame): raw data
        - lat (float): latitud
        - fname (string): filename where constituents will be saved

    Returns:
        - constituents (dict): the amplitude, phase and errors
    """

    lib_spec = importlib.util.find_spec("utide")
    if lib_spec is not None:
        from utide import solve
    else:
        raise ValueError(
            "You will require utide library. You can downloaded it from https://pypi.org/project/UTide/"
        )

    data.dropna(inplace=True)

    constituents = solve(
        data.index,
        data,
        lat=lat,
        nodal=True,
        trend=False,
        method="ols",
        conf_int="MC",
        Rayleigh_min=0.95,
        verbose=False,
    )

    if file_name is not None:
        save.to_json(constituents, file_name, True)

    return constituents


def reconstruction_tidal_level(df: pd.DataFrame, tidalConstituents: dict):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        tidalConstituents (dict): [description]

    Returns:
        [type]: [description]
    """
    from utide import reconstruct

    # try:
    #     time = mdates.date2num(df.index.to_pydatetime())
    # except:
    #     time = mdates.date2num(df.index)
    tidalLevel = reconstruct(df.index, tidalConstituents)

    df["ma"] = tidalLevel["h"] - tidalConstituents["mean"]
    return df
