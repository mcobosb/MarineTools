import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.temporal import analysis
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.utils import save
from matplotlib.dates import date2num, num2julian
from scipy.interpolate import Rbf
from scipy.optimize import minimize


def max_moving(data: pd.DataFrame, dur: int):
    """Selects the peaks of the time series for a given duration

    Args:
        - data (pd.DataFrame): timeseries
        - dur(int): duration of the moving window

    Returns:
        - results (pd.DataFrame): peaks and time of maximum values
    """

    n = len(data)
    id_ = []
    for k in range(0, n - dur + 1):
        idx = data.iloc[k : k + dur + 1].idxmax()
        if idx == data.index[k + int(dur / 2)]:
            id_.append(idx)

    results = pd.DataFrame(data.loc[id_], index=id_)
    return results


def gaps(data, variables, fname="gaps", buoy=False):
    """Creates a table with the main characteristics of gaps for variables

    Args:
        * data (pd.DataFrame): time series
        * variables (string): with the variables where gap-info is required
        * fname (string): name of the output file with the information table

    Returns:
        * tbl_gaps (pd.DataFrame): gaps info
    """

    if not isinstance(variables, list):
        variables = [variables]

    if not buoy:
        columns_ = [
            "Cadency (min)",
            "Accuracy*",
            "Period",
            "No. years",
            "Gaps (%)",
            "Med. gap (hr)",
            "Max. gap (d)",
        ]
    else:
        columns_ = [
            "Cadency (min)",
            "Accuracy*",
            "Period",
            "No. years",
            "Gaps (%)",
            "Med. gap (hr)",
            "Max. gap (d)",
            "Quality data (%)",
        ]

    tbl_gaps = pd.DataFrame(
        0,
        columns=columns_,
        index=variables,
    )
    tbl_gaps.index.name = "var"

    for i in variables:
        dt_nan = data[i].dropna()
        if buoy:
            quality = np.sum(data.loc[dt_nan.index, "Qc_e"] <= 2)

        dt0 = (dt_nan.index[1:] - dt_nan.index[:-1]).total_seconds() / 3600
        dt = dt0[dt0 > np.median(dt0) + 0.1].values
        if dt.size == 0:
            dt = 0
        acc = st.mode(np.diff(dt_nan.sort_values().unique()))[0]

        tbl_gaps.loc[i, "Cadency (min)"] = np.round(st.mode(dt0)[0]*60, decimals=2)
        tbl_gaps.loc[i, "Accuracy*"] = np.round(acc, decimals=2)
        tbl_gaps.loc[i, "Period"] = str(dt_nan.index[0]) + "-" + str(dt_nan.index[-1])
        tbl_gaps.loc[i, "No. years"] = dt_nan.index[-1].year - dt_nan.index[0].year
        tbl_gaps.loc[i, "Gaps (%)"] = np.round(
            np.sum(dt) / data[i].shape[0] * 100, decimals=2
        )
        tbl_gaps.loc[i, "Med. gap (hr)"] = np.round(np.median(dt), decimals=2)
        tbl_gaps.loc[i, "Max. gap (d)"] = np.round(np.max(dt)/24, decimals=2)

        if buoy:
            tbl_gaps.loc[i, "Quality data (%)"] = np.round(
                quality / len(dt_nan) * 100, decimals=2
            )

    if not fname:
        logger.info(tbl_gaps)
    else:
        save.to_xlsx(tbl_gaps, fname)

    return tbl_gaps


def nonstationary_ecdf(
    data: pd.DataFrame,
    variable: str,
    wlen: float = 14 / 365.25,
    equal_windows: bool = False,
    pemp: list = None,
):
    """Computes the empirical percentiles selecting a moving window

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable
        * wlen (float): length of window in days. Defaults to 14 days.
        * pemp (list, optional): given empirical percentiles

    Returns:
        * res (pd.DataFrame): values of the given non-stationary percentiles
        * pemp (list): chosen empirical percentiles
    """

    timestep = 1 / 365.25
    if equal_windows:
        timestep = wlen

    if pemp is None:
        pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        if variable.lower().startswith("d") | (variable == "Wd"):
            pemp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        if (variable == "Hs") | (variable == "Hm0"):
            pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995])

    # res = pd.DataFrame(0, index=np.arange(0, 1, timestep), columns=pemp)
    res = pd.DataFrame(0, index=data.n.unique(), columns=pemp)

    for i in res.index:
        if i >= (1 - wlen):
            final_offset = i + wlen - 1
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] <= final_offset
            )
        elif i <= wlen:
            initial_offset = i - wlen
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] >= 1 + initial_offset
            )
        else:
            mask = (data["n"] >= i - wlen) & (data["n"] <= i + wlen)
        res.loc[i, pemp] = data[variable].loc[mask].quantile(q=pemp).values

    return res, pemp


def best_params(data: pd.DataFrame, bins: int, distrib: str, tail: bool = False):
    """Computes the best parameters of a simple probability model attending to the rmse of the pdf

    Args:
        * data (pd.DataFrame): raw time series
        * bins (int): no. of bins for the histogram
        * distrib (string): name of the probability model
        * tail (bool, optional): If it is fit a tail or not. Defaults to False.

    Returns:
        * params (list): the estimated parameters
    """

    dif_, sser = 1e2, 1e3
    nlen = int(len(data) / 200)

    data = data.sort_values(ascending=True).values
    while (dif_ > 1) & (sser > 30) & (0.95 * nlen < len(data)):
        results = fit_(data, bins, distrib)
        sse, params = results[0], results[1:]
        dif_, sser = np.abs(sser - sse), sse

        if tail:
            data = data[int(nlen / 4) :]
        else:
            data = data[nlen:-nlen]
    return params


def ecdf(df: pd.DataFrame, variable: str, no_perc: int or bool = False):
    """Computes the empirical cumulative distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * variable (pd.DataFrame): name of the variable
        * no_perc (optional, False|integer): number of empirical percentiles to be interpolated. Defaults all data

    Returns:
        * dfs (pd.DataFrame): sorted values of the time series and the non-excedence probability of every value
    """
    dfs = df[variable].sort_values().to_frame()
    dfs.index = np.arange(1, len(dfs) + 1) / (len(dfs) + 1)
    if not isinstance(no_perc, bool):
        percentiles = np.linspace(1 / no_perc, 1 - (1 / no_perc), no_perc)
        values = np.interp(percentiles, dfs.index, dfs[variable])
        dfs = pd.DataFrame(values, columns=[variable], index=percentiles)
    return dfs


def nonstationary_epdf(
    data: pd.DataFrame, variable: str, wlen: float = 14 / 365.25, no_values: int = 14
):
    """Computes the empirical percentiles selecting a moving window

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable
        * wlen (float): length of window in days. Defaults to 14 days.

    Returns:
        * res (pd.DataFrame): values of the given non-stationary percentiles
        * pemp (list): chosen empirical percentiles
    """

    nlen = len(data)
    ndates = np.arange(0, 1, 1 / 365.25)
    values_ = np.linspace(data[variable].min(), data[variable].max(), no_values)
    pdf_ = pd.DataFrame(-1, index=ndates, columns=(values_[:-1] + values_[1:]) / 2)

    columns = pdf_.columns
    for ind_, col_ in enumerate(columns[:-1]):
        for i in pdf_.index:
            if i > (1 - wlen):
                final_offset = i + wlen - 1
                mask = (
                    (data["n"] > i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] <= final_offset)
            elif i < wlen:
                initial_offset = i - wlen
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] >= 1 + initial_offset)
            else:
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                )
            pdf_.loc[i, col_] = np.sum(mask) / nlen

    return pdf_


def epdf(df: pd.DataFrame, variable: str, no_values: int = 14):
    """Computes the empirical probability distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * variable (pd.DataFrame): name of the variable
        * no_values (optional, False|integer): number of empirical percentiles to be interpolated. Defaults all data

    Returns:
        * dfs (pd.DataFrame): sorted values of the time series and the probability
    """
    dfs = df[variable].sort_values().to_frame()
    dfs["prob"] = np.arange(1, len(dfs) + 1)
    count_ = pd.DataFrame(-1, index=dfs[variable].unique(), columns=["prob"])

    for _, ind_ in enumerate(count_.index):
        count_.loc[ind_] = np.sum(dfs[variable] == ind_)

    values_ = np.linspace(df[variable].min(), df[variable].max(), no_values)
    pdf_ = pd.DataFrame(-1, index=(values_[:-1] + values_[1:]) / 2, columns=["prob"])
    for ind_, index_ in enumerate(pdf_.index):
        # range_ = np.interp(values_, pdf_["prob"], pdf_[variable])
        val_ = np.sum(
            count_.loc[
                ((count_.index < values_[ind_ + 1]) & (count_.index > values_[ind_]))
            ].values
        )
        pdf_.loc[index_, "prob"] = val_
    pdf_.loc[:, "prob"] = pdf_["prob"] / (np.sum(pdf_).values * np.diff(values_))
    return pdf_


def acorr(data, maxlags=24):
    """[summary]

    Args:
        data ([type]): [description]
        maxlags (int, optional): [description]. Defaults to 24.

    Returns:
        [type]: [description]
    """

    lags, c_, _, _ = plt.acorr(data, usevlines=False, maxlags=maxlags, normed=True)
    plt.close()
    # lags, c_ = lags[-maxlags:], c_[-maxlags:]
    return lags, c_


def bidimensional_ecdf(data1, data2, nbins):
    """Compute the empirical 2d CDF

    Args:
        data1 ([type]): data of first variable
        data2 ([type]): data of second variable
        nbins ([type]): number of bins

    Returns:
        [type]: [description]
    """
    f, xedges, yedges = np.histogram2d(data1, data2, bins=nbins)
    Fe = np.cumsum(np.cumsum(f, axis=0), axis=1) / (np.sum(f) + 1)
    Fe = np.flipud(np.rot90(Fe))

    xmid, ymid = (xedges[0:-1] + xedges[1:]) / 2, (yedges[0:-1] + yedges[1:]) / 2
    xe, ye = np.meshgrid(xmid, ymid)

    return xe, ye, Fe


def get_params_by_labels(
    df, param, par, imod, cossen, first=None, pos=[0, 0, 0], ppf=False
):
    """Gets the parameters of the probability models by its name

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par (dict): the guess parameters of the probability model
        * imod (list): combination of modes for fitting
        * cossen (np.ndarray): the variability of the modes
        * first (boolean or None, optional): For fitting of first or n-order. Defaults to None (n-order).
        * pos (list, optional): location of the different probability models. Defaults to [0, 0, 0].
        * ppf (bool, optional): True if it is required the numerically computation of the cdf in a mesh. Defaults to False.

    Returns:
        * df (pd.DataFrame): the parameters
        * esc (list): weight of the probability models
    """

    for i in param["fun"].keys():
        if isinstance(param["fun"][i], str):
            param["fun"][i] = getattr(st, param["fun"][i])
        else:
            param["fun"][i] = param["fun"][i]

    modo, esc = imod, {}
    if param["reduction"]:
        # Parametros para el regimen extremal con un modelo de probabilidad de bi- o triparamétrico
        if first:  # TODO: la obtención de parametros para el orden 1
            df["shape"] = []
            df["loc"] = []
            if param["no_param"][0] == 2:
                df["xi2"] = []
            else:
                df["scale"] = []
                df["xi2"] = []
        else:
            df["shape"] = par[0] + np.dot(
                par[1 : modo[0] * 2 + 1], cossen[0 : modo[0] * 2, :]
            )
            df["shape_0"] = par[0]
            for ii, jj in enumerate(np.arange(1, modo[0] * 2, 2)):
                df["shape_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["shape_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["shape_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[jj - 1 : jj + 1, :]
                )
            df["loc"] = par[modo[0] * 2 + 1] + np.dot(
                par[modo[0] * 2 + 2 : modo[0] * 4 + 2], cossen[0 : modo[0] * 2, :]
            )
            df["loc_0"] = par[modo[0] * 2 + 1]
            for ii, jj in enumerate(np.arange(modo[0] * 2 + 2, modo[0] * 4 + 2, 2)):
                df["loc_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["loc_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["loc_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                )

            if param["no_param"][0] == 2:
                df["xi2"] = []
            else:
                df["scale"] = par[modo[0] * 4 + 2] + np.dot(
                    par[modo[0] * 4 + 3 : modo[0] * 6 + 3], cossen[0 : modo[0] * 2, :]
                )
                df["scale_0"] = par[modo[0] * 4 + 2]
                for ii, jj in enumerate(np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)):
                    df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["scale_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["scale_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )
                df["xi2"] = par[modo[0] * 6 + 3] + np.dot(
                    par[modo[0] * 6 + 4 : modo[0] * 6 + modo[1] * 2 + 4],
                    cossen[0 : modo[1] * 2, :],
                )
                df["xi2_0"] = par[modo[0] * 6 + 3]
                for ii, jj in enumerate(
                    np.arange(modo[0] * 6 + 4, modo[0] * 6 + modo[1] * 2 + 4, 2)
                ):
                    df["xi2_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["xi2_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["xi2_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )

        esc[1] = st.norm.cdf(par[-2])
        esc[2] = 1 - st.norm.cdf(par[-1])
        if param["no_param"][0] == 2:
            df["u1"] = []
            df["u2"] = []

            df["siggp1"] = []
            df["xi1"] = []
            df["siggp2"] = []
        else:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"], df["scale"])
            df["u2"] = param["fun"][1].ppf(
                st.norm.cdf(par[-1]), df["shape"], df["loc"], df["scale"]
            )

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"], df["scale"]
            )
            df["u1"][
                df["u1"] < df["siggp1"]
            ] = np.nan  # TODO: limitacion discontinuidades
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"], df["scale"]
            )

            pars_p = ["u1", "u2", "siggp1", "xi1", "siggp2"]
            for pari in pars_p:
                fft = np.fft.fft(df[pari]) / len(df[pari])
                a0 = np.real(fft[0])
                an = 2 * np.real(fft[1:])
                bn = -2 * np.imag(fft[1:])
                mod = np.sqrt(an**2 + bn**2)
                pha = np.arctan2(bn, an)

                df["{}_0".format(pari)] = a0
                for ii, jj in enumerate(np.arange(1, modo[0] + 1)):
                    df["{}_{}_mod".format(pari, jj)] = mod[ii]
                    df["{}_{}_pha".format(pari, jj)] = pha[ii]
                    df["{}_{}".format(pari, jj)] = np.dot(
                        [an[ii], bn[ii]], cossen[2 * ii : 2 * ii + 2, :]
                    )
    else:
        # TODO separar por orden de fourier
        # Parámetros para uno modelo de probabilidad bi- o triparamétrico
        # for i in range(param['no_fun']):
        if param["order"] == 0:
            modo = [0, 0, 0]
        elif not first:
            modo = [imod[pos[2]], imod[pos[2]], imod[pos[2]]]

        df["shape_0"] = par[pos[0]]
        if modo[pos[1]] != 0:
            df["shape"] = par[pos[0]] + np.dot(
                par[pos[0] + 1 : pos[0] + modo[pos[1]] * 2 + 1],
                cossen[0 : modo[pos[1]] * 2, :],
            )
            for ii, jj in enumerate(np.arange(1, modo[pos[1]] * 2, 2)):
                df["shape_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["shape_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["shape_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[jj - 1 : jj + 1, :]
                )
        df["loc_0"] = par[pos[0] + modo[pos[1]] * 2 + 1]

        if modo[pos[1] + 1] != 0:
            df["loc"] = par[pos[0] + modo[pos[1]] * 2 + 1] + np.dot(
                par[
                    pos[0]
                    + modo[pos[1]] * 2
                    + 2 : pos[0]
                    + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                    + 2
                ],
                cossen[0 : modo[pos[1] + 1] * 2, :],
            )
            for ii, jj in enumerate(np.arange(modo[0] * 2 + 2, modo[0] * 4 + 2, 2)):
                df["loc_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["loc_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["loc_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                )

        if param["no_param"][pos[2]] == 3:
            df["scale_0"] = par[pos[0] + modo[pos[1]] * 2 + 2]

            if first:
                if modo[pos[1] + 1] != 0:
                    df["scale"] = par[
                        pos[0] + np.sum(modo[pos[1] : pos[1] + 2]) * 2 + 2
                    ]
                if modo[pos[1] + 2] != 0:
                    df["scale"] = par[pos[0] + modo[pos[1]] * 2 + 2] + np.dot(
                        par[
                            pos[0]
                            + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                            + 3 : pos[0]
                            + np.sum(modo[pos[1] : pos[1] + 3]) * 2
                            + 3
                        ],
                        cossen[0 : modo[pos[1] + 2] * 2, :],
                    )
                    for ii, jj in enumerate(
                        np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)
                    ):
                        df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                            par[jj] ** 2 + par[jj + 1] ** 2
                        )
                        df["scale_{}_pha".format(ii + 1)] = np.arctan2(
                            par[jj + 1], par[jj]
                        )
                        df["scale_{}".format(ii + 1)] = np.dot(
                            par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                        )

            else:
                # print(imod, len(par), pos[0]+np.sum(modo[pos[1]:pos[1]+2])*2+2, pos[0]+np.sum(modo)*2+3)
                df["scale"] = par[
                    pos[0] + np.sum(modo[pos[1] : pos[1] + 2]) * 2 + 2
                ] + np.dot(
                    par[
                        pos[0]
                        + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                        + 3 : pos[0]
                        + np.sum(modo) * 2
                        + 3
                    ],
                    cossen[0 : modo[pos[2]] * 2, :],
                )
                for ii, jj in enumerate(np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)):
                    df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["scale_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["scale_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )

        if param["no_fun"] == 1:
            esc = param["weights"][0]
        else:
            if pos[2] == 0:  # TODO: para solo dos funciones
                # esc = par[-len(param['pesos'])+1]
                esc = param["weights"][0]
            elif pos[2] == len(param["weights"]) - 1:
                # esc = 1- np.sum(par[-len(param['pesos'])+1:])
                esc = param["weights"][1]
            else:
                # esc = np.sum(par[-len(param['pesos'])+1:-len(param['pesos'])+1+pos[2]])
                esc = param["weights"][1]

    return df, esc


def bias_adjustment(
    obs,
    hist,
    rcp,
    variable,
    funcs=["gumbel_l", "gumbel_r"],
    quantiles=[0.1, 0.9],
    params=None,
):
    """IT IS REQUIRED THE SAME NUMBER OF VALUES FOR OBSERVED AND HISTORICAL DATA

    Args:
        obs ([type]): [description]
        hist ([type]): [description]
        rcp ([type]): [description]
        variable ([type]): [description]
        funcs (list, optional): [description]. Defaults to ["gumbel_l", "gumbel_r"].
        quantiles (list, optional): [description]. Defaults to [0.1, 0.9].
        params ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    funcs = funcs.copy()
    for index, fun in enumerate(funcs):
        funcs[index] = getattr(st, fun)

    hist["unbiased"] = 0
    rcp["unbiased"] = 0

    # Low tail
    low_tail_hist = hist.loc[
        hist[variable] <= hist[variable].quantile(quantiles[0]), variable
    ]

    if isinstance(params, dict):
        if "obs_low" in params:
            params_obs_low = params["obs_low"]
    else:
        params_obs_low = funcs[0].fit(
            obs.loc[obs[variable] <= obs[variable].quantile(quantiles[0]), variable]
        )

    if isinstance(params, dict):
        if "hist_low" in params:
            params_hist_low = params["hist_low"]
    else:
        params_hist_low = funcs[0].fit(low_tail_hist)

    hist.loc[hist[variable] <= hist[variable].quantile(quantiles[0]), "unbiased"] = (
        funcs[0].ppf(funcs[0].cdf(low_tail_hist, *params_hist_low), *params_obs_low)
    )
    low_tail_rcp = rcp.loc[
        rcp[variable] <= rcp[variable].quantile(quantiles[0]), variable
    ]
    rcp.loc[rcp[variable] <= rcp[variable].quantile(quantiles[0]), "unbiased"] = funcs[
        0
    ].ppf(funcs[0].cdf(low_tail_rcp, *params_hist_low), *params_obs_low)

    # High tail
    high_tail_hist = hist.loc[
        hist[variable] >= hist[variable].quantile(quantiles[1]), variable
    ]

    if isinstance(params, dict):
        if "obs_high" in params:
            params_obs_high = params["obs_high"]
    else:
        params_obs_high = funcs[1].fit(
            obs.loc[obs[variable] >= obs[variable].quantile(quantiles[1]), variable]
        )

    if isinstance(params, dict):
        if "hist_high" in params:
            params_hist_high = params["hist_high"]
    else:
        params_hist_high = funcs[1].fit(high_tail_hist)

    hist.loc[hist[variable] >= hist[variable].quantile(quantiles[1]), "unbiased"] = (
        funcs[1].ppf(funcs[1].cdf(high_tail_hist, *params_hist_high), *params_obs_high)
    )
    high_tail_rcp = rcp.loc[
        rcp[variable] >= rcp[variable].quantile(quantiles[1]), variable
    ]
    rcp.loc[rcp[variable] >= rcp[variable].quantile(quantiles[1]), "unbiased"] = funcs[
        1
    ].ppf(funcs[1].cdf(high_tail_rcp, *params_hist_high), *params_obs_high)

    # Body
    n_obs = len(obs)
    obs_sort = np.sort(obs[variable])

    # body_obs_quantile = np.linspace(quantiles[0], quantiles[1], 20)

    # cdf_obs = ecdf(obs, variable)
    # body_obs_values = obs[variable].quantile(body_obs_quantile)
    cdf_hist = ecdf(hist, variable)
    # body_hist_values = hist[variable].quantile(body_obs_quantile)

    values_body_hist = hist.loc[
        (hist[variable] > hist[variable].quantile(quantiles[0]))
        & (hist[variable] < hist[variable].quantile(quantiles[1])),
        variable,
    ]
    hist_adj_list = list()
    Fe_Fhist_hist = np.interp(values_body_hist, cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Fhist_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        hist_adj_list.append(obs_sort[pos])
    hist.loc[
        (hist[variable] > hist[variable].quantile(quantiles[0]))
        & (hist[variable] < hist[variable].quantile(quantiles[1])),
        "unbiased",
    ] = np.asarray(hist_adj_list)

    values_body_rcp = rcp.loc[
        (rcp[variable] > rcp[variable].quantile(quantiles[0]))
        & (rcp[variable] < rcp[variable].quantile(quantiles[1])),
        variable,
    ]
    rcp_adj_list = list()
    Fe_Frcp_hist = np.interp(values_body_rcp, cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Frcp_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        rcp_adj_list.append(obs_sort[int(pos)])
    rcp.loc[
        (rcp[variable] > rcp[variable].quantile(quantiles[0]))
        & (rcp[variable] < rcp[variable].quantile(quantiles[1])),
        "unbiased",
    ] = np.asarray(rcp_adj_list)

    fit_params = {
        "hist_low": params_hist_low,
        "hist_high": params_hist_high,
        "obs_low": params_obs_low,
        "obs_high": params_obs_high,
    }

    return hist, rcp, fit_params


def probability_mapping(obs, hist, rcp, variable, func):
    """[summary]

    Args:
        obs ([type]): [description]
        hist ([type]): [description]
        rcp ([type]): [description]
        variable ([type]): [description]
        func ([type]): [description]

    Returns:
        [type]: [description]
    """

    func = getattr(st, func)

    params_obs = func.fit(obs[variable])
    params_hist = func.fit(hist[variable])

    rcp["unbiased"] = func.ppf(func.cdf(rcp[variable], *params_hist), *params_obs)
    hist["unbiased"] = func.ppf(func.cdf(hist[variable], *params_hist), *params_obs)

    return hist, rcp


def empirical_cdf_mapping(obs, hist, rcp, variable):
    """[summary]

    Args:
        obs ([type]): [description]
        hist ([type]): [description]
        rcp ([type]): [description]
        variable ([type]): [description]
    """

    n_obs = len(obs)
    obs_sort = np.sort(obs[variable])

    cdf_hist = ecdf(hist, variable)

    hist_adj_list = list()
    Fe_Fhist_hist = np.interp(hist[variable], cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Fhist_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        hist_adj_list.append(obs_sort[pos])
    hist["unbiased"] = np.asarray(hist_adj_list)

    rcp_adj_list = list()
    Fe_Frcp_hist = np.interp(rcp[variable], cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Frcp_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        rcp_adj_list.append(obs_sort[pos])
    rcp["unbiased"] = np.asarray(rcp_adj_list)

    return hist, rcp


def rotate_geo2nav(ang):
    """Rotate angles from geographical (0º pointing to the East, 90º pointing to the North) to navigational (0º pointing to the North, 90º pointing to the East)

    Args:
        * ang (pd.DataFrame or np.array): vector with the angles

    Return:
        rotated angles
    """
    ang = np.fmod(270 - ang + 360, 360)
    return ang


def uv2Uang(u, v, labels=["u", "ang"]):
    """[summary]

    Args:
        u ([type]): [description]
        v ([type]): [description]
        labels (list, optional): [description]. Defaults to ['u', 'ang'].

    Returns:
        [type]: [description]
    """
    ang = np.fmod(np.arctan2(v, u) * 180 / np.pi + 360, 360)
    data = pd.DataFrame(
        np.vstack([np.sqrt(u**2 + v**2), ang]).T, columns=labels, index=ang.index
    )  # TODO: verify
    return data


def optimize_rbf_epsilon(coords, data, num, method="gaussian", smooth=0.5, eps0=1):
    """[summary]

    Args:
        coords ([type]): [description]
        data ([type]): [description]
        num ([type]): [description]
        method (str, optional): [description]. Defaults to 'gaussian'.
        smooth (float, optional): [description]. Defaults to 0.5.
        eps0 (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """

    res_ = minimize(
        rmse_rbf,
        eps0,
        args=(coords, data, num, method, smooth),
        bounds=[[1e-9, 1e5]],
        method="SLSQP",
        options={"ftol": 1e-7, "eps": 1e-4, "maxiter": 1e4},
    )
    # print(res_)
    return res_["x"]


def rmse_rbf(eps0, coords, data, num, method, smooth):
    """[summary]

    Args:
        eps0 ([type]): [description]
        coords ([type]): [description]
        data ([type]): [description]
        num ([type]): [description]
        method ([type]): [description]
        smooth ([type]): [description]

    Returns:
        [type]: [description]
    """
    func = Rbf(
        *coords[:num, :].T, data[:num], function=method, smooth=smooth, epsilon=eps0
    )
    Vdt = func(*coords[num:, :].T)

    # func = Rbf(coords[:num, :], data[:num, :], function=method, smooth=smooth, epsilon=eps0)
    # Vdt = func(coords[num:, :])
    error = np.sqrt(1 / len(Vdt) * np.sum((Vdt - data[num:]) ** 2))
    # error = 1/len(Vdt)*np.sum(np.abs(Vdt - data[num:]))
    # print(eps0, error)
    return error


def outliers_detection(
    data, outliers_fraction, method="Local Outlier Factor", scaler="MinMaxScaler"
):
    """[summary]

    Args:
        data ([type]): [description]
        outliers_fraction ([type]): [description]
        method (str, optional): [description]. Defaults to "Local Outlier Factor".
        scaler (str, optional): [description]. Defaults to "MinMaxScaler".

    Returns:
        [type]: [description]
    """

    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    algorithms = {
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "One-Class SVM": svm.OneClassSVM(
            nu=outliers_fraction, kernel="rbf", gamma="scale"
        ),
        "Isolation Forest": IsolationForest(
            contamination=outliers_fraction, behaviour="new"
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=25, contamination=outliers_fraction
        ),
    }

    if not scaler:
        transformed_data, scale = scaler(data, method=scaler)
    else:
        transformed_data = data.copy()

    algorithms[method].fit(np.asarray([data[:, i], data[:, i + 1]]).T)
    if method == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(
            np.asarray([cases_deep_normalize[:, i], cases_shallow_normalize[:, i]]).T
        )
        # y_pred = algorithm.fit_predict(cases_deep_normalize, cases_shallow_normalize)
    else:
        y_pred = algorithm.fit(
            np.asarray([cases_deep_normalize[:, i], cases_shallow_normalize[:, i]]).T
        ).predict(
            np.asarray([cases_deep_normalize[:, i], cases_shallow_normalize[:, i]]).T
        )

    inliners = y_pred == 1
    outliers = data[~inliners, i]
    return outliers


def scaler(data, method="MinMaxScaler", transform=True, scale=False):
    """[summary]

    Args:
        data ([type]): [description]
        method (str, optional): [description]. Defaults to "MinMaxScaler".
        transform (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                       StandardScaler)

    algorithms = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }

    np_array = False
    if not isinstance(data, pd.DataFrame):
        np_array = True

    if transform & (not scale):
        scale = algorithms[method].fit(data)

    if not transform:
        if np_array:
            transformed_data = scale.inverse_transform(data)
        else:
            transformed_data = pd.DataFrame(
                scale.inverse_transform(data), index=data.index, columns=data.columns
            )
            scale = None
    else:
        if np_array:
            transformed_data = scale.transform(data)
        else:
            transformed_data = pd.DataFrame(
                scale.transform(data), index=data.index, columns=data.columns
            )

    return transformed_data, scale


def fall_velocity(d, T, S):
    """Estimate fall velocity based on Soulsby's (1997) optimization.
      w = kvis/d* [sqrt(10.36^2 + 1.049 D^3) - 10.36]

    Args:
        d = grain diameter (mm)
        T = temperature (deg C)
        S = Salinity (ppt)
    """

    g = 9.81
    rho = density(T, S)
    kvis = kvisfun(T)
    rhos = 2650
    d = d / 1000
    s = rhos / rho
    D = (g * (s - 1) / kvis**2) ** (1 / 3) * d
    w = kvis / d * (np.sqrt(10.36**2 + 1.049 * D**3) - 10.36)
    return w


def density(T, S):
    """Estimates water density from temperature and salinity
    Approximation from VanRijn, L.C. (1993) Handbook for Sediment Transport
    by Currents and Waves

    Args:
        T = temperature (C)
        S = salinity (o/oo)
    """

    CL = (S - 0.03) / 1.805  # VanRijn

    rho = 1000 + 1.455 * CL - 6.5e-3 * (T - 4 + 0.4 * CL) ** 2
    return rho


def kvisfun(T):
    """Estimates kinematic viscosity of water Approximation from VanRijn, L.C. (1989) Handbook of Sediment Transport

    Args:
        kvis = kinematic viscosity (m^2/sec)
        T = temperature (C)
    """

    kvis = 1e-6 * (1.14 - 0.031 * (T - 15) + 6.8e-4 * (T - 15) ** 2)
    return kvis


def str2fun(param: dict, var_: str = None):
    """Create an object of scipy.stats function given a string with the name

    Args:
        * param (dict): dictionary with parameters
        * var_ (str): name of the variable

    Return:
        * The input dictionary updated
    """

    if var_ is None:
        for key in param["fun"].keys():
            if isinstance(param["fun"][key], str):
                if param["fun"][key] == "wrap_norm":
                    param["fun"][key] = stf.wrap_norm()
                else:
                    param["fun"][key] = getattr(st, param["fun"][key])
    else:
        for key in param[var_]["fun"].keys():
            if isinstance(param[var_]["fun"][key], str):
                if param[var_]["fun"][key] == "wrap_norm":
                    param[var_]["fun"][key] = stf.wrap_norm()
                else:
                    param[var_]["fun"][key] = getattr(st, param[var_]["fun"][key])

    return param


def zero_cross(ts, dt):
    """Analize the time series using th zero-upcrossing method.

    Args:
        - ts: vector con los datos de superficie libre [L]
        - dt: resolución temporal (1/fs, fs=frecuencia de muestreo) [T]

    Returns:
        Una tupla de la forma:
        * ``H``: vector con la serie de alturas de ola [L]
        * ``T``: vector con la serie de periodos de paso por cero [T]
        * ``Ac``: vector con las amplitudes de cresta [L]
        * ``As``: vector con las amplitudes de seno [L]
        * ``Tc``: vector con los periodos de cresta [T]
        * ``Ts``: vector con los periodos de seno [T]

    """

    ndat = len(ts)
    t = np.linspace(0, ndat - 1, ndat - 1) * dt

    # Cálculo del nº de pasos ascendentes por cero.
    sg = np.sign(ts)
    ps = sg[0 : ndat - 1] * sg[1:ndat]

    i1 = np.asarray([i for i, x in enumerate(ps) if x < 0])
    i2 = np.zeros(len(i1), dtype=int)

    vmx = np.zeros(len(i1) - 1)
    imx = np.zeros(len(i1) - 1, dtype=int)
    for j in range(0, len(i1) - 2):
        vmx[j] = max(abs(ts[i1[j] + 1 : i1[j + 1] + 1]))
        io = [
            float(item)
            for item, jj in enumerate(abs(ts[i1[j] : i1[j + 2]]))
            if jj == vmx[j]
        ]
        i2[j] = int(i1[j] + 1)
        imx[j] = int(i1[j] + io[0])
    imx[-1] = i1[-1]

    tc = t[i1] - dt * ts[i1] / (ts[i2] - ts[i1])
    dtc = np.diff(tc)

    if ts[0] < 0:  # El primero es ascendente
        Tc = dtc[0::2]
        Ts = dtc[1::2]
        ic = imx[0::2]
        it = imx[1::2]
        Ac = vmx[0::2]
        As = vmx[1::2]
        if ts[-1] < 0:  # El último es ascendente
            Tc, Ac, ic = Tc[0:-1], Ac[0:-1], ic[0:-1]
            T = Tc + Ts
            H = Ac + As
            i1 = i1[0:-1]
        else:
            T = Tc + Ts
            H = Ac + As

    else:  # El primero es descendente
        Tc = dtc[1::2]
        Ts = dtc[2::2]
        ic = imx[1::2]
        it = imx[2::2]
        Ac = vmx[1::2]
        As = vmx[2::2]
        if ts[-1] < 0:  # El último es ascendente
            Tc, Ac, ic = Tc[0:-1], Ac[0:-1], ic[0:-1]
            T = Tc + Ts
            H = Ac + As
            i1 = i1[1:-1]
        else:
            T = Tc + Ts
            H = Ac + As
            i1 = i1[1:]

    return H, T, Ac, As, Tc, Ts, ic, it, i1


def dataoverthreshold(data, variable, threshold, duration):
    """Compute the events over threshold that have a minimum duration

    Args:
        data ([pd.DataFrame]): time series
        variable ([string]): variable where the method is applied
        threshold ([numeric value]): a threshold number
        duration ([numeric value]): the minimum duration to be consider an extreme event

    Returns:
        events: pd.DataFrame with the time series over the threshold
        eventsno: pd.DataFrame with the number of events per year
    """

    asc_roots = data.loc[
        ((data.iloc[:-1].shift(1) < threshold) & (data.iloc[1:] >= threshold)), variable
    ]
    dec_roots = data.loc[
        ((data.iloc[:-1] >= threshold) & (data.iloc[1:].shift(-1) < threshold)),
        variable,
    ]

    if asc_roots.index[0] > dec_roots.index[0]:
        dec_roots.drop(index=dec_roots.index[0])

    if asc_roots.index[-1] > dec_roots.index[-1]:
        asc_roots.drop(index=asc_roots.index[-1])

    events = pd.DataFrame(columns=[variable])
    eventsno = pd.DataFrame(0, index=asc_roots.index.year.unique(), columns=["eventno"])
    for indexdate in range(len(asc_roots)):
        aux = pd.DataFrame(
            data.loc[asc_roots.index[indexdate] : dec_roots.index[indexdate]],
            columns=[variable],
        )
        if (aux.index[-1] - aux.index[0]).total_seconds() / 3600 >= duration:
            events = pd.concat([events, aux])
            eventsno.loc[aux.index[0].year] += 1

    return events, eventsno


def isolines(data, iso_lines=[0]):
    """[summary]

    Args:
        data ([type]): [description]
        iso_lines (list, optional): [description]. Defaults to [0].

    Returns:
        [type]: [description]
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    cs = ax.contour(data["x"], data["y"], data["z"], iso_lines)
    # ax.contourf(data['x'], data['y'], data['z'])

    isolevels = dict()
    for i, j in enumerate(iso_lines):
        p = cs.collections[i].get_paths()[0]  # usually the longest path
        v = p.vertices
        isolevels[j] = pd.DataFrame(
            v[:, :2], index=np.arange(len(v)), columns=["x", "y"]
        )

    plt.close(fig)
    return isolevels


def pre_ensemble_plot(models, param, var_, fname=None):
    """[summary]

    Args:
        models ([type]): [description]
        param ([type]): [description]
        var_ ([type]): [description]
        fname ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    probs = [0.05, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    if var_.lower().startswith("d"):
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n = np.linspace(0, 1, 24 * 365.25)

    ppfs = dict()
    for prob in probs:
        df = pd.DataFrame(np.ones(len(n)) * prob, index=n, columns=["prob"])
        df["n"] = n
        ppfs[str(prob)] = df.copy()
        for model in models:
            param[var_][model] = str2fun(param[var_][model], None)
            res = stf.ppf(df.copy(), param[var_][model])
            # Transformed timeserie if required
            if param[var_][model]["transform"]["make"]:
                res[var_] = analysis.inverse_transform(res[[var_]], param[var_][model])
            ppfs[str(prob)].loc[:, model] = res[var_]

        ppfs[str(prob)]["mean"] = ppfs[str(prob)].loc[:, models].mean(axis=1)
        ppfs[str(prob)]["std"] = ppfs[str(prob)].loc[:, models].std(axis=1)

    return ppfs


def mkdir(path: str):
    """Create a folder with path name if not exists

    Args:
        path (str): name of the new folder
    """
    if path != "":
        os.makedirs(path, exist_ok=True)
    return


def smooth_1d(data, window_len=None, poly_order=3):
    """[summary]

    Args:
        data ([type]): [description]
        window_len ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    if window_len is None:
        # TODO: cambiar el 51 por un valor objetivo
        window_len = int(len(data) / 51)

    if not window_len % 2:
        window_len += 1

    from scipy.signal import savgol_filter

    sm_data = savgol_filter(data, window_len, poly_order)
    return sm_data


def nearest(data, point):
    idx_ = np.argmin(np.sqrt((data.x - point[0]) ** 2 + (data.y - point[1]) ** 2))
    return idx_


def date2julian(dates, calendar="julian"):
    """Convert datetimes to julian dates

    Args:
        dates ([type]): [description]
    """
    from datetime import date

    if calendar == "julian":
        julian_dates = np.zeros(len(dates))
        for i, date_ in enumerate(dates):
            julian_dates[i] = (
                date_.toordinal()
                - date(1, 1, 1).toordinal()
                + date_.hour / (24)
                + date_.second / (3600 * 24)
                + 367
            )
        dates = julian_dates

    elif calendar == "gregorian":
        if isinstance(dates, pd.DatetimeIndex):
            dates = num2julian(date2num(dates))
            dates.toordinal() - date(1, 1, 1).toordinal()
        elif isinstance(dates, pd.DataFrame):
            dates.index = dates.index.to_julian_date()
        else:
            raise ValueError("DatetimeIndex or DataFrame are required.")
    else:
        raise ValueError("Calendar can be Julian or Gregorian.")
    return dates


def get_params_bylabel(
    df, param, par, imod, cossen, first=None, pos=[0, 0, 0], ppf=False
):
    """Gets the parameters of the probability models by its name

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par (dict): the guess parameters of the probability model
        * imod (list): combination of modes for fitting
        * cossen (np.ndarray): the variability of the modes
        * first (boolean or None, optional): For fitting of first or n-order. Defaults to None (n-order).
        * pos (list, optional): location of the different probability models. Defaults to [0, 0, 0].
        * ppf (bool, optional): True if it is required the numerically computation of the cdf in a mesh. Defaults to False.

    Returns:
        * df (pd.DataFrame): the parameters
        * esc (list): weight of the probability models
    """

    for i in param["fun"].keys():
        if isinstance(param["fun"][i], str):
            param["fun"][i] = getattr(st, param["fun"][i])
        else:
            param["fun"][i] = param["fun"][i]

    modo, esc = imod, {}
    if param["extreme"]:
        # Parametros para el regimen extremal con un modelo de probabilidad de bi- o triparamétrico
        if first:  # TODO: la obtención de parametros para el orden 1
            df["shape"] = []
            df["loc"] = []
            if param["no_param"][0] == 2:
                df["xi2"] = []
            else:
                df["scale"] = []
                df["xi2"] = []
        else:
            df["shape"] = par[0] + np.dot(
                par[1 : modo[0] * 2 + 1], cossen[0 : modo[0] * 2, :]
            )
            df["shape_0"] = par[0]
            for ii, jj in enumerate(np.arange(1, modo[0] * 2, 2)):
                df["shape_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["shape_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["shape_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[jj - 1 : jj + 1, :]
                )
            df["loc"] = par[modo[0] * 2 + 1] + np.dot(
                par[modo[0] * 2 + 2 : modo[0] * 4 + 2], cossen[0 : modo[0] * 2, :]
            )
            df["loc_0"] = par[modo[0] * 2 + 1]
            for ii, jj in enumerate(np.arange(modo[0] * 2 + 2, modo[0] * 4 + 2, 2)):
                df["loc_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["loc_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["loc_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                )

            if param["no_param"][0] == 2:
                df["xi2"] = []
            else:
                df["scale"] = par[modo[0] * 4 + 2] + np.dot(
                    par[modo[0] * 4 + 3 : modo[0] * 6 + 3], cossen[0 : modo[0] * 2, :]
                )
                df["scale_0"] = par[modo[0] * 4 + 2]
                for ii, jj in enumerate(np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)):
                    df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["scale_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["scale_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )
                df["xi2"] = par[modo[0] * 6 + 3] + np.dot(
                    par[modo[0] * 6 + 4 : modo[0] * 6 + modo[1] * 2 + 4],
                    cossen[0 : modo[1] * 2, :],
                )
                df["xi2_0"] = par[modo[0] * 6 + 3]
                for ii, jj in enumerate(
                    np.arange(modo[0] * 6 + 4, modo[0] * 6 + modo[1] * 2 + 4, 2)
                ):
                    df["xi2_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["xi2_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["xi2_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )

        esc[1] = st.norm.cdf(par[-2])
        esc[2] = 1 - st.norm.cdf(par[-1])
        if param["no_param"][0] == 2:
            df["u1"] = []
            df["u2"] = []

            df["siggp1"] = []
            df["xi1"] = []
            df["siggp2"] = []
        else:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"], df["scale"])
            df["u2"] = param["fun"][1].ppf(
                st.norm.cdf(par[-1]), df["shape"], df["loc"], df["scale"]
            )

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"], df["scale"]
            )
            df["u1"][
                df["u1"] < df["siggp1"]
            ] = np.nan  # TODO: limitacion discontinuidades
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"], df["scale"]
            )

            pars_p = ["u1", "u2", "siggp1", "xi1", "siggp2"]
            for pari in pars_p:
                fft = np.fft.fft(df[pari]) / len(df[pari])
                a0 = np.real(fft[0])
                an = 2 * np.real(fft[1:])
                bn = -2 * np.imag(fft[1:])
                mod = np.sqrt(an**2 + bn**2)
                pha = np.arctan2(bn, an)

                df["{}_0".format(pari)] = a0
                for ii, jj in enumerate(np.arange(1, modo[0] + 1)):
                    df["{}_{}_mod".format(pari, jj)] = mod[ii]
                    df["{}_{}_pha".format(pari, jj)] = pha[ii]
                    df["{}_{}".format(pari, jj)] = np.dot(
                        [an[ii], bn[ii]], cossen[2 * ii : 2 * ii + 2, :]
                    )
    else:
        # TODO separar por orden de fourier
        # Parámetros para uno modelo de probabilidad bi- o triparamétrico
        # for i in range(param['no_fun']):
        if param["order"] == 0:
            modo = [0, 0, 0]
        elif not first:
            modo = [imod[pos[2]], imod[pos[2]], imod[pos[2]]]

        df["shape_0"] = par[pos[0]]
        if modo[pos[1]] != 0:
            df["shape"] = par[pos[0]] + np.dot(
                par[pos[0] + 1 : pos[0] + modo[pos[1]] * 2 + 1],
                cossen[0 : modo[pos[1]] * 2, :],
            )
            for ii, jj in enumerate(np.arange(1, modo[pos[1]] * 2, 2)):
                df["shape_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["shape_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["shape_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[jj - 1 : jj + 1, :]
                )
        df["loc_0"] = par[pos[0] + modo[pos[1]] * 2 + 1]

        if modo[pos[1] + 1] != 0:
            df["loc"] = par[pos[0] + modo[pos[1]] * 2 + 1] + np.dot(
                par[
                    pos[0]
                    + modo[pos[1]] * 2
                    + 2 : pos[0]
                    + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                    + 2
                ],
                cossen[0 : modo[pos[1] + 1] * 2, :],
            )
            for ii, jj in enumerate(np.arange(modo[0] * 2 + 2, modo[0] * 4 + 2, 2)):
                df["loc_{}_mod".format(ii + 1)] = np.sqrt(
                    par[jj] ** 2 + par[jj + 1] ** 2
                )
                df["loc_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                df["loc_{}".format(ii + 1)] = np.dot(
                    par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                )

        if param["no_param"][pos[2]] == 3:
            df["scale_0"] = par[pos[0] + modo[pos[1]] * 2 + 2]

            if first:
                if modo[pos[1] + 1] != 0:
                    df["scale"] = par[
                        pos[0] + np.sum(modo[pos[1] : pos[1] + 2]) * 2 + 2
                    ]
                if modo[pos[1] + 2] != 0:
                    df["scale"] = par[pos[0] + modo[pos[1]] * 2 + 2] + np.dot(
                        par[
                            pos[0]
                            + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                            + 3 : pos[0]
                            + np.sum(modo[pos[1] : pos[1] + 3]) * 2
                            + 3
                        ],
                        cossen[0 : modo[pos[1] + 2] * 2, :],
                    )
                    for ii, jj in enumerate(
                        np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)
                    ):
                        df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                            par[jj] ** 2 + par[jj + 1] ** 2
                        )
                        df["scale_{}_pha".format(ii + 1)] = np.arctan2(
                            par[jj + 1], par[jj]
                        )
                        df["scale_{}".format(ii + 1)] = np.dot(
                            par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                        )

            else:
                # print(imod, len(par), pos[0]+np.sum(modo[pos[1]:pos[1]+2])*2+2, pos[0]+np.sum(modo)*2+3)
                df["scale"] = par[
                    pos[0] + np.sum(modo[pos[1] : pos[1] + 2]) * 2 + 2
                ] + np.dot(
                    par[
                        pos[0]
                        + np.sum(modo[pos[1] : pos[1] + 2]) * 2
                        + 3 : pos[0]
                        + np.sum(modo) * 2
                        + 3
                    ],
                    cossen[0 : modo[pos[2]] * 2, :],
                )
                for ii, jj in enumerate(np.arange(modo[0] * 4 + 3, modo[0] * 6 + 3, 2)):
                    df["scale_{}_mod".format(ii + 1)] = np.sqrt(
                        par[jj] ** 2 + par[jj + 1] ** 2
                    )
                    df["scale_{}_pha".format(ii + 1)] = np.arctan2(par[jj + 1], par[jj])
                    df["scale_{}".format(ii + 1)] = np.dot(
                        par[jj : jj + 2], cossen[2 * ii : 2 * ii + 2, :]
                    )

        if param["no_fun"] == 1:
            esc = param["weights"][0]
        else:
            if pos[2] == 0:  # TODO: para solo dos funciones
                # esc = par[-len(param['pesos'])+1]
                esc = param["weights"][0]
            elif pos[2] == len(param["weights"]) - 1:
                # esc = 1- np.sum(par[-len(param['pesos'])+1:])
                esc = param["weights"][1]
            else:
                # esc = np.sum(par[-len(param['pesos'])+1:-len(param['pesos'])+1+pos[2]])
                esc = param["weights"][1]

    return df, esc


def mean_dt_param(B, Q):
    nmodels = len(B)
    norders = np.max([np.shape(Bm) for Bm in B], axis=0)

    Bs = np.zeros([norders[0], norders[1], nmodels])
    # Qs = np.zeros([norders[0], norders[0], nmodels])

    for i in range(nmodels):
        Bs[:, : np.shape(B[i])[1], i] = B[i]

    B, Q = np.mean(Bs, axis=2), np.mean(Q, axis=0)
    return B, Q, int((norders[1] - 1) / norders[0])


def rmse(a, b):

    len_ = len(a)
    value_ = np.sqrt(np.sum((a - b) ** 2) / len_)
    return value_
