import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.utils import auxiliar, read, save
from scipy.integrate import quad
from scipy.optimize import differential_evolution, dual_annealing, minimize, shgo

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

warnings.filterwarnings("ignore")

"""This file is part of MarineTools.

MarineTools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MarineTools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MarineTools.  If not, see <https://www.gnu.org/licenses/>.
"""


def st_analysis(df: pd.DataFrame, param: dict):
    """Fits stationary simple or mixed models

    Args:
        - df (pd.DataFrame): raw time series
        - param (dict): guess parameters for the analysis. More detailed information in 'analysis'.

    Returns:
        * The parameters and the mode of the fit

    """
    par0 = []

    df_sort = df[param["var"]].sort_values(ascending=True)
    p = np.linspace(0, 1, len(df_sort))

    if (param["basis_function"]["order"] == 0) and (param["reduction"] == False):
        if param["no_fun"] == 1:
            par0 = param["fun"][0].fit(df[param["var"]])
        elif param["no_fun"] > 1:
            percentiles = np.hstack([0, np.cumsum(param["ws_ps"]), 1])
            for i in param["fun"].keys():
                filtro = (p >= percentiles[i]) & (p <= percentiles[i + 1])
                par = param["fun"][i].fit(df_sort[filtro])
                par0 = np.hstack([par0, par]).tolist()
    elif param["piecewise"]:
        thresholds = np.hstack(
            [df[param["var"]].min(), param["ws_ps"], df[param["var"]].max()]
        )
        # df, _ = auxiliar.nonstationary_ecdf(df, percentiles, param["var"])
        for i in param["fun"].keys():
            filtro = (df[param["var"]] >= thresholds[i]) & (
                df[param["var"]] <= thresholds[i + 1]
            )
            par = best_params(df.loc[filtro, param["var"]], 25, param["fun"][i])
            par0 = np.hstack([par0, par]).tolist()

        if not param["fix_percentiles"]:
            par0 = np.hstack([par0, param["ws_ps"]])
    else:
        if param["reduction"]:
            percentiles = np.hstack([0, param["ws_ps"]])
            par1 = param["fun"][1].fit(
                df_sort[((p >= percentiles[1]) & (p <= percentiles[2]))]
            )

            par2 = 1 / np.max(df[param["var"]])
            # par2 = param["fun"][2].fit(df_sort[(p >= percentiles[2])])[0]
            par0 = [
                st.norm.ppf(param["ws_ps"][0]),
                st.norm.ppf(param["ws_ps"][-1]),
            ]
            par0 = np.hstack([par1, par2, par0]).tolist()

            # bnds = [[] for _ in range(len(par0))]
            # if param["optimization"]["bounds"] is False:
            #     bnds = None
            # else:
            #     for i in range(0, len(par0)):
            #         # if i >= param["no_tot_param"]:
            #         bnds[i] = [
            #             par0[i] - param["optimization"]["bounds"],
            #             par0[i] + param["optimization"]["bounds"],
            #         ]
            #         # else:
            #         #     bnds[i] = [
            #         #         0,
            #         #         par0[i] + param["optimization"]["bounds"],
            #         #     ]

            #     bnds = tuple(bnds)

            # t_expans = params_t_expansion([0, 0, 0], param, df)
            # res = minimize(
            #     nllf,
            #     par0.copy(),
            #     args=(df, [0, 0, 0], param, t_expans),
            #     method="SLSQP",
            #     bounds=bnds,
            #     options={
            #         # "ftol": param["optimization"]["ftol"],
            #         "eps": 1e-2,  # param["optimization"]["eps"],
            #         # "maxiter": param["optimization"]["maxiter"],
            #     },
            # )
            # par0 = res.x.tolist()
        else:
            if param["no_fun"] == 1:
                par0 = param["fun"][0].fit(df[param["var"]])
            else:
                # percentiles = np.hstack([0, param["ws_ps"], 1])
                res, _ = auxiliar.nonstationary_ecdf(
                    df, param["var"], pemp=param["ws_ps"]
                )
                if len(param["ws_ps"]) == 1:
                    res.columns = ["u1"]
                    df["u1"] = 0.0
                    for n_ in res.index:
                        df.loc[df.n == n_, "u1"] = res.loc[n_, "u1"]
                elif len(param["ws_ps"]) == 2:
                    # res.columns = ["u1", "u2"]
                    df["u1"], df["u2"] = 0, 0
                    for n_ in res.index:
                        df.loc[df.n == n_, "u1"] = res.loc[n_, "u1"]
                        df.loc[df.n == n_, "u2"] = res.loc[n_, "u2"]
                if param["no_fun"] == 2:
                    ibody = df[param["var"]] <= df["u1"]
                    iutail = df[param["var"]] > df["u1"]

                    parb = param["fun"][0].fit(df.loc[ibody, param["var"]])
                    if not param["type"] == "circular":
                        part = param["fun"][1].fit(
                            df.loc[iutail, param["var"]] - df.loc[iutail, "u1"]
                        )
                    else:
                        part = param["fun"][1].fit(df.loc[iutail, param["var"]])
                    par0 = np.hstack([par0, parb, part]).tolist()
                    if not param["fix_percentiles"]:
                        par0 = np.hstack([par0, st.norm.ppf(param["ws_ps"])])

                else:
                    # ----------------------------------------------------------------------
                    # With three PMs it is required that thresholds be related to the data
                    # ----------------------------------------------------------------------
                    iltail = df[param["var"]] < df["u1"]
                    ibody = (df[param["var"]] >= df["u1"]) & (
                        df[param["var"]] <= df["u2"]
                    )
                    iutail = df[param["var"]] > df["u2"]

                    if not param["type"] == "circular":
                        parl = param["fun"][0].fit(
                            df.loc[iltail, "u1"] - df.loc[iltail, param["var"]]
                        )
                    else:
                        parl = param["fun"][0].fit(df.loc[iltail, param["var"]])
                    parb = param["fun"][1].fit(df.loc[ibody, param["var"]])

                    if not param["type"] == "circular":
                        part = param["fun"][2].fit(
                            df.loc[iutail, param["var"]] - df.loc[iutail, "u2"]
                        )
                    else:
                        part = param["fun"][2].fit(df.loc[iutail, param["var"]])
                    par0 = np.hstack([par0, parl, parb, part]).tolist()

                    if not param["fix_percentiles"]:
                        par0 = np.hstack([par0, st.norm.ppf(param["ws_ps"])])

    # if param[
    #     "guess"
    # ]:  # checked outside because some previous parameters are required later
    #     logger.info("Initial guess computed: " + str(par0))
    #     par0 = param["p0"]
    #     logger.info("Initial guess given: " + str(par0))
    # else:
    #     logger.info("Initial guess computed: " + str(par0))
    mode = np.zeros(param["no_fun"], dtype=int).tolist()

    # if (not param["guess"]) & (any(np.abs(par0) > 20)):
    #     warnings.warn(
    #         "Parameters of the initial guess are high. The convergence is not ensured."
    #         + "It is recommended: (i) modify the percentiles, or (ii) increase the"
    #         + "bounds."
    #     )

    return df, par0, mode


def fit_(data: pd.DataFrame, bins: int, model: str):
    """Fits a simple probability model and computes the sse with the empirical pdf

    Args:
        * data (pd.DataFrame): raw time series
        * bins (int): no. of bins for the histogram
        * model (string): name of the probability model

    Returns:
        * results (np.array): the parameters computed
    """

    y, x = np.histogram(data, bins=bins, density=True)
    xq = np.diff(x)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    yq = np.cumsum(xq * y)

    if model is st.genpareto:
        params = model.fit(data, 0.01, loc=np.min(data))
    else:
        params = model.fit(data)
    cdf = model.cdf(x, loc=params[-2], scale=params[-1], *params[:-2])
    sse = np.sum((yq - cdf) ** 2)
    if np.isnan(sse):
        sse = 1e10

    results = np.zeros(len(params) + 1)
    results[: int(len(params) + 1)] = np.append(sse, params)
    return results


def nonst_analysis(df: pd.DataFrame, param: dict):
    """Makes a non-stationary analysis for several modes

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): guess parameters of the model

    Returns:
        * param (dict): parameters of the model
    """

    par, bic, nllf = nonst_fit(df, param)
    if not param["initial_parameters"]["make"]:
        if param["basis_function"]["order"] > 1:
            if param["bic"]:
                mode = min(bic, key=bic.get)
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
            else:
                mode = min(nllf, key=nllf.get)
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
        elif param["basis_function"]["order"] == 1:
            if param["bic"]:
                mode = list(bic.keys())[0]
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
            else:
                mode = list(nllf.keys())[0]
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
        else:
            mode = np.zeros(param["no_fun"], dtype=int)
            param["par"] = list(par)
            param["mode"] = [int(i) for i in mode]

        all_ = []
        for i in bic.keys():
            all_.append([str(i), bic[i], list(par[i])])

        param["all"] = all_
    else:
        param["par"] = list(par)
        param["all"] = [str(param["initial_parameters"]["mode"]), bic, list(par)]

    return param


def nonst_fit(df: pd.DataFrame, param: dict):
    """Fits a non-stationary probability model

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the guess parameters of the model

    Returns:
        * par (list): the best parameters
        * bic (float): the Bayesian information criteria
    """

    par, bic = {}, {}
    # ----------------------------------------------------------------------------------
    # Check if a mode is not given. Run the general analysis
    # ----------------------------------------------------------------------------------
    if not param["initial_parameters"]["make"]:

        if param["basis_function"]["order"] >= 1:
            par, nllf, mode = fourier_expansion(df, par, param)

            logndat = np.log(len(df[param["var"]]))
            for i in mode:
                bic[tuple(i)] = 2 * nllf[tuple(i)] + logndat * (len(par[tuple(i)]))
        else:
            par = param["par"]
            bic[0] = 0
    else:
        # ------------------------------------------------------------------------------
        # Check if a mode is given. Optimize a specific mode (and parameters)
        # ------------------------------------------------------------------------------
        if not any(param["initial_parameters"]["par"]):
            param = fourier_initialization(df, param)
        nllf = 1e9
        par, nllf, _ = fourier_expansion(df, param["par"], param)
        bic = 2 * nllf + np.log(len(df[param["var"]])) * (len(par))

    return par, bic, nllf


def fourier_initialization(df, param):
    """Initialize the parameters for trigonometric expansion at a given order. Makes
    fast the optimization

    Args:
        df (_type_): _description_
        param (_type_): _description_
    """
    # To be added
    timestep = 1 / 365.25
    wlen = 14 / 365.25  # ventana mensual
    time_ = np.arange(0, 1, timestep)
    logger.info("Initializing parameters through Fourier Series on a moving window.")

    for index_, i in enumerate(time_):
        # print(index_, len(time_))
        if i >= (1 - wlen):
            final_offset = i + wlen - 1
            mask = ((df["n"] >= i - wlen) & (df["n"] <= i + wlen)) | (
                df["n"] <= final_offset
            )
        elif i <= wlen:
            initial_offset = i - wlen
            mask = ((df["n"] >= i - wlen) & (df["n"] <= i + wlen)) | (
                df["n"] >= 1 + initial_offset
            )
        else:
            mask = (df["n"] >= i - wlen) & (df["n"] <= i + wlen)

        _, par_, _ = st_analysis(df.loc[mask], param)
        if index_ == 0:
            res = pd.DataFrame(0, index=time_, columns=np.arange(len(par_)))

        res.loc[i, :] = par_
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(res)
    # plt.show()
    parameters = []
    for var_ in range(param["no_tot_param"]):
        # Compute the Fast Fourier Transform
        mean_ = np.mean(res.loc[:, var_])
        coefs = np.fft.fft(res.loc[:, var_] - mean_)

        N = len(res.loc[:, var_])
        # Choose one side of the spectra
        cn = np.ravel(coefs[0 : N // 2] / N)

        an, bn = 2 * np.real(cn), -2 * np.imag(cn)

        an = an[: param["initial_parameters"]["mode"][0] + 1]
        bn = bn[: param["initial_parameters"]["mode"][0] + 1]

        parameters = np.hstack([parameters, mean_])
        for order_k in range(param["initial_parameters"]["mode"][0]):
            parameters = np.hstack([parameters, an[order_k + 1], bn[order_k + 1]])

        if param["initial_parameters"]["plot"]:
            # ---- Checking function ----
            t = res.index.values
            cosine_funtion = np.zeros(len(t)) + np.mean(res.loc[:, var_])
            for i, ii in enumerate(an):
                cosine_funtion += an[i] * np.cos(2 * np.pi * i * t) + bn[i] * np.sin(
                    2 * np.pi * i * t
                )

            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(t, cosine_funtion, label=str(var_))
            plt.plot(res[var_])
            plt.legend()
            plt.show()

    if param["reduction"]:
        # Adding the weights
        parameters = np.hstack([parameters, res.iloc[0, -2], res.iloc[0, -1]])
        param["initial_parameters"]["mode"] = (
            param["initial_parameters"]["mode"][0],
            param["initial_parameters"]["mode"][0],
        )

    param["initial_parameters"]["par"] = parameters.tolist()
    param["par"] = param["initial_parameters"]["par"]
    param["mode"] = param["initial_parameters"]["mode"]
    # logger.info("Initial parameters obtained.")
    # logger.info(param["par"])
    return param


def fourier_expansion(data: pd.DataFrame, par: list, param: dict):
    """Prepares the initial guess and estimates the n-order non-stationary parameters

    Args:
        * data (pd.DataFrame): raw data
        * par (list): the guess parameters of the probability model
        * param (dict): the parameters of the analysis

    Returns:
        * par (dict): parameters of the first order
        * llf (dict): log-likelihood value of the first order
        * mode (dict): parameter of the first order
    """
    nllf = {}
    if not param["initial_parameters"]["make"]:
        mode = []
        if param["no_fun"] == 1:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((i,))
        elif param["no_fun"] == 2:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((i, 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append((param["basis_function"]["order"], i))
        else:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((1, i, 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append((i, param["basis_function"]["order"], 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append(
                    (
                        param["basis_function"]["order"],
                        param["basis_function"]["order"],
                        i,
                    )
                )

        ind_ = np.diff(np.vstack([mode[0], mode]), axis=0)

        ind_[ind_ < 0] = 0
        par0, pos = {}, []
        # ------------------------------------------------------------------------------
        # Compute the position after which the parameters will be expanded
        # ------------------------------------------------------------------------------
        for ind_mode, imode in enumerate(mode):
            no_prob_model = np.where(ind_[ind_mode] == 1)[0]

            if not no_prob_model.size == 0:
                pos.append(no_prob_model[0])
            else:
                pos.append(0)

            if not ind_mode:
                nllf[imode] = 1e10
                par[imode] = param["par"]

        # ------------------------------------------------------------------------------
        # Expand in Fourier Series and fit the parameters
        # ------------------------------------------------------------------------------
        for ind_mode, imode in enumerate(mode):
            if ind_mode - 1 < 0:
                comp_ = None
            else:
                comp_ = mode[ind_mode - 1]
            par0[imode] = initial_params(param, par, pos[ind_mode], imode, comp_)
            logger.info("Mode " + str(imode) + " non-stationary")
            if ind_mode == 0:
                par[imode], nllf[imode] = fit(
                    data, param, par0[imode], imode, nllf[imode]
                )
            else:
                par[imode], nllf[imode] = fit(
                    data, param, par0[imode], imode, nllf[comp_]
                )
    else:
        mode = [tuple(param["initial_parameters"]["mode"])]
        logger.info("Mode " + str(mode[0]) + " non-stationary")
        par, nllf = fit(data, param, par, mode[0], 1e10)

    return par, nllf, mode[1:]


def initial_params(param: dict, par: list, pos: int, imode: list, comp: list):
    """Prepares the parameters from previous fit to the following

    Args:
        * param (dict): the parameters of the analysis
        * par (list): the guess parameters of the probability model
        * pos (int): location to place the new terms for the fit
        * imode (list): combination of modes for fitting
        * comp (list): components of the current mode

    Returns:
        * par0 (list): initial guess parameters
    """

    if bool(
        [
            meth
            for meth in ["trigonometric", "modified"]
            if (meth in param["basis_function"]["method"])
        ]
    ):
        add_ = [np.random.rand(1)[0] * 1e-5, np.random.rand(1)[0] * 1e-5]
        npars_ = 2
    else:
        add_ = np.random.rand(1)[0] * 1e-5
        npars_ = 1

    if comp is None:
        comp = imode
        par0 = par[comp]
        for i in range(param["no_tot_param"]):
            par0 = np.insert(
                par0,
                i * (npars_ + 1) + 1,
                add_,
            )

    elif param["reduction"]:
        if pos == 0:
            loc = npars_ * (imode[0] - 1) + 1
            par0 = np.insert(
                par[comp],
                loc,
                add_,
            )
            for _ in range(param["no_param"][0] - 1):
                loc = loc + npars_ * (imode[0] - 1) + (npars_ + 1)
                par0 = np.insert(
                    par0,
                    loc,
                    add_,
                )
        else:
            loc = (
                param["no_param"][0] * (imode[0] * npars_ + 1)
                + npars_ * (imode[1] - 1)
                + 1
            )
            par0 = np.insert(
                par[comp],
                loc,
                add_,
            )
    else:
        par0 = par[comp]
        loc = 0
        for j in range(pos):
            loc += (npars_ * imode[j] + 1) * param["no_param"][j]

        for i in range(param["no_param"][pos]):
            loc = loc + npars_ * (imode[pos] - 1) + 1
            par0 = np.insert(
                par0,
                loc,
                add_,
            )
            loc += npars_
    return par0


def matching_lower_bound(par: dict):
    """Matching conditions between two probability models (PMs). Lower refers to the
    low tail-body PMs in the case of fitting three PMs.

    Args:
        par (dict): parameters of the usual dictionary format

    Returns:
        [type]: [description]
    """

    # ----------------------------------------------------------------------------------
    # Obtaining the parameters
    # ----------------------------------------------------------------------------------
    t_expans = params_t_expansion(
        mode, param, df.sort_values(by="n").drop_duplicates(subset=["n"]).loc[:, "n"]
    )
    df_, _ = get_params(
        df.sort_values(by="n").drop_duplicates(subset=["n"]),
        param,
        par,
        mode,
        t_expans,
    )
    # ----------------------------------------------------------------------------------
    # Applying the restrictions along "n"
    # ----------------------------------------------------------------------------------
    if not param["reduction"]:
        # ------------------------------------------------------------------------------
        # Using two PMs, the body PM is the first one and the second PM is used as upper
        # tail model. Using three PMS, the body PM is the center one.
        # ------------------------------------------------------------------------------
        if len(df_) == 2:
            f_body, f_tail = 0, 1
        else:
            f_body, f_tail = 1, 0

        if len(df_) == 2:
            if param["no_param"][f_body] == 2:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
            else:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )

            if param["no_param"][f_tail] == 2:
                ft_u1 = param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
            else:
                ft_u1 = param["fun"][f_tail].pdf(
                    0,
                    df_[f_tail]["s"],
                    df_[f_tail]["l"],
                    df_[f_tail]["e"],
                )
        else:
            if param["no_param"][f_body] == 2:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
            else:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )

            if param["no_param"][f_tail] == 2:
                ft_u1 = param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
            else:
                ft_u1 = param["fun"][f_tail].pdf(
                    0, df_[f_tail]["s"], df_[f_tail]["l"], df_[f_tail]["e"]
                )

    constraints_ = np.sqrt(1 / len(ft_u1) * np.sum((fc_u1 - Fc_u1 * ft_u1) ** 2))

    return constraints_


# def matching_upper_bound(par: dict):
#     """Matching conditions between two probability models (PMs). Upper refers to the
#     low tail-body PMs for fitting three PMs.

#     Args:
#         par (dict): parameters of the usual dictionary format

#     Returns:
#         [type]: [description]
#     """
#     # ----------------------------------------------------------------------------------
#     # Obtaining the parameters
#     # ----------------------------------------------------------------------------------
#     t_expans = params_t_expansion(
#         mode, param, df.sort_values(by="n").drop_duplicates(subset=["n"]).loc[:, "n"]
#     )
#     df_, _ = get_params(
#         df.sort_values(by="n").drop_duplicates(subset=["n"]),
#         param,
#         par,
#         mode,
#         t_expans,
#     )

#     # ----------------------------------------------------------------------------------
#     # Applying the restrictions along "n"
#     # ----------------------------------------------------------------------------------
#     if not param["reduction"]:
#         # ------------------------------------------------------------------------------
#         # Using two PMs, the body PM is the first one and the second PM is used as upper
#         # tail model. Using three PMS, the body PM is the center one.
#         # ------------------------------------------------------------------------------
#         f_body, f_tail = 1, 2

#         if param["no_param"][f_body] == 2:
#             fc_u2 = param["fun"][f_body].pdf(
#                 df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"]
#             )
#             Fc_u2 = param["fun"][f_body].cdf(
#                 df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"]
#             )
#         else:
#             fc_u2 = param["fun"][f_body].pdf(
#                 df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"], df_[f_body]["e"]
#             )
#             Fc_u2 = param["fun"][f_body].cdf(
#                 df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"], df_[f_body]["e"]
#             )

#         if param["no_param"][f_tail] == 2:
#             ft_u2 = -param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
#         else:
#             ft_u2 = -param["fun"][f_tail].pdf(
#                 0, df_[f_tail]["s"], df_[f_tail]["l"], df_[f_tail]["e"]
#             )

#     constraints_ = np.sqrt(1 / len(ft_u2) * np.sum((ft_u2 * Fc_u2 - fc_u2) ** 2))

#     return constraints_


def fit(df_: pd.DataFrame, param_: dict, par0: list, mode_: list, ref: int):
    """Fits the data to the given probability model

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par0 (list): the guess parameters of the probability model
        * mode (list): components of the current mode
        * ref (int): log-likelihood value of the reference order

    Returns:
        * res['x'] (list): the fit parameters
        * res['fun'] (float): the value of the log-likelihood function
    """
    # ----------------------------------------------------------------------------------
    # Creating boundaries for optimization
    # ----------------------------------------------------------------------------------
    bnds = [[] for _ in range(len(par0))]
    if param_["optimization"]["bounds"] is False:
        bnds = None
    else:
        for i in range(0, len(par0)):
            bnds[i] = (
                par0[i] - param_["optimization"]["bounds"],
                par0[i] + param_["optimization"]["bounds"],
            )

    global df, param, mode, t_expans
    df, param, mode = df_, param_, mode_
    t_expans = params_t_expansion(mode, param, df["n"])

    # ----------------------------------------------------------------------------------
    # For assuring that local minimum of previous optimizations not be accepted during
    # the current optimization mode
    # ----------------------------------------------------------------------------------
    if ref == 1e10:
        ref = 10 * len(df)
    else:
        ref -= np.abs(ref) * 1e-4

    nllf_, j = 1e10, 0
    fixed = par0
    res, nllfv = {}, {}
    if param["constraints"]:
        if param["no_fun"] == 1:
            constraints_ = []
        elif param["no_fun"] == 2:
            constraints_ = []  # [
            #     {"type": "eq", "fun": lambda x: matching_lower_bound(x)},
            # ]
        else:
            constraints_ = []  # [
            #     {"type": "eq", "fun": lambda x: matching_lower_bound(x)},
            #     {"type": "eq", "fun": lambda x: matching_upper_bound(x)},
            # ]
    else:
        constraints_ = []

    while not ((nllf_ < ref) | (j >= param["optimization"]["giter"])):
        j += 1
        # ------------------------------------------------------------------------------
        # Optimize NLLF using the given algorithm
        # ------------------------------------------------------------------------------
        if param["optimization"]["method"] == "SLSQP":
            res[j] = minimize(
                nllf,
                par0,
                args=(df, mode, param, t_expans),
                bounds=bnds,
                constraints=constraints_,
                method="SLSQP",
                options={
                    "ftol": param["optimization"]["ftol"],
                    "eps": param["optimization"]["eps"],
                    "maxiter": param["optimization"]["maxiter"],
                },
            )
        elif param["optimization"]["method"] == "dual_annealing":
            res[j] = dual_annealing(
                nllf,
                bnds,
                x0=par0,
                args=(df, mode, param, t_expans),
                maxiter=int(param["optimization"]["maxiter"]),
            )
        elif param["optimization"]["method"] == "differential_evolution":
            res[j] = differential_evolution(
                nllf, bnds, args=(df, mode, param, t_expans)
            )
        elif param["optimization"]["method"] == "shgo":
            res[j] = shgo(nllf, par0, args=(df, mode, param, t_expans))

        # ------------------------------------------------------------------------------
        # Check whether the algorithm succesfully run
        # ------------------------------------------------------------------------------
        nllfv[j] = res[j]["fun"]
        if res[j]["success"] and not (
            res[j]["fun"] == 1e10 or res[j]["fun"] == 0.0 or res[j]["fun"] == -0.0
        ):
            nllf_ = res[j]["fun"]
        elif res[j]["message"] == "Iteration limit exceeded":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif res[j]["message"] == "Iteration limit reached":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif res[j]["message"] == "Positive directional derivative for linesearch":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif (res[j]["message"] == "Inequality constraints incompatible") & (
            res[j]["fun"] < ref
        ):
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        else:
            nllf_ = 1e10

        if (
            res[j]["fun"] == 1e10
            or res[j]["fun"] == 0.0
            or res[j]["fun"] == -0.0
            or any(np.isnan(res[j].x))
        ):
            res[j]["message"] = "No valid combination of parameters"
            nllf_ = 1e10

        par0 = fixed + (-1) ** np.random.uniform(1) * (np.random.rand(1) * 1e-4 * j)
        if param_["optimization"]["bounds"] is False:
            bnds = None
        else:
            for i in range(0, len(par0)):
                bnds[i] = (
                    par0[i] - param["optimization"]["bounds"],
                    par0[i] + param["optimization"]["bounds"],
                )

        print_message(res[j], j, mode, ref, param)

    nllf_min = min(nllfv, key=nllfv.get)
    if nllfv[nllf_min] > ref:
        res["x"] = par0
        res["fun"] = ref
    else:
        res = res[nllf_min]

    return res["x"], res["fun"]


def nllf(par, df, imod, param, t_expans):
    """Computes the negative log-likelyhood value

    Args:
        * par (dict): the guess parameters of the probability model
        * df (pd.DataFrame): raw data
        * imod (list): combination of modes for fitting
        * param (dict): the parameters of the analysis
        * t_expans (np.ndarray): the variability of the modes

    Returns:
        * nllf (float): negative log-likelyhood value
    """
    # ----------------------------------------------------------------------------------
    # Obtaining the parameters
    # ----------------------------------------------------------------------------------
    df, esc = get_params(df, param, par, imod, t_expans)
    cweight_ = False
    # for key_ in esc.keys():
    #     if any(esc[key_] <= 0):
    #         cweight_ = True

    if np.isnan(par).any() | cweight_:
        nllf = 1e10
    elif param["reduction"]:
        # ------------------------------------------------------------------------------
        # Compute the NLLF reducing some parameters of the probability models
        # ------------------------------------------------------------------------------
        idb = (df[param["var"]] <= df["u2"]) & (df[param["var"]] >= df["u1"])
        idgp1 = df[param["var"]] < df["u1"]
        idgp2 = df[param["var"]] > df["u2"]

        if (
            any(
                df["xi1"][idgp1]
                * (df[param["var"]][idgp1] - df["u1"][idgp1])
                / df["siggp1"][idgp1]
                >= 1
            )
            | any(
                df["xi2"][idgp2]
                * (df[param["var"]][idgp2] - df["u2"][idgp2])
                / df["siggp2"][idgp2]
                <= -1
            )
            | any(df["u1"] <= 0)
            | any(df["u1"] >= df["u2"])
        ):
            nllf = 1e10
        else:
            if param["no_param"][0] == 2:
                lpdf = param["fun"][1].logpdf(
                    df[param["var"]][idb], df["shape"][idb], df["loc"][idb]
                )
            else:
                lpdf = param["fun"][1].logpdf(
                    df[param["var"]][idb],
                    df["shape"][idb],
                    df["loc"][idb],
                    df["scale"][idb],
                )

            lpdf = np.append(
                lpdf,
                np.log(esc[1])
                + param["fun"][0].logpdf(
                    df["u1"][idgp1] - df[param["var"]][idgp1],
                    df["xi1"][idgp1],
                    scale=df["siggp1"][idgp1],
                ),
            )
            lpdf = np.append(
                lpdf,
                np.log(esc[2])
                + param["fun"][2].logpdf(
                    df[param["var"]][idgp2],
                    df["xi2"][idgp2],
                    loc=df["u2"][idgp2],
                    scale=df["siggp2"][idgp2],
                ),
            )
            nllf = -np.sum(lpdf)
    else:
        # ------------------------------------------------------------------------------
        # Compute the NLLF without reducing any parameter of the probability models
        # ------------------------------------------------------------------------------
        nllf, en, lpdf = 0, 0, 0

        if param["constraints"]:
            # if not (param["type"] == "circular"):
            # --------------------------------------------------------------------------
            # Two PMs
            # --------------------------------------------------------------------------
            if len(df) == 2:
                idgp1 = df[0][param["var"]] <= df[0]["u1"]
                idgp2 = df[0][param["var"]] > df[0]["u1"]

                if param["no_param"][0] == 2:
                    lpdf += np.sum(
                        param["fun"][0].logpdf(
                            df[0].loc[idgp1, param["var"]],
                            df[0].loc[idgp1, "s"],
                            df[0].loc[idgp1, "l"],
                        )
                        + np.log(esc[0])
                    )
                else:
                    lpdf += np.sum(
                        param["fun"][0].logpdf(
                            df[0].loc[idgp1, param["var"]],
                            df[0].loc[idgp1, "s"],
                            df[0].loc[idgp1, "l"],
                            df[0].loc[idgp1, "e"],
                        )
                        + np.log(esc[0])
                    )

                if param["no_param"][1] == 2:
                    lpdf += np.sum(
                        param["fun"][1].logpdf(
                            df[1].loc[idgp2, param["var"]] - df[1].loc[idgp2, "u1"],
                            df[1].loc[idgp2, "s"],
                            df[1].loc[idgp2, "l"],
                        )
                        + np.log(esc[1])
                    )
                else:
                    lpdf += np.sum(
                        param["fun"][1].logpdf(
                            df[1].loc[idgp2, param["var"]] - df[1].loc[idgp2, "u1"],
                            df[1].loc[idgp2, "s"],
                            df[1].loc[idgp2, "l"],
                            df[1].loc[idgp2, "e"],
                        )
                        + np.log(esc[1])
                    )
            else:
                # ----------------------------------------------------------------------
                # Three PMs
                # ----------------------------------------------------------------------
                iltail = df[0][param["var"]] < df[0]["u1"]
                ibody = (df[1][param["var"]] >= df[1]["u1"]) & (
                    df[1][param["var"]] <= df[1]["u2"]
                )
                iutail = df[2][param["var"]] > df[2]["u2"]

                if param["no_param"][0] == 2:
                    # lpdf += np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][0].logpdf(
                    #             df[0].loc[iltail, "u1"]
                    #             - df[0].loc[iltail, param["var"]],
                    #             df[0].loc[iltail, "s"],
                    #             df[0].loc[iltail, "l"],
                    #         )
                    #         + np.log(esc[0]),
                    #     ]
                    # )
                    lpdf += param["fun"][0].logpdf(
                        df[0].loc[iltail, "u1"] - df[0].loc[iltail, param["var"]],
                        df[0].loc[iltail, "s"],
                        df[0].loc[iltail, "l"],
                    ) + np.log(esc[0])
                else:
                    # lpdf = np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][0].logpdf(
                    #             df[0].loc[iltail, "u1"]
                    #             - df[0].loc[iltail, param["var"]],
                    #             df[0].loc[iltail, "s"],
                    #             df[0].loc[iltail, "l"],
                    #             df[0].loc[iltail, "e"],
                    #         )
                    #         + np.log(esc[0]),
                    #     ]
                    # )
                    lpdf += param["fun"][0].logpdf(
                        df[0].loc[iltail, "u1"] - df[0].loc[iltail, param["var"]],
                        df[0].loc[iltail, "s"],
                        df[0].loc[iltail, "l"],
                        df[0].loc[iltail, "e"],
                    ) + np.log(esc[0])

                if param["no_param"][1] == 2:
                    # lpdf = np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][1].logpdf(
                    #             df[1].loc[ibody, param["var"]],
                    #             df[1].loc[ibody, "s"],
                    #             df[1].loc[ibody, "l"],
                    #         ),
                    #     ]
                    # )
                    lpdf += param["fun"][1].logpdf(
                        df[1].loc[ibody, param["var"]],
                        df[1].loc[ibody, "s"],
                        df[1].loc[ibody, "l"],
                    )
                else:
                    # lpdf = np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][1].logpdf(
                    #             df[1].loc[ibody, param["var"]],
                    #             df[1].loc[ibody, "s"],
                    #             df[1].loc[ibody, "l"],
                    #             df[1].loc[ibody, "e"],
                    #         ),
                    #     ]
                    # )
                    lpdf += param["fun"][1].logpdf(
                        df[1].loc[ibody, param["var"]],
                        df[1].loc[ibody, "s"],
                        df[1].loc[ibody, "l"],
                        df[1].loc[ibody, "e"],
                    )

                if param["no_param"][2] == 2:
                    # lpdf = np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][2].logpdf(
                    #             df[1].loc[iutail, param["var"]]
                    #             - df[1].loc[iutail, "u2"],
                    #             df[1].loc[iutail, "s"],
                    #             df[1].loc[iutail, "l"],
                    #         )
                    #         + np.log(esc[2]),
                    #     ]
                    # )
                    lpdf += param["fun"][2].logpdf(
                        df[2].loc[iutail, param["var"]] - df[2].loc[iutail, "u2"],
                        df[2].loc[iutail, "s"],
                        df[2].loc[iutail, "l"],
                    ) + np.log(esc[2])
                else:
                    # lpdf = np.hstack(
                    #     [
                    #         lpdf,
                    #         param["fun"][2].logpdf(
                    #             df[1].loc[iutail, param["var"]]
                    #             - df[1].loc[iutail, "u2"],
                    #             df[1].loc[iutail, "s"],
                    #             df[1].loc[iutail, "l"],
                    #             df[1].loc[iutail, "e"],
                    #         )
                    #         + np.log(esc[2]),
                    #     ]
                    # )
                    lpdf += param["fun"][2].logpdf(
                        df[2].loc[iutail, param["var"]] - df[2].loc[iutail, "u2"],
                        df[2].loc[iutail, "s"],
                        df[2].loc[iutail, "l"],
                        df[2].loc[iutail, "e"],
                    ) + np.log(esc[2])

            if (np.isnan(lpdf).any()) | (np.isinf(lpdf).any()):
                nllf = 1e10
            else:
                # if lpdf.size == 1:
                #     nllf = 1e10
                # else:
                nllf = -np.sum(lpdf)

        else:
            if param["no_fun"] == 1:
                if param["no_param"][0] == 2:
                    lpdf = param["fun"][0].logpdf(
                        df[0][param["var"]], df[0]["s"], df[0]["l"]
                    )
                else:
                    lpdf = param["fun"][0].logpdf(
                        df[0][param["var"]], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
            else:
                for i in range(param["no_fun"]):
                    if i == 0:
                        df_ = df[i].loc[df[i][param["var"]] < df[i]["u" + str(i + 1)]]
                        if (not param["piecewise"]) & (param["type"] == "circular"):
                            nplogesci = np.log(esc[i])
                        else:
                            nplogesci = esc[i][
                                df[i][param["var"]] < df[i]["u" + str(i + 1)]
                            ]
                    elif i == param["no_fun"] - 1:
                        df_ = df[i].loc[df[i][param["var"]] >= df[i]["u" + str(i)]]
                        if (not param["piecewise"]) & (param["type"] == "circular"):
                            nplogesci = np.log(esc[i])
                        else:
                            nplogesci = esc[i][
                                df[i][param["var"]] >= df[i]["u" + str(i)]
                            ]
                    else:
                        df_ = df[i].loc[
                            (
                                (df[i][param["var"]] >= df[i]["u" + str(i)])
                                & (df[i][param["var"]] < df[i]["u" + str(i + 1)])
                            )
                        ]
                        if (not param["piecewise"]) & (param["type"] == "circular"):
                            nplogesci = np.log(esc[i])
                        else:
                            nplogesci = esc[i][
                                (
                                    (df[i][param["var"]] >= df[i]["u" + str(i)])
                                    & (df[i][param["var"]] < df[i]["u" + str(i + 1)])
                                )
                            ]

                    # ------------------------------------------------------------------
                    # Solution for piecewise PMs in circular variables
                    # ------------------------------------------------------------------
                    if (
                        (not param["no_fun"] == 1)
                        & (param["type"] == "circular")
                        & (param["scipy"][i] == True)
                    ):
                        en = np.log(
                            param["fun"][i].cdf(
                                df_["u" + str(i + 1)] * 0 + 2 * np.pi,
                                df_["s"],
                                df_["l"],
                            )
                            - param["fun"][i].cdf(
                                df_["u" + str(i)] * 0, df_["s"], df_["l"]
                            )
                        )
                        # np.log(
                        #     param["fun"][i].cdf(
                        #         df_["u" + str(i + 1)], df_["s"], df_["l"]
                        #     )
                        #     - param["fun"][i].cdf(df_["u" + str(i)], df_["s"], df_["l"])
                        # )
                    else:
                        en = 0

                    if param["no_param"][i] == 2:
                        lpdf = np.hstack(
                            [
                                lpdf,
                                param["fun"][i].logpdf(
                                    df_[param["var"]], df_["s"], df_["l"]
                                )
                                + nplogesci
                                - en,
                            ]
                        )
                    else:
                        lpdf = np.hstack(
                            [
                                lpdf,
                                param["fun"][i].logpdf(
                                    df_[param["var"]], df_["s"], df_["l"], df_["e"]
                                )
                                + nplogesci
                                - en,
                            ]
                        )

            if np.isnan(lpdf).any() | np.isinf(lpdf).any():
                nllf = 1e10  # * (sum(np.isnan(lpdf) + sum(np.isinf(lpdf))))
            else:
                if lpdf.size == 1:
                    nllf = 1e10
                else:
                    nllf += -np.sum(lpdf * param["weighted"]["values"])

    # print(nllf)
    return nllf


def get_params(df: pd.DataFrame, param: dict, par: list, imod: list, t_expans):
    """Gets the parameters of the probability models for fitting

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par (dict): the guess parameters of the probability model
        * imod (list): combination of modes for fitting
        * t_expans (np.ndarray): the variability of the modes

    Returns:
        * df (pd.DataFrame): the parameters
        * esc (list): weight of the probability models
    """
    mode, esc = imod, {}
    pos = [0, 0, 0]
    if param["reduction"]:
        # --------------------------------------------------------------------------
        # After first order, all parameters of the same function are developed to
        # the next order at the same fit
        # --------------------------------------------------------------------------
        if param["basis_function"]["method"] in [
            "trigonometric",
            "sinusoidal",
            "modified",
        ]:
            pars_fourier = 1
            # Fourier expansion has two parameters per mode
            if bool(
                [
                    meth
                    for meth in ["trigonometric", "modified"]
                    if (meth in param["basis_function"]["method"])
                ]
            ):
                pars_fourier = 2
            # Check if detrend is required

            df["shape"] = par[0] + np.dot(
                par[1 : mode[0] * pars_fourier + 1],
                t_expans[0 : mode[0] * pars_fourier, :],
            )
            df["loc"] = par[mode[0] * 2 + 1] + np.dot(
                par[mode[0] * pars_fourier + 2 : mode[0] * pars_fourier * 2 + 2],
                t_expans[0 : mode[0] * pars_fourier, :],
            )

            # Check the parameters no. of the probability model for the body
            if param["no_param"][0] == 2:
                df["xi2"] = par[mode[0] * pars_fourier * 2 + 2] + np.dot(
                    par[
                        mode[0] * pars_fourier * 2
                        + 3 : mode[0] * pars_fourier * 2
                        + mode[1] * pars_fourier
                        + 3
                    ],
                    t_expans[0 : mode[1] * pars_fourier, :],
                )
            else:
                df["scale"] = par[mode[0] * pars_fourier * 2 + 2] + np.dot(
                    par[
                        mode[0] * pars_fourier * 2 + 3 : mode[0] * pars_fourier * 3 + 3
                    ],
                    t_expans[0 : mode[0] * pars_fourier, :],
                )
                df["xi2"] = par[mode[0] * pars_fourier * 3 + 3] + np.dot(
                    par[
                        mode[0] * pars_fourier * 3
                        + 4 : mode[0] * pars_fourier * 3
                        + mode[1] * pars_fourier
                        + 4
                    ],
                    t_expans[0 : mode[1] * pars_fourier, :],
                )
        else:
            # Polynomial expansion has one parameters per mode
            df["shape"] = params_t_expansion(par[0 : mode[0] + 1], param, df["n"])
            df["loc"] = params_t_expansion(
                par[mode[0] + 1 : mode[0] * 2 + 2], param, df["n"]
            )

            if param["no_param"][0] == 2:
                df["xi2"] = params_t_expansion(
                    par[mode[0] * 2 + 2 : mode[0] * 2 + mode[1] + 3], param, df["n"]
                )
            else:
                df["scale"] = params_t_expansion(
                    par[mode[0] * 2 + 2 : mode[0] * 3 + 3], param, df["n"]
                )
                df["xi2"] = params_t_expansion(
                    par[mode[0] * 3 + 3 : mode[0] * 3 + mode[1] + 4], param, df["n"]
                )

        esc[1] = st.norm.cdf(par[-2])
        esc[2] = 1 - st.norm.cdf(par[-1])
        if param["no_param"][0] == 2:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"])
            df["u2"] = param["fun"][1].ppf(st.norm.cdf(par[-1]), df["shape"], df["loc"])

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"]
            )
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"]
            )
        else:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"], df["scale"])
            df["u2"] = param["fun"][1].ppf(
                st.norm.cdf(par[-1]), df["shape"], df["loc"], df["scale"]
            )

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"], df["scale"]
            )
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"], df["scale"]
            )
    else:
        # ------------------------------------------------------------------------------
        # Obtaining the parameters in a handy form for bi or tri-parametric models
        # ------------------------------------------------------------------------------
        df_, esc = {}, {}
        pars_fourier = 1
        if param["basis_function"]["order"] == 0:
            mode = [0, 0, 0]
        # elif not first:
        if len(imod) == 1:
            mode = [imod[0], imod[0], imod[0]]
        elif len(imod) == 2:
            mode = [imod[0], imod[1], imod[0]]
        else:
            mode = [imod[0], imod[1], imod[2]]

        for i in range(param["no_fun"]):
            if param["basis_function"]["method"] in [
                "trigonometric",
                "sinusoidal",
                "modified",
            ]:
                # Fourier expansion has two parameters per mode
                if bool(
                    [
                        meth
                        for meth in ["trigonometric", "modified"]
                        if (meth in param["basis_function"]["method"])
                    ]
                ):
                    pars_fourier = 2

                df_[i] = df.copy()
                df_[i]["s"] = par[pos[0]]

                df_[i]["s"] = df_[i]["s"] + np.dot(
                    par[pos[0] + 1 : pos[0] + mode[i] * pars_fourier + 1],
                    t_expans[0 : mode[i] * pars_fourier, :],
                )
                df_[i]["l"] = par[pos[0] + mode[i] * pars_fourier + 1]

                df_[i]["l"] = df_[i]["l"] + np.dot(
                    par[
                        pos[0]
                        + mode[i] * pars_fourier
                        + 2 : pos[0]
                        + mode[i] * pars_fourier * 2
                        + 2
                    ],
                    t_expans[0 : mode[i] * pars_fourier, :],
                )

                if param["no_param"][i] == 3:
                    df_[i]["e"] = par[pos[0] + mode[i] * pars_fourier * 2 + 2]

                    df_[i]["e"] = df_[i]["e"] + np.dot(
                        par[
                            pos[0]
                            + mode[i] * pars_fourier * 2
                            + 3 : pos[0]
                            + mode[i] * pars_fourier * 3
                            + 3
                        ],
                        t_expans[0 : mode[i] * pars_fourier, :],
                    )
            else:
                df_[i] = df.copy()
                df_[i]["s"] = params_t_expansion(
                    par[pos[0] : pos[0] + mode[i] * pars_fourier + 1], param, df["n"]
                )

                df_[i]["l"] = params_t_expansion(
                    par[
                        pos[0]
                        + mode[i] * pars_fourier
                        + 1 : pos[0]
                        + 2 * mode[i] * pars_fourier
                        + 2
                    ],
                    param,
                    df["n"],
                )

                if param["no_param"][i] == 3:
                    df_[i]["e"] = params_t_expansion(
                        par[
                            pos[0]
                            + 2 * mode[i] * pars_fourier
                            + 2 : pos[0]
                            + 3 * mode[i] * pars_fourier
                            + 3
                        ],
                        param,
                        df["n"],
                    )

            # --------------------------------------------------------------------------
            # Updating index for the next probability model
            # pos[0]: the index of the first parameter of the i-PM
            # pos[1]: cumulative sum of the number of parameters of the PM
            # pos[2]: i-PM
            # --------------------------------------------------------------------------
            if param["basis_function"]["order"] == 0:
                pos[0] = pos[0] + int(param["no_param"][i])
                pos[1] = pos[1] + param["no_param"][i]
                pos[2] = i + 1
            else:
                pos[0] = (
                    pos[0]
                    + int(param["no_param"][i])
                    + pars_fourier * imod[i] * int(param["no_param"][i])
                )
                pos[1] += 1
                pos[2] += 1

        df = df_.copy()

        # ---------------------------------------------------------------------------
        # Obtaining percentiles from parameters.
        # ---------------------------------------------------------------------------
        if param["fix_percentiles"]:
            if param["no_fun"] == 1:
                esc[0] = 1
            elif param["no_fun"] == 2:
                esc[0] = param["ws_ps"][0]
                esc[1] = 1 - param["ws_ps"][0]
            else:
                esc[0] = param["ws_ps"][0]
                esc[1] = param["ws_ps"][1] - param["ws_ps"][0]
                esc[2] = 1 - esc[1] - esc[0]
        else:
            if param["no_fun"] == 1:
                esc[0] = 1
            if param["no_fun"] == 2:
                # Two restrictions
                esc[0] = st.norm.cdf(par[-1])
                esc[1] = 1 - st.norm.cdf(par[-1])
            else:
                # Three restrictions
                esc[0] = st.norm.cdf(par[-2])
                esc[1] = st.norm.cdf(par[-1]) - st.norm.cdf(par[-2])
                esc[2] = 1 - st.norm.cdf(par[-1])

        # Compute the threshold according to Cobos et al. 2022 "A method to characterize
        # climate, Earth or environmental vector random processes"
        # if param[
        #     "piecewise"
        # ]:  # TODO: modify for a more refined and understandable version
        #     if param["no_fun"] == 2:
        #         if (param["no_param"][0] == 2) & (param["no_param"][1] == 2):
        #             a1 = param["fun"][0].pdf(param["ws_ps"], df[0]["s"], df[0]["l"])
        #             b1 = param["fun"][1].pdf(param["ws_ps"], df[1]["s"], df[1]["l"])
        #             c1 = param["fun"][0].cdf(param["ws_ps"], df[0]["s"], df[0]["l"])
        #             c2 = 1 - param["fun"][1].cdf(param["ws_ps"], df[1]["s"], df[1]["l"])

        #         elif (param["no_param"][0] == 3) & (param["no_param"][1] == 2):
        #             a1 = param["fun"][0].pdf(
        #                 param["ws_ps"], df[0]["s"], df[0]["l"], df[0]["e"]
        #             )
        #             b1 = param["fun"][1].pdf(param["ws_ps"], df[1]["s"], df[1]["l"])
        #             c1 = param["fun"][0].cdf(
        #                 param["ws_ps"], df[0]["s"], df[0]["l"], df[0]["e"]
        #             )
        #             c2 = 1 - param["fun"][1].cdf(param["ws_ps"], df[1]["s"], df[1]["l"])
        #         else:
        #             a1 = param["fun"][0].pdf(
        #                 param["ws_ps"], df[0]["s"], df[0]["l"], df[0]["e"]
        #             )
        #             b1 = param["fun"][1].pdf(
        #                 param["ws_ps"], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )
        #             c1 = param["fun"][0].cdf(
        #                 param["ws_ps"], df[0]["s"], df[0]["l"], df[0]["e"]
        #             )
        #             c2 = 1 - param["fun"][1].cdf(
        #                 param["ws_ps"], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )

        #         esc[0] = c1 + b1 / a1 * c2
        #         esc[1] = a1 / b1 * (c1 + b1 / a1 * c2)
        #         df[0]["u1"] = df[0]["s"] * 0 + param["ws_ps"]
        #         df[1]["u1"] = df[1]["s"] * 0 + param["ws_ps"]
        #     elif param["no_fun"] == 3:
        #         if param["no_param"][1] == 2:
        #             df[0]["u1"] = param["fun"][1].pdf(esc[0], df[1]["s"], df[1]["l"])
        #             df[1]["u1"] = param["fun"][1].pdf(esc[0], df[1]["s"], df[1]["l"])
        #             df[1]["u2"] = param["fun"][1].pdf(
        #                 1 - esc[2], df[1]["s"], df[1]["l"]
        #             )
        #             df[2]["u2"] = param["fun"][1].ppf(
        #                 1 - esc[2], df[1]["s"], df[1]["l"]
        #             )
        #         else:
        #             df[0]["u1"] = param["fun"][1].ppf(
        #                 esc[0], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )
        #             df[1]["u1"] = param["fun"][1].ppf(
        #                 esc[0], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )
        #             df[1]["u2"] = param["fun"][1].ppf(
        #                 1 - esc[2], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )
        #             df[2]["u2"] = param["fun"][1].ppf(
        #                 1 - esc[2], df[1]["s"], df[1]["l"], df[1]["e"]
        #             )

        if (
            ((not param["fix_percentiles"]) | (param["constraints"]))
            & (not param["reduction"])
            & (not param["type"] == "circular")
        ):
            if param["no_fun"] == 2:
                if param["no_param"][0] == 2:
                    df[0]["u1"] = param["fun"][0].ppf(esc[0], df[0]["s"], df[0]["l"])
                    df[1]["u1"] = param["fun"][1].ppf(esc[0], df[1]["s"], df[1]["l"])
                else:
                    df[0]["u1"] = param["fun"][0].ppf(
                        esc[0], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
                    df[1]["u1"] = param["fun"][1].ppf(
                        esc[1], df[0]["s"], df[1]["l"], df[1]["e"]
                    )
            elif param["no_fun"] == 3:
                if param["no_param"][1] == 2:
                    df[0]["u1"] = param["fun"][0].ppf(esc[0], df[0]["s"], df[0]["l"])
                    df[1]["u1"] = param["fun"][1].ppf(esc[0], df[1]["s"], df[1]["l"])
                    df[1]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"]
                    )
                    df[2]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[2]["s"], df[2]["l"]
                    )
                else:
                    df[0]["u1"] = param["fun"][0].ppf(
                        esc[0], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
                    df[1]["u1"] = param["fun"][1].ppf(
                        esc[0], df[1]["s"], df[1]["l"], df[1]["e"]
                    )
                    df[1]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"], df[1]["e"]
                    )
                    df[2]["u2"] = param["fun"][2].ppf(
                        1 - esc[2], df[2]["s"], df[2]["l"], df[2]["e"]
                    )

    return df, esc


def ppf(df: pd.DataFrame, param: dict):
    """Computes the inverse of the probability function

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the parameters of the probability model

    Return:
        df (pd.DataFrame): inverse of the cdf
    """

    t_expans = params_t_expansion(param["mode"], param, df["n"])

    if param["reduction"]:
        # ------------------------------------------------------------------------------
        # If reduction is possible
        # ------------------------------------------------------------------------------
        df, esc = get_params(df, param, param["par"], param["mode"], t_expans)

        # Choose data below esc[1]
        idu1 = df["prob"] < esc[1]
        df.loc[idu1, param["var"]] = df.loc[idu1, "u1"] - param["fun"][0].ppf(
            (esc[1] - df.loc[idu1, "prob"]) / esc[1],
            df.loc[idu1, "xi1"],
            scale=df.loc[idu1, "siggp1"],
        )

        # Choose data above esc[2]
        idu2 = df["prob"] > 1 - esc[2]
        df.loc[idu2, param["var"]] = df.loc[idu2, "u2"] + param["fun"][2].ppf(
            (df.loc[idu2, "prob"] - 1 + esc[2]) / esc[2],
            df.loc[idu2, "xi2"],
            loc=0,
            scale=df.loc[idu2, "siggp2"],
        )
        idb = (df["prob"] <= 1 - esc[2]) & (df["prob"] >= esc[1])

        if param["no_param"][0] == 2:
            # Computes the ppf for biparametric PMs
            df.loc[idb, param["var"]] = param["fun"][1].ppf(
                df.loc[idb, "prob"], df.loc[idb, "shape"], df.loc[idb, "loc"]
            )
        else:
            # Computes the ppf for triparametric PMs
            df.loc[idb, param["var"]] = param["fun"][1].ppf(
                df.loc[idb, "prob"],
                df.loc[idb, "shape"],
                df.loc[idb, "loc"],
                df.loc[idb, "scale"],
            )
    else:
        if param["no_fun"] == 1:
            # --------------------------------------------------------------------------
            # If only one probability model is given
            # --------------------------------------------------------------------------
            df, esc = get_params(df, param, param["par"], param["mode"], t_expans)
            if param["no_param"][0] == 2:
                df[0][param["var"]] = param["fun"][0].ppf(
                    df[0]["prob"], df[0]["s"], df[0]["l"]
                )
            else:
                df[0][param["var"]] = param["fun"][0].ppf(
                    df[0]["prob"], df[0]["s"], df[0]["l"], df[0]["e"]
                )
            df = df[0]
        else:
            # --------------------------------------------------------------------------
            # Wheter more than one probability model are given
            # --------------------------------------------------------------------------
            if param["type"] == "circular":
                data = np.linspace(param["minimax"][0], param["minimax"][1], 100)
            else:
                data = np.linspace(param["minimax"][0], param["minimax"][1], 1000)
            df[param["var"]] = -1

            # --------------------------------------------------------------------------
            # Due to the computational cost of cdf wrap-norm, the file will be saved
            # --------------------------------------------------------------------------
            if (param["type"] == "circular") & (
                all(value == 0 for value in param["scipy"].values())
            ):
                if os.path.isfile(param["file_name"] + "_cdf_wrapnorm.temp.npy"):
                    cdfs = read.npy(param["file_name"] + "_cdf_wrapnorm.temp")
                else:
                    cdfs = cdf(df, param, ppf=True)
                    save.to_npy(cdfs, param["file_name"] + "_cdf_wrapnorm.temp")
            else:
                cdfs = cdf(df, param, ppf=True)

            posi = np.zeros(len(df), dtype=int)
            dfn = np.sort(df["n"].unique())

            for i, j in enumerate(df.index):  # Seeking every n (dates)
                posn = np.argmin(np.abs(df["n"][j] - dfn))
                posj = np.argmin(np.abs(df["prob"][j] - cdfs[posn, :].T))
                posi[i] = posj
                if not posj:
                    posi[i] = posi[i - 1]

            df.loc[:, param["var"]] = data[posi]

    return df


def cdf(df: pd.DataFrame, param: dict, ppf: bool = False):
    """Computes the cumulative distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the parameters of the probability model
        * ppf (boolean, optional): boolean for selecting the method for the computation. Defaults is False (creating a previous mesh of data). False is computed from the probability models (sometimes it is not possible).

    Return:
        * prob (pd.DataFrame): the non-excedence probability
    """

    t_expans = params_t_expansion(param["mode"], param, df["n"])
    if not any(df.columns == "prob"):
        df["prob"] = 0

    if param["reduction"]:
        # Reduction allowed
        df, esc = get_params(df, param, param["par"], param["mode"], t_expans)

        if not "data" in df.columns:
            df.rename(columns={param["var"]: "data"}, inplace=True)
        fu1 = df["data"] < df["u1"]
        df.loc[fu1, "prob"] = esc[1] * (
            1
            - df.loc[fu1, "xi1"]
            / df.loc[fu1, "siggp1"]
            * (df.loc[fu1, "data"] - df.loc[fu1, "u1"])
        ) ** (-1.0 / df.loc[fu1, "xi1"])

        fu2 = df["data"] > df["u2"]
        df.loc[fu2, "prob"] = (
            1
            - esc[2]
            + esc[2]
            * (
                1
                - (
                    1
                    + df.loc[fu2, "xi2"]
                    / df.loc[fu2, "siggp2"]
                    * (df.loc[fu2, "data"] - df.loc[fu2, "u2"])
                )
                ** (-1.0 / df.loc[fu2, "xi2"])
            )
        )

        fuc = (df["data"] >= df["u1"]) & (df["data"] <= df["u2"])
        if param["no_param"][0] == 2:
            df.loc[fuc, "prob"] = param["fun"][1].cdf(
                df.loc[fuc, "data"], df.loc[fuc, "shape"], df.loc[fuc, "loc"]
            )
        else:
            df.loc[fuc, "prob"] = param["fun"][1].cdf(
                df.loc[fuc, "data"],
                df.loc[fuc, "shape"],
                df.loc[fuc, "loc"],
                df.loc[fuc, "scale"],
            )
        cdf_ = df["prob"]
    else:
        # Not reduction applied
        if param["no_fun"] == 1:
            # One PM
            df, esc = get_params(df, param, param["par"], param["mode"], t_expans)

            if param["no_param"][0] == 2:
                df[0]["prob"] = param["fun"][0].cdf(
                    df[0]["data"], df[0]["s"], df[0]["l"]
                )
            else:
                df[0]["prob"] = param["fun"][0].cdf(
                    df[0]["data"], df[0]["s"], df[0]["l"], df[0]["e"]
                )
            cdf_ = df[0]["prob"]
        else:
            # More than one PMs
            if ppf:
                if param["type"] == "circular":
                    data = np.linspace(param["minimax"][0], param["minimax"][1], 100)
                else:
                    data = np.linspace(param["minimax"][0], param["minimax"][1], 1000)
                dfn = np.sort(df["n"].unique())
                t_expans = params_t_expansion(param["mode"], param, dfn)
                aux = pd.DataFrame(-1, index=dfn, columns=["s"])
                aux["n"] = df["n"]
                dff, esc = get_params(
                    aux,
                    param,
                    param["par"],
                    param["mode"],
                    t_expans,
                )
                cdf_ = np.zeros([len(dfn), len(data)])
                if param["constraints"]:
                    if len(dff) == 2:
                        for k, j in enumerate(dfn):
                            fu1 = data <= dff[0].loc[j, "u1"]
                            fu2 = data > dff[0].loc[j, "u1"]

                            if param["no_param"][0] == 2:
                                cdf_[k, fu1] = param["fun"][0].cdf(
                                    data[fu1], dff[0].loc[j, "s"], dff[0].loc[j, "l"]
                                )
                            else:
                                cdf_[k, fu1] = param["fun"][0].cdf(
                                    data[fu1],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                    dff[0].loc[j, "e"],
                                )

                            if param["no_param"][1] == 2:
                                cdf_[k, fu2] = esc[0] + esc[1] * param["fun"][1].cdf(
                                    data[fu2] - dff[0].loc[j, "u1"],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                )
                            else:
                                cdf_[k, fu2] = esc[0] + esc[1] * param["fun"][1].cdf(
                                    data[fu2] - dff[0].loc[j, "u1"],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                    dff[1].loc[j, "e"],
                                )
                    else:
                        for k, j in enumerate(dfn):
                            fu0 = data < dff[0].loc[j, "u1"]
                            fu1 = (data >= dff[1].loc[j, "u1"]) & (
                                data <= dff[1].loc[j, "u2"]
                            )
                            fu2 = data > dff[2].loc[j, "u2"]

                            if param["no_param"][0] == 2:
                                cdf_[k, fu0] = esc[0] * param["fun"][0].cdf(
                                    dff[0].loc[j, "u1"] - data[fu0],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                )
                            else:
                                cdf_[k, fu0] = esc[0] * param["fun"][0].cdf(
                                    dff[0].loc[j, "u1"] - data[fu0],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                    dff[0].loc[j, "e"],
                                )

                            if param["no_param"][1] == 2:
                                cdf_[k, fu1] = param["fun"][1].cdf(
                                    data[fu1], dff[1].loc[j, "s"], dff[1].loc[j, "l"]
                                )
                            else:
                                cdf_[k, fu1] = param["fun"][1].cdf(
                                    data[fu1],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                    dff[1].loc[j, "e"],
                                )

                            if param["no_param"][2] == 2:
                                cdf_[k, fu2] = (
                                    1
                                    - esc[2]
                                    + esc[2]
                                    * param["fun"][2].cdf(
                                        data[fu2] - dff[2].loc[j, "u2"],
                                        dff[2].loc[j, "s"],
                                        dff[2].loc[j, "l"],
                                    )
                                )
                            else:
                                cdf_[k, fu2] = (
                                    1
                                    - esc[2]
                                    + esc[2]
                                    * param["fun"][2].cdf(
                                        data[fu2] - dff[2].loc[j, "u2"],
                                        dff[2].loc[j, "s"],
                                        dff[2].loc[j, "l"],
                                        dff[2].loc[j, "e"],
                                    )
                                )
                else:
                    # For piecewise PMs
                    for k, j in enumerate(dfn):
                        logger.info(
                            "Computing the cdf numerically | "
                            + str(np.round((k + 1) / len(dfn) * 100, decimals=2))
                            + " %"
                        )
                        for i in range(param["no_fun"]):
                            esci = esc[i]

                            if param["no_param"][i] == 2:
                                if (param["type"] == "circular") & (
                                    param["fun"][i].name == "wrap_norm"
                                ):
                                    cdf_[k, :] += esci * param["fun"][i].cdf(
                                        data, dff[i].loc[j, "s"], dff[i].loc[j, "l"]
                                    )
                                else:
                                    en = param["fun"][i].cdf(
                                        param["minimax"][1],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                    ) - param["fun"][i].cdf(
                                        param["minimax"][0],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                    )
                                    cdf_[k, :] += (
                                        esci
                                        * (
                                            param["fun"][i].cdf(
                                                data,
                                                dff[i].loc[j, "s"],
                                                dff[i].loc[j, "l"],
                                            )
                                            - param["fun"][i].cdf(
                                                param["minimax"][0],
                                                dff[i].loc[j, "s"],
                                                dff[i].loc[j, "l"],
                                            )
                                        )
                                        / en
                                    )
                            else:
                                if param["piecewise"]:
                                    en = 1
                                else:
                                    en = param["fun"][i].cdf(
                                        param["minimax"][1],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                        dff[i].loc[j, "e"],
                                    ) - param["fun"][i].cdf(
                                        param["minimax"][0],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                        dff[i].loc[j, "e"],
                                    )
                                cdf_[k, :] += (
                                    esci
                                    * (
                                        param["fun"][i].cdf(
                                            data,
                                            dff[i].loc[j, "s"],
                                            dff[i].loc[j, "l"],
                                            dff[i].loc[j, "e"],
                                        )
                                        - param["fun"][i].cdf(
                                            param["minimax"][0],
                                            dff[i].loc[j, "s"],
                                            dff[i].loc[j, "l"],
                                            dff[i].loc[j, "e"],
                                        )
                                    )
                                    / en
                                )

            else:
                df, esc = get_params(
                    df,
                    param,
                    param["par"],
                    param["mode"],
                    t_expans,
                )
                df[0]["prob"] = 0
                # ----------------------------------------------------------------------
                # Different approach using restrictions
                # ----------------------------------------------------------------------
                if param["constraints"]:
                    # For 2 PMs
                    if len(df) == 2:  # Agregué prob, data y u1
                        fu1 = df[0]["data"] <= df[0]["u1"]
                        fu2 = df[0]["data"] > df[0]["u1"]

                        if param["no_param"][0] == 2:
                            df[0].loc[fu1, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu1, "data"],
                                df[0].loc[fu1, "s"],
                                df[0].loc[fu1, "l"],
                            )
                        else:
                            df[0].loc[fu1, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu1, "data"],
                                df[0].loc[fu1, "s"],
                                df[0].loc[fu1, "l"],
                                df[0].loc[fu1, "e"],
                            )

                        if param["no_param"][1] == 2:
                            df[0].loc[fu2, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                df[1].loc[fu2, "s"],
                                df[1].loc[fu2, "l"],
                            )
                        else:
                            df[0].loc[fu2, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                df[1].loc[fu2, "s"],
                                df[1].loc[fu2, "l"],
                                df[1].loc[fu2, "e"],
                            )
                    else:
                        df[0]["u2"] = df[2]["u2"]
                        # For 3 PMs
                        fu0 = df[0]["data"] < df[0]["u1"]
                        fu1 = (df[0]["data"] >= df[0]["u1"]) & (
                            df[0]["data"] <= df[0]["u2"]
                        )
                        fu2 = df[0]["data"] > df[0]["u2"]

                        if param["no_param"][0] == 2:
                            df[0].loc[fu0, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu0, "u1"] - df[0].loc[fu0, "data"],
                                df[0]["s"],
                                df[0]["l"],
                            )
                        else:
                            df[0].loc[fu0, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu0, "u1"] - df[0].loc[fu0, "data"],
                                df[0]["s"],
                                df[0]["l"],
                                df[0]["e"],
                            )

                        if param["no_param"][1] == 2:
                            df[0].loc[fu1, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu1, "data"], df[1]["s"], df[1]["l"]
                            )
                        else:
                            df[0].loc[fu1, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu1, "data"],
                                df[1]["s"],
                                df[1]["l"],
                                df[1]["e"],
                            )

                        if param["no_param"][2] == 2:
                            df[0].loc[fu2, "prob"] = (
                                1
                                - esc[2]
                                + esc[2]
                                * param["fun"][2].cdf(
                                    df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                    df[2]["s"],
                                    df[2]["l"],
                                )
                            )
                        else:
                            df[0].loc[fu2, "prob"] = (
                                1
                                - esc[2]
                                + esc[2]
                                * param["fun"][2].cdf(
                                    df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                    df[2]["s"],
                                    df[2]["l"],
                                    df[2]["e"],
                                )
                            )

                else:
                    # ----------------------------------------------------------------------
                    # First approach of the non-stationary analysis
                    # ----------------------------------------------------------------------
                    for i in range(param["no_fun"]):
                        if param["no_param"][i] == 2:
                            en = param["fun"][i].cdf(
                                param["minimax"][1], df[i]["s"], df[i]["l"]
                            ) - param["fun"][i].cdf(
                                param["minimax"][0], df[i]["s"], df[i]["l"]
                            )
                            df[0]["prob"] += (
                                esc[i]
                                * (
                                    param["fun"][i].cdf(
                                        df[i]["data"], df[i]["s"], df[i]["l"]
                                    )
                                    - param["fun"][i].cdf(
                                        param["minimax"][0], df[i]["s"], df[i]["l"]
                                    )
                                )
                                / en
                            )
                        else:
                            en = param["fun"][i].cdf(
                                param["minimax"][1], df[i]["s"], df[i]["l"], df[i]["e"]
                            ) - param["fun"][i].cdf(
                                param["minimax"][0], df[i]["s"], df[i]["l"], df[i]["e"]
                            )
                            df[0]["prob"] += (
                                esc[i]
                                * (
                                    param["fun"][i].cdf(
                                        df[i]["data"],
                                        df[i]["s"],
                                        df[i]["l"],
                                        df[i]["e"],
                                    )
                                    - param["fun"][i].cdf(
                                        param["minimax"][0],
                                        df[i]["s"],
                                        df[i]["l"],
                                        df[i]["e"],
                                    )
                                )
                                / en
                            )

                cdf_ = df[0]["prob"]

    return cdf_


def transform(data: pd.DataFrame, params: dict):
    """Normalized the input data given a normalized method (Box-Cox or Yeo-Johnson)

    Args:
        data ([pd.DataFrame]): input timeseries
        params ([dict]): parameters which can include

    Returns:
        data: normalized input data
        params: a dictionary with lambda of transformation
    """
    from sklearn import preprocessing

    # In some cases, it is usually that pd.Series appears. Transform to pd.DataFrame
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if "lambda" in params["transform"].keys():
        # Use the lambda of a previous transformation if it is given
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        )
        powertransform.lambdas_ = [params["transform"]["lambda"]]
    else:
        # Compute lambda of transformation
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        ).fit(data.values.reshape(-1, 1))
        params["transform"]["lambda"] = powertransform.lambdas_[0]

    # Normalized the input data
    data = pd.DataFrame(
        {data.columns[0]: powertransform.transform(data.values.reshape(-1, 1))[:, 0]},
        index=data.index,
    )

    return data, params


def inverse_transform(data: pd.DataFrame, params: dict, ensemble: bool = False):
    """Compute the inverse of the transformation

    Args:
        data (pd.DataFrame): raw timeseries
        params (dict): parameters of the model
        ensemble (bool, optional): True or False

    Returns:
        pd.DataFrame: the inverse of the transform of data
    """
    from sklearn import preprocessing

    if ensemble:
        powertransform = preprocessing.PowerTransformer(
            params["method_ensemble"], standardize=False
        )
        powertransform.lambdas_ = [params["lambda_ensemble"]]
    else:
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        )
        powertransform.lambdas_ = [params["transform"]["lambda"]]

    data = pd.DataFrame(
        {
            data.columns[0]: powertransform.inverse_transform(
                data.values.reshape(-1, 1)
            )[:, 0]
        },
        index=data.index,
    )
    return data


def numerical_cdf_pdf_at_n(n: float, param: dict, variable: str, alpha: float = 0.01):
    """Compute the cdf and pdf numerically at a given time "n" (normalized year)

    Args:
        n (float): a value in normalized period
        param (dict): input parameters
        variable (str): Name of the variable.
        alpha (float, optional): Resolution. Defaults to 0.01.

    Returns:
        pd.DataFrame: the numerical pdf and cdf
    """

    param = auxiliar.str2fun(param, None)
    # pemp = np.linspace(0 + alpha / 2, 1 - alpha / 2)
    values_ = np.linspace(param["minimax"][0], param["minimax"][1], 360)

    # df = pd.DataFrame(pemp, index=pemp, columns=["prob"])
    df = pd.DataFrame(values_, index=values_, columns=["data"])
    df[param["var"]] = values_
    df["n"] = np.ones(len(values_)) * n
    if (param["non_stat_analysis"] == True) | (param["no_fun"] > 1):
        df["cdf"] = cdf(df, param)
        # df = ppf(df, param)
        # df.rename(columns={"prob": "pdf"}, inplace=True)
        df["pdf"] = df["cdf"].diff() / df[param["var"]].diff()
    else:
        df = pd.DataFrame(
            param["fun"][0].ppf(df["prob"], *param["par"]),
            index=df.index,
            columns=[variable],
        )

    # Transformed timeseries
    if (not param["transform"]["plot"]) & param["transform"]["make"]:
        if "scale" in param:
            df[param["var"]] = df[param["var"]] * param["scale"]

        df[param["var"]] = df[param["var"]] + param["transform"]["min"]
        df[param["var"]] = inverse_transform(df[[param["var"]]], param)
    elif ("scale" in param) & (not param["transform"]["plot"]):
        df[param["var"]] = df[param["var"]] * param["scale"]

    # if param["circular"]:
    #     values_ = (
    #         np.diff(np.rad2deg(res[param["var"]].values))
    #         + np.rad2deg(res[param["var"]].values)[:-1]
    #     )
    #     index_ = np.diff(res[param["var"]].index)
    # else:
    #     values_ = np.diff(res[param["var"]].index) / np.diff(res[param["var"]].values)
    #     index_ = np.diff(res[param["var"]].values) + res[param["var"]].values[:-1]

    # res_ = pd.DataFrame(values_, index=index_, columns=["pdf"])
    # res_["cdf"] = (res[param["var"]].index[:-1] + res[param["var"]].index[1:]) / 2
    return df


def ensemble_cdf(
    data: pd.DataFrame, param: dict, variable: str, nodes: list = [4383, 900]
):
    """Computes the ensemble of the cumulative distribution function for several models

    Args:
        * data (pd.DataFrame): raw data
        * param (dict): the parameters of the probability model
        * variable (string): name of the variable
        * nodes (list): no of nodes in the time and variable axes. Defaults to [4383, 250].

    Returns:
        * cdfs (pd.DataFrame): the non-stationary non-excedance probability
    """
    cdfs = 0
    data_model = np.tile(data, nodes[0])
    index_ = np.repeat(np.linspace(0, 1 - 1 / nodes[0], nodes[0]), nodes[1])

    for model in param["TS"]["models"]:
        dfm = pd.DataFrame(
            np.asarray([index_, data_model]).T, index=index_, columns=["n", "data"]
        )

        indexes = np.where(dfm["data"] == 0)[0]
        param[variable] = auxiliar.str2fun(param[variable], model)
        if param[variable][model]["range"] > 10:
            dfm["data"] = dfm["data"] / (param[variable][model]["range"] / 3)
        cdf_model = cdf(dfm.copy(), param[variable][model])
        cdf_model.iloc[indexes[np.isnan(cdf_model.iloc[indexes]).values]] = 0
        cdf_model.loc[np.isnan(cdf_model)] = 1
        cdfs += cdf_model

    cdfs = cdfs / len(param["TS"]["models"])
    return cdfs


def ensemble_ppf(
    df: pd.DataFrame, param: dict, variable: str, nodes: list = [4383, 900]
):
    """Computes the inverse of the cumulative distribution function

    Args:
        * data (pd.DataFrame): raw data
        * param (dict): the parameters of the probability model
        * variable (string): name of the variable
        * nodes (list): no of nodes in the time and variable axes. Defaults to [4383, 250].

    Returns:
        * df (pd.DataFrame): the raw time series
    """

    data = np.linspace(
        param["TS"]["minimax"][variable][0],
        param["TS"]["minimax"][variable][1],
        nodes[1],
    )
    df[variable] = -1

    try:
        cdfs = read.csv(param["TS"]["F_files"] + variable)
    except:
        if 1:
            cdfs = ensemble_cdf(data, param, variable, nodes)
            # pd.DataFrame(cdfs).to_csv(param["TS"]["F_files"] + variable + ".csv")
        else:
            raise ValueError(
                "F-files are not found at {}.".format(
                    param["TS"]["F_files"] + variable + ".csv"
                )
            )

    dfn = np.sort(cdfs.index.unique())
    logger.info("Estimating pdf of ensemble for %s" % (variable))

    posn = np.zeros(df.shape[0], dtype=np.int16)
    posi = np.zeros(df.shape[0], dtype=np.int16)
    for no_, ind_ in enumerate(df["n"].index):  # Find the date for every n
        posn[no_] = np.argmin(np.abs(df.loc[ind_, "n"] - dfn))
        posi[no_] = np.argmin(np.abs(df["prob"][ind_] - cdfs.loc[dfn[posn[no_]]]))
    df.loc[:, variable] = data[posi]

    return df[[variable]]


def params_t_expansion(mod: int, param: dict, nper: pd.DataFrame):
    """Computes the oscillatory dependency in time of the parameters

    Args:
        * mod (int): maximum mode of the oscillatory dependency
        * param (dict): the parameters
        * nper (pd.DataFrame): time series of normalized year

    Return:
        * t_expans (np.ndarray): the variability of every mode
    """

    if param["basis_function"]["method"] == "trigonometric":
        t_expans = np.zeros([np.max(mod) * 2, len(nper)])
        for i in range(0, np.max(mod)):
            n = (
                nper
                * np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )
            t_expans[2 * i, :] = np.cos(2 * np.pi * n.T)
            t_expans[2 * i + 1, :] = np.sin(2 * np.pi * n.T)
    elif param["basis_function"]["method"] == "modified":
        t_expans = np.zeros([np.max(mod) * 2, len(nper)])
        for i in range(0, np.max(mod)):
            per_ = (
                np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )
            n = (
                nper
                * np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )

            t_expans[2 * i, :] = np.cos(np.pi * (n.T - 1))
            t_expans[2 * i + 1, :] = np.sin(
                2 * np.pi * n.T - per_ * np.pi - np.pi * nper + 0.5 * np.pi
            )
    elif param["basis_function"]["method"] == "sinusoidal":
        t_expans = np.zeros([np.max(mod), len(nper)])
        for i in range(0, np.max(mod)):
            n = (i + 1) * (nper + 1)
            t_expans[i, :] = np.sin(np.pi * n.T)

    elif param["basis_function"]["method"] == "chebyshev":
        nper = 2 * (nper - 0.5)
        # nper = nper / 2 + 0.5
        t_expans = np.polynomial.chebyshev.chebval(nper, mod)
    elif param["basis_function"]["method"] == "legendre":
        nper = 2 * (nper - 0.5)
        t_expans = np.polynomial.legendre.legval(nper, mod)
    elif param["basis_function"]["method"] == "laguerre":
        nper = 2 * (nper - 0.5)
        t_expans = np.polynomial.laguerre.lagval(nper, mod)
    elif param["basis_function"]["method"] == "hermite":
        nper = 2 * (nper - 0.5)
        t_expans = np.polynomial.hermite.hermval(nper, mod)
    elif param["basis_function"]["method"] == "ehermite":
        nper = 2 * (nper - 0.5)
        t_expans = np.polynomial.hermite_e.hermeval(nper, mod)
    elif param["basis_function"]["method"] == "polynomial":
        nper = 2 * (nper - 0.5)
        t_expans = np.polynomial.polynomial.polyval(nper, mod)
    else:
        t_expans = np.ones(len(nper)) * mod[0]

    return t_expans


def print_message(res: dict, j: int, mode: list, ref: float, param: dict):
    """Prints the messages during the computation

    Args:
        * res (dict): result from the fitting algorithm
        * j (int): no. of iteration
        * mode (list): the current mode
        * ref (float): reference value of previous nllf

    Returns:
        None
    """
    if param["verbose"]:
        if not ((res["fun"] == 1e10) | (np.abs(res["fun"]) == np.inf)):

            if (ref < 0) & (res["fun"] < 0):
                improve = np.round((ref - res["fun"]) / -ref * 100, decimals=3)
                if improve > 0:
                    improve = "Yes (" + str(improve) + " %)"
                else:
                    improve = "No or not significant "
            elif (ref > 0) & (res["fun"] > 0):
                improve = np.round((ref - res["fun"]) / ref * 100, decimals=3)
                if improve > 0:
                    improve = "Yes (" + str(improve) + " %)"
                else:
                    improve = "No or not significant "
            elif (ref > 0) | (res["fun"] < 0):
                improve = np.round((ref - res["fun"]) / ref * 100, decimals=3)
                improve = "Yes (" + str(improve) + " %)"
            else:
                improve = "No or not significant "

            if not (improve == "No or not significant "):
                logger.info("mode:     " + str(mode))
                logger.info("fun:      " + str(res["fun"]))
                logger.info("improve:  " + improve)
                logger.info("message:  " + str(res["message"]))
                logger.info("feval:    " + str(res["nfev"]))
                logger.info("niter:    " + str(res["nit"]))
                logger.info("giter:    " + str(j))
                logger.info("params:   " + str(res["x"]))
                logger.info(
                    "=============================================================================="
                )

    return


class wrap_norm(st.rv_continuous):
    """Wrapped normal probability model"""

    def __init__(self):
        """Initializate the main parameters and properties"""
        self.loc = 0
        self.scale = 0
        self.x = 0
        self.numargs = 0
        self.no = 20
        self.name = "wrap_norm"

        self.n_ = np.linspace(-np.pi, np.pi, self.no)

    def pdf(self, x, *args):
        """Compute the probability density

        Args:
            x ([type]): data

        Returns:
            [type]: [description]
        """

        if args:
            self.loc = args[0][0]
            self.scale = args[0][1]

        if isinstance(x, (float, int)):
            self.len = 1
        else:
            self.len = len(x)
        n_ = np.tile(self.n_, (self.len, 1)).T

        w = np.tile(np.exp(1j * np.pi * (x - self.loc) / (2 * np.pi)), (self.no, 1))
        q = np.tile(np.exp(-self.scale**2 / 2), (self.no, 1))
        f = (w**2) ** n_ * q ** (n_**2)

        f = np.sum(f, axis=0)
        return np.abs(f)

    def logpdf(self, x, *args):
        """Compute the logarithmic probability density

        Args:
            x ([type]): data

        Returns:
            [type]: [description]
        """
        if args:
            self.loc = args[0][0]
            self.scale = args[0][1]

        lpdf = np.log(self.pdf(x))
        return lpdf

    def cdf(self, x, loc, scale):
        """Compute the cumulative distribution

        Args:
            x ([type]): [description]
            loc ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """

        self.loc = loc
        self.scale = scale

        if isinstance(x, (float, int)):
            self.len = 1
        else:
            self.len = len(x)

        cdf_ = np.zeros(self.len)

        if self.len == 1:
            cdf_ = quad(self.pdf, 0, x, limit=100)[0]
            cdf_max = quad(self.pdf, 0, 2 * np.pi, limit=100)[0]
            cdf_ = cdf_ / cdf_max
        else:
            cdf_[-1] = quad(self.pdf, 0, x[-1], limit=100)[0]
            for ind_, val_ in enumerate(x):
                cdf_[ind_] = quad(self.pdf, 0, val_, limit=100)[0]
            cdf_ = cdf_ / cdf_[-1]
        return cdf_

    def ppf(self, x, loc, scale):
        """Compute the inverse of the cumulative distribution

        Args:
            x ([type]): [description]
            loc ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """

        data = np.linspace(0, 2 * np.pi, 360)
        df[param["var"]] = -1

        # cdfs = self.cdf(data, loc, scale)

        posi = np.zeros(len(loc), dtype=int)
        # dfn = np.sort(df["n"].unique())

        for i, j in enumerate(loc):  # Seeking every n (dates)
            cdfs = self.cdf(data, j, scale[i])
            # posn = np.argmin(np.abs(df["n"][j] - dfn))
            posj = np.argmin(np.abs(x - cdfs))
            posi[i] = posj
            if not posj:
                posi[i] = posi[i - 1]

        df.loc[:, param["var"]] = data[posi]
        return df

    def nllf(self, x0):
        """Compute the negative likelyhood function

        Args:
            x0 ([type]): [description]

        Returns:
            [type]: [description]
        """
        nllf_ = -np.sum(self.logpdf(self.x, x0))
        return nllf_

    def fit(self, x):
        """Fit the loc and scale parameters to the data

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.loc = st.circmean(x)
        self.scale = st.circstd(x)
        self.x = x
        x0 = [self.loc, self.scale]

        res = minimize(
            self.nllf, x0, bounds=[(0, 2 * np.pi), (0, 2 * np.pi)], method="SLSQP"
        )
        return res["x"]

    def name():
        """[summary]

        Returns:
            [type]: [description]
        """
        return "wrap_norm"
