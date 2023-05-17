import os

import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.utils import auxiliar
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.optimize import fsolve
from scipy.special import gamma


def confidence_intervals(boot, alpha, resample, *args):
    """Compute the confidence intervals using percentiles or bias-corrected and  accelerated (BCa) bootstrap method

    Args:
        - boot: matrix with the resampling
        - alpha: significance level
        - resample: method for estimateing the confidence intervals (standard or bca)
        - args[opt]: optional arguments use in bca method:

            * ``orig``: initial estimation of probability model parameters
            * ``nosim``: simulation number
            * ``peaks``: peaks timeseries

    Returns:
        lower and upper bound (confidence intervals)

    """

    if resample == "standard":
        ci = np.percentile(boot, np.array((alpha / 2, 1 - alpha / 2)) * 100, axis=0)
    elif resample == "bca":
        orig, nosim, peaks, tri, tipo, nyears = args[0]
        npeaks, norig = np.size(peaks), np.size(orig)
        pz0 = np.zeros(norig)
        for i in range(0, norig):
            mask = boot[:, i] < orig[i]
            pz0[i] = np.sum(mask) / float(nosim)
        z0 = st.norm.ppf(pz0)
        u = np.zeros([npeaks, norig])
        id_ = np.arange(npeaks)
        for i in range(0, npeaks):
            u[i, :] = ajusta_lmom(peaks[id_ != i], tri, tipo, nyears) + orig
        a = np.sum(u ** 3) / np.sum(u ** 2) ** (3 / 2) / 6
        zalpha = st.norm.ppf((alpha / 2, 1 - alpha / 2))
        prob = np.zeros([2, norig])
        prob[0, :] = st.norm.cdf(z0 + (z0 + zalpha[0]) / (1 - a * (z0 + zalpha[0])))
        prob[1, :] = st.norm.cdf(z0 + (z0 + zalpha[1]) / (1 - a * (z0 + zalpha[1])))
        ci = np.zeros([2, norig])
        for i in range(0, norig):
            if not (np.isnan(prob[0, i]) | np.isnan(prob[1, i])):
                ci[:, i] = np.percentile(boot[:, i], prob[:, i] * 100, axis=0)

    return ci


def bootstrapping(peaks, tri, method, func, nyears, nosim, resample):
    """Compute the parameters and the return period values given a probability model, resampling and fitting methods

    Args:
        - peaks: annual maxima timeseries
        - tri: return periods array
        - method: fitting method of the probability model
        - func: string with the name of the probabiliy model
        - nyears: number of years for simulation
        - nosim: number of simulations for the resampling method
        - resample: method of resampling

    Returns:
        Parameters and return period values of resampling
    """

    npeaks = np.size(peaks)
    orig = probability_model_fit(peaks, tri, method, func, nyears)
    boot = np.zeros([nosim, np.size(orig)])
    if resample == "parametric":
        if method == "MLE":
            params = orig[: func.numargs + 2]
            logger.info(
                "Computing bootstrapping of {} simulations using MLE and parametric methods for {} probability model. It will take a while.".format(
                    str(nosim), func.name
                )
            )
            for j in range(0, nosim):
                boot[j, :] = probability_model_fit(
                    func.rvs(*params, npeaks), tri, method, func
                )
        elif method == "L-MOM":
            logger.info(
                "Computing bootstrapping of {} simulations using L-MOM and parametric methods for {} probability model.".format(
                    str(nosim), func
                )
            )
            if (func == "expon") | (func == "genpareto"):
                for j in range(0, nosim):
                    boot[j, :] = probability_model_fit(
                        st.genpareto.rvs(orig[0], orig[1], orig[2], npeaks),
                        tri,
                        method,
                        func,
                        nyears,
                    )
            elif (func == "genextreme") | (func == "gumbel_r"):
                for j in range(0, nosim):
                    boot[j, :] = probability_model_fit(
                        st.genextreme.rvs(orig[0], orig[1], orig[2], npeaks),
                        tri,
                        method,
                        func,
                    )
    elif resample == "non-parametric":
        for j in range(0, nosim):
            logger.info(
                "Computing bootstrapping of {} simulations using {} and non-parametric methods for {} probability model. It will take a while.".format(
                    str(nosim), method, func.name
                )
            )
            boot[j, :] = probability_model_fit(
                np.random.choice(peaks, npeaks), tri, method, func, nyears
            )

    return boot, orig


def probability_model_fit(data, tr, method, func, *nyears):
    """Estimated the parameters of the probability model

    Args:
        - data: timeseries
        - tr: return periods
        - method: MLE or L-MOM for fitting the probability model (func)
        - func: probability model
        - nyears [opt]: number of years of the timeseries

    Returns:
        Parameters and return period values of data
    """

    if method == "MLE":
        params = func.fit(data)
        pri = 1 - 1.0 / tr
        qtr = func.ppf(pri, *params)
        nu = 1

    elif method == "L-MOM":
        params = l_mom(data, func)
        if (func == "expon") | (func == "genpareto"):
            nu = np.size(data) / nyears[0]
            pri = 1 - 1 / (tr * nu)
            qtr = st.genpareto.ppf(pri, *params)
        elif (func == "genextreme") | (func == "gumbel_r"):
            pri = 1 - 1.0 / tr
            qtr = st.genextreme.ppf(pri, *params)
            nu = 1

    return np.hstack((*params, nu, qtr))


def l_mom(data, func):
    """Estimated the parameters using l-moments method

    Args:
        - data: timeseries
        - func: string with the name of the probability model

    Returns:
        The fitting parameters of the data
    """

    data = np.sort(data)
    n = np.size(data)
    p = (np.arange(1, n + 1) - 0.35) / n
    lam1, lam2 = np.mean(data), np.mean((2 * p - 1) * data)
    if func == "genpareto":
        lam3 = np.mean((1 - 6 * p + 6 * p ** 2) * data)
        tau3 = lam3 / lam2
        k = (3 * tau3 - 1) / (1 + tau3)
        sig = lam2 * (1 - k) * (2 - k)
        mu = lam1 - sig / (1 - k)
    elif func == "expon":
        sig = 2 * lam2
        mu = lam1 - sig
        k = 0
    elif func == "genextreme":
        lam3 = np.mean((1 - 6 * p + 6 * p ** 2) * data)
        tau3 = lam3 / lam2

        def fun(x, tau3):
            return 2.0 * (1 - 3.0 ** -x) / (1 - 2.0 ** -x) - 3 - tau3

        c = 2 / (3 + tau3) - np.log(2) / np.log(3)
        k = 7.859 * c + 2.9554 * c ** 2
        k = fsolve(fun, k, args=(tau3))
        sig = lam2 * k / (1 - 2 ** (-k)) / gamma(1 + k)
        mu = lam1 - sig * (1 - gamma(1 + k)) / k
    elif func == "gumbel_r":
        sig = lam2 / np.log(2)
        mu = lam1 - 0.5772 * sig
        k = 0

    return [k, mu, sig]


def lmom_genpareto(data, tr, anios):
    """Funcion que calcula el valor de los parametros de la pareto generalizada a partir del metodo de los l-moments

    Args:
        - data: serie de peaks
        - tr: periodos de retorno
        - anios: numero de anios de la serie

    Returns:
        Devuelve todos los parametros del ajuste
    """

    k, mu, sig = l_mom(data, "genpareto")
    mrlp = np.mean(data - mu)
    sigmodif = sig - k * mu
    nu = np.size(data) / anios
    pri = 1 - 1 / (tr * nu)
    qtr = st.genpareto.ppf(pri, k, mu, sig)
    return np.hstack((k, mu, sig, mrlp, sigmodif, nu, qtr))


def pot_method(
    df: pd.DataFrame,
    var_: str,
    window_size: int,
    alpha: float = 0.05,
    sim_no: int = 10000,
    method: str ="nearest"
):
    """Compute the parameters of Generalized Pareto Distribution using l-moment. The
    confidence intervals are obtained using bootstrapping techniques.

    Args:
        - df: dataframe con los datos
        - alpha: nivel de significancia
        - sim_no: number of bootstrapping sample

    Returns:
        Devuelve una tupla con los valores del test de Anderson-Darling, test de Anderson-Darling modificado, el valor
        medio, los intervalos de confianza, el umbral definido, periodos de retorno y el numero de anios de la muestra

    """
    # Initialize some variables variables
    tr_eval = np.array([1,50, 100, 1000])

    thresholds = np.percentile(
        df[var_], np.hstack([np.linspace(90, 99, 10), np.linspace(99.1, 99.9, 9)])
    )

    no_thres = np.size(thresholds)

    results = {}
    results["mean_value_lmom"], results["upper_lim"], results["lower_lim"], results["au2_lmom"], results["au2pv_lmom"] = (
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros(no_thres),
        np.zeros(no_thres),
    )

    # Load p-value table for Anderson-Darling test and create the interpolator
    # data = loadmat(
    #     os.path.join(os.path.dirname(__file__), "../", "utils/misc/PVAL_AU2_LMOM.mat")
    # )

    # pvallmom, au2val, ko, ndata = (
    #     data["PVAL"],
    #     data["AU2VAL"][0],
    #     data["Ko"][0],
    #     data["NDATA"][0],
    # )

    # interpolator = RegularGridInterpolator(
    #     (ndata, ko, au2val),
    #     pvallmom,
    #     method=method,
    #     bounds_error=False,
    #     fill_value=None,
    # )

    # Compute the POT
    logger.info(
        "Computing POT analysis for a window size of "
        + str(df.index[window_size] - df.index[0])
    )
    df_max_events = auxiliar.max_moving(df[var_], window_size)
    event = df_max_events.loc[
        df_max_events[var_] > np.percentile(df[var_], 90), var_
    ].values[:]
    nyears = df.index.year.max() - df.index.year.min()

    # Make the analysis
    for i in range(no_thres):
        logger.info(
            "Threshold "
            + str(i + 1)
            + " of "
            + str(no_thres)
            + " ("
            + str(np.round(thresholds[i], decimals=3))
            + ")."
        )
        events_for_threshold_i = event[event > thresholds[i]]
        n_evi = np.size(events_for_threshold_i)
        results["mean_value_lmom"][i, :] = lmom_genpareto(events_for_threshold_i, tr_eval, nyears)

        # Bootstraping using l-moments for Generalized Pareto Distributions
        boot = np.zeros([sim_no, np.size(tr_eval) + 6])

        for j in range(0, sim_no):
            boot[j, :] = lmom_genpareto(
                np.random.choice(events_for_threshold_i, n_evi), tr_eval, nyears
            )

        results["upper_lim"][i, :] = np.percentile(boot, (1 - alpha / 2) * 100, axis=0)
        results["lower_lim"][i, :] = np.percentile(boot, (alpha / 2) * 100, axis=0)

        # Anderson-Darling Upper without confidence intervals for l-moments
        # results["au2_lmom"][i] = au2(results["mean_value_lmom"][i, 0:3], events_for_threshold_i)
        # results["au2pv_lmom"][i] = interpolator(
        #     [
        #         np.max((10, np.min((500, n_evi)))),
        #         np.sign(results["mean_value_lmom"][i, 0])
        #         * np.min((np.abs(results["mean_value_lmom"][i, 0]), 0.5)),
        #         results["au2_lmom"][i],
        #     ]
        # )
    
    results["nyears"] = nyears
    results["thresholds"] = thresholds
    results["tr_eval"] = tr_eval

    return results


def au2(par, data):
    """Funcion que calcula el valor del estadistico Anderson-Darling

    Args:
        - par: parametros medios de la funcion de distribucion de pareto
        - data: peaks sobre umbral

    Returns:
        Devuelve el valor del test

    """

    ndat = np.size(data)
    cdf = np.sort(st.genpareto.cdf(data, *par))
    cdf -= 1e-6  # TODO: para evitar los 1.0 en la cdf
    estadist = (
        ndat / 2
        - 2 * np.sum(cdf)
        - np.sum((2 - (2 * np.arange(1, ndat + 1) - 1.0) / ndat) * np.log(1 - cdf))
    )

    return estadist


def automatico_lmom_boot(df_eventos, alpha, umb, param, bca, nyears):
    """Funcion que calcula los parametros de la funcion de Pareto generalizada a partir de tecnicas de remuestreo,
    seleccionado el umbral, la tecnica de remuestreo y calcula de los intervalos de confianza

    Args:
        - df_eventos: serie de datos de entrada
        - alpha: nivel de significancia para los intervalos de confianza
        - umbral: umbral seleccionado para el regimen extremal
        - param: tipo de tecnica de remuestreo (parametrica o no-parametrica)
        - bca: metodo de calculo de los intervalos de confianza
        - nyears: numero de anios de la muestra

    Returns:
        Devuelve una tupla con:

            * ``boot``: matriz con los resultados de la aplicacion de la tecnica de remuestreo
            * ``orig``: parametros originales de la funcion, ci, tr, peaks, npeaks, eventanu

    """
    tr = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
    nosim = 10000

    #  peaks SOBRE EL UMBRAL CON PWM Y C.I. BOOTSTRAPPING
    # Identifica eventos
    numb = np.size(umb)
    event = df_eventos.values[:]

    boot, orig, ci = (
        [list() for i in range(2)],
        [list() for i in range(2)],
        [list() for i in range(2)],
    )
    for i in range(numb):
        # df_pot = df_eventos[df_eventos.values[:, 0] > umb[i]]  # TODO: ver las funciones siguientes para eliminar peaks
        # y dejar solo df_pot
        peaks = event[event > umb[i]]
        npeaks = np.size(peaks)
        eventanu = npeaks / nyears

        # Valor central y Bootstrapping "casero" con umbral predefinido
        boot_s, orig_s = bootstrapping(peaks, tr, "L-MOM", "genpareto", nyears, nosim, param)
        ci_s = confidence_intervals(
            boot_s, alpha, bca, (orig_s, nosim, peaks, tr, "genpareto", nyears)
        )
        boot[0].append(boot_s), orig[0].append(orig_s), ci[0].append(ci_s)

        # Si exponencial es posible, la calcula
        expo = (ci_s[0, 0] < 0) & (ci_s[1, 0] > 0)
        if expo:
            boot_e, orig_e = bootstrapping(peaks, tr, "L-MOM", "expon", nyears, nosim, param)
            ci_e = confidence_intervals(
                boot_e, alpha, bca, (orig_e, nosim, peaks, tr, "expon", nyears)
            )
            boot[1].append(boot_e), orig[1].append(orig_e), ci[1].append(ci_e)

    return boot, orig, ci, tr, peaks, npeaks, eventanu


def annual_maxima_method(df, alpha, method, func, resample, ci_method, tr = None):
    """Funcion que calcula los parametros de la funcion de GEV, los intervalos de confianza y los valores para
    distintos periodos de retorno

    Args:
        - df: serie de maximos anuales
        - alpha: nivel de significancia para los intervalos de confianza
        - resample: metodo de remuestreo
        - ci_method: method for computing the confidence intervals


    Returns:
        Devuelve una tupla con la matriz de los parametros ajustados de la aplicacion de la tecnica de remuestreo, de
        la serie original y sus intervalos de confianza

            * ``boot``: matriz con los resultados de la aplicacion de la tecnica de remuestreo
            * ``orig``: parametros originales de la funcion, ci, tr, peaks, npeaks, eventanu

    """

    if not ((method == "MLE") | (method == "L-MOM")):
        raise ValueError(
            'Fitting methods for probability models are "MLE" or "L-MOM". Given {}.'.format(
                method
            )
        )

    if method == "L-MOM":
        if not func in ["genpareto", "genextreme", "gumbel_r", "expon"]:
            raise ValueError(
                'Function {} is not included in L-MOM methods for fitting. Use "genpareto", "genextreme", "gumbel_r" or "expon".'.format(
                    method
                )
            )
    else:
        try:
            func = getattr(st, func)
        except:
            raise ValueError("Function {} is not included scipy.stats.".format(func))

    if not ((resample == "parametric") | (resample == "non-parametric")):
        raise ValueError(
            'Resampling methods are "parametric" or "non-parametric". Given {}'.format(
                resample
            )
        )

    if not ((ci_method == "standard") | (ci_method == "bca")):
        raise ValueError(
            'Confidence interval methods are "standard" or "bca". Given {}'.format(
                ci_method
            )
        )

    if tr is None:
        tr = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
        tr = np.hstack((np.arange(1.1, 2 + 1e-6, 0.1), tr[tr > 2]))
    annumax = df.groupby(df.index.year).max()
    nyears = len(annumax)
    pannumax = np.arange(1.0, nyears + 1.0) / (nyears + 1.0)

    nosim = 10
    boot, orig, ci = (
        [list() for i in range(2)],
        [list() for i in range(2)],
        [list() for i in range(2)],
    )
    boot_a, orig_a = bootstrapping(annumax, tr, method, func, nyears, nosim, resample)
    ci_a = confidence_intervals(
        boot_a, alpha, ci_method, (orig_a, nosim, annumax, tr, func, nyears)
    )
    boot[0].append(boot_a), orig[0].append(orig_a), ci[0].append(ci_a)

    # Wheter the parameters indicate that gumbel_r can be estimated, repeat the method with gumbel_r probability model
    if (ci_a[0, 0] < 0) & (ci_a[1, 0] > 0):
        if method == "MLE":
            func = st.gumbel_r
        else:
            func = "gumbel_r"
        boot_g, orig_g = bootstrapping(
            annumax, tr, method, func, nyears, nosim, resample
        )
        ci_g = confidence_intervals(
            boot_g, alpha, ci_method, (orig_g, nosim, annumax, tr, func, nyears)
        )
        boot[1].append(boot_g), orig[1].append(orig_g), ci[1].append(ci_g)

    return tr, pannumax, annumax, boot, orig, ci
