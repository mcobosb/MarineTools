import datetime
import time

import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.utils import auxiliar, read, save

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


def show_init_message():
    message = (
        "\n"
        + "=============================================================================\n"
        + " Initializing MarineTools.temporal, v.1.0.0\n"
        + "=============================================================================\n"
        + "Copyright (C) 2021 Environmental Fluid Dynamics Group (University of Granada)\n"
        + "=============================================================================\n"
        + "This program is free software; you can redistribute it and/or modify it under\n"
        + "the terms of the GNU General Public License as published by the Free Software\n"
        + "Foundation; either version 3 of the License, or (at your option) any later \n"
        + "version.\n"
        + "This program is distributed in the hope that it will be useful, but WITHOUT \n"
        + "ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS\n"
        + "FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
        + "You should have received a copy of the GNU General Public License along with\n"
        + "this program; if not, write to the Free Software Foundation, Inc., 675 Mass\n"
        + "Ave, Cambridge, MA 02139, USA.\n"
        + "============================================================================="
    )
    return message


def marginalfit(df: pd.DataFrame, parameters: dict):
    """Fits a stationary (or not), simple or mixed probability model to data. Additional
    information can be found in Cobos et al., 2022, 'MarineTools.temporal: A Python
    package to simulate Earth and environmental time series'. Environmental Modelling
    and Software.

    Args:
        * df (pd.DataFrame): the raw time series
        * parameters (dict): the initial guess parameters of the probability models.
            - 'var' key is a  string with the name of the variable,
            - 'type': it defines circular or linear variables
            - 'fun' is a list within strings with the name of the probability model,
            - 'non_stat_analysis' stand for stationary (False) or not (True),
            - 'ws_ps': initial guess of percentiles or weights of PMs
            - 'basis_function' is an option to specify the GFS expansion that includes:
                - 'method': a string with an option of the GFS
                - 'no_terms': number of terms of GFS
                - 'periods': is a list with periods of oscillation for NS-PMs ,
            - 'transform' stand for normalization that includes:
                - 'make': True or False
                - 'method': box-cox or yeo-jonhson
                - 'plot': True or False
            - 'detrend': stand for removing trends of time series through the statistical parameters
                - 'make': True or False
                - 'method': a string with an option of the GFS (commonly polynomials
                    approaches)
            - 'optimization': a dictionary with some initial parameters for the optimization
            method (see scipy.optimize.minimize), some options are:
                - 'method': "SLSQP",
                - 'maxiter': 1e2,
                - 'ftol': 1e-4
                - 'eps': 1e-7
                - 'bounds': 0.5
                - 'weighted': a boolean for weighted data along the time axis. A low number
            of values at some times might give incongruent results.
            - giter: number of global iterations. Repeat the minimization algorithm
            changing the initial guess
            - 'initial_parameters': when initial parameters of a unique optimization
            mode are given
                - 'make': True or False
                - 'mode': a list with the mode to be computed independently,
                - 'par': initial guess of the parameters for the mode given
            (optional)
            - 'file_name': string where it will be saved the analysis (optional)

    Example:
        * param = {'Hs': {'var': 'Hs',
                        'fun': {0: 'norm'},
                        'type': 'linear' (default) or 'circular',
                        'non_stat_analysis': True (default), False,
                        'basis_function': None or a list with:
                            {"method": "trigonometric", "sinusoidal", ...
                            "no_terms": int,
                            "periods": [1, 2, 4, ...]}
                        'ws_ps': 1 or a list,
                        'transform': None or a dictionary with:
                            {"make": True,
                            "plot": False,
                            "method": "box-cox" or "yeo-johnson"},
                        'detrend': None or a dictionary with:
                            {"make": True,
                            "method": "chebyschev", "legendre", "polynomial", ...},
                        'initial_parameters': None or a list with:
                            {'mode': [6] or [2,2],
                            'par': a list with initial parameters if mode is given},
                        'optimization': {'method': 'SLSQP' (default), 'dual_annealing',
                            'differential_evolution' or 'shgo',
                            'eps', 'ftol', 'maxiter', 'bounds'},
                        'giter': 10,
                        'scale': False,
                        'bic': True or False,
                        'file_name': if not given, it is created from input parameters
                        }
                    }

    Returns:
        * dict: the fitting parameters
    """
    # Initial computational time
    start_time = time.time()
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(show_init_message())
    logger.info("Current Time = %s\n" % current_time)

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Check if negative and positive values are in the timeseries for fitting purpouses
    if df[df < 0].any().values[0]:
        logger.info(
            "Dataset has negative values. Check that the chosen distribution functions adequately fit negative values."
        )

    if parameters["type"] == "circular":
        # Transform angles to radian
        df = np.deg2rad(df)
        # Compute the percentile of change between probability models
        ecdf = auxiliar.ecdf(df, parameters["var"], no_perc=1000)
        # Smooth the ecdf
        ecdf["soft"] = auxiliar.smooth_1d(ecdf[parameters["var"]], 100)
        # Compute the difference
        ecdf["dif"] = ecdf["soft"].diff()
        # Obtain the index of the max
        max_ = auxiliar.max_moving(ecdf["dif"], 250)
        parameters["ws_ps"] = [max_.index[0]]

    # Check that the input dictionary is well defined
    parameters = check_marginal_params(parameters)

    # Normalized the data using one of the normalization method if it is required
    if parameters["transform"]["make"]:
        df, parameters = stf.transform(df, parameters)
        parameters["transform"]["min"] = df.min().values[0] - 1e-2
        df -= parameters["transform"]["min"]
        # if parameters["piecewise"]:
        #     for ind_, val_ in enumerate(parameters["ws_ps"]):
        #         parameters["ws_ps"][ind_] = val_ - parameters["transform"]["min"]

    # Scale and shift time series for ensuring the use of any PM
    parameters["range"] = float((df.max() - df.min()).values[0])
    if parameters["scale-shift"]:
        if parameters["range"] > 10:
            df = df / (parameters["range"] / 3)
            parameters["scale"] = parameters["range"] / 3
            # if parameters["piecewise"]:
            #     for ind_, val_ in enumerate(parameters["ws_ps"]):
            #         parameters["ws_ps"][ind_] = val_ / (parameters["range"] / 3)

    # Bound the variable with some reference values
    if parameters["type"] == "circular":
        parameters["minimax"] = [0, 2 * np.pi]
    else:
        parameters["minimax"] = [
            float(df[parameters["var"]].min()),
            float(
                np.max(
                    [df[parameters["var"]].max(), df[parameters["var"]].max() * 1.25]
                )
            ),
        ]

    # Calculate the normalize time along the reference oscillating period
    df["n"] = np.fmod(
        (df.index - datetime.datetime(df.index[0].year, 1, 1, 0)).total_seconds().values
        / (parameters["basis_period"][0] * 365.25 * 24 * 3600),
        1,
    )
    # Create and additional time line for detrending
    if parameters["detrend"]["make"]:
        df["n_detrend"] = (df.index - df.index[0]) / (df.index[-1] - df.index[0])

    # Compute the temporaL weights if it is required
    if parameters["weighted"]["make"]:
        if parameters["weighted"]["window"] == "month":
            counts = df.groupby("n").count()  # TODO: con pesos promedio mensuales.
        else:
            counts = df.groupby("n").count()
            nmean = counts.mean()
            weights = nmean / counts
            df["counts"] = np.nan
            for n in counts.index:
                nn = len(df.loc[df["n"] == n, "counts"])
                df.loc[df["n"] == n, "counts"] = np.ones(nn) * weights.loc[n].values
        parameters["weighted"]["values"] = df["counts"]
    else:
        parameters["weighted"]["values"] = 1

    if not parameters["non_stat_analysis"]:
        # Make the stationary analysis
        logger.info("MARGINAL STATIONARY FIT")
        logger.info(
            "=============================================================================="
        )
        # Write the information about the variable, PMs and method
        term = (
            "Stationary fit of "
            + parameters["var"]
            + " to a "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += " - " + str(parameters["fun"][i].name)
        term += " - genpareto " * parameters["reduction"]
        term += " probability model"
        logger.info(term)
        # Make the stationary analysis
        df, parameters["par"], parameters["mode"] = stf.st_analysis(df, parameters)

        # Write the information about the variable, PMs and method
    elif parameters["non_stat_analysis"] & (
        not parameters["initial_parameters"]["make"]
    ):
        # Make the stationary analysis first
        df, parameters["par"], parameters["mode"] = stf.st_analysis(df, parameters)
        # Make the non-stationary analysis
        logger.info("MARGINAL NON-STATIONARY FIT")
        logger.info(
            "=============================================================================="
        )
        term = (
            "\nNon-stationary fit of "
            + parameters["var"]
            + " with the "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += " - " + str(parameters["fun"][i].name)
        if parameters["reduction"]:
            term += " - genpareto " * parameters["reduction"]
        term += " probability model"
        logger.info(term)
        logger.info(
            "with the " + parameters["optimization"]["method"] + " optimization method."
        )
        logger.info(
            "=============================================================================="
        )

        # Make the non-stationary analysis
        parameters = stf.nonst_analysis(df, parameters)

    else:
        # Make the non-stationary analysis of a given mode
        term = (
            "Non-stationary fit of "
            + parameters["var"]
            + " with the "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += "-" + str(parameters["fun"][i].name)
        term += " and mode:"
        for mode in parameters["initial_parameters"]["mode"]:
            term += " " + str(mode)
        logger.info(term)
        logger.info(
            "=============================================================================="
        )
        # Make the non-stationary analysis
        parameters = stf.nonst_analysis(df, parameters)

    # List all the parameters in a standar format
    if parameters["reduction"]:
        if not parameters["non_stat_analysis"]:
            pars, esc = stf.get_params(df, parameters, parameters["par"], [0, 0, 0], 0)
        else:  # Los parámetros del no estacionario para la cola inferior no tienen amplitud y fase
            t_expans = stf.params_t_expansion(
                parameters["mode"],
                parameters,
                df["n"],
            )
            pars, esc = stf.get_params(
                df,
                parameters,
                parameters["par"],
                [parameters["mode"][0], parameters["mode"][0], parameters["mode"][0]],
                t_expans,
            )

        parameters_ = pars.iloc[0, 2:]
        parameters_["ps_0"] = esc[1]
        parameters_["ps_1"] = 1 - esc[2]
        parameters["standar_parameters"] = parameters_.to_dict()

    # Change the object function for its string names
    parameters["fun"] = {i: parameters["fun"][i].name for i in parameters["fun"].keys()}
    parameters["status"] = "Distribution models fitted succesfully"

    # Final computational time
    logger.info("End of the fitting process")
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    # Save the parameters in the file if "file_name" is given in params
    # auxiliar.mkdir(parameters["folder_name"])

    if not "file_name" in parameters.keys():
        generate_outputfilename(parameters)

    del parameters["weighted"]["values"]
    save.to_json(parameters, parameters["file_name"])

    # Return the dictionary with the parameters of the analysis
    return parameters


def check_marginal_params(param: dict):
    """Checks the input parameters and includes some required arguments for the computation

    Args:
        * param (dict): the initial guess parameters of the probability models (see the docs of marginalfit)

    Returns:
        * param (dict): checked and updated parameters
    """

    param["no_param"] = {}
    param["scipy"] = {}
    param["reduction"] = False
    param["no_tot_param"] = 0
    param["constraints"] = True

    logger.info("USER OPTIONS:")
    k = 1

    # Checking the transform parameters if any
    if not "transform" in param.keys():
        param["transform"] = {}
        param["transform"]["make"] = False
        param["transform"]["plot"] = False
    else:
        if param["transform"]["make"]:
            if not param["transform"]["method"] in ["box-cox", "yeo-johnson"]:
                raise ValueError(
                    "The power transformation methods available are 'yeo-johnson' and 'box-cox', {} given.".format(
                        param["transform"]["method"]
                    )
                )
            else:
                logger.info(
                    str(k)
                    + " - Data is previously normalized ("
                    + param["transform"]["method"]
                    + " method given)".format(str(k))
                )
                k += 1

    # Checking the detrend parameters if any
    if not "detrend" in param.keys():
        param["detrend"] = {}
        param["detrend"]["make"] = False
    else:
        if param["detrend"]["make"]:
            if not param["detrend"]["method"] in [
                "chebyshev",
                "legendre",
                "laguerre",
                "hermite",
                "ehermite",
                "polynomial",
            ]:
                raise ValueError(
                    "Methods available are:\
                    chebyshev, legendre, laguerre, hermite, ehermite or polynomial.\
                    Given {}.".format(
                        param["detrend"]["method"]
                    )
                )
            else:
                logger.info(
                    str(k)
                    + " - Detrend timeseries is appliedData is previously normalized ("
                    + param["detrend"]["method"]
                    + " method given)".format(str(k))
                )
                k += 1

    # Check if it can be reduced the number of parameters using Solari (2011) analysis
    if not "fun" in param.keys():
        raise ValueError("Probability models are required in a list in fun.")
    if len(param["fun"].keys()) == 3:
        if (param["fun"][0] == "genpareto") & (param["fun"][2] == "genpareto"):
            param["reduction"] = True
            logger.info(
                str(k)
                + " - The combination of PMs given enables the reduction"
                + " of parameters during the optimization"
            )
            k += 1

    # Check the number of probability models
    if len(param["fun"].keys()) > 3:
        raise ValueError(
            "No more than three probability models are allowed in this version"
        )

    if param["reduction"]:
        # Particular case where the number of parameters to be optimized is reduced
        param["fun"] = {
            0: getattr(st, param["fun"][0]),
            1: getattr(st, param["fun"][1]),
            2: getattr(st, param["fun"][2]),
        }
        param["no_fun"] = 2
        param["no_param"][0] = int(param["fun"][1].numargs + 2)
        if param["no_param"][0] > 5:
            raise ValueError(
                "Probability models with more than 3 parameters are not allowed in this version"
            )
        param["no_param"][1] = 1
        param["no_tot_param"] = int(param["fun"][1].numargs + 3)
    else:
        param["no_fun"] = len(param["fun"].keys())
        for i in range(param["no_fun"]):
            if isinstance(param["fun"][i], str):
                if param["fun"][i] == "wrap_norm":
                    param["fun"][i] = stf.wrap_norm()
                    param["scipy"][i] = False
                    param["constraints"] = False
                else:
                    param["fun"][i] = getattr(st, param["fun"][i])
                    param["scipy"][i] = True
            param["no_param"][i] = int(param["fun"][i].numargs + 2)
            if param["no_param"][i] > 5:
                raise ValueError(
                    "Probability models with more than 3 parameters are not allowed in this version"
                )
            param["no_tot_param"] += int(param["fun"][i].numargs + 2)

    if param["non_stat_analysis"] == False:
        param["basis_period"] = None
        param["basis_function"] = {"method": "None", "order": 0, "no_terms": 0}

    if not "basis_period" in param:
        param["basis_period"] = [1]
    elif param["basis_period"] == None:
        param["order"] = 0
        if param["non_stat_analysis"] == False:
            param["basis_period"] = [1]
    elif isinstance(param["basis_period"], int):
        param["basis_period"] = list(param["basis_period"])

    if (not "basis_function" in param.keys()) & param["non_stat_analysis"]:
        raise ValueError("Basis function is required when non_stat_analysis is True.")

    if (not "method" in param["basis_function"]) & param["non_stat_analysis"]:
        raise ValueError("Method is required when non_stat_analysis is True.")
    elif param["non_stat_analysis"]:
        if param["basis_function"]["no_terms"] == 0:
            raise ValueError(
                "Number of period higher than zero is required when non_stat_analysis is True."
            )
        elif param["basis_function"]["method"] in [
            "trigonometric",
            "sinusoidal",
            "modified",
        ]:
            if ((not "no_terms") & (not "periods")) in param["basis_function"].keys():
                raise ValueError(
                    "Number of terms or periods are required for Fourier Series approximation."
                )
            else:
                if not "periods" in param["basis_function"]:
                    param["basis_function"]["periods"] = list(
                        1 / np.arange(1, param["basis_function"]["no_terms"] + 1)
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                else:
                    param["basis_function"]["no_terms"] = len(
                        param["basis_function"]["periods"]
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                # param["approximation"]["periods"].sort(reverse=True)
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]
        else:
            if param["basis_function"]["method"] not in [
                "chebyshev",
                "legendre",
                "laguerre",
                "hermite",
                "ehermite",
                "polynomial",
            ]:
                raise ValueError(
                    "Method available are:\
                    trigonometric, modified, sinusoidal, \
                    chebyshev, legendre, laguerre, hermite, ehermite or polynomial.\
                    Given {}.".format(
                        param["basis_function"]["method"]
                    )
                )
            else:
                if not "degree" in param["basis_function"].keys():
                    raise ValueError("The polynomial methods require the degree")
                param["basis_function"]["order"] = param["basis_function"]["degree"]
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]

        logger.info(
            str(k)
            + " - The basis function given is {}.".format(
                param["basis_function"]["method"]
            )
        )
        k += 1

        logger.info(
            str(k)
            + " - The number of terms given is {}.".format(
                param["basis_function"]["order"]
            )
        )
        k += 1

    # Check if initial parameters are given
    if not "initial_parameters" in param.keys():
        param["initial_parameters"] = {}
        param["initial_parameters"]["make"] = False
    elif "initial_parameters" in param.keys():
        if not "make" in param["initial_parameters"]:
            raise ValueError(
                "The evaluation of a certain mode requires that initial parameter 'make' set to True. Not given."
            )
        if not "par" in param["initial_parameters"].keys():
            param["initial_parameters"]["par"] = []
            logger.info(
                str(k)
                + " - Parameters of optimization not given. It will be applied the Fourier initialization."
            )
            k += 1
        else:
            logger.info(
                str(k)
                + " - Parameters of optimization given ({}).".format(
                    param["initial_parameters"]["par"]
                )
            )
            k += 1
        if not "mode" in param["initial_parameters"].keys():
            raise ValueError(
                "The evaluation of a mode requires the initial mode 'mode'. Give the mode."
            )
        else:
            logger.info(
                str(k)
                + " - Mode of optimization given ({}).".format(
                    param["initial_parameters"]["mode"]
                )
            )
            k += 1
        if not "plot" in param["initial_parameters"].keys():
            param["initial_parameters"]["plot"] = False
    # TODO: chequear lo siguiente, no tiene mucho sentido
    # else:
    #     param["initial_parameters"] = {}
    #     param["initial_parameters"][
    #         "make"
    #     ] = False  # TODO: chequear que la opción con True funciona correctamente
    #     param["initial_parameters"]["mode"] = [param["basis_function"]["order"]]
    #     param["initial_parameters"]["par"] = []
    #     logger.info(
    #         str(k)
    #         + " - Initial parameters will be computed using series expansion of the given order ({}).".format(
    #             param["basis_function"]["order"]
    #         )
    #     )
    #     k += 1

    if not "optimization" in param.keys():
        param["optimization"] = {}
        param["optimization"]["method"] = "SLSQP"
        param["optimization"]["eps"] = 1e-3
        param["optimization"]["maxiter"] = 1e2
        param["optimization"]["ftol"] = 1e-3
        if param["initial_parameters"]["make"]:
            logger.info(
                str(k)
                + " - Optimization method more adequate using the Fourier Series initialization is {}.".format(
                    "dual_annealing"
                )
            )
            param["optimization"]["method"] = "dual_annealing"

            k += 1
    else:
        if param["optimization"] is None:
            param["optimization"] = {}
            param["optimization"]["method"] = "SLSQP"
            param["optimization"]["eps"] = 1e-3
            param["optimization"]["maxiter"] = 1e2
            param["optimization"]["ftol"] = 1e-3
        else:
            if not "eps" in param["optimization"].keys():
                param["optimization"]["eps"] = 1e-3
            if not "maxiter" in param["optimization"].keys():
                param["optimization"]["maxiter"] = 1e2
            if not "ftol" in param["optimization"].keys():
                param["optimization"]["ftol"] = 1e-3

    # if not "method" in param["optimization"]:
    #     param["optimization"]["method"] = "SLSQP"
    # else:
    #     logger.info(
    #         "{} - Optimization method was given by user ({})".format(
    #             str(k), str(param["optimization"]["method"])
    #         )
    #     )
    #     k += 1

    if not "giter" in param["optimization"].keys():
        param["optimization"]["giter"] = 10
    else:
        if not isinstance(param["optimization"]["giter"], int):
            raise ValueError("The number of global iterations should be an integer.")
        else:
            logger.info(
                "{} - Global iterations were given by user ({})".format(
                    str(k), str(param["optimization"]["giter"])
                )
            )
            k += 1

    # if param["initial_parameters"]["make"]:
    #     param["optimization"]["method"] = "dual_annealing"

    if not "bounds" in param["optimization"].keys():
        if param["type"] == "circular":
            param["optimization"]["bounds"] = 0.1
        else:
            param["optimization"]["bounds"] = 0.5
    else:
        if not isinstance(param["optimization"]["bounds"], (float, int, bool)):
            raise ValueError("The bounds should be a float, integer or False.")
        else:
            logger.info(
                "{} - Bounds were given by user (bounds = {})".format(
                    str(k), str(param["optimization"]["bounds"])
                )
            )
            k += 1

    if "piecewise" in param:
        if not param["reduction"]:
            if param["piecewise"]:
                param["constraints"] = False
                logger.info(
                    str(k)
                    + " - Piecewise analysis of PMs defined by user. Piecewise is set to True."
                )
                k += 1
        else:
            logger.info(
                str(k)
                + " - Piecewise analysis is not recommended when reduction is applied. Piecewise is set to False."
            )
            param["piecewise"] = False
            k += 1
    else:
        param["piecewise"] = False

    if param["no_fun"] == 1:
        param["constraints"] = False

    if param["reduction"]:
        param["constraints"] = False

    if not "transform" in param.keys():
        param["transform"] = {"make": False, "method": None, "plot": False}

    # if "debug" in param.keys():
    #     if param["debug"]:
    #         logger.add(
    #             "debug_file.log", format="{message}", level="DEBUG", rotation="5 MB"
    #         )
    #         logger.info("{} - Debug mode ON.".format(str(k)))
    #         k += 1

    if param["reduction"]:
        if len(param["ws_ps"]) != 2:
            raise ValueError(
                "Expected two percentiles for the analysis. Got {}.".format(
                    str(len(param["ws_ps"]))
                )
            )
    else:
        if (not "ws_ps" in param) & (param["no_fun"] - 1 == 0):
            param["ws_ps"] = []
        elif (not "ws_ps" in param) & (param["no_fun"] - 1 != 0):
            raise ValueError(
                "Expected {} weight\\s for the analysis. However ws_ps option is not given.".format(
                    str(param["no_fun"] - 1)
                )
            )

        if len(param["ws_ps"]) != param["no_fun"] - 1:
            raise ValueError(
                "Expected {} weight\\s for the analysis. Got {}.".format(
                    str(param["no_fun"] - 1), str(len(param["ws_ps"]))
                )
            )

    # Check if the variable is circular or linear
    if param["type"] == "circular":
        logger.info("{} - Type is set to 'circular'.".format(str(k)))
    else:
        logger.info("{} - Type is set to 'linear'.".format(str(k)))
    k += 1

    if (any(np.asarray(param["ws_ps"]) > 1) or any(np.asarray(param["ws_ps"]) < 0)) & (
        not param["piecewise"]
    ):
        raise ValueError(
            "Percentiles cannot be lower than 0 or bigger than one. Got {}.".format(
                str(param["ws_ps"])
            )
        )

    if not "guess" in param.keys():
        param["guess"] = False

    if not "bic" in param.keys():
        param["bic"] = False

    if param["constraints"]:
        if (not param["optimization"]["method"] == "SLSQP") & (param["no_fun"] > 1):
            raise ValueError(
                "Constraints are just available for SLSQP method in this version."
            )

    if "fix_percentiles" in param.keys():
        if param["fix_percentiles"]:
            logger.info(
                "{} - Percentiles are fixed. Fix_percentiles is set to True.".format(
                    str(k)
                )
            )
            k += 1
    elif not param["non_stat_analysis"]:
        param["fix_percentiles"] = True
        logger.info(
            "{} - Percentiles are fixed. Fix_percentiles is set to True.".format(str(k))
        )
        k += 1
    else:
        param["fix_percentiles"] = False

    # if not "folder_name" in param.keys():
    #     param["folder_name"] = "marginalfit/"

    if not "scale-shift" in param.keys():
        param["scale-shift"] = False

    if not param["scale-shift"]:
        param["scale"] = 1
        param["shift"] = 0

    if not "weighted" in param.keys():
        param["weighted"] = {}
        param["weighted"]["make"] = False
    else:
        if not "make" in param["weighted"].keys():
            param["weighted"]["make"] = False
        elif isinstance(param["weighted"]["make"], bool):
            logger.info("{} - Weighted data along time is set to True.".format(str(k)))
            k += 1
            if not "window" in param["weighted"]:
                param["weighted"]["window"] = "timestep"
            elif not (
                (param["weighted"]["window"] == "timestep")
                | (param["weighted"]["window"] == "month")
            ):
                raise ValueError("Weighted window options are 'timestep' or 'month'.")
            logger.info(
                "{} - Weighted window for every {}.".format(
                    str(k), param["weighted"]["window"]
                )
            )
            k += 1
        else:
            raise ValueError("Weighted options are True or False.")

    if k == 1:
        logger.info("None.")

    logger.info(
        "==============================================================================\n"
    )

    return param


# def init_fourier_coefs():
#     """Compute an estimation of the initial parameters for trigonometric expansions"""
#     timestep = 1 / 365.25
#     wlen = 14 / 365.25  # 14-days window
#     res = pd.DataFrame(
#         0, index=np.arange(0, 1, timestep), columns=["s", "loc", "scale"]
#     )
#     for ii, i in enumerate(res.index):
#         if i >= (1 - wlen):
#             final_offset = i + wlen - 1
#             mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
#                 data["n"] <= final_offset
#             )
#         elif i <= wlen:
#             initial_offset = i - wlen
#             mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
#                 data["n"] >= 1 + initial_offset
#             )
#         else:
#             mask = (data["n"] >= i - wlen) & (data["n"] <= i + wlen)

#         model = st.gamma
#         result = st.fit(
#             model,
#             data[station].loc[mask],
#             bounds=[(0, 5), bound, (0, 100)],
#         )
#         res.loc[i, :] = result.params.a, result.params.loc, result.params.scale

#     coefs = np.fft.fft(res.loc[:, paramName] - np.mean(res.loc[:, paramName]))

#     N = len(res.loc[:, paramName])
#     # Choose one side of the spectra
#     cn = np.ravel(coefs[0 : N // 2] / N)

#     an, bn = 2 * np.real(cn), -2 * np.imag(cn)

#     an = an[: index + 1]
#     bn = bn[: index + 1]

#     parameters = np.mean(res.loc[:, paramName])
#     for order_k in range(index):
#         parameters = np.hstack([parameters, an[order_k + 1], bn[order_k + 1]])
#     return


def nanoise(
    data: pd.DataFrame, variable: str, remove: bool = False, filter_: str = None
):
    """Adds noise to time series for better estimations

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): variable to apply noise
        * remove (bool): if True filtered data is removed
        * filter_ (list): lower limit of values to be filtered. See query pandas DataFrame.

    Returns:
        * df_out (pd.DataFrame): time series with noise
    """

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if isinstance(variable, str):
        variable = [variable]

    # Filtering data
    if filter_ is not None:
        data = data.query(filter_)

    # Remove nans
    for var_ in variable:
        df = data[var_].dropna().to_frame()
        if not df.empty:
            increments = st.mode(np.diff(np.sort(data[var_].unique())))[0]
            df[var_] = df[var_] + np.random.rand(len(df[var_])) * increments
        else:
            raise ValueError("Input time series is empty.")

    # Removing data
    if remove:
        df = df.loc[df[variable] == filter_, variable]

    return df


def look_models(data, variable, percentiles=[1], fname="models_out", funcs="natural"):
    """Fits many of probability model to data and sorts in descending order of estimation following the sse

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of variables
        * percentiles (list): value of the cdf at the transition between different probability models
        * fname (string): name of the output file with the parameters
        * funcs (string): a parameter with the value of 'None' or 'natural' that stands for the estimation of the whole range of the probability model in scipy.stats or more frequently used in the literature

    Returns:
        * results (pd.DataFrame): the parameters of the estimation
    """

    # TODO: for mixed functions
    if not funcs:
        funcs = st._continuous_distns._distn_names
    elif funcs == "natural":
        funcs = [
            "alpha",
            "beta",
            "expon",
            "genpareto",
            "genextreme",
            "gamma",
            "gumbel_r",
            "gumbel_l",
            "triang",
            "lognorm",
            "norm",
            "rayleigh",
            "weibull_min",
            "weibull_max",
        ]

    results = dict()
    cw = np.hstack([0, np.cumsum(percentiles)])
    dfs = data.sort_values(variable, ascending=True)
    dfs["p"] = np.linspace(0, 1, len(dfs))
    # for i, j in enumerate(percentiles):
    # filt = ((dfs['p'] > cw[i]) & (dfs['p'] < j))
    # Create a table with the parameters of the best estimations and sse
    results = pd.DataFrame(
        0,
        index=np.arange(0, len(funcs)),
        columns=["models", "sse", "a", "b", "c", "d", "e", "f"],
    )
    results.index.name = "id"

    # Computeh the best estimations for the given models
    for k, name in enumerate(funcs):
        model = getattr(st, name)
        out = stf.fit_(dfs.loc[:, variable], 25, model)
        results.loc[k, "models"] = name
        results.iloc[k, 1 : len(out) + 1] = out
    results.sort_values(by="sse", inplace=True)
    results["position"] = np.arange(1, len(funcs) + 1)

    # for i, j in enumerate(percentiles):
    results.replace(0, "-", inplace=True)

    # Save to a xlsx file
    save.to_xlsx(results, fname)

    return results


def storm_series(data, cols, info):
    """Computes the storm events of the data

    Args:
        * data (pd.DataFrame): raw time series
        * cols (list): names of the concomitant required variables
        * info (dict): the storm duration, interarrival time and threshold of storm events computed following Lira-Loarca et al (2020)

    Returns:
        * df (pd.DataFrame): storm time series
    """

    # TODO: modificar la función para la separación por tormentas --> no rellene juec
    from extreme_events import extremal

    check_items = all(item in data.columns for item in cols)
    if not check_items:
        raise ValueError("Any label of cols is not in data")

    if info["time_step"] == "D":
        strTimeStep, intDur = " days", "D"
    else:
        strTimeStep, intDur = " hours", "h"
    min_duration_td = pd.Timedelta(str(info["min_duration"]) + strTimeStep)
    min_interarrival_td = pd.Timedelta(str(info["inter_time"]) + strTimeStep)

    cycles_ini, _, infoc = extremal.extreme_events(
        data,
        cols[0],
        info["threshold"],
        min_interarrival_td,
        min_duration_td,
        interpolation=info["interpolation"],
        truncate=True,
        extra_info=True,
    )

    times = pd.date_range(data.index[0], data.index[-1], freq=info["time_step"])
    df = pd.DataFrame(-99.99, index=times, columns=cols)
    df["nstorm"] = 0
    tini = times[0]
    for i, j in enumerate(cycles_ini):
        tini = j.index[0]
        if i == len(cycles_ini) - 1:
            tfin = data.index[-1]
        else:
            tfin = cycles_ini[i + 1].index[0]
        df.loc[tini:tfin, "nstorm"] = i + 1
        if not info["interpolation"]:
            df.loc[j.index[0] : j.index[-1], cols] = infoc["data_cycles"].loc[
                j.index[0] : j.index[-1], cols
            ]
        else:
            df.loc[j.index[1] : j.index[-2], cols] = infoc["data_cycles"].loc[
                j.index[1] : j.index[-2], cols
            ]

    df.index.name = "date"
    df.replace(-99.99, np.nan, inplace=True)

    if "fname" in info.keys():
        save.to_csv(df.dropna(), info["fname"])
    df.dropna(inplace=True)
    return df


def storm_properties(data, cols, info):
    """[summary]

    Args:
        data (pd.DataFrame): raw data
        cols (list): names of the variables to applied the methodology
        info (dict): A dictionary with the key parameters of the analysis
            * inter_time: minimum interarrival time between storms to be considerend independient,
            * threshold: minimum value of the storm data,
            * min_duration: minimum duration of the storm to be considered a rare event,
            * data_timestep: timestep of the input timeseries,
            * interpolation: if interpolation is applied to found the ini/end of every storm

    Returns:
        dict: the timeseries split by storms that fulfilled the requeriments
    """

    from extreme_events import extremal
    from marinetools.temporal.classification import class_storm_seasons

    if not "interpolation" in info:
        info["interpolation"] = False

    if not "class_type" in info:
        info["class_type"] = "WSSF"

    if info["time_step"] == "D":
        strTimeStep, intDur = " days", "D"
    else:
        strTimeStep, intDur = " hours", "h"
    min_duration_td = pd.Timedelta(str(info["min_duration"]) + strTimeStep)
    min_interarrival_td = pd.Timedelta(str(info["inter_time"]) + strTimeStep)

    cycles_ini, calms_ini, _ = extremal.extreme_events(
        data,
        cols[0],
        info["threshold"],
        min_interarrival_td,
        min_duration_td,
        interpolation=info["interpolation"],
        truncate=True,
        extra_info=True,
    )

    # TODO: quitar cuando Pedro haya arreglado los indices de las calmas
    if cycles_ini[-1].index[-1] == data.index[-1]:
        del cycles_ini[-1]
        del calms_ini[-1]

    for ii, _ in enumerate(cycles_ini):
        if ii == len(cycles_ini) - 1:
            calms_ini[ii] = calms_ini[ii].rename(
                {calms_ini[ii].index[0]: cycles_ini[ii].index[-1]}
            )
        else:
            calms_ini[ii] = calms_ini[ii].rename(
                {
                    calms_ini[ii].index[0]: cycles_ini[ii].index[-1],
                    calms_ini[ii].index[-1]: cycles_ini[ii + 1].index[0],
                }
            )

    # Duration of storm and interarrival time
    cycles, calms = cycles_ini, calms_ini
    dur_cycles = extremal.events_duration(list(cycles))
    dur_calms = extremal.events_duration(list(calms))

    dur_cycles = pd.DataFrame(
        dur_cycles.values, index=dur_cycles.index, columns=["dur_storm"]
    )
    dur_cycles = class_storm_seasons(dur_cycles, info["class_type"])

    dur_cycles["dur_storm"] = dur_cycles["dur_storm"] / np.timedelta64(1, intDur)
    dur_calms = dur_calms / np.timedelta64(1, intDur)

    durs_storm_calm = pd.DataFrame(
        -1,
        index=np.arange(len(dur_cycles)),
        columns=["dur_storm", "dur_calms", "season"],
    )
    durs_storm_calm["season"] = dur_cycles["season"].values
    durs_storm_calm["dur_storm"] = dur_cycles["dur_storm"].values
    durs_storm_calm["dur_calms"] = dur_calms.values

    maxs, medians, ini, end = [], [], [], []
    for event in cycles:
        maxs.append(event.max())
        medians.append(event.median())
        ini.append(event.index[0])
        end.append(event.index[-1])

    durs_storm_calm["max_value"] = maxs
    durs_storm_calm["median_value"] = medians
    durs_storm_calm["storm_ini"] = ini
    durs_storm_calm["storm_end"] = end

    if "fname" in info.keys():
        save.to_csv(durs_storm_calm, info["fname"])

    return durs_storm_calm


def normalize(data, variables, circular=False):
    """Normalizes data using the maximum distance between values

    Args:
        * data (pd.DataFrame): raw time series

    Returns:
        * datan (pd.DataFrame): normalized variable
    """

    datan = data.copy()
    for i in variables:
        if circular:
            datan[i] = np.deg2rad(data[i]) / np.pi
        else:
            datan[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

    return datan


def dependencies(df: pd.DataFrame, param: dict):
    """Computes the temporal dependency using a VAR model (Solari & van Gelder, 2011;
    Solari & Losada, 2011).

    Args:
        - df (pd.DataFrame): raw time series
        - param (dict): parameters of dt.
            - 'mvar' is the main variable,
            - 'threshold' stands for the threshold of the main variable,
            - 'vars' is a list with the name of all variables,
            - 'order' is the order of the VAR model,
            - 'events' is True or False standing for storm analysis (Lira-Loarca et al, 2020)
            or Full simulation,
            - 'not_save_error' stands for not include error time series in json file
            - 'file_name' is output file name.
        - method (string): name of the multivariate method of dependence. Defaults to "VAR".

    Returns:
        - df_dt (dict): parameters of the fitting process
    """
    logger.info(show_init_message())

    logger.info("UNI/MULTIVARIATE & TEMPORAL DEPENDENCY")
    logger.info(
        "=============================================================================="
    )

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Remove nans
    df.dropna(inplace=True)

    # Check that the input dictionary is well defined
    param["TD"] = check_dependencies_params(param["TD"])

    # Compute: (1) the univariate and temporal analysis is one variable is given,
    #          (2) the multivariate and temporal analysis is more than one is given
    logger.info(
        "Computing the parameters of the stationary {} model up to {} order.".format(
            param["TD"]["method"], param["TD"]["order"]
        )
    )
    logger.info(
        "=============================================================================="
    )

    variables_ = df.columns
    # Compute the normalize time using the maximum period
    df["n"] = (
        (df.index.dayofyear + df.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    # Transform angles into radians
    for var_ in param["TD"]["vars"]:
        if param[var_]["type"] == "circular":
            df[var_] = np.deg2rad(df[var_])

    cdf_ = pd.DataFrame(index=df.index, columns=param["TD"]["vars"])

    for var_ in param["TD"]["vars"]:
        param[var_]["order"] = np.max(param[var_]["mode"])
        param = auxiliar.str2fun(param, var_)

        variable = pd.DataFrame(df[var_].values, index=df.index, columns=["data"])
        variable["n"] = df["n"].values
        # variable[var_] = df[var_].values

        # Transformed timeserie
        if param[var_]["transform"]["make"]:
            variable["data"], _ = stf.transform(variable["data"], param[var_])
            variable["data"] -= param[var_]["transform"]["min"]

        if "scale" in param[var_]:
            variable["data"] = variable["data"] / param[var_]["scale"]

        # Compute the CDF using the estimated parameters
        cdf_[var_] = stf.cdf(variable, param[var_])

        # Remove outlayers
        if any(cdf_[var_] >= 1 - 1e-6):
            logger.info(
                "Casting {} probs of {} next to one (F({}) > 1-1e-6).".format(
                    str(np.sum(cdf_[var_] >= 1 - 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] >= 1 - 1e-6, var_] = 1 - 1e-6

        if any(cdf_[var_] <= 1e-6):
            logger.info(
                "Casting {} probs of {} next to zero (F({}) < 1e-6).".format(
                    str(np.sum(cdf_[var_] <= 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] <= 1e-6, var_] = 1e-6

        # If "events" is True, the conditional analysis over threshold following the
        # steps given in Lira-Loarca et al (2019) is applied
        if (var_ == param["TD"]["mvar"]) & param["TD"]["events"]:
            logger.info(
                "Computing conditioned probability models to the threshold of the main variable"
            )
            cdfj = cdf_[var_].copy()
            variable = pd.DataFrame(
                np.ones(len(df["n"])) * param["TD"]["threshold"],
                index=df.index,
                columns=["data"],
            )
            variable[var_] = df[var_].values
            variable["n"] = df["n"].values
            cdfu = stf.cdf(variable, param[var_])
            cdf_umbral = pd.DataFrame(cdfu)
            cdf_umbral["n"] = variable["n"]
            cdf_[var_] = (cdfj - cdfu) / (1 - cdfu)

    # Remove nans in CDF
    if any(np.sum(np.isnan(cdf_))):
        logger.info(
            "Some Nan ("
            + str(np.sum(np.sum(np.isnan(cdf_))))
            + " values) are founds before the normalization."
        )
        cdf_[np.isnan(cdf_)] = 0.5

    if param["TD"]["method"] == "VAR":
        z_ = pd.DataFrame(-1, columns=variables_, index=df.index)
        z = np.zeros(np.shape(cdf_))
        # Normalize the CDF of every variable
        for ind_, var_ in enumerate(cdf_):
            z[:, ind_] = st.norm.ppf(cdf_[var_].values)
            z_.loc[:, var_] = st.norm.ppf(cdf_[var_].values)

        # Save simulation file
        if param["TD"]["save_z"]:
            save.to_txt(
                z,
                "z_values" + ".csv",
            )

        # Fit the parameters of the AR/VAR(p) model
        df_dt = varfit(z_, param["TD"]["order"])
        for key_ in param["TD"].keys():
            df_dt[key_] = param["TD"][key_]
    else:
        logger.info("No more methods are yet available.")

    # Save to json file

    # auxiliar.mkdir("dependency")

    if not "file_name" in param["TD"].keys():
        auxiliar.mkdir("dependency")
        filename = "dependency/"
        for var_ in param["TD"]["vars"]:
            filename += var_ + "_"
        filename += str(param["TD"]["order"]) + "_"
        filename += param["TD"]["method"]
    else:
        filename = param["TD"]["file_name"]

    param["TD"]["file_name"] = filename

    if param["TD"]["not_save_error"]:
        df_dt.pop("y", None)
        df_dt.pop("y*", None)

    save.to_json(df_dt, param["TD"]["file_name"], True)

    # Clean memory usage
    del cdf_, param

    return df_dt


def check_dependencies_params(param: dict):
    """Checks the input parameters and includes some required arguments for the computation of multivariate dependencies

    Args:
        * param (dict): the initial guess parameters of the probability models

    Returns:
        * param (dict): checked and updated parameters
    """

    logger.info("USER OPTIONS:")
    logger.info(
        "==============================================================================\n"
    )
    k = 1

    if not "method" in param.keys():
        param["method"] = "VAR"
        logger.info(str(k) + " - VAR method used")
        k += 1

    if not "not_save_error" in param.keys():
        param["not_save_error"] = True

    if not "events" in param.keys():
        param["events"] = False

    if not "mvar" in param.keys():
        param["mvar"] = None

    if not "save_z" in param.keys():
        param["save_z"] = False

    logger.info(
        "==============================================================================\n"
    )
    global text_warning
    text_warning = True

    return param


def varfit(data: np.ndarray, order: int):
    """Computes the coefficientes of the VAR(p) model and chooses the model with lowest BIC.

    Args:
        * data (np.ndarray): normalize data with its probability model
        * order (int): maximum order (p) of the VAR model

    Returns:
        * par_dt (dict): parameter of the temporal dependency using VAR model
    """

    from statsmodels.tsa.ar_model import AutoReg as AR
    from statsmodels.tsa.vector_ar.var_model import VAR

    # Create the list of output parameters
    data_ = data.values.T
    [dim, t] = np.shape(data_)
    t = t - order
    bic, r2adj = np.zeros(order), []
    if dim == 1:
        model = AR(data)
    else:
        model = VAR(data)

    par_dt = [list() for i in range(order)]
    for p in range(1, order + 1):
        # Create the matrix of input data for p-order
        y = data_[:, order:]
        z0 = np.zeros([p * dim, t])
        for i in range(1, p + 1):
            z0[(i - 1) * dim : i * dim, :] = data_[:, order - i : -i]
        z = np.vstack((np.ones(t), z0))
        # Estimated the parameters using the ordinary least squared error analysis
        par_dt[p - 1], bic[p - 1], r2a = varfit_OLS(y, z)
        r2adj.append(r2a)
        res = model.fit(p)
        # print(res.summary())

        # Computed using statmodels
        par_dt[p - 1]["B"] = res.params.values.T
        par_dt[p - 1]["U"] = y - np.dot(res.params.values.T, z)
        # Estimate de covariance matrix
        par_dt[p - 1]["Q"] = np.cov(par_dt[p - 1]["U"])
        bic[p - 1] = res.bic
        # lag_order = res.k_ar

    # Select the minimum BIC and return the parameter associated to it
    id_ = np.argmin(bic)
    par_dt = par_dt[id_]
    par_dt["id"] = int(id_) + 1
    par_dt["bic"] = [float(bicValue) for bicValue in bic]
    par_dt["R2adj"] = r2adj[par_dt["id"]]
    logger.info(
        "Minimum BIC ("
        + str(par_dt["bic"][par_dt["id"]])
        + ") obtained for p-order "
        + str(par_dt["id"] + 1)  # Python starts at zero
        + " and R2adj: "
        + str(par_dt["R2adj"])
    )
    logger.info(
        "=============================================================================="
    )

    if id_ + 1 == order:
        logger.info("The lower BIC is in the higher order model. Increase the p-order.")

    return par_dt


def varfit_OLS(y, z):
    """Estimates the parameters of VAR using the RMSE described in Lutkepohl (ecs. 3.2.1 and 3.2.10)

    Args:
        * y: X Matrix in Lutkepohl
        * z: Z Matrix in Lutkepohl

    Returns:
        * df (dict): matrices B, Q, y U
        * bic (float): Bayesian Information Criteria
        * R2adj (float): correlation factor
    """

    df = dict()
    m1, m2 = np.dot(y, z.T), np.dot(z, z.T)

    # Estimate the parameters
    df["B"] = np.dot(m1, np.linalg.inv(m2))

    nel, df["dim"] = np.shape(df["B"].T)
    df["U"] = y - np.dot(df["B"], z)
    # Estimate de covariance matrix
    df["Q"] = np.cov(df["U"])
    df["y"] = y
    if df["dim"] == 1:
        error_ = np.random.normal(np.zeros(df["dim"]), df["Q"][0], z.shape[1]).T
    else:
        error_ = np.random.multivariate_normal(
            np.zeros(df["dim"]), df["Q"], z.shape[1]
        ).T
    df["y*"] = np.dot(df["B"], z) + error_

    # Estimate R2 and R2-adjusted parameters
    R2 = np.sum((df["y*"] - np.mean(y)) ** 2, axis=1) / np.sum(
        (y - np.mean(y)) ** 2, axis=1
    )
    R2adj = 1 - (1 - R2) * (len(z.T) - 1) / (len(z.T) - nel - 1)

    # rmse = np.sqrt(np.sum((st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))) ** 2, axis=1)/y.shape[1])
    # mae = np.sum(np.abs(st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))), axis=1)/y.shape[1]
    # logger.info(rmse, mae)

    # Compute the LLF
    multivariatePdf = st.multivariate_normal.pdf(
        df["U"].T, mean=np.zeros(df["dim"]), cov=df["Q"]
    )
    mask = multivariatePdf > 0

    global text_warning
    if len(multivariatePdf) != len(multivariatePdf[mask]):
        if text_warning:
            logger.info(
                "Casting {} zero-values of the multivariate pdf. Removed.".format(
                    str(np.sum(~mask))
                )
            )
            text_warning = False

        llf = np.sum(np.log(multivariatePdf[mask]))
    else:
        llf = np.sum(np.log(multivariatePdf))

    # aic = df['dim']*np.log(np.sum(np.abs(y - np.dot(df['B'], z)))) + 2*nel
    # Compute the BIC
    bic = -2 * llf + np.log(np.size(y)) * np.size(np.hstack((df["B"], df["Q"])))

    return df, bic, R2adj.tolist()


def ensemble_dt(models: dict, percentiles="equally"):
    """Compute the ensemble of multivariate and temporal dependency parameters

    Args:
        * models (dict): models where parameters of temporal dependencies are saved
        * percentiles (string or list): "equally" is equally probability is given for RCMs
            and a list with percentiles of every RCMs if not

    Returns:
        [type]: [description]
    """
    # Initialize matrices
    B, Q = [], []
    # Read the parameter of every ensemble model
    for model_ in models.keys:
        df_dt = read.rjson(models[model_], "td")
        B.append(df_dt["B"])
        Q.append(df_dt["Q"])

    nmodels = len(B)
    norders = np.max([np.shape(Bm) for Bm in B], axis=0)

    Bs = np.zeros([norders[0], norders[1], nmodels])

    # Compute the ensemble using percentiles
    for i in range(nmodels):
        if percentiles == "equally":
            Bs[:, : np.shape(B[i])[1], i] = B[i]
        else:
            Bs[:, : np.shape(B[i])[1], i] = B[i] * percentiles[i]

    # Compute the ensemble using percentiles
    if percentiles == "equally":
        B, Q = np.mean(Bs, axis=2), np.mean(Q, axis=0)
    else:
        B, Q = np.sum(Bs, axis=2), np.sum(Q, axis=0)

    # Create a dictionary with parameters of the ensemble
    df_dt_ensemble = dict()
    df_dt_ensemble["B"], df_dt_ensemble["Q"], df_dt_ensemble["id"] = (
        B,
        Q,
        int((norders[1] - 1) / norders[0]),
    )  # ord_

    # Create the fit directory and save the parameters to a json file
    auxiliar.mkdir("fit")
    save.to_json(df_dt_ensemble, "fit/ensemble_df_dt", True)
    return B, Q, int((norders[1] - 1) / norders[0])


def iso_rmse(
    reference: pd.DataFrame,
    variable: str,
    param: dict = None,
    data: pd.DataFrame = None,
    daysWindowsLength: int = 14,
):
    """Compute the rmse of the iso-probability lines of the non-stationary cdf

    Args:
        reference (pd.DataFrame): _description_
        variable (str): _description_
        param (dict, optional): _description_. Defaults to None.
        data (pd.DataFrama, optional):
        daysWindowsLength (int, optional): _description_. Defaults to 14.

    Returns:
        _type_: _description_
    """

    if param is not None:
        if param["basis_period"] is not None:
            T = np.max(param["basis_period"])
        emp_non_st = False
    else:
        emp_non_st = True
        T = 1
        data["n"] = np.fmod(
            (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
            .total_seconds()
            .values
            / (T * 365.25 * 24 * 3600),
            1,
        )

    reference["n"] = np.fmod(
        (reference.index - datetime.datetime(reference.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    xp, pemp = auxiliar.nonstationary_ecdf(
        reference,
        variable,
        wlen=daysWindowsLength / (365.25 * T),
    )

    # A empirical model
    if emp_non_st:
        data_check, _ = auxiliar.nonstationary_ecdf(
            data,
            variable,
            wlen=daysWindowsLength / (365.25 * T),
        )
    else:
        # A theoretical model
        # ----------------------------------------------------------------------------------
        for j, i in enumerate(pemp):
            if not emp_non_st:
                if param["transform"]["plot"]:
                    xp[i], _ = stf.transform(xp[[i]], param)
                    xp[i] -= param["transform"]["min"]
                    if "scale" in param:
                        xp[i] = xp[i] / param["scale"]

                param = auxiliar.str2fun(param, None)
        data_check = pd.DataFrame(0, index=n, columns=pemp)

        for i, j in enumerate(pemp):
            df = pd.DataFrame(np.ones(dt) * pemp[i], index=n, columns=["prob"])
            df["n"] = n
            if (param["non_stat_analysis"] == True) | (param["no_fun"] > 1):
                res = stf.ppf(df, param)
            else:
                res = pd.DataFrame(
                    param["fun"][0].ppf(df["prob"], *param["par"]),
                    index=df.index,
                    columns=[variable],
                )

            # Transformed timeserie
            if (not param["transform"]["plot"]) & param["transform"]["make"]:
                if "scale" in param:
                    res[param["var"]] = res[param["var"]] * param["scale"]

                res[param["var"]] = res[param["var"]] + param["transform"]["min"]
                res[param["var"]] = stf.inverse_transform(res[[param["var"]]], param)
            elif ("scale" in param) & (not param["transform"]["plot"]):
                res[param["var"]] = res[param["var"]] * param["scale"]

            data_check[j] = res[param["var"]]

    # ----------------------------------------------------------------------------------
    rmse = pd.DataFrame(-1, index=pemp, columns=["rmse"])
    for j in pemp:
        rmse.loc[j] = auxiliar.rmse(xp[j], data_check[j])

    return rmse


def confidence_bands(rmse, n, confidence_level):
    """_summary_

    Args:
        rmse (_type_): número de puntos de datos
        n (_type_): RMSE de tu modelo
        confidence_level (_type_): nivel de confianza
    """

    # Paso 2: Calcular el error estándar de los residuos
    ser = rmse / np.sqrt(n)

    # Paso 4: Calcular el valor crítico de la distribución t de Student
    degrees_of_freedom = n - 1
    alpha = 1 - confidence_level
    t_critical = st.t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Paso 5: Calcular el margen de error
    margin_of_error = t_critical * ser

    return margin_of_error


def generate_outputfilename(parameters):
    """_summary_

    Args:
        parameters (_type_): _description_
    """

    filename = parameters["var"] + "_" + str(parameters["fun"][0])
    for i in range(1, parameters["no_fun"]):
        filename += "_" + str(parameters["fun"][i])
    filename += "_genpareto" * parameters["reduction"]

    # for i in parameters["ws_ps"]:
    # filename += "_" + str(i)

    filename += "_st_" * (not parameters["non_stat_analysis"])
    filename += "_nonst" * parameters["non_stat_analysis"]

    filename += "_" + str(parameters["basis_period"][0])

    filename += "_" + parameters["basis_function"]["method"]
    if "no_terms" in parameters["basis_function"].keys():
        filename += "_" + str(parameters["basis_function"]["no_terms"])
    else:
        filename += "_" + str(parameters["basis_function"]["degree"])
    filename += "_" + parameters["optimization"]["method"]

    parameters["file_name"] = filename
    return
