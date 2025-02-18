import datetime
import os
import re
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sp
import xarray as xr
from loguru import logger

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

global g, nu, kappa, gamma
g = 9.81
nu = 1.0e-6
kappa = 0.4
gamma = 1000.0


def _clock(initial, it, telaps, dConfig):
    """Compute the advances in time from the beggining of the run

    Args:
        initial (datetime): initial time of the run
        it (int): number of iterations
        telaps (float): elapsed time
        dConfig (dict): parameters specifications

    Returns:
        print in the shell the advances
    """
    elapsed_ = datetime.datetime.now() - initial

    # For debugging, the shell will not clean
    if not dConfig["bLog"]:
        os.system("cls" if os.name == "nt" else "printf '\033c'")

    total_seconds = elapsed_.total_seconds()
    hours = str(int(total_seconds // 3600)).zfill(2)
    minutes = str(int((total_seconds % 3600) // 60)).zfill(2)
    seconds = str(int(total_seconds % 60)).zfill(2)
    return logger.info(
        f"{hours}:{minutes}:{seconds} - Time steps: "
        + "{0:4d}".format(it)
        + " - Elapsed time: "
        + "{0:9.2f}".format(telaps)
        + " - "
        + "{0:5.2f}".format(np.round(telaps / dConfig["fFinalTime"] * 100, decimals=2))
        + " % completed"
    )


def _configure_time(dConfig):
    """Convert input time configurations into time vector with its units

    Args:
        dConfig (dict): parameters specifications

    Returns:
        dict: dictionary updated
    """
    ts, units = dConfig["fTimeStep"].split(" ")
    ts = float(ts)
    if units.startswith("second"):
        dConfig["fTimeStep"] = ts
        dConfig["time_multiplier_factor"] = 1
    elif units.startswith("hour"):
        dConfig["fTimeStep"] = ts * 3600
        dConfig["fFinalTime"] = dConfig["fFinalTime"] * 3600
        dConfig["time_multiplier_factor"] = 3600
    elif units.startswith("day"):
        dConfig["fTimeStep"] = ts * 3600 * 24
        dConfig["fFinalTime"] = dConfig["fFinalTime"] * 3600 * 24
        dConfig["time_multiplier_factor"] = 3600 * 24

    dConfig["iTime"] = np.arange(int(dConfig["fFinalTime"] / dConfig["fTimeStep"]) + 1)
    return dConfig


def createConfigFile():
    dConfig = ConfigParser()

    dConfig["DEFAULT"] = {
        "iInitialEstuaryCondition": 1,  # Identificador de condición inicial: 1=de archivo, 2= Asignado eta y Q=0
        "iInitialBoundaryCondition": 2,  # Frontera inicial: 1= reflejante o 2=abierta
        "bMurilloCondition": 1,  # Algoritmo condicion dx-n
        "bSurfaceGradientMethod": 0,  # Surface Gradient Method
        "bMcComarckLimiterFlux": 0,  # Con esquema McCormack (sin limitador), 1: Con esquema TVD-McCormack (con limitador)
        "bSourceTermBalance": 1,  # Balance de terminos fuente'
        "bDryBed": 0,  # Algoritmo lecho seco
        "fCourantNo": 0.9,  # Número de fCourantNo
        "dtst": 3600,  # Espaciamiento de los datos de salida, en segundos
        "iFormulaLimiterFlux": 4,  # Fórmula de limitador de flujo fi 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
        "iPsiFormula": 1,  # Fórmula para Psi: 1 = García-Navarro, 2 = Tseng
    }
    #     "geometryFilename"
    #    "hydroFilename"

    if dConfig["iPsiFormula"] == 1:
        dConfig["delta"] = 0.2

    with open("config.ini", "w") as configfile:
        dConfig.write(configfile)
    return


def _initial_condition(dbt, db, df, aux, dConfig):
    """Update databases and auxiliary variables with the initial conditions.
    Initial conditions are set into the dictionary (dConfig). Options are:
        iInitialEstuaryCondition (int):

    Args:
        dbt (xr): _description_
        db (xr): the information about the geometry of the sections in arrays
        df (pd.DataFrame):
        aux (dict):
        dConfig (dict): parameters specifications

    Returns:
        _type_: the updated databases and auxiliary dictionary
    """

    if dConfig["iInitialEstuaryCondition"] == 0:
        # Check that the initial water flux is given
        if not "fQci" in dConfig.keys():
            raise ValueError(
                "Initial estuarine conditions was set to 0. Parameter "
                + "'fQci' should be given with the inputs."
            )

        logger.info(
            "Computing initial condition for Q = " + str(dConfig["fQci"]) + " m3/s"
        )
        dbt["Q"][:, 0] = dConfig["fQci"]
        for i in range(dConfig["nx"]):
            facman = (
                dbt["Q"][i, 0] * df["nmann"][i] / np.sqrt(pd.to_numeric(df["S0"][i]))
            )
            facsec = db["A"][i, :] * db["Rh"][i, :] ** (2.0 / 3.0)
            araux = db["A"][i, :]

            dbt["A"][i, 0] = np.interp(facman, facsec, araux)

    elif dConfig["iInitialEstuaryCondition"] == 1:
        # Check that the initial water level is given
        if not "fEtaci" in dConfig.keys():
            raise ValueError(
                "Initial estuarine conditions was set to 1. Parameter "
                + "'fEtaci' should be given with the inputs."
            )

        logger.info(
            "Computing initial condition for Q = 0 and eta = "
            + str(dConfig["fEtaci"])
            + " m"
        )

        for i in range(dConfig["nx"]):
            dbt["A"][i, 0] = np.interp(
                dConfig["fEtaci"] - df["z"][i], db["eta"][i, :], db["A"][i, :]
            )
            dbt["Rh"][i, 0] = np.interp(dbt["A"][i, 0], db["A"][i, :], db["Rh"][i, :])

        # Calculate Q given the Area
        dbt["Q"][:, 0] = (
            dbt["A"][:, 0]
            * np.sqrt(df["S0"].values)
            * dbt["Rh"][:, 0] ** (2.0 / 3.0)
            / df["nmann"]
        )

    # In some estuaries, both options should be given at the same time. To consider this
    # option, run the model with some previous time steps
    aux["U"][0, :] = dbt["A"][:, 0]
    aux["U"][1, :] = dbt["Q"][:, 0]
    aux["F"][0, :] = dbt["Q"][:, 0]

    return aux


def _tidal_level(db, aux, df, dConfig, fTelaps):
    """Obtain the total water level at the seaward due to tidal elevations

    Args:
        db (_type_): _description_
        aux (_type_): _description_
        df (_type_): _description_
        dConfig ():
        iTime (_type_): _description_

    Returns:
        aux (dict): updated variable with the seaward boundary condition
    """

    # Compute the tidal level at iTime
    aux["eta_tide"] = np.interp(
        fTelaps,
        aux["tidal_level"].index,
        aux["tidal_level"]["level"],
    )

    # Compute the total water level at seaward location
    aux["tidal_area_seaward"] = np.interp(
        -df.loc[dConfig["nx"] - 1, "z"] + aux["eta_tide"],
        db["eta"][-1, :],
        db["A"][-1, :],
    )

    # Compute the total water level next to the seaward location
    aux["tidal_area_next_seaward"] = np.interp(
        -df.loc[dConfig["nx"] - 2, "z"] + aux["eta_tide"],
        db["eta"][-2, :],
        db["A"][-2, :],
    )
    return aux


def _read_oldfiles(filename, version_="v48"):
    """_summary_

    Args:
        filename (_type_): _description_
        version_ (str, optional): _description_. Defaults to "v48".

    Returns:
        _type_: _description_
    """
    dConfig = {}
    with open(filename, "r") as file:
        data = file.readlines()

    dConfig["sGeometryFilename"] = re.split("\s+", data[2][:-1])[0]
    dConfig["sHydroFilename"] = re.split("\s+", data[3][:-1])[0]
    dConfig["iInitialEstuaryCondition"] = float(
        re.split("\s+", data[4][:-1])[0]
    )  # Identificador de condición inicial: 1=de archivo, 2=Manning, 3= Asignado eta y Q=0

    if dConfig["iInitialEstuaryCondition"] == 1:  # lee de archivo
        dConfig["ciFilename"] = re.split("\s+", data[5][:-1])[0]
    elif dConfig["iInitialEstuaryCondition"] == 2:  # Q inicial
        dConfig["fQci"] = float(re.split("\s+", data[6][:-1])[0])
        dConfig["iInitialEstuaryCondition"] = 0
    else:
        dConfig["fEtaci"] = float(re.split("\s+", data[7][:-1])[0])
        dConfig["iInitialEstuaryCondition"] = 1

    dConfig["ioLocations"] = re.split("\s+", data[8][:-1])[
        0
    ]  # Archivo con coordenadas X de los puntos para guardar series temporales
    if version_ == "v44":
        k = -1
        dConfig["fCourantNo"] = 0.3
    if version_ == "v48":
        k = 0
        dConfig["fCourantNo"] = float(re.split("\s+", data[9][:-1])[0])

    dConfig["outputFilename"] = re.split("\s+", data[11 + k][:-1])[0]
    dConfig["tesp"] = float(re.split("\s+", data[12 + k][:-1])[0])

    dConfig["bSedimentTransport"] = float(re.split("\s+", data[20 + k][:-1])[0])
    dConfig["sedimentpropertiesFilename"] = re.split("\s+", data[21 + k][:-1])[0]

    # ------------------------- Opciones del programa: -----------------------
    dConfig["bMcComarckLimiterFlux"] = float(
        re.split("\s+", data[23 + k][:-1])[0]
    )  # Sin limitador de flujo (MacCormack)
    dConfig["bMcComarckLimiterFlux"] = 1  # TODO: probando, quitar
    dConfig["bSurfaceGradientMethod"] = float(
        re.split("\s+", data[24 + k][:-1])[0]
    )  # Surface Gradient Method
    dConfig["bSourceTermBalance"] = float(
        re.split("\s+", data[25 + k][:-1])[0]
    )  # Balance de terminos fuente
    dConfig["idbeta"] = float(
        re.split("\s+", data[26 + k][:-1])[0]
    )  # uso de betas, 0=no, 1=s�
    dConfig["bDryBed"] = float(
        re.split("\s+", data[28 + k][:-1])[0]
    )  # Algoritmo lecho seco
    dConfig["bMurilloCondition"] = float(
        re.split("\s+", data[29 + k][:-1])[0]
    )  # verifica n de Manning para cumplir condici�n de Murillo para dx
    # ------------------------- Opciones de frontera final: ------------------
    dConfig["iFinalBoundaryCondition"] = float(
        re.split("\s+", data[31 + k][:-1])[0]
    )  # Ident. Q en frontera final: 1) Abierta, 2) Fijo, 3) Marea
    # dConfig["iFinalBoundaryCondition"] = 3  # TODO: probando la marea aguas abajo ,quitar
    if dConfig["iFinalBoundaryCondition"] == 2:
        dConfig["qfijo"] = float(re.split("\s+", data[32 + k][:-1])[0])

    dConfig["tideFilename"] = re.split("\s+", data[37 + k][:-1])[0]

    dConfig["iInitialBoundaryCondition"] = float(
        re.split("\s+", data[44 + k][:-1])[0]
    )  # Tipo de frontera aguas arriba: 1=reflejante, 2=abierta
    dConfig["dtmin"] = 1e10

    dConfig["iFormulaLimiterFlux"] = 4  # Fórmula de limitador de flujo fi
    # 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
    dConfig["iPsiFormula"] = 2  # Fórmula para Psi: 1 = Garc�a-Navarro, 2 = Tseng

    if dConfig["iPsiFormula"] == 1:
        dConfig["delta"] = 0.2  # Para calcular Psi con f�rmula Garc�a-Navarro,
        # "delta = small positive number betwen 0.1 and 0.3"
    return dConfig


def _calculate_Is(db, dConfig):
    """Compute the source terms I1 (difference between pressure thrusts applied over the
    frontiers x1 and x2) and I2 (pressure forces due the channel width variation)

    Args:
        db (xr): the information about the geometry of the sections in arrays
        dConfig (dict): parameters specifications
    """
    # -----------------------------------------------------------------------------------
    # Calculate I1
    # -----------------------------------------------------------------------------------
    db["I1"][:, 1:] = np.cumsum(
        (db["sigma"][:, 1:] + db["sigma"][:, :-1])
        / 2
        * db["eta"][:, 1:]
        * (db["eta"][:, 1:] - db["eta"][:, :-1]),
        axis=1,
    )

    # -----------------------------------------------------------------------------------
    # Calculate I2
    # -----------------------------------------------------------------------------------
    # Central finite-difference
    dhdx, dI1dx = np.zeros([db.sizes["x"], db.sizes["z"]]), np.zeros(
        [db.sizes["x"], db.sizes["z"]]
    )
    dhdx[1:-1, :] = (db["eta"][2:, :] - db["eta"][:-2, :]) / (2 * dConfig["dx"])
    dI1dx[1:-1, :] = (db["I1"][2:, :] - db["I1"][:-2, :]) / (2 * dConfig["dx"])
    # Upward finite-difference
    dhdx[0, :] = (db["eta"][1, :] - db["eta"][0, :]) / dConfig["dx"]
    dI1dx[0, :] = (db["I1"][1, :] - db["I1"][0, :]) / dConfig["dx"]

    # Downward finite-difference
    dhdx[-1, :] = (db["eta"][-1, :] - db["eta"][-2, :]) / dConfig["dx"]
    dI1dx[-1, :] = (db["I1"][-1, :] - db["I1"][-2, :]) / dConfig["dx"]

    db["I2"][:] = dI1dx - db["A"][:].values * dhdx
    return


def _hydraulic_parameters(dbt, db, iTime, predictor=True):
    """Compute the hydraulic sections as function of A

    Args:
        dbt (xr): the database with the variables in space and time
        db (xr): information about the geometry of the sections in arrays
        iTime (int): _description_
        pred (bool, optional): for predictor (True) or corrector (False). Defaults to True.

    Returns:
        The updated databases
    """

    vars = ["Rh", "B", "eta", "beta", "I1", "I2", "xl", "xr"]
    A = np.tile(dbt["A"][:, iTime].values, [db.sizes["z"], 1]).T

    # Compute the index where the given area is found
    # TODO: Check the following incorrect formula
    indexes_db = np.argmin(np.abs(A - db["A"].values), axis=1, keepdims=True)

    mask = indexes_db + 1 >= db.sizes["z"]
    # Check that computed area is below the maximum area given in the inputs
    if sum(mask) > 0:
        str_ = ""
        for val in dbt["A"][mask.T[0], iTime].values:
            str_ = str(val) + ", "
        str_ = str_[:-2]
        str_ += " m2"
        raise ValueError("Flooding area outside the range given in the inputs: " + str_)

    # Obtain the proportional factor between areas
    facpr = (
        dbt["A"][:, iTime]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    ) / (
        np.take_along_axis(db["A"][:].values, indexes_db + 1, axis=1)[:, 0]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    )

    # Calculate the mean value of variables
    for var in vars:
        if predictor:
            varname = var
        else:
            varname = var + "p"

        dbt[varname][:, iTime] = (
            np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            + (
                np.take_along_axis(db[var][:].values, indexes_db + 1, axis=1)[:, 0]
                - np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            )
            * facpr
        )

    return


def _read_cross_sectional_geometry(dConfig):
    """Read the cross-sectional geometry of the estuary.
    This information will not change during the script.

    Args:
        dConfig (dict): parameters specifications

    Returns:
        db (xr): The information about the geometry of the sections in arrays
    """
    df = pd.read_csv("cross_sections_" + dConfig["sGeometryFilename"])

    dConfig["nx"] = len(df["x"].unique())
    x = np.reshape(df["x"].values, [dConfig["nx"], len(df["eta"].unique())])
    z = np.reshape(df["z"].values, np.shape(x))

    # Output Variables
    vars_ = ["eta", "A", "P", "B", "Rh", "sigma", "xl", "xr", "beta", "I1", "I2"]

    # Creating an empty xr
    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "z"], zeros.copy())

    db = xr.Dataset(
        dict_, coords={"xlocal": (["x", "z"], x), "zlocal": (["x", "z"], z)}
    )

    # Filling the xr
    db["eta"][:, :] = np.reshape(df["eta"].values, np.shape(x))
    db["A"][:, :] = np.reshape(df["A"].values, np.shape(x))
    db["P"][:, :] = np.reshape(df["P"].values, np.shape(x))
    db["B"][:, :] = np.reshape(df["B"].values, np.shape(x))
    db["Rh"][:, :] = np.reshape(df["Rh"].values, np.shape(x))
    db["sigma"][:, :] = np.reshape(df["sigma"].values, np.shape(x))
    db["xl"][:, :] = np.reshape(df["xl"].values, np.shape(x))
    db["xr"][:, :] = np.reshape(df["xr"].values, np.shape(x))
    db["beta"][:, :] = np.reshape(df["beta"].values, np.shape(x))

    return db


def _read_geometry_oldfiles(dConfig):
    """Read the geometrical properties of the original Guadalfortran files

    Args:
        dConfig (dict): parameters specifications

    Returns:
        df (pd.DataFrame): _description_
        db (xr): the information about the geometry of the sections in arrays
    """
    with open(dConfig["geometryFilename"], "r") as file:
        data = file.readlines()

    nx, nz = [int(i) for i in re.split("\s+", data[5][:-1])]
    dConfig["nx"], dConfig["nz"] = nx, nz

    df = pd.DataFrame(
        -1.0,
        index=np.arange(nx),
        columns=["x", "z", "nmann", "xutm", "yutm", "AngMd", "AngMi"],
        dtype=np.float64,
    )

    eta, A, P, B, Rh, sigma, I1, I2, xleft, xright, beta = (
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
        np.zeros([nx, nz], dtype=np.float32),
    )
    lines = len(data)
    line = 5
    ix = -1
    is_ = nz
    # Reading the
    while line < lines:  # lines - 1 in Adra
        if is_ == nz:
            line += 1
            ix += 1
            aux_ = [
                float(i) for i in re.split("\s+", data[line][1:-1])
            ]  # :-1 en Adra, v48
            df.loc[ix, :] = aux_
            is_ = 0
        else:
            (
                eta[ix, is_],
                A[ix, is_],
                P[ix, is_],
                B[ix, is_],
                Rh[ix, is_],
                sigma[ix, is_],
                I1[ix, is_],
                I2[ix, is_],
                xleft[ix, is_],
                xright[ix, is_],
                beta[ix, is_],
            ) = [
                float(i) for i in re.split("\s+", data[line][:-3])
            ]  # -1 en Adra, v48
            is_ += 1
        line += 1

    x, z = np.meshgrid(eta[0, :], df["x"].values)
    vars_ = ["eta", "A", "P", "B", "Rh", "sigma", "I1", "I2", "xl", "xr", "beta"]
    if isinstance(vars_, str):
        vars_ = [vars_]

    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "z"], zeros.copy())
    db = xr.Dataset(
        dict_, coords={"xlocal": (["x", "z"], x), "zlocal": (["x", "z"], z)}
    )

    db["eta"][:, :] = eta
    db["A"][:, :] = A
    db["P"][:, :] = P
    db["B"][:, :] = B
    db["Rh"][:, :] = Rh
    db["sigma"][:, :] = sigma
    db["I1"][:, :] = I1
    db["I2"][:, :] = I2
    db["xl"][:, :] = xleft
    db["xr"][:, :] = xright
    db["beta"][:, :] = beta
    del eta, A, P, B, Rh, sigma, I1, I2, xleft, xright, beta
    return df, db


def _initialize_netcdf(df, dConfig):
    """Create an empty xr with the variables of the SV model in space and time

    Args:
        df (pd.DataFrame): geometrical properties per section
        dConfig (dict): parameters specifications

    Returns:
        dbt (xr): The output variables in space and time
    """
    t = dConfig["iTime"] * dConfig["fTimeStep"]
    t, x = np.meshgrid(t, df["x"])

    # Output variables
    vars_ = [
        "A",
        "Ap",
        "Ac",
        "Q",
        "q",
        "Qp",
        "Qc",
        "Rh",
        "Rhp",
        "I1",
        "I1p",
        "B",
        "Bp",
        "eta",
        "etap",
        "beta",
        "betap",
        "I2",
        "I2p",
        "U",
        "c",
        "S",
        "Qb",
        "Qs",
        "Qt",
        "rho",
        "rhop",
        "xl",
        "xr",
        "xlp",
        "xrp",
    ]

    # Creating an empty xr
    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "t"], zeros.copy())
    dbt = xr.Dataset(
        dict_,
        coords={"xlocal": (["x", "t"], x), "time": (["x", "t"], t)},
    )

    # Creating the attributes of the main variables
    dbt.A.attrs = {
        "description": "Cross-sectional flooding area",
        "longname": "area",
        "units": "m2",
    }
    dbt.Q.attrs = {
        "description": "Cross-sectional averaged water discharge",
        "longname": "water discharge",
        "units": "m3/s",
    }
    dbt.q.attrs = {
        "description": "Fluvial contributions along the estuary",
        "longname": "water flux contributions",
        "units": "m3/s",
    }
    dbt.Rh.attrs = {
        "description": "Hydraulic radius of the flooding area",
        "longname": "hydraulic radius",
        "units": "m",
    }
    dbt.I1.attrs = {
        "description": "Term I1 of the balance",
        "longname": "I1",
        "units": "",
    }
    dbt.B.attrs = {
        "description": "Maximum flooding width",
        "longname": "flooding width",
        "units": "m",
    }
    dbt.eta.attrs = {
        "description": "Free surface elevation",
        "longname": "flooding water depth",
        "units": "m",
    }
    dbt.beta.attrs = {
        "description": "Beta coefficient",
        "longname": "beta",
        "units": "",
    }
    dbt.I2.attrs = {
        "description": "Term I2 of the balance",
        "longname": "I2",
        "units": "",
    }
    dbt.U.attrs = {
        "description": "Cross-sectional averaged water velocity",
        "longname": "mean water velocity",
        "units": "m/s",
    }
    dbt.S.attrs = {
        "description": "Cross-sectional averaged salinity",
        "longname": "salinity",
        "units": "psu",
    }
    dbt.Qb.attrs = {
        "description": "Bedload sediment transport",
        "longname": "bedload sediment transport",
        "units": "m3/s",
    }
    dbt.Qs.attrs = {
        "description": "Suspended sediment transport",
        "longname": "suspended sediment transport",
        "units": "m3/s",
    }
    dbt.Qt.attrs = {
        "description": "Total sediment transport",
        "longname": "total sediment transport",
        "units": "m3/s",
    }
    dbt.rho.attrs = {
        "description": "Fluid density. Sum of salinity and bedload and suspended sediment",
        "longname": "fluid density",
        "units": "kg/m3",
    }
    dbt.xl.attrs = {
        "description": "Distance from the thalweg to the left riverbank",
        "longname": "x-left",
        "units": "m",
    }
    dbt.xr.attrs = {
        "description": "Distance from the thalweg to the right riverbank",
        "longname": "x-right",
        "units": "m",
    }

    return dbt


def _read_hydro(dConfig):
    """Read hydrographs of water supply to the estuary

    Args:
        dConfig (dict): parameters specifications

    Returns:
        pd.DataFrame: Timeseries of water supply at every estuarine section
    """
    hydro = pd.read_csv(dConfig["sHydroFilename"], sep=" ", index_col=[0], header=0)
    hydro.columns = [int(col) for col in hydro.columns]
    return hydro


def _fluvial_contribution(dbt, hydro, dConfig):
    """_summary_

    Args:
       dbt (xr): the database with the variables in space and time
        hydro (_type_): _description_
        dConfig (dict): parameters specifications
    """
    for node in hydro.columns:
        dbt["q"][node, :] = (
            np.interp(
                dConfig["iTime"] * dConfig["time_multiplier_factor"],
                hydro.index.values,
                hydro[node],
            )
            / dConfig["dx"]
        )

    return


def _initialize_auxiliary(db, df, dConfig):
    """Initialize auxiliary variables

    Args:
        db (xr): the information about the geometry of the sections in arrays
        df (pd.DataFrame): geometrical properties per section
        dConfig (dict): parameters specifications

    Returns:
        _type_: _description_
    """

    dConfig["dx"] = (df.x.values[-1] - df.x.values[0]) / (
        dConfig["nx"] - 1
    )  # distance between sections

    if dConfig["bSurfaceGradientMethod"] == 0:
        df["elev"] = np.zeros(dConfig["nx"])

    aux = dict()
    if dConfig["bSourceTermBalance"] == 1:
        df["zmedp"] = df["z"].diff(periods=-1) / dConfig["dx"]
        df.loc[dConfig["nx"] - 1, "zmedp"] = df.loc[dConfig["nx"] - 2, "zmedp"]

        df["zmedc"] = -df["z"].diff() / dConfig["dx"]
        df.loc[0, "zmedc"] = df.loc[1, "zmedc"]

        aux["Qmed"], aux["Amed"], aux["Rmed"] = (
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
        )

    aux["U"], aux["F"], aux["Gv"] = (
        np.zeros([2, dConfig["nx"]]),
        np.zeros([2, dConfig["nx"]]),
        np.zeros([2, dConfig["nx"]]),
    )

    aux["Up"], aux["Fp"] = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])
    aux["Uc"], aux["Gvp"] = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])
    aux["Un"], aux["D"] = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])

    # Compute the I1 and I2 from geometrical properties
    _calculate_Is(db, dConfig)

    # Bed and friction slopes using the finite differences (centered, foreward and backward)
    df["S0"] = 0.0
    df.loc[1 : dConfig["nx"] - 2, "S0"] = (
        df.loc[: dConfig["nx"] - 3, "z"].values - df.loc[2:, "z"].values
    ) / (2 * dConfig["dx"])
    df.loc[dConfig["nx"] - 1, "S0"] = (
        -(df.loc[dConfig["nx"] - 1, "z"] - df.loc[dConfig["nx"] - 2, "z"])
        / dConfig["dx"]
    )
    df.loc[0, "S0"] = (df.loc[0, "z"] - df.loc[1, "z"]) / dConfig["dx"]

    # Minimum slope
    # mask = np.abs(df["S0"]) < 0.01 / dConfig["dx"]
    # df["signS0"] = np.sign(df["S0"])
    # df["S0"] = np.abs(df["S0"])
    # df.loc[mask, "S0"] = 0.01 / dConfig["dx"]

    # Murillo condition for dx
    df["nmann1"] = df["nmann"].values

    return db, df, aux


def _read_sediments(dConfig):
    """Read the sediments file

    Args:
        dConfig (dict): parameters specifications

    Return:
        sedProp (dict): sediment properties
    """

    sedProp = pd.read_csv(
        dConfig["sSedimentFilename"], sep=" ", index_col=[0], header=0
    )["parameter"].to_dict()

    # sedProp = {}
    # sedProp["Dmed"] = 0.0141
    # sedProp["D90"] = 0.02791
    # sedProp["D84"] = 0.02278
    # sedProp["D50"] = 0.00754
    # sedProp["D16"] = 0.00169
    # sedProp["sigma"] = 3.7347
    # sedProp["rhos"] = 2.65
    sedProp["Diamx"] = sedProp["D50"] * (
        (sedProp["rhos"] - 1.0) * g / (1.0e-6) ** 2.0
    ) ** (1.0 / 3.0)
    return sedProp


def _dry_soil(dbt, iTime, var_=""):
    """_summary_

    Args:
       dbt (xr): the database with the variables in space and time
        iTime (_type_): _description_
        var_ (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    Adry = dbt["B"][:, iTime] * 0.01  # 1 cm de alto por el ancho, 1e-8
    Qdry = 1e-5  # to ensure that U = Q/A ~ 0
    # mask = np.abs(dbt["Q" + var_][:, iTime]) < Qdry
    # dbt["Q" + var_][mask, iTime] = Qdry

    mask = dbt["A" + var_][:, iTime] <= Adry
    dbt["A" + var_][mask, iTime] = Adry[mask]

    # mask = ((dbt["A" + var_][:, iTime] < Adry) | (dbt["Q" + var_][:, iTime] < Qdry))
    # dbt["A" + var_][mask, iTime] = Adry
    dbt["Q" + var_][mask, iTime] = Qdry

    return mask


def _TVD_MacCormack(dbt, df, dConfig, iTime):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        df (_type_): _description_
        dConfig (dict): parameters specifications
        iTime (_type_): _description_

    Returns:
        _type_: _description_
    """

    # umed, Amed, cmed = np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"])
    # a1med, a2med = np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"]),
    e1med, e2med, D = (
        np.ones([2, dConfig["nx"] - 1]),
        np.ones([2, dConfig["nx"] - 1]),
        np.zeros([2, dConfig["nx"] + 1]),
    )
    # alfa1med, alfa2med= np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"])
    psi1med, psi2med = np.zeros(dConfig["nx"] - 1), np.zeros(dConfig["nx"] - 1)
    r1med, r2med = np.zeros(dConfig["nx"] - 1), np.zeros(dConfig["nx"] - 1)

    # for i in range(dConfig["nx"]):
    umed = (
        dbt["Q"][1:, iTime].values / np.sqrt(dbt["A"][1:, iTime].values)
        + dbt["Q"][:-1, iTime].values / np.sqrt(dbt["A"][:-1, iTime].values)
    ) / (np.sqrt(dbt["A"][1:, iTime].values) + np.sqrt(dbt["A"][:-1, iTime].values))

    Amed = (dbt["A"][1:, iTime].values + dbt["A"][:-1, iTime].values) / 2.0
    cmed = (dbt["c"][1:, iTime].values + dbt["c"][:-1, iTime].values) / 2.0
    a1med = umed + cmed
    a2med = umed - cmed
    e1med[1, :] = a1med
    e2med[1, :] = a2med
    if not dConfig["bSurfaceGradientMethod"]:
        alfa1med = (
            (dbt["Q"][1:, iTime].values - dbt["Q"][:-1, iTime].values)
            + (-umed + cmed) * (Amed - dbt["A"][:-1, iTime].values)
        ) / (2.0 * cmed)
        alfa2med = -(
            (dbt["Q"][1:, iTime].values - dbt["Q"][:-1, iTime].values)
            + (-umed - cmed) * (Amed - dbt["A"][:-1, iTime].values)
        ) / (2.0 * cmed)
    else:
        alfa1med = (
            # dbt["B"][:-1, iTime] *
            (
                (
                    dbt["Q"][1:, iTime].values / dbt["B"][1:, iTime].values
                    - dbt["Q"][:-1, iTime].values / dbt["B"][:-1, iTime].values
                )
                + (-umed + cmed) * (df["elev"][1:].values - df["elev"][:-1].values)
            )
            / (2.0 * cmed)
        )
        alfa2med = (
            # -dbt["B"][:-1, iTime] *
            -(
                (
                    dbt["Q"][1:, iTime].values / dbt["B"][1:, iTime].values
                    - dbt["Q"][:-1, iTime].values / dbt["B"][:-1, iTime].values
                )
                + (-umed - cmed) * (df["elev"][1:].values - df["elev"][:-1].values)
            )
            / (2.0 * cmed)
        )

    # ! C�lculo de Psi, seleccionar opci�n al inicio de la subrutina
    if dConfig["iPsiFormula"] == 1:  #                  ! Garc�a-Navarro
        delta1 = dConfig["delta"]
        delta2 = dConfig["delta"]
    elif dConfig["iPsiFormula"] == 2:  #              ! Tseng
        delta1 = np.max(
            [
                np.zeros(dConfig["nx"] - 1),
                a1med - (dbt["U"][:-1, iTime] + dbt["c"][:-1, iTime]),
                (dbt["U"][1:, iTime] + dbt["c"][1:, iTime]) - a1med,
            ],
            axis=0,
        )
        delta2 = np.max(
            [
                np.zeros(dConfig["nx"] - 1),
                a2med - (dbt["U"][:-1, iTime] - dbt["c"][:-1, iTime]),
                (dbt["U"][1:, iTime] - dbt["c"][1:, iTime]) - a2med,
            ],
            axis=0,
        )

    mask = np.abs(a1med) >= delta1
    psi1med[mask] = np.abs(a1med[mask])
    mask = np.abs(a1med) <= delta1
    psi1med[mask] = delta1[mask]

    mask = np.abs(a2med) >= delta2
    psi2med[mask] = np.abs(a2med[mask])
    mask = np.abs(a2med) <= delta2
    psi2med[mask] = delta2[mask]

    # Computing r
    mask = alfa1med == 0
    r1med[mask] = 1

    imask = np.where(a1med < 0)[0]
    imask = np.asarray([i for i in imask if i != dConfig["nx"] - 2])
    if len(imask) != 0:
        r1med[imask] = alfa1med[imask + 1] / alfa1med[imask]

    mask = a1med == 0
    r1med[mask] = 1

    imask = np.where(a1med > 0)[0]
    imask = np.asarray([i for i in imask if i != 0])
    if len(imask) != 0:
        r1med[imask] = alfa1med[imask - 1] / alfa1med[imask]
    r1med[-1] = 1

    mask = alfa2med == 0
    r2med[mask] = 1

    imask = np.where(a2med < 0)[0]
    imask = np.asarray([i for i in imask if i != dConfig["nx"] - 2])
    if len(imask) != 0:
        r2med[imask] = alfa2med[imask + 1] / alfa2med[imask]

    mask = a2med == 0
    r2med[mask] = 1

    imask = np.where(a2med > 0)[0]
    imask = np.asarray([i for i in imask if i != 0])
    if len(imask) != 0:
        r2med[imask] = alfa2med[imask - 1] / alfa2med[imask]
    r2med[-1] = 1

    # Cálculo de fi
    fi1med = np.max(
        [np.zeros(len(r1med)), np.min([np.ones(len(r1med)), r1med], axis=0)], axis=0
    )
    fi2med = np.max(
        [np.zeros(len(r2med)), np.min([np.ones(len(r2med)), r2med], axis=0)], axis=0
    )

    if dConfig["iFormulaLimiterFlux"] == 1:  #               ! MinMod
        fi1med = np.max(0.0, np.min(1.0, r1med))
        fi2med = np.max(0.0, np.min(1.0, r2med))
    elif dConfig["iFormulaLimiterFlux"] == 2:  #          ! Roe's Superbee
        fi1med = np.max(0.0, np.min(2 * r1med, 1.0), np.min(r1med, 2.0))
        fi2med = np.max(0.0, np.min(2 * r2med, 1.0), np.min(r2med, 2.0))
    elif dConfig["iFormulaLimiterFlux"] == 2:  #           ! Van Leer
        fi1med = (np.abs(r1med) + r1med) / (1 + np.abs(r1med))
        fi2med = (np.abs(r2med) + r2med) / (1 + np.abs(r2med))
    elif dConfig["iFormulaLimiterFlux"] == 4:  #         ! Van Albada
        fi1med = (r1med**2.0 + r1med) / (1 + r1med**2.0)
        fi2med = (r2med**2.0 + r2med) / (1 + r2med**2.0)

    # ! C�lculo de D
    fac1pr = alfa1med * psi1med * (1 - dConfig["lambda"] * np.abs(a1med)) * (1 - fi1med)
    fac2pr = alfa2med * psi2med * (1 - dConfig["lambda"] * np.abs(a2med)) * (1 - fi2med)

    D[0, :-2] = 0.5 * (fac1pr * e1med[0, :] + fac2pr * e2med[0, :])
    D[1, :-2] = 0.5 * (fac1pr * e1med[1, :] + fac2pr * e2med[1, :])

    # D[0, 0] = D[0, 1]
    # D[1, 0] = D[1, 1]

    # D[0, -1] = D[0, -2]
    # D[1, -1] = D[1, -2]

    return D


def _density(dbt, dConfig, sedProp, iTime, predictor=True):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        dConfig (dict): parameters specifications
        sedProp (_type_): _description_
        iTime (_type_): _description_
        predictor (bool, optional): _description_. Defaults to True.
    """
    if predictor:
        var_ = ""
    else:
        var_ = "p"

    rhow = 1000 * (1 + dConfig["fSalinityBeta"] * dbt["S"][:, iTime])
    dbt["rho" + var_][:, iTime] = (
        rhow
        + (1 - rhow / 1000 / sedProp["rhos"])
        * dbt["Qt"][:, iTime]
        / (dbt["A" + var_][:, iTime] * dConfig["dx"])
        * dConfig["dtmin"]
    )
    return


def _salinity(dbt, aux, dConfig, iTime):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        dConfig (dict): parameters specifications
        iTime (_type_): _description_
        var_: "p", "c", ""

    Returns:
        _type_: _description_
    """
    # Dry bed algorithm
    if dConfig["bDryBed"]:
        mask_dry = _dry_soil(dbt, iTime)
        aux["ast"][mask_dry] = 0
    # ----------------------------------------------------------------------

    if not "iInitialSalinityCondition" in dConfig.keys():
        raise ValueError("Initial salinity conditions in psu is required.")
    else:
        aux["ast"][0] = dConfig["iInitialSalinityCondition"]

    if not "iFinalSalinityCondition" in dConfig.keys():
        raise ValueError("Final salinity conditions in psu is required.")
    else:
        aux["ast"][-1] = dConfig["iFinalSalinityCondition"]

    dbt["S"][:, iTime] = aux["ast"] / dbt["A"][:, iTime].values + aux["salinity"]
    aux["salinity"] = dbt["S"][:, iTime].values

    # Bound the minimum and maximum values of salinity to 0 a 35 psu,
    mask = dbt["S"][:, iTime] < 0
    dbt["S"][mask, iTime] = 0

    mask = dbt["S"][:, iTime] > 35
    dbt["S"][mask, iTime] = 35
    return aux


def _salinity_gradient(dbt, aux, dConfig, iTime):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        dConfig (dict): parameters specifications
        iTime (_type_): _description_

    Returns:
        _type_: _description_
    """

    kasdif_forward = dbt["A"][1:, iTime].values * (
        dbt["S"][1:, iTime].values - dbt["S"][:-1, iTime].values
    )
    kasdif_forward = np.hstack([kasdif_forward, kasdif_forward[-1]])

    kasdif_backward = dbt["A"][:-1, iTime].values * (
        dbt["S"][1:, iTime].values - dbt["S"][:-1, iTime].values
    )
    kasdif_backward = np.hstack([kasdif_backward[0], kasdif_backward])

    ausdif = (
        (
            dbt["Q"][2:, iTime].values * dbt["S"][2:, iTime].values
            - dbt["Q"][:-2, iTime].values * dbt["S"][:-2, iTime].values
        )
        * dConfig["lambda"]
        / 2
    )
    ausdif = np.hstack(
        [
            # ausdif[0],
            (
                dbt["Q"][1, iTime].values * dbt["S"][1, iTime].values
                - dbt["Q"][0, iTime].values * dbt["S"][0, iTime].values
            )
            * dConfig["lambda"],  # dConfig["dx"]
            ausdif,
        ]
    )

    ausdif = np.hstack(
        [
            ausdif,
            (
                dbt["Q"][-1, iTime].values
                * dbt["S"][-1, 0].values  # TODO: este valor debería ser el del océano
                - dbt["Q"][-2, iTime].values * dbt["S"][-2, iTime].values
            )
            * dConfig["lambda"],
        ]
    )

    aux["ast"] = (
        dConfig["fKH"]
        * dConfig["lambda"] ** 2
        / dConfig["dtmin"]
        * (kasdif_forward - kasdif_backward)
        - ausdif
    )
    return aux


def _vanRijn(sedProp, dbt, df, iTime):
    """Compute the bedload and suspended sediment transport using the van Rijn
    equation (van Rijn, 1992)

    Args:
        sedProp (dict): sediment properties
        dbt (xr): the database with the variables in space and time
        df (_type_): _description_
        iTime (_type_): _description_
    """

    #  ! Cuidado con el signo, regresar despu�s
    vel = np.abs(dbt["U"][:, iTime].values)
    #  !Sf=abs(vel)
    # direc = np.sign(vel)

    # Cálculo del transporte de fondo POR ARRASTRE: ######################
    c_1 = 18 * np.log((12 * dbt["Rh"][:, iTime].values) / (3 * sedProp["D90"]))
    mask = c_1 < 1e-3
    c_1[mask] = 1e-3
    u_star = ((g**0.5) / (c_1)) * vel

    shields_crit = 0.0013 * sedProp["Diamx"] ** 0.29
    if sedProp["Diamx"] < 150:
        shields_crit = 0.055

    u_star_crit = np.sqrt(shields_crit * (sedProp["rhos"] - 1) * g * sedProp["D50"])

    T = (u_star**2 - u_star_crit**2) / (u_star_crit**2)
    mask = T < 0  # no hay tte por fondo
    T[mask] = 0  #! Corrección

    #! Transporte de fondo (m�sico) Creo que es volumétrico ya
    gb = (
        0.053
        * ((sedProp["rhos"] - 1.0) * g * sedProp["D50"] ** 3.0) ** 0.5
        * T**2.1
        * sedProp["Diamx"] ** -0.3
    )
    mask = T >= 3
    gb[mask] = (
        0.100
        * ((sedProp["rhos"] - 1.0) * g * sedProp["D50"] ** 3.0) ** 0.5
        * T[mask] ** 1.5
        * sedProp["Diamx"] ** -0.3
    )

    dbt["Qb"][:, iTime] = (gb * dbt["B"][:, iTime]) / (  # *df["signS0"])
        sedProp["rhos"] * gamma
    )  # Volum�trico

    # -----------------------------------------------------------------------------------
    # Cálculo del transporte de fondo EN SUSPENSIÓN
    # -----------------------------------------------------------------------------------
    #! Altura del salto
    deltab = sedProp["Dmed"] * 0.3 * sedProp["Diamx"] ** 0.7 * T**0.5

    #! Velocidad al cortante
    Ux = (
        g * dbt["Rh"][:, iTime].values * np.abs(df["Sf"].values)
    ) ** 0.5  # TODO: direction
    # mask = np.abs(Ux) < 1e-4  # TODO: poniendo limites a los ceros
    # Ux[mask] = 1e-4

    #! Rugosidad equivalente de Nikurazde
    # TODO: MCB - no estoy del todo convencido de que h deba ser dbt["eta"][:, iTime]
    # ks = (
    #     12.0 * dbt["eta"][:, iTime].values / np.exp(vel / (2.5 * vel))
    # )

    #! Nivel de referencia
    # a = ks  #         ! Elegir entre    OJO
    a = deltab  # uno y otro
    mask = a < 0.01 * dbt["eta"][:, iTime]
    a[mask] = 0.01 * dbt["eta"][mask, iTime].values  #! correci�n

    mask = a > 0.5 * dbt["eta"][:, iTime]
    a[mask] = 0.5 * dbt["eta"][mask, iTime].values

    #! Concentraci�n de referencia, en la elevaci�n z=a
    Ca = 0.117 * gamma * sedProp["rhos"] * T / sedProp["Diamx"]  # 		! En fondo plano
    # Ca=0.015*gamma*rhos*D50*T**1.5/(a*Diamx**0.3) #	! En fondo con ondulaciones

    #! Di�metro representativo de las part�culas en suspensi�n
    sedProp["Ds"] = sedProp["D50"] * (1.0 + 0.011 * (sedProp["sigma"] - 1.0))

    if T.any() >= 25:
        sedProp["Ds"] = sedProp["D50"]

    #! Velocidad de ca�da del di�metro representativo
    ws = (sedProp["rhos"] - 1.0) * g * sedProp["Ds"] ** 2.0 / (18.0 * nu)
    if (sedProp["Ds"] >= 0.0001) & (sedProp["Ds"] <= 0.001):
        ws = (
            10.0
            * nu
            * (
                (
                    1.0
                    + 0.01
                    * g
                    * (sedProp["rhos"] - 1.0)
                    * sedProp["Ds"] ** 3.0
                    / nu**2.0
                )
                ** 0.5
                - 1.0
            )
            / sedProp["Ds"]
        )
    elif sedProp["Ds"] >= 0.001:
        ws = 1.1 * ((sedProp["rhos"] - 1.0) * g * sedProp["Ds"]) ** 0.5

    # ! Factor que toma en cuenta la diferencia entre la difusi�n de una part�cula
    # ! de fluido y la de las part�culas de sedimentos
    beta = np.ones(len(Ux)) * 2
    mask = Ux != 0
    beta[mask] = 1.0 + 2.0 * (ws / Ux[mask]) ** 2.0
    mask = beta >= 2
    beta[mask] = 2.0

    # ! Par�metro de suspensi�n de Rouse
    zr = np.zeros(len(Ux))
    mask = Ux > 0
    zr[mask] = ws / (beta[mask] * kappa * Ux[mask])

    # ! M�xima concentraci�n posible en el nivel de referencia
    C0 = 0.65 * gamma * sedProp["rhos"]

    # ! Factor global de correci�n
    psi = np.zeros(len(Ux))
    psi[mask] = 2.5 * (ws / Ux[mask]) ** 0.8 * (Ca[mask] / C0) ** 0.4

    # ! Modificaci�n de la z de Rouse
    zp = zr + psi

    # ! Coeficiente adimensional A (Adim)
    Adim = a / dbt["eta"][:, iTime].values

    # ! Coeficiente adimensional F
    F = np.zeros(len(Adim))
    mask = np.abs(((1.0 - Adim) ** zp * (1.2 - zp))) >= 1e-4
    F[mask] = (Adim[mask] ** zp[mask] - Adim[mask] ** 1.2) / (
        (1.0 - Adim[mask]) ** zp[mask] * (1.2 - zp[mask])
    )  # TODO: verificar que la máscara es teóricamente aceptable o debo imponer un F máximo

    # Transporte del fondo en suspensión, calculado con esfuerzos
    gbs1 = Ca * dbt["eta"][:, iTime] * vel * F

    # Velocidad crítica
    # velc = (
    #     0.082516
    #     * sedProp["D50"] ** 0.1
    #     * np.log(4.0 * dbt["eta"][:, iTime].values / sedProp["D90"])
    # )
    # if sedProp["D50"] >= 0.0005:
    #     velc = (
    #         3.691500
    #         * sedProp["D50"] ** 0.6
    #         * np.log(4.0 * dbt["eta"][:, iTime].values / sedProp["D90"])
    #     )

    # ! Transporte de fondo en suspensi�n, calculado con la velocidad media
    # gbs2 = (
    #     0.012
    #     * gamma
    #     * sedProp["rhos"]
    #     * vel
    #     * sedProp["D50"]
    #     * ((vel - velc) / np.sqrt(g * (sedProp["rhos"] - 1.0) * sedProp["D50"])) ** 2.4
    #     / sedProp["Diamx"] ** 0.6
    # )
    # mask = gbs2 < 0
    # gbs2[mask] = 0

    # ! Elecci�n de transporte   ! OJO; manualmente
    gbs = gbs1  # !gbs=gbs2
    dbt["Qs"][:, iTime] = (
        gbs / (sedProp["rhos"] * gamma) * dbt["B"][:, iTime].values  # * df["signS0"]
    )  # ! Volum�trico
    dbt["Qt"][:, iTime] = dbt["Qs"][:, iTime] + dbt["Qb"][:, iTime]
    mask = dbt["Qt"][:, iTime] < 1e-6
    dbt["Qt"][mask, iTime] = 0
    return


def _gAS_terms(dbt, df, aux, config, iTime, predictor=True):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        df (_type_): _description_
        aux (_type_): _description_
        config (_type_): _description_
        iTime (_type_): _description_
        predictor (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Variables de ecuaciones (vectores U, F y G) y cálculo del paso de tiempo (dt)
    if predictor:
        var_ = ""
    else:
        var_ = "p"

    if not config["bSourceTermBalance"]:
        df["Sf"] = (
            dbt["Q" + var_][:, iTime].values
            * np.abs(dbt["Q" + var_][:, iTime].values)
            * df["nmann1"] ** 2.0
            / (
                dbt["A" + var_][:, iTime].values ** 2.0
                * dbt["Rh" + var_][:, iTime].values ** (4.0 / 3.0)
            )
        )
        aux["gAS0"] = g * dbt["A" + var_][:, iTime] * df["S0"][:]
        aux["gASf"] = g * dbt["A" + var_][:, iTime] * df["Sf"][:]
    else:
        # Source term balance
        if var_ == "":
            aux["Qmed"][:-1] = (
                dbt["Q" + var_][1:, iTime].values + dbt["Q" + var_][:-1, iTime].values
            ) / 2
            aux["Amed"][:-1] = (
                dbt["A" + var_][1:, iTime].values + dbt["A" + var_][:-1, iTime].values
            ) / 2
            aux["Rmed"][:-1] = (
                dbt["Rh" + var_][1:, iTime].values + dbt["Rh" + var_][:-1, iTime].values
            ) / 2

            aux["Qmed"][-1] = dbt["Q"][-1, iTime].values
            aux["Amed"][-1] = dbt["A"][-1, iTime].values
            aux["Rmed"][-1] = dbt["Rh"][-1, iTime].values
        else:
            aux["Qmed"][1:] = (
                dbt["Q" + var_][1:, iTime].values + dbt["Q" + var_][:-1, iTime].values
            ) / 2
            aux["Amed"][1:] = (
                dbt["A" + var_][1:, iTime].values + dbt["A" + var_][:-1, iTime].values
            ) / 2
            aux["Rmed"][1:] = (
                dbt["Rh" + var_][1:, iTime].values + dbt["Rh" + var_][:-1, iTime].values
            ) / 2

            aux["Qmed"][0] = dbt["Qp"][0, iTime].values
            aux["Amed"][0] = dbt["Ap"][0, iTime].values
            aux["Rmed"][0] = dbt["Rhp"][0, iTime].values

        if predictor:
            aux["gAS0"] = g * aux["Amed"] * df["zmedp"].values  # * df["signS0"]
        else:
            aux["gAS0"] = g * aux["Amed"] * df["zmedc"].values  # * df["signS0"]
        aux["gASf"] = (
            g
            * df["nmann1"].values ** 2.0
            * aux["Qmed"] ** 2
            # * np.abs(aux["Qmed"])
            / (aux["Amed"] * aux["Rmed"] ** (4.0 / 3.0))
        )  # * df["signSf"]

    return aux


def _F_terms(dbt, aux, dConfig, iTime, predictor=True):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        aux (_type_): _description_
        dConfig (_type_): _description_
        iTime (_type_): _description_
        predictor (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if predictor:
        var_ = ""
    else:
        var_ = "p"

    aux["F" + var_][0, :] = dbt["Q" + var_][:, iTime]
    aux["F" + var_][1, :] = (
        dbt["Q" + var_][:, iTime].values ** 2.0 / dbt["A" + var_][:, iTime].values
        + g * dbt["I1" + var_][:, iTime].values
    )
    if dConfig["bBeta"]:
        aux["F" + var_][1, :] = dbt["beta"][:, iTime].values * aux["F" + var_][1, :]

    return aux


def _Gv_terms(dbt, aux, iTime, predictor=True):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        aux (_type_): _description_
        iTime (_type_): _description_
        predictor (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if predictor:
        var_ = ""
    else:
        var_ = "p"
    aux["Gv" + var_][0, :] = dbt["q"][:, iTime].values
    aux["Gv" + var_][1, :] = (
        g * dbt["I2" + var_][:, iTime].values + aux["gAS0"] - aux["gASf"]
    )
    return aux


def _conditionMurillo(dbt, df, config, iTime):
    """_summary_

    Args:
        dbt (_type_): _description_
        df (_type_): _description_
        config (_type_): _description_
        iTime (_type_): _description_

    Returns:
        _type_: _description_
    """
    cdx = 0.6
    nmr = 0
    nmur = cdx * np.sqrt(
        2 * dbt["Rh"][:, iTime].values ** (2.0 / 3.0) / (g * config["dx"])
    )
    mask = nmur < df["nmann"]
    df.loc[mask, "nmann1"] = nmur[mask]
    # facmr[mask] = nmur[mask] / df.loc[mask, "nmann"]
    dbt["I1"][mask, iTime] = dbt["I1"][mask, iTime] * nmur[mask] / df.loc[mask, "nmann"]
    vmr = mask

    return vmr, nmur


def _courant_number(dbt, df, dConfig, iTime):
    """Compute the time step through the Courant number. The Courant number includes:
    mean water velocity, perturbation velocity and the salinity dispersion

    Args:
        dbt (_type_): _description_
        df (_type_): _description_
        dConfig (_type_): _description_
        iTime (_type_): _description_

    Returns:
        aux (dict): update the input dictionary with the configuration
    """
    # Cálculo del valor del paso de tiempo, dependiente de Courant, de la velocidad
    # media del flujo, de propagación de las perturbaciones y de la dispersión salina
    dbt["U"][:, iTime] = (
        dbt["Q"][:, iTime] / dbt["A"][:, iTime]  # * df["signS0"]
    )  # mean velocity
    dbt["c"][:, iTime] = np.sqrt(
        g * dbt["A"][:, iTime] / dbt["B"][:, iTime]
    )  #  perturbation celerity

    # Remove locations where bed is dry
    mask = dbt["A"][:, iTime].values != 1e-8
    sum_mask = sum(mask)

    if sum_mask == 0:
        mask[-1] = True

    dtpr = (
        dConfig["fCourantNo"]
        * dConfig["dx"]
        / (np.abs(dbt["U"][mask, iTime].values) + dbt["c"][mask, iTime].values)
    )
    dConfig["dtmin"] = np.min(dtpr)

    if dConfig["bDensity"] & (sum_mask != 0):
        aux_ = np.min(
            dConfig["fCourantNo"]
            * dConfig["dx"]
            / (dConfig["fKH"] / dbt["B"][mask, iTime].values)
        )
        dConfig["dtmin"] = np.min([dConfig["dtmin"], aux_])
    return dConfig


def _check_dt(dConfig, iTime, fTelaps):
    """Bound the dt for ensuring convergency

    Args:
        dConfig (_type_): _description_
        iTime ():
        fTelaps (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Bound the maximum timestep to half the given timestep (for convergency)
    # if dConfig["dtmin"] > dConfig["fTimeStep"] / 2:
    #     dConfig["dtmin"] = dConfig["fTimeStep"] / 2

    # Check the time step for ensuring the save times
    dConfig["next_timestep"] = False
    if fTelaps == 0:
        dConfig["next_timestep"] = True
    elif iTime + 1 > dConfig["iTime"][-1]:
        if (
            fTelaps + dConfig["dtmin"]
            >= dConfig["iTime"][iTime]
            * dConfig["time_multiplier_factor"]
            * dConfig["fTimeStep"]
        ):
            dConfig["dtmin"] = (
                dConfig["iTime"][iTime]
                * dConfig["time_multiplier_factor"]
                * dConfig["fTimeStep"]
                - fTelaps
            )
            dConfig["next_timestep"] = True
    elif (
        fTelaps + dConfig["dtmin"]
        >= dConfig["iTime"][iTime + 1]
        * dConfig["time_multiplier_factor"]
        * dConfig["fTimeStep"]
    ):
        dConfig["dtmin"] = (
            dConfig["iTime"][iTime + 1]
            * dConfig["time_multiplier_factor"]
            * dConfig["fTimeStep"]
            - fTelaps
        )
        dConfig["next_timestep"] = True

    fTelaps += dConfig["dtmin"]

    dConfig["lambda"] = dConfig["dtmin"] / dConfig["dx"]
    return dConfig, fTelaps


def _boundary_conditions(dbt, aux, dConfig, iTime, predictor=True):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        aux (_type_): _description_
        config (_type_): _description_
        iTime (_type_): _description_
        var_ (str, optional): _description_. Defaults to "p".

    Returns:
        _type_: _description_
    """

    if predictor:
        # Upward boundary condition
        if dConfig["iInitialBoundaryCondition"] == 0:  # Open boundary condition
            aux["Up"][1, 0] = aux["Up"][1, 1]
        elif dConfig["iInitialBoundaryCondition"] == 1:  # Reflecting boundary condition
            aux["Up"][1, 0] = dbt["q"][0, iTime]

        # Downward boundary condition
        if dConfig["iFinalBoundaryCondition"] == 0:  # Open flux
            aux["Up"][0, -1] = aux["Up"][0, -2]
            aux["Up"][1, -1] = aux["Up"][1, -2]
        elif dConfig["iFinalBoundaryCondition"] == 1:  # Tidal level
            aux["Up"][0, -1] = aux["tidal_area_seaward"]
            aux["Up"][0, -2] = aux["tidal_area_next_seaward"]
        elif dConfig["iFinalBoundaryCondition"] == 2:  # TODO: Discharge flux
            aux["Up"][1, -1] = qff + qver
            aux["Up"][1, -2] = qff + qver  # Improvement for the numerical scheme
    else:
        # Upward boundary condition
        if dConfig["iInitialBoundaryCondition"] == 0:  # Open boundary condition
            aux["Uc"][1, 0] = aux["Uc"][1, 1]
            aux["Uc"][0, 0] = aux["Up"][0, 0]
        elif dConfig["iInitialBoundaryCondition"] == 1:  # Reflecting boundary condition
            aux["Uc"][1, 0] = dbt["q"][0, iTime]
            aux["Uc"][0, 0] = aux["Up"][0, 0]

        # Downward boundary condition
        if dConfig["iFinalBoundaryCondition"] == 0:  # Open flux
            aux["Uc"][0, -1] = aux["Uc"][0, -2]
            aux["Uc"][1, -1] = aux["Uc"][1, -2]
        elif dConfig["iFinalBoundaryCondition"] == 1:  # Tidal level
            aux["Uc"][0, -1] = aux["tidal_area_seaward"]
            aux["Uc"][0, -2] = aux["tidal_area_next_seaward"]
            aux["Uc"][1, -1] = aux["Up"][1, -1]
        elif dConfig["iFinalBoundaryCondition"] == 2:  # TODO: Discharge flux
            aux["Uc"][1, -1] = qff + qver
            aux["Uc"][1, -2] = qff + qver  # Improvement for the numerical scheme

    return aux


def _init_logplots():
    """Initialize log plots

    Returns:
        matplotlib.pyplot axes
    """
    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(16, 8))
    axs = axs.flatten()
    axs[0].set_title("U")
    axs[1].set_title("Up")
    axs[2].set_title("Uc")
    axs[3].set_title("Un")
    axs[4].set_title("A")
    axs[5].set_title("F")
    axs[6].set_title("Gv")
    axs[7].set_title("D")
    return axs


def _log_plots(dbt, aux, dConfig, iTime, axs):
    """Plot the log figure every timestep

    Args:
        dbt (_type_): _description_
        aux (_type_): _description_
        dConfig (_type_): _description_
        iTime (_type_): _description_
        axs (_type_): _description_
    """
    axs[0].plot(aux["U"][1, :])
    axs[1].plot(aux["Up"][1, :])
    axs[2].plot(aux["Uc"][1, :])
    axs[3].plot(aux["Un"][1, :])
    axs[4].plot(dbt["A"][:, iTime])
    axs[5].plot(
        (
            aux["Fp"][1, 1:] * dbt["rhop"][1:, iTime]
            - aux["Fp"][1, :-1] * dbt["rhop"][:-1, iTime]
        )
        / dbt["rhop"][1:, iTime]
    )
    axs[6].plot(dConfig["dtmin"] * aux["Gvp"][1, 1:])
    axs[7].plot(aux["D"][1, :])
    plt.pause(0.01)
    return


def _savefiles(dbt, db, df, dConfig):
    """_summary_

    Args:
        dbt (xr): the database with the variables in space and time
        db (xr): the information about the geometry of the sections in arrays
        df (pd.DataFrame): geometrical properties per section
        dConfig (dict): parameters specifications
    """
    if dConfig["bLog"]:
        dbt.to_netcdf(dConfig["sOutputFilename"])
        db.to_netcdf("auxiliary_file.nc")
        df.to_csv("df.csv")
    else:
        dbt[
            [
                "A",
                "Q",
                "q",
                "Rh",
                "I1",
                "B",
                "eta",
                "beta",
                "I2",
                "U",
                "S",
                "Qb",
                "Qs",
                "Qt",
                "rho",
                "xl",
                "xr",
            ]
        ].to_netcdf(dConfig["sOutputFilename"])
        df.to_csv("df.csv")
    return


def param_(xref, lambda_, x, x0, xb, B0, Bb):
    """Compute the parameters of the Prandtl and Rahman (1980)

    Args:
        xref (_type_): _description_
        lambda_ (_type_): _description_
        x (_type_): _description_
        x0 (_type_): _description_
        xb (_type_): _description_
        B0 (_type_): _description_
        Bb (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = 0.0
    n = (np.log(Bb / lambda_) - np.log(B0 / lambda_)) / np.log(xref / lambda_)
    x0 = (xref + x0) / lambda_
    xb = (xref - xb) / lambda_
    y0 = 4 * np.pi / (2 - m) * x0 * ((2 - m) / 2)
    yb = 4 * np.pi / (2 - m) * xb * ((2 - m) / 2)
    yx = 4 * np.pi / (2 - m) * (x / lambda_) ** ((2 - m) / 2)
    ys = np.array([y0, yb])
    nu = (n + 1) / (2 - m)
    return ys, yx, nu, m


def eta(r, A0, ys, yx, t, nu):
    k = np.sqrt(1 - 1j * r[0] / (2 * np.pi))
    zx = np.real(
        A0
        * ((k * yx) / (k * ys[0])) ** (1 - nu)
        * (
            sp.yv(nu - 1, k * ys[0]) * sp.jv(nu, k * ys[1])
            - sp.jv(nu - 1, k * yx) * sp.yv(nu, k * ys[1])
        )
        / (
            sp.yv(nu - 1, k * ys[0]) * sp.jv(nu, k * ys[1])
            - sp.jv(nu - 1, k * ys[0]) * sp.yv(nu, k * ys[1])
        )
        * np.exp(1j * (2 * np.pi * t))
    )
    return zx


def u(r, A0, ys, yx, t, nu, x, m):
    k = np.sqrt(1 - 1j * r[0] / (2 * np.pi))
    ux = np.real(
        -1j
        * A0
        * x ** (-m / 2)
        * ((k * yx) / (k * ys[0])) ** (1 - nu)
        * (
            sp.jv(nu, k * yx) * sp.yv(nu, k * ys[1])
            - sp.jv(nu, k * ys[1]) * sp.yv(nu, k * yx)
        )
        / (
            sp.jv(nu, k * ys[1]) * sp.yv(nu - 1, k * ys[0])
            - sp.jv(nu - 1, k * ys[0]) * sp.yv(nu, k * ys[1])
        )
        * np.exp(1j * (2 * np.pi * t))
    )
    return ux


def model_eta_u(x, t, tidal_params, params, constituents):
    """Computing the free surface elevation and mean water current using the methodology
    given in Prandtl and Rahman (1980)

    Args:
        x (_type_): _description_
        t (_type_): _description_
        constituents (_type_): _description_
        params (dict):
            - h0:
            - xref:
            - x0:
            - xb:
            - B0:
            - Bb:

    Returns:
        _type_: _description_
    """
    eta_, u_ = pd.DataFrame(0.0, index=t, columns=x), pd.DataFrame(
        0.0, index=t, columns=x
    )

    for const in constituents:
        T0, A0, p0, r = (
            tidal_params["T"][const],
            tidal_params["A0"][const],
            tidal_params["p0"][const],
            tidal_params["r"][const],
        )
        g, T = 9.81, T0 * 3600
        lambda_ = np.sqrt(g * params["h0"]) * T

        A0, p0 = A0 / params["h0"], p0 * np.pi / 180.0
        A0 = A0 * np.exp(-1j * p0)

        time = eta_.index - eta_.index[0]
        time = np.array([(j.seconds / 3600 + j.days * 24) / T0 for j in time])
        for j, t in enumerate(eta_.index):
            ys, yx, nu, m = param_(
                params["xref"],
                lambda_,
                x,
                params["x0"],
                params["xb"],
                params["B0"],
                params["Bb"],
            )

            eta_.loc[t, :] = (
                eta_.loc[t, :] + eta(r, A0, ys, yx, time[j], nu) * params["h0"]
            )
            u_.loc[t, :] = (
                u_.loc[t, :] + u(r, A0, ys, yx, time[j], nu, x, m) * lambda_ / T
            )
    return eta_, u_


def run_average(x, num):
    """Computed the moving average of the data

    Args:
        x: data
        num: width of the moving window

    Returns:
        The averaged signal
    """
    cont = 0
    av = x.copy()
    for i in range(0, int(num / 2)):
        av.iloc[i] = x.iloc[cont : i + int(num / 2)].mean()
    for i in range(int(num / 2), len(x) - int(num / 2)):
        av.iloc[i] = x.iloc[i - int(num / 2) : i + int(num / 2)].mean()
    for i in range(len(x) - int(num / 2), len(x)):
        av.iloc[i] = x.iloc[i - int(num / 2) :].mean()
    return av


def load_general_data(time_str, time_it):
    x_edges = np.array([57.6, 47.1, 35.3, 26.2, 17.3, 0])
    # coefs = x_edges[::-1]*(1.4)/57.6 - 0.4

    coefs = np.array(
        [
            -0.25,
            -0.7641 / 3.2608,
            -0.3951 / 3.0339,
            0.4737 / 0.8648,
            0.5112 / 0.5248,
            0.3056 / 0.2416,
        ]
    )[::-1]
    aux_info = {
        "coefs_stokes": coefs,
        "lat": 36,
        "ang_thalw": np.array(
            [
                80 * np.pi / 180,
                20 * np.pi / 180,
                70 * np.pi / 180,
                80 * np.pi / 180,
                100 * np.pi / 180,
                90 * np.pi / 180,
            ]
        ),
    }
    # 80, 20, 70, 80, 30, 90
    # 90, 20, 100, 40, 80, 10 , medido de la figura:
    LOCX = {"xedg": x_edges, "xbox": np.diff(x_edges) / 2.0 + x_edges[0:-1]}
    RHO = {"river": 999.7, "sea": None}

    PATH = {}
    PATH["root"] = os.path.join("..", "Data", time_str[0] + "-" + time_str[1])
    PATH["pre"] = os.path.join(
        "..", "Data", time_str[0] + "-" + time_str[1], "Pretreated"
    )
    PATH["box"] = os.path.join("..", "Data", time_str[0] + "-" + time_str[1], "Boxes")
    PATH["results"] = os.path.join(
        "..", "Data", time_str[0] + "-" + time_str[1], "Results"
    )

    return aux_info, LOCX, RHO, PATH


def signif_freqs(nfile, path):
    """Calcula las frecuencias significativas"""

    pathf = os.path.join(path["res"], "Table2-Significant_frequencies.xlsx")
    create_dir(path["signif_freqs"])
    freqs = []
    for i in nfile:
        # if 'completo' in i:
        n_wav = 24 * 5
        delta_st = 1 / 8766 * (n_wav)
        # else:
        #     n_wav = 24
        #     dst = 1/8766*(n_wav)
        pathr = os.path.join(path["recons"], i + "_recons.pkl")
        path0 = os.path.join(path["signif_freqs"], i + "_wvl_output.pkl")

        try:
            nivelr = pickle.load(open(pathr, "rb"))
        except ImportError:
            print("La senal {} aun no esta reconstruida.".format(i))

        wvl, glbl, fft, sig95 = wavelets(nivelr.iloc[::n_wav, 0], delta_st, 0.95)
        freqs.append(wvl_freq_ci(nivelr.iloc[::n_wav, 0], fft, glbl))
        wvl_info = {"wvl": wvl, "glbl": glbl, "fft": fft, "sig95": sig95}
        pickle.dump(wvl_info, open(path0, "wb"))

    freqs = pd.DataFrame(freqs, index=nfile)
    writer = pd.ExcelWriter(pathf)
    freqs.to_excel(writer)
    writer.save()
    return


def wavelets(data, delta_st, slevel):
    """Wavelet analysis and the statistical approach suggested by Torrence and Compo (1998) using the wavelet module"""
    std = data.std()  # Standard deviation
    std2 = std**2  # var_iance
    dat = (data.values - data.mean()) / std  # Calculatituding anomaly and normalizing

    prop = {}
    prop["no_measures"] = dat.size  # Number of measurements

    prop["delta_j"] = 1 / 12  # Twelve sub-octaves per octaves
    prop["s0"] = -1  # 2 * delta_t                    # Starting scale, here 6 months
    prop["J"] = (
        -1
    )  # 7 / dj                     # Seven powers of two with dj sub-octaves
    #  alpha = 0.0                       # Lag-1 autocorrelatitudion for white noise
    try:
        alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelatitudion for red noise
    except Warning:
        # When the dataset is too short, or there is a strong trend, ar1 raises a
        # warning. In this case, we assume a white noise background spectrum.
        alpha = 1.0

    prop["mother"] = wavelet.Morlet(6)  # Morlet mother wavelet with m=6

    # The following routines perform the wavelet transform and significance
    # analysis for the chosen data set.
    wvl, fft = {}, {}
    wvl["wvl"], scales, wvl["freqs"], wvl["coi"], fft["fft"], fft["freqs"] = (
        wavelet.cwt(
            dat, delta_st, prop["delta_j"], prop["s0"], prop["J"], prop["mother"]
        )
    )
    # iwave = wavelet.icwt(wave, scales, dst, dj, mother)

    # Normalized wavelet and Fourier power spectra
    wvl["power"] = (np.abs(wvl["wvl"])) ** 2
    fft["power"] = np.abs(fft["fft"]) ** 2
    wvl["period"] = 1 / wvl["freqs"]

    # Significance test. Where ratio power/sig95 > 1, power is significant.
    fft["signif"], fft["theor"] = wavelet.significance(
        1.0,
        delta_st,
        scales,
        0,
        alpha,
        significance_level=slevel,
        wavelet=prop["mother"],
    )
    sig95 = wvl["power"] / np.ones([1, prop["no_measures"]]) * fft["signif"][:, None]
    # sig95 = wvl['power'] / sig95

    # Power rectification as of Liu et al. (2007). TODO: confirm if significance
    # test ratio should be calculatituded first.
    # power /= scales[:, None]

    # Calculatitudes the global wavelet spectrum and determines its significance level.
    glbl = {}
    glbl["power"] = wvl["power"].mean(axis=1)
    dof = prop["no_measures"] - scales  # Correction for padding at edges
    glbl["signif"], _ = wavelet.significance(
        std2,
        delta_st,
        scales,
        1,
        alpha,
        significance_level=slevel,
        dof=dof,
        wavelet=prop["mother"],
    )

    # Scale average between avg1 and avg2 periods and significance level
    # sel = find((period >= avg1) & (period < avg2))
    # Cdelta = mother.cdelta
    # scale_avg = (scales * np.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    # scale_avg = power / scale_avg
    # scale_avg = std2 * dj * dst / Cdelta * scale_avg[sel, :].sum(axis=0)
    # scale_avg_signif, tmp = wavelet.significance(
    #     std2, dst, scales, 2, alpha, significance_level=slevel, dof=[scales[sel[0]], scales[sel[-1]]], wavelet=mother)

    return wvl, glbl, fft, sig95


import os
import pickle
from datetime import timedelta

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_rhoe(x, xboxes, B, h, rho, rhos, dt):
    """Compute the density in every box

    Args:
        x: location of the box (km)
        B: width at different locations
        h: mean depth of every box (m)
        rho: densities

    Returns:
        The density gradient (drhoe), the volumen of the boxes (Vol) and the mean width (Am)
    """

    # rho.index = rho.index.to_datetime(rho.index)
    colnames = [str(i) + " km" for i in xboxes]
    rhoe = pd.DataFrame(-1, index=rho.index, columns=colnames)
    drhoe = pd.DataFrame(0, index=rho.index, columns=colnames)
    rhos = rhos.tz_localize(None)
    if any("0.0 km" in s for s in rho.columns):
        rhoe["sea"] = rhos.loc[:, "sea"] * 0.25 + rho.loc[:, "0.0 km"] * 0.75
    else:
        rhoe["sea"] = rhos.loc[:, "sea"] * 0.25 + rho.loc[:, 0] * 0.75
    drhoe["sea"] = np.append(0.0, np.diff(rhoe.loc[:, "sea"]) / dt)
    Vol = pd.DataFrame(0, index=["hm3"], columns=colnames)
    for i, j in enumerate(colnames):
        print(i, j)
        rho_c, Bc, xc, hc = (
            rho.iloc[:, i + 1 : i + 3],
            B[i : i + 2],
            x[i : i + 2],
            h.iloc[0, i : i + 2].values,
        )
        f1 = Bc * hc * rho_c
        f2 = Bc * hc
        rhoe.loc[:, j] = np.trapz(f1, xc, 2) / np.trapz(f2, xc)
        drhoe.loc[1:, j] = np.diff(rhoe.loc[:, j]) / dt
        drhoe.loc[drhoe.index[0], j] = drhoe.loc[drhoe.index[1], j]
        Vol[j] = np.trapz(f2, xc[::-1] * 1000)

    return rhoe, drhoe, Vol


def calc_phi(rhoe, h, box):
    """Compute the potential energy anomaly

    Args:
        rhoe: estimated density of every box
        h: depth of boxes
        rho_s: density of the salted water

    Returns:
        Averaged box potential energy anomaly
    """

    g = 9.806
    # rhoe.insert(0, 'river', rhoe.iloc[:, 0]*0 + rhor)
    if box:
        Phi_m = pd.DataFrame(-1, index=rhoe.index, columns=rhoe.columns[:-1])
        for i, j in enumerate(Phi_m.columns):
            Phi_m.loc[:, j] = -g * h[i] * (rhoe.iloc[:, i] - rhoe.iloc[:, i + 1]) / 2
    else:
        Phi_m = pd.DataFrame(-1, index=rhoe.index, columns=rhoe.columns[1:-1])
        for i, j in enumerate(Phi_m.columns):
            Phi_m.loc[:, j] = (
                -g * h.iloc[0, i] * (rhoe.iloc[:, i + 1] - rhoe.iloc[:, i + 2]) / 2
            )

    return Phi_m


def spatial_interpolation(xedges, M, V):
    """Computed the spatial interpolation of data to the mean and edge locations of every box

    Args:
        xboxes: new location of data query
        M: tides
        V: velocities at surface in the river axis
        Rho: densities

    Returns:
        Three dataframes with tides, velocities at the surface and densities in the new points
    """

    nam_sl = ["Bonanza", "5.3 km", "26.8 km", "36.45 km", "51.8 km", "62.55 km"]
    xsl = np.array([0, 5.3, 26.8, 36.45, 51.8, 62.55])
    # nam_ctds = ['0 km', '17.3 km', '26.2 km', '35.3 km', '47.1 km', '57.6 km', '84.3 km']
    # xctds = np.array([0, 17.3, 26.2, 35.3, 47.1, 57.6, 84.3])
    nam_us = ["14.3 km", "20.8 km", "31.8 km", "39.8 km", "49.3 km", "63.8 km"]
    xus = np.array([14.3, 20.8, 31.8, 39.8, 49.3, 63.8])
    M, V = M.reindex(columns=nam_sl), V.reindex(columns=nam_us)
    # Rho = Rho.reindex(columns=nam_ctds)

    # colnames = [str(i)+' km' for i in xboxes]
    colnames = [str(i) + " km" for i in xedges]
    Mx, Vx = pd.DataFrame(-1, index=M.index, columns=colnames), pd.DataFrame(
        -1, index=M.index, columns=colnames
    )
    for j in Vx.index:
        Mx.loc[j, colnames] = np.interp(xedges, xsl, M.loc[j, nam_sl])
        Vx.loc[j, colnames] = np.interp(xedges, xus, V.loc[j, nam_us])
        # Rhox.iloc[j, :] = np.interp(xedges, xctds, Rho.iloc[j])

    return Mx, Vx


def tidal_average(signal, tm, ind_):
    """Computed the tidal average of the signal

    Args:
        ref: tidal signal of reference
        signal: series to interporlate

    Returns:
        The averaged signal
        :rtype: object
    """
    if tm is None:
        signalm = signal - np.mean(signal)
        sg = np.sign(signalm.values)
        ps = sg[0:-1] * sg[1:]

        # Sign change
        ind_ = np.where(ps < 0)[0]
        sc = sg[ind_]

        if sc[0] > 0:
            ind_ = ind_[1:]

        if np.remainder(len(ind_), 2) == 0:
            ind_ = ind_[0:-1]

        nid = len(ind_)
        val_ = np.zeros(int((nid - 1) / 2.0))
        tm = []
        for j in range(0, nid - 1, 2):
            val_[int(j / 2)] = np.max(
                signal.iloc[ind_[j] : ind_[j + 2]], axis=0
            ) - np.min(signal.iloc[ind_[j] : ind_[j + 2]], axis=0)
            dt = timedelta(
                seconds=(signal.index[ind_[j + 2]] - signal.index[ind_[j]]).seconds
                / 2.0
            )
            tm.append(signal.index[ind_[j]] + dt)
    else:
        nid = len(ind_)
        val_ = np.zeros([int((nid - 1) / 2.0), len(np.atleast_1d(signal.iloc[0]))])
        for j in range(0, nid - 1, 2):
            val_[int(j / 2), :] = np.mean(signal.iloc[ind_[j] : ind_[j + 2]])

    average = pd.DataFrame(val_, index=tm, columns=signal.columns)
    return average, ind_


def setting_up_boxes(M_, V_, Rho_, Q_, Qs_, W_, D_, C_, loc, aux_info):
    """Compute the main characteristics of the boxes. These characteristics are tidal-averaged and given in xboxes
    location.

    Args:
        rho: density measured at each sensor
        xboxes: location of the boxes (km)

    """

    B_box = 795.15 * np.exp(-loc["xbox"] / 65.5)
    h_box = 5839.4 * np.exp(-loc["xbox"] / 60.26) / B_box

    B_edges = 795.15 * np.exp(-loc["xedg"] / 65.5)
    h_edges = pd.DataFrame(
        [5839.4 * np.exp(-loc["xedg"] / 60.26) / B_edges],
        index=["m"],
        columns=Rho_.columns[::-1],
    )
    h_edges.rename(columns={"0 km": "0.0 km"}, inplace=True)
    A_edges = B_edges * h_edges

    Rho_["river"] = Rho_["river"]
    if any(Rho_.columns) == "23.6 km":
        Rho_.drop("23.6 km", 1, inplace=True)

    Rho_.rename(columns={"0 km": "0.0 km"}, inplace=True)
    Rhox = Rho_[Rho_.columns[::-1]].copy()
    Mx, Vx = spatial_interpolation(loc["xedg"], M_, V_)
    # Mx.insert(np.shape(Mx)[1], 'sea', M_['Bonanza'])
    # Vx.insert(np.shape(Vx)[1], 'sea', Vx.iloc[:, -1])
    rho["sea"] = rho["sea"].tz_localize(None)
    Rhox = Rhox.tz_localize(None)
    Rhox = Rhox.assign(sea=rho["sea"].values)

    Ac, ind_ = tidal_average(M_["Bonanza"].to_frame(), None, 0)
    Q, _ = tidal_average(Q_, Ac.index, ind_)
    Q.rename(columns={"flow (m3/s)": "river"}, inplace=True)

    # Rhox, _ = tidal_average(Rhox, Ac.index, ind_)
    dt = 1.0
    dt1 = np.asarray(
        [(Rhox.index[i + 1] - Rhox.index[i]).seconds for i in range(len(Rhox) - 1)]
    )
    #    dt1 = np.asarray(dt1)
    #    dt1 = (Rhox.index[1:] - Rhox.index[0:-1]).seconds
    rhoe, drhoe, Vol = calc_rhoe(
        loc["xedg"], loc["xbox"], B_edges, h_edges, Rhox, rho["sea"], dt1
    )
    phie = calc_phi(rhoe, h_box, True)
    phi = calc_phi(Rhox, h_edges, False)

    # Se multiplica por el ancho y se incluye el bombeo mareal
    # ttes_adv = ttes_advectivos(phi, Mx, Vx, Ac, h_edges, B_edges, loc['xedg'], ind_, dt, aux_info['lat'])
    # ttes_adv['stokes'] = ttes_adv['stokes']*(1+aux_info['coefs_stokes'])*B_edges

    # ttes_dif = ttes_difusivos(Rhox, B_edges, loc['xedg'], rhoe.columns, Ac, ind_)

    ter_fuente = terminos_fuente(
        loc["xedg"], B_edges, C_, Vx, W_.to_frame(), Qs_, Ac, ind_, rhoe.columns
    )

    # promedios mareales
    Mx, _ = tidal_average(Mx, Ac.index, ind_)
    Vx, _ = tidal_average(Vx, Ac.index, ind_)
    W, _ = tidal_average(W_.to_frame(), Ac.index, ind_)
    D, _ = tidal_average(D_.to_frame(), Ac.index, ind_)
    # Vl, _ = tidal_average(Vl, Ac.index, ind_)
    Qs, _ = tidal_average(Qs_, Ac.index, ind_)
    phi, _ = tidal_average(phie, Ac.index, ind_)
    rhoe, _ = tidal_average(rhoe, Ac.index, ind_)
    drhoe, _ = tidal_average(drhoe, Ac.index, ind_)
    Rhox, _ = tidal_average(Rhox, Ac.index, ind_)
    h_box = pd.DataFrame([h_box], index=["m"], columns=rhoe.columns[:-1])

    var_names = [
        "rhoe",
        "drhoe",
        "ttes_adv",
        "ttes_dif",
        "Phi",
        "Q",
        "Qs",
        "Ac",
        "Vol",
        "h_box",
        "h_edges",
        "Mx",
        "Vx",
        "W",
        "D",
        "Rhox",
        "A_edges",
    ]
    saving_data(
        [
            rhoe,
            drhoe,
            ttes_adv,
            ttes_dif,
            phie,
            Q,
            Qs,
            Ac,
            Vol,
            h_box,
            h_edges,
            Mx,
            Vx,
            W,
            D,
            Rhox,
            A_edges,
        ],
        var_names,
    )
    saving_data([ind_], ["ind_"], path, "npy")
    hs = {"box": h_box, "edges": h_edges}

    return (
        rhoe,
        Rhox,
        drhoe,
        ttes_adv,
        ttes_dif,
        phi,
        Q,
        Qs,
        Ac,
        Vol,
        A_edges,
        hs["edges"],
        Vx,
        W,
        D,
    )
