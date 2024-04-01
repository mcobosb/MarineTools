import datetime
import os
import re
import time
from configparser import ConfigParser

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from marinetools.utils import read

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

global g
g = 9.81


def clock(initial, telap, dConfig):
    """_summary_

    Args:
        initial (_type_): _description_
        telap (_type_): _description_
        dConfig (_type_): _description_

    Returns:
        _type_: _description_
    """
    elapsed_ = datetime.datetime.now() - initial
    if not dConfig["bLog"]:
        os.system("cls" if os.name == "nt" else "printf '\033c'")
    total_seconds = elapsed_.total_seconds()
    hours = str(int(total_seconds // 3600)).zfill(2)
    minutes = str(int((total_seconds % 3600) // 60)).zfill(2)
    seconds = str(int(total_seconds % 60)).zfill(2)
    return logger.info(
        f"{hours}:{minutes}:{seconds} - Time steps: "
        + "{0:4d}".format(telap)
        + " - "
        + "{0:5.2f}".format(
            np.round(
                telap * dConfig["fTimeStep"] / dConfig["fFinalTime"] * 100, decimals=2
            )
        )
        + " % completed"
    )


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


def initial_condition(dbt, db, df, aux, dConfig):
    """_summary_

    Args:
        dbt (_type_): _description_
        db (_type_): _description_
        dConfig (_type_): _description_

    Returns:
        _type_: _description_
    """

    if dConfig["iInitialEstuaryCondition"] == 2:
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

    elif dConfig["iInitialEstuaryCondition"] == 3:
        logger.info(
            "Computing initial condition for Q = 0 and eta = "
            + str(dConfig["fEtaci"])
            + " m"
        )
        # dbt["Q"][
        #     :, 0
        # ] = 40  # dConfig["qci"] # TODO: En el GRE el nivel es muy superior al caudal, meter ambas opciones a la vez
        for i in range(dConfig["nx"]):
            # areava = db["A"][i, :]
            # etava = db["eta"][i, :]
            # etavv = dConfig["etaci"] - df["z"][i]
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

    aux["U"][0, :] = dbt["A"][:, 0]
    aux["U"][1, :] = dbt["Q"][:, 0]
    aux["F"][0, :] = dbt["Q"][:, 0]
    # time.sleep(1)
    return aux


def read_oldfiles(filename, version_="v48"):
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
    else:
        dConfig["fEtaci"] = float(re.split("\s+", data[7][:-1])[0])

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
    # dConfig["idi2"] = float(
    #     re.split("\s+", data[27][:-1])[0]
    # )  # Para usar I2 de archivo o calcularlo
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

    # dConfig["idmarea"] = float(
    #     re.split("\s+", data[33][:-1])[0]
    # )  # orma de introducir la marea (idmarea): 1=senoide, 2=archivo
    # if dConfig["idmarea"] == 1:
    #     dConfig["amplitud"] = float(re.split("\s+", data[34][:-1])[0])
    #     dConfig["periodo"] = float(re.split("\s+", data[35][:-1])[0])
    #     dConfig["nivelref"] = float(re.split("\s+", data[36][:-1])[0])
    # else:
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


def calculate_Is(db, dConfig):
    """Compute t
    I1: permite calcular la diferencia entre los empujes de presión aplicados sobre las fronteras x1 y x2
    I2: permite calcular la fuerza de presión debido a la variación del ancho del canal
    TODO: En ambos se debe incluir la densidad rho(z)

    Args:
        db (_type_): _description_
        dConfig (_type_): _description_
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


def seccionhid(db, dbt, telaps, predictor=True):
    """Compute the hydraulic sections as function of A

    Args:
        db (_type_): _description_
        dbt (_type_): _description_
        telaps (_type_): _description_
        pred (bool, optional): for predictor (True) or corrector (False). Defaults to True.
    """

    vars = ["Rh", "B", "eta", "beta", "I1", "I2", "xl", "xr"]
    A = np.tile(dbt["A"][:, telaps].values, [db.sizes["z"], 1]).T
    # Compute the index where the given area is found
    indexes_db = np.argmin(np.abs(A - db["A"].values), axis=1, keepdims=True)
    # Obtain the proportional factor between areas
    # MCB - para asegurar que no se sale ningún valor por estar al límite de valores eta para la interpolación
    mask = indexes_db + 1 >= db.sizes["z"]
    if sum(mask) > 0:
        str_ = ""
        for val in dbt["A"][mask.T[0], telaps].values:
            str_ = str(val) + ", "
        str_ = str_[:-2]
        str_ += " m2"
        raise ValueError("Area outside the range " + str_)

    facpr = (
        dbt["A"][:, telaps]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    ) / (
        np.take_along_axis(db["A"][:].values, indexes_db + 1, axis=1)[:, 0]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    )

    # Calculate the mean value of variables
    for var in vars:
        if predictor:
            # Para calcular el predictor necesito calcular las variables iniciales, para
            # el corrector se tira de las predichas
            varname = var
        else:
            varname = var + "p"

        dbt[varname][:, telaps] = (
            np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            + (
                np.take_along_axis(db[var][:].values, indexes_db + 1, axis=1)[:, 0]
                - np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            )
            * facpr
        )

    return


def read_cross_sectional_geometry(config):
    df = pd.read_csv("cross_sections_" + config["sGeometryFilename"])
    config["nx"] = len(df["x"].unique())
    x = np.reshape(df["x"].values, [config["nx"], len(df["eta"].unique())])
    z = np.reshape(df["z"].values, np.shape(x))
    vars_ = ["eta", "A", "P", "B", "Rh", "sigma", "xl", "xr", "beta", "I1", "I2"]
    if isinstance(vars_, str):
        vars_ = [vars_]

    import xarray as xr

    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "z"], zeros.copy())
    db = xr.Dataset(
        dict_, coords={"xlocal": (["x", "z"], x), "zlocal": (["x", "z"], z)}
    )

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


def read_geometry_oldfiles(config):
    with open(config["geometryFilename"], "r") as file:
        data = file.readlines()

    nx, nz = [int(i) for i in re.split("\s+", data[5][:-1])]
    config["nx"], config["nz"] = nx, nz

    import pandas as pd

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


def initialize_netcdf(df, config):
    t = config["iTime"] * config["fTimeStep"]
    t, x = np.meshgrid(t, df["x"])
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
    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "t"], zeros.copy())
    dbt = xr.Dataset(
        dict_,
        coords={"xlocal": (["x", "t"], x), "time": (["x", "t"], t)},
    )
    # Create the attributes of the main variables
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


def read_hydro(config):
    #     nxhidros1 # número de puntos con hidrogramas
    #     nthidros # numero de instantes
    #     # x1
    #     # t1  Q1,1
    #     # t2  Q2,1
    #     #  2  0.00
    #     # Tini 0   	0
    #     # Ttot 2678400 0 --> Por lo tanto, los caudales son nulos
    #     # prhid, hidroaportes1
    #     # pasa de x a nodos con xanodos(x,np,nxhidros1,hidrosxs,hidronodos1) linea 283
    #     # Crea una variable con la localización de los hidrogramas -- nodos

    #     # Revisión de hidrogramas: si hay hidrogramas en nodos repetidos los suma
    # hidroaportes = np.zeros([config["nx"], config["nz"]])
    # hidroaportes1 = np.zeros([config["nx"], config["nz"]])
    # hidronodos, hidronodos1 = np.zeros(config["nx"]), np.zeros(config["nx"])
    # with open(config["hydroFilename"], "r") as file:
    #     data = file.readlines()

    # config["nxhidros"] = int(re.split("\s+", data[7][:-1])[0])  # Numero de hidrogramas
    # config["nthidros"] = int(re.split("\s+", data[8][:-1])[0])  # Tiempo del hidrograma
    # config["hidrosxs"] = float(
    #     re.split("\s+", data[8][:-1])[1]
    # )  # Se lee Localización del hidrograma
    # hidronodos[0] = hidronodos1[0]
    # for j in range(config["nthidros"]):
    #     hidroaportes[j, config["nxhidros"]] = hidroaportes1[j, 0]

    #     for i in range(config["nxhidros"]):
    #         if hidronodos1[i] == hidronodos1[i - 1]:
    #             for j in range(config["nthidros"]):
    #                 hidroaportes[j, config["nxhidros"]] = (
    #                     hidroaportes[j, config["nxhidros"]] + hidroaportes1[j, i]
    #                 )
    #         else:
    #             config["nxhidros"] += 1
    #             hidronodos[config["nxhidros"]] = hidronodos1[i]
    #             for j in range(config["nthidros"]):
    #                 hidroaportes[j, config["nxhidros"]] = hidroaportes1[j, i]

    # hidroaportes = read.csv("hydro.csv", sep=" ", header=None)
    hydro = pd.read_csv(
        config["sHydroFilename"], sep=" ", index_col=[0], header=0
    )  # TODO: incluir dConfig["dx"]
    hydro.columns = [int(col) for col in hydro.columns]
    return hydro


def fluvial_contribution(dbt, hydro, dConfig):
    for node in hydro.columns:
        dbt["q"][node, :] = (
            np.interp(dConfig["iTime"] * 3600, hydro.index.values, hydro[node])
            / dConfig["dx"]
        )
    # for ii in range(dConfig["nxhidros"]):
    #     qhidpr = hydro.values[:, 0]
    #     # Caudales por aportes de afluentes (x)
    #     Qaport[ii] = np.interp(telaps * 3600, hydro.index.values, qhidpr)

    #     if dConfig["nxhidros"] >= 2:
    #         for ii in range(2, dConfig["nxhidros"]):
    #             qpuntual[hydronodes[ii]] = Qaport[ii] / dConfig["dx"]

    # para contorno aguas arriba abierto:
    # if dConfig["iInitialBoundaryCondition"] == 2:
    #     qpuntual[0] = Qaport[0] / dConfig["dx"]
    return


def initialize_auxiliary(config, db, df):

    config["dx"] = (df.x.values[-1] - df.x.values[0]) / (
        config["nx"] - 1
    )  # a partir dle número total de perfiles

    if config["bSurfaceGradientMethod"] == 0:
        df["elev"] = np.zeros(config["nx"])

    if config["bSourceTermBalance"] == 1:
        df["zmedp"] = df["z"].diff(periods=-1) / config["dx"]
        df.loc[config["nx"] - 1, "zmedp"] = df.loc[config["nx"] - 2, "zmedp"]

        df["zmedc"] = -df["z"].diff() / config["dx"]
        df.loc[0, "zmedc"] = df.loc[1, "zmedc"]

    aux = dict()
    aux["Qmed"], aux["Amed"], aux["Rmed"] = (
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
    )
    # aux["Qmedc"], aux["Amedc"], aux["Rmedc"] = (
    #     np.zeros(config["nx"]),
    #     np.zeros(config["nx"]),
    #     np.zeros(config["nx"]),
    # )

    aux["U"], aux["F"], aux["Gv"] = (
        np.zeros([2, config["nx"]]),
        np.zeros([2, config["nx"]]),
        np.zeros([2, config["nx"]]),
    )

    aux["Up"], aux["Fp"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])
    aux["Uc"], aux["Gvp"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])
    aux["Un"], aux["D"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])

    calculate_Is(db, config)

    # Pendientes con diferencias centradas, adelantadas y atrasadas
    df["S0"] = 0.0
    df.loc[1 : config["nx"] - 2, "S0"] = (
        df.loc[: config["nx"] - 3, "z"].values - df.loc[2:, "z"].values
    ) / (2 * config["dx"])
    df.loc[config["nx"] - 1, "S0"] = (
        -(df.loc[config["nx"] - 1, "z"] - df.loc[config["nx"] - 2, "z"]) / config["dx"]
    )
    df.loc[0, "S0"] = (df.loc[0, "z"] - df.loc[1, "z"]) / config["dx"]
    # TODO: MCB - estoy trabajando los signos para permitir la realización de las raíces y para todas
    # aquellas variables direccionales. Además, debo limitar los valores mínimos para que no se dispare
    # Impongo el mínimo en un centímetro por cada dx
    mask = np.abs(df["S0"]) < 0.01 / config["dx"]
    df["signS0"] = np.sign(df["S0"])
    df["S0"] = np.abs(df["S0"])
    df.loc[mask, "S0"] = 0.01 / config["dx"]
    # -----------------------------------------------------------------------------------

    # Para condición de Murillo para dx
    df["nmann1"] = df["nmann"].values

    return db, df, aux


# def read_sedimentos():
#     # Archivo con las características de los sedimentos, unidades de longitud en metros
#     # 0.00032799899		Diámetro medio, Dmed
#     # 0.00043999989		D90
#     # 0.0004059913		D84
#     # 0.000319		D50
#     # 0.00024899864		D16
#     # 1.6305			Desviación estándar de lamuestra, sigmag
#     # 2.65 			Peso específico relativo, Ss
#     Diamx=D50*((rhos-1.)*g/(1.e-6)**2.0)**(1./3.)
# 	return


def fdry(dbt, telaps, var_=""):
    # TODO: cambiar la rutina para que modifique los datos en la variable original
    Adry = 1e-3
    Qdry = 1e-5  # para asegurar que U=Q/A ~ 0
    # Qseco = 0.001
    # ndy = 0
    mask = (dbt["A" + var_][:, telaps] < Adry) | (dbt["Q" + var_][:, telaps] < Qdry)
    dbt["A" + var_][mask, telaps] = Adry

    # dbt["Q"][:, telaps]
    # TODO: aunque crea un qseco después no lo usa.
    # TODO: la primera vez que se entra a fdry no tenemos Q
    dbt["Q" + var_][mask, telaps] = Qdry
    # dbt["Q"][mask, telaps] =
    # mask = A < Aseco
    # A2 = A
    # Q2 = Q
    # A2[mask] = Aseco
    # Q2[mask] = Q[mask]
    # vdy = np.unique(np.cumsum(mask))
    return mask


def limitador(dbt, df, dConfig, telaps):
    g = 9.81

    # umed, Amed, cmed = np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"])
    # a1med, a2med = np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"]),
    e1med, e2med, D = (
        np.ones([2, dConfig["nx"] - 1]),
        np.ones([2, dConfig["nx"] - 1]),
        np.zeros([2, dConfig["nx"] - 1]),
    )
    # alfa1med, alfa2med= np.zeros(dConfig["nx"]), np.zeros(dConfig["nx"])
    psi1med, psi2med = np.zeros(dConfig["nx"] - 1), np.zeros(dConfig["nx"] - 1)
    r1med, r2med = np.zeros(dConfig["nx"] - 1), np.zeros(dConfig["nx"] - 1)

    # for i in range(dConfig["nx"]):
    umed = (
        dbt["Q"][1:, telaps] / np.sqrt(dbt["A"][1:, telaps])
        + dbt["Q"][:-1, telaps] / np.sqrt(dbt["A"][:-1, telaps])
    ) / (np.sqrt(dbt["A"][1:, telaps]) + np.sqrt(dbt["A"][:-1, telaps]))
    Amed = (np.sqrt(dbt["A"][1:, telaps]) + np.sqrt(dbt["A"][:-1, telaps])) / 2.0
    cmed = (np.sqrt(dbt["c"][1:, telaps]) + np.sqrt(dbt["c"][:-1, telaps])) / 2.0
    a1med = umed + cmed
    a2med = umed - cmed
    e1med[1, :] = a1med
    e2med[1, :] = a2med
    if not dConfig["bSurfaceGradientMethod"]:  #         ! Sin Surface-Gradient Method
        alfa1med = (
            (dbt["Q"][1:, telaps] - dbt["Q"][:-1, telaps])
            + (-umed + cmed) * (Amed - dbt["A"][:-1, telaps])
        ) / (2.0 * cmed)
        alfa2med = -(
            (dbt["Q"][1:, telaps] - dbt["Q"][:-1, telaps])
            + (-umed - cmed) * (Amed - dbt["A"][:-1, telaps])
        ) / (2.0 * cmed)
    else:  #     ! Con Surface-Gradient Method
        alfa1med = (
            dbt["B"][:-1, telaps]
            * (
                (
                    dbt["Q"][1:, telaps] / dbt["B"][1:, telaps]
                    - dbt["Q"][:-1, telaps] / dbt["B"][:-1, telaps]
                )
                + (-umed.values + cmed.values)
                * (df["elev"][1:].values - df["elev"][:-1].values)
            )
            / (2.0 * cmed)
        )
        alfa2med = (
            -dbt["B"][:-1, telaps]
            * (
                (
                    dbt["Q"][1:, telaps] / dbt["B"][1:, telaps]
                    - dbt["Q"][:-1, telaps] / dbt["B"][:-1, telaps]
                )
                + (-umed.values - cmed.values)
                * (df["elev"][1:].values - df["elev"][:-1].values)
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
                a1med - (dbt["U"][:-1, telaps] + dbt["c"][:-1, telaps]),
                (dbt["U"][1:, telaps] + dbt["c"][1:, telaps]) - a1med,
            ],
            axis=0,
        )
        delta2 = np.max(
            [
                np.zeros(dConfig["nx"] - 1),
                a2med - (dbt["U"][:-1, telaps] - dbt["c"][:-1, telaps]),
                (dbt["U"][1:, telaps] - dbt["c"][1:, telaps]) - a2med,
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

    #! C�lculo de r
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
    # for i in range(dConfig["nx"]-1):
    #     j=i+1
    fac1pr = alfa1med * psi1med * (1 - dConfig["lambda"] * np.abs(a1med)) * (1 - fi1med)
    fac2pr = alfa2med * psi2med * (1 - dConfig["lambda"] * np.abs(a2med)) * (1 - fi2med)
    D[0, :-1] = 0.5 * (fac1pr[:-1] * e1med[0, :-1] + fac2pr[:-1] * e2med[0, :-1])
    D[1, :-1] = 0.5 * (fac1pr[:-1] * e1med[1, :-1] + fac2pr[:-1] * e2med[1, :-1])

    D[0, -1] = D[0, -2]
    D[1, -1] = D[1, -2]

    return D


def density(dbt, dConfig, sedProp, telaps, predictor=True):
    if predictor:
        var_ = ""
    else:
        var_ = "p"

    rhow = 1000 * (1 + dConfig["beta"] * dbt["S"][:, telaps])
    dbt["rho" + var_][:, telaps] = (
        rhow
        + (1 - rhow / 1000 / sedProp["rhos"])
        * dbt["Qt"][:, telaps]
        / (dbt["A" + var_][:, telaps] * dConfig["dx"])
        * dConfig["dtmin"]
    )
    return


def salinity(dbt, dConfig, telaps, var_="p"):
    """_summary_

    Args:
        dbt (_type_): _description_
        dConfig (_type_): _description_
        telaps (_type_): _description_
        var_: "p", "c", ""

    Returns:
        _type_: _description_
    """

    asdif = (
        dbt["A" + var_][2:, telaps].values * dbt["S"][2:, telaps].values
        + dbt["A" + var_][:-2, telaps].values * dbt["S"][:-2, telaps].values
    ) / 2
    asdif = np.hstack([asdif[0], asdif])
    asdif = np.hstack([asdif[0], asdif])

    dbt["S"][:, telaps] = asdif / dbt["A" + var_][:, telaps].values
    mask = dbt["S"][:, telaps] < 0
    dbt["S"][mask, telaps] = 0

    mask = dbt["S"][:, telaps] > 35
    dbt["S"][mask, telaps] = 35
    dbt["S"][-1, telaps] = 20
    return dbt


def salinity_gradient(dbt, dConfig, telaps):
    """_summary_

    Args:
        dbt (_type_): _description_
        dConfig (_type_): _description_
        telaps (_type_): _description_

    Returns:
        _type_: _description_
    """

    kasdif_forward = dbt["A"][1:, telaps].values * (
        dbt["S"][1:, telaps].values - dbt["S"][:-1, telaps].values
    )
    kasdif_forward = np.hstack([kasdif_forward, kasdif_forward[-1]])

    # Added for computing salinity - Diez Minguito et al 2013
    # Funciona
    # -----------
    # asdif_forward = dbt["A"][2:, telaps].values * (
    #     dbt["S"][2:, telaps].values - dbt["S"][1:-1, telaps].values
    # )
    # asdif_forward = np.hstack(
    #     [
    #         dbt["A"][1, telaps].values
    #         * (dbt["S"][1, telaps].values - dbt["S"][0, telaps].values),
    #         asdif_forward,
    #     ]
    # )
    # asdif_forward = np.hstack([asdif_forward[0], asdif_forward])
    # -------------------------
    # kasdif = 

    kasdif_backward = dbt["A"][:-1, telaps].values * (
        dbt["S"][1:, telaps].values - dbt["S"][:-1, telaps].values
    )
    kasdif_backward = np.hstack([kasdif_backward[0], kasdif_backward])

    # Funciona
    # -----------
    # asdif_backward = dbt["A"][:-2, telaps].values * (
    #     dbt["S"][1:-1, telaps].values - dbt["S"][:-2, telaps].values
    # )
    # asdif_backward = np.hstack(
    #     [
    #         asdif_backward,
    #         dbt["A"][-2, telaps].values
    #         * (dbt["S"][-1, telaps].values - dbt["S"][-2, telaps].values),
    #     ]
    # )
    # asdif_backward = np.hstack([asdif_backward[0], asdif_backward])
    # -----------

    ausdif = (
        (
            dbt["Q"][2:, telaps].values * dbt["S"][2:, telaps].values
            - dbt["Q"][:-2, telaps].values * dbt["S"][:-2, telaps].values
        )
        * dConfig["lambda"]
        / 2
    )
    ausdif = np.hstack(
        [
            # ausdif[0],
            (
                dbt["Q"][1, telaps].values * dbt["S"][1, telaps].values
                - dbt["Q"][0, telaps].values * dbt["S"][0, telaps].values
            )
            * dConfig["lambda"],  # dConfig["dx"]
            ausdif,
        ]
    )
    # TODO: forzar salinidad lado mar

    # Funciona bien
    # ---------------------------------------------------------
    # ausdif = np.hstack(
    #     [
    #         ausdif,
    #         (
    #             dbt["Q"][-1, telaps].values * dbt["S"][-1, telaps].values
    #             - dbt["Q"][-2, telaps].values * dbt["S"][-2, telaps].values
    #         )
    #         * dConfig["lambda"],
    #     ]
    # )
    # ---------------------------------
    ausdif = np.hstack(
        [
            ausdif,
            (
                dbt["Q"][-1, telaps].values
                * dbt["S"][-1, 0].values  # TODO: este valor debería ser el del océano
                - dbt["Q"][-2, telaps].values * dbt["S"][-2, telaps].values
            )
            * dConfig["lambda"],
        ]
    )

    # import matplotlib.pyplot as plt

    # plt.plot(
    #     -dConfig["kh"]
    #     * dConfig["lambda"] ** 2
    #     / dConfig["dtmin"]
    #     * (kasdif_forward - kasdif_backward),
    #     label="KAS",
    # )

    # plt.plot(ausdif, label="AUS")
    # plt.legend()
    # plt.show()

    ast = (
        dConfig["kh"]
        * dConfig["lambda"] ** 2
        / dConfig["dtmin"]
        * (kasdif_forward - kasdif_backward)
        - ausdif # El signo debe ser negativo
    )
    return ast


def vanRijn(sedProp, dbt, df, telaps):

    g = 9.81
    nu = 1.0e-6
    kappa = 0.4
    gamma = 1000.0

    #  ! Cuidado con el signo, regresar despu�s
    vel = np.abs(dbt["U"][:, telaps].values)
    #  !Sf=abs(vel)
    # direc = np.sign(vel)

    # Cálculo del transporte de fondo POR ARRASTRE: ######################
    c_1 = 18 * np.log((12 * dbt["Rh"][:, telaps].values) / (3 * sedProp["D90"]))
    u_star = ((g**0.5) / (c_1)) * dbt["U"][:, telaps].values

    sedProp["Diamx"] = sedProp["D50"] * (
        (sedProp["rhos"] - 1.0) * g / (1.0e-6) ** 2.0
    ) ** (1.0 / 3.0)
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

    dbt["Qb"][:, telaps] = (df["signS0"] * gb * dbt["B"][:, telaps]) / (
        sedProp["rhos"] * gamma
    )  # Volum�trico

    # -----------------------------------------------------------------------------------
    # Cálculo del transporte de fondo EN SUSPENSIÓN
    # -----------------------------------------------------------------------------------
    #! Altura del salto
    deltab = sedProp["Dmed"] * 0.3 * sedProp["Diamx"] ** 0.7 * T**0.5

    #! Velocidad al cortante
    Ux = (g * dbt["Rh"][:, telaps].values * df["Sf"].values) ** 0.5
    # mask = np.abs(Ux) < 1e-4  # TODO: poniendo limites a los ceros
    # Ux[mask] = 1e-4

    #! Rugosidad equivalente de Nikurazde
    # TODO: MCB - no estoy del todo convencido de que h deba ser dbt["eta"][:, telaps]
    # ks = (
    #     12.0 * dbt["eta"][:, telaps].values / np.exp(vel / (2.5 * vel))
    # )

    #! Nivel de referencia
    # a = ks  #         ! Elegir entre    OJO
    a = deltab  # uno y otro
    mask = a < 0.01 * dbt["eta"][:, telaps]
    a[mask] = 0.01 * dbt["eta"][mask, telaps].values  #! correci�n

    mask = a > 0.5 * dbt["eta"][:, telaps]
    a[mask] = 0.5 * dbt["eta"][mask, telaps].values

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
    Adim = a / dbt["eta"][:, telaps].values

    # ! Coeficiente adimensional F
    F = np.zeros(len(Adim))
    mask = np.abs(((1.0 - Adim) ** zp * (1.2 - zp))) >= 1e-4
    F[mask] = (Adim[mask] ** zp[mask] - Adim[mask] ** 1.2) / (
        (1.0 - Adim[mask]) ** zp[mask] * (1.2 - zp[mask])
    )  # TODO: verificar que la máscara es teóricamente aceptable o debo imponer un F máximo

    # Transporte del fondo en suspensión, calculado con esfuerzos
    gbs1 = Ca * dbt["eta"][:, telaps] * vel * F

    # Velocidad crítica
    # velc = (
    #     0.082516
    #     * sedProp["D50"] ** 0.1
    #     * np.log(4.0 * dbt["eta"][:, telaps].values / sedProp["D90"])
    # )
    # if sedProp["D50"] >= 0.0005:
    #     velc = (
    #         3.691500
    #         * sedProp["D50"] ** 0.6
    #         * np.log(4.0 * dbt["eta"][:, telaps].values / sedProp["D90"])
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
    dbt["Qs"][:, telaps] = (
        df["signS0"] * gbs / (sedProp["rhos"] * gamma) * dbt["B"][:, telaps].values
    )  # ! Volum�trico
    dbt["Qt"][:, telaps] = dbt["Qs"][:, telaps] + dbt["Qb"][:, telaps]
    mask = dbt["Qt"][:, telaps] < 1e-6
    dbt["Qt"][mask, telaps] = 0
    return


def gASterms(dbt, df, aux, config, telaps, predictor=True):
    # Variables de ecuaciones (vectores U, F y G) y cálculo del paso de tiempo (dt)
    if predictor:
        var_ = ""
    else:
        var_ = "p"
    if not config["bSourceTermBalance"]:  #      ! Sin balance de términos fuente
        # df["Sf"] = (
        #     dbt["Q"][:, telaps]
        #     * np.abs(dbt["Q"][:, telaps])
        #     * df["nmann1"] ** 2.0
        #     / (dbt["A"][:, telaps] ** 2.0 * dbt["Rh"][:, telaps] ** (4.0 / 3.0))
        # )
        aux["gAS0"] = g * dbt["A" + var_][:, telaps] * df["S0"][:] * df["signS0"]
        aux["gASf"] = g * dbt["A" + var_][:, telaps] * df["Sf"][:] * df["signSf"]
    else:  #   ! Con balance de términos fuente
        aux["Qmed"][:-1] = (
            dbt["Q" + var_][1:, telaps].values + dbt["Q" + var_][:-1, telaps].values
        ) / 2
        aux["Amed"][:-1] = (
            dbt["A" + var_][1:, telaps].values + dbt["A" + var_][:-1, telaps].values
        ) / 2
        aux["Rmed"][:-1] = (
            dbt["Rh" + var_][1:, telaps].values + dbt["Rh" + var_][:-1, telaps].values
        ) / 2

        aux["Qmed"][-1] = dbt["Q" + var_][-1, telaps].values
        aux["Amed"][-1] = dbt["A" + var_][-1, telaps].values
        aux["Rmed"][-1] = dbt["Rh" + var_][-1, telaps].values

        if predictor:
            aux["gAS0"] = g * aux["Amed"] * df["zmedp"].values * df["signS0"]
        else:
            aux["gAS0"] = g * aux["Amed"] * df["zmedc"].values * df["signS0"]
        aux["gASf"] = (
            g
            * df["nmann1"].values ** 2.0
            * aux["Qmed"]
            * np.abs(aux["Qmed"])
            / (aux["Amed"] * aux["Rmed"] ** (4.0 / 3.0))
        ) * df["signSf"]

    # if not dConfig["bSourceTermBalance"]:
    #     # Sin balance de términos de fuente
    #     # -------------------------------------------------------------------------------
    #     df["Sfp"] = (
    #         dbt["Qp"][:, telaps]
    #         * np.abs(dbt["Qp"][:, telaps])
    #         * df["nmann1"] ** 2.0
    #         / (dbt["Ap"][:, telaps] ** 2.0 * dbt["Rhp"][:, telaps] ** (4.0 / 3.0))
    #     )
    #     gAS0 = g * dbt["Ap"][:, telaps] * df["S0"]
    #     gASfp = g * dbt["Ap"][:, telaps] * df["Sfp"]
    # else:
    #     # Con balance de términos de fuente
    #     # -------------------------------------------------------------------------------
    #     aux["Qmedc"][:-1] = (dbt["Qp"][1:, telaps] + dbt["Qp"][:-1, telaps]) / 2
    #     aux["Amedc"][:-1] = (dbt["Ap"][1:, telaps] + dbt["Ap"][:-1, telaps]) / 2
    #     aux["Rmedc"][:-1] = (dbt["Rhp"][1:, telaps] + dbt["Rhp"][:-1, telaps]) / 2

    #     aux["Qmedc"][-1] = dbt["Qp"][-1, telaps]
    #     aux["Amedc"][-1] = dbt["Ap"][-1, telaps]
    #     aux["Rmedc"][-1] = dbt["Rhp"][-1, telaps]

    #     gAS0p = g * aux["Amedc"] * df["zmedc"]
    #     gASfp = (
    #         g
    #         * df["nmann1"] ** 2.0
    #         * aux["Qmedc"]
    #         * np.abs(aux["Qmedc"])
    #         / (aux["Amedc"] * aux["Rmedc"] ** (4.0 / 3.0))
    return aux


def Fterms(dbt, aux, dConfig, telaps, predictor=True):
    if predictor:
        var_ = ""
    else:
        var_ = "p"

    aux["F" + var_][0, :] = dbt["Q" + var_][:, telaps]
    aux["F" + var_][1, :] = (
        dbt["Q" + var_][:, telaps].values ** 2.0 / dbt["A" + var_][:, telaps].values
        + g * dbt["I1" + var_][:, telaps].values
    )
    if dConfig["bBeta"]:
        aux["F" + var_][1, :] = dbt["beta"][:, telaps].values * aux["F" + var_][1, :]

    return aux


def Gvterms(dbt, aux, telaps, predictor=True):
    if predictor:
        var_ = ""
    else:
        var_ = "p"
    aux["Gv" + var_][0, :] = dbt["q"][:, telaps].values  # qpuntual[:]
    aux["Gv" + var_][1, :] = (
        g * dbt["I2" + var_][:, telaps].values + aux["gAS0"] - aux["gASf"]
    )
    return aux


def conditionMurillo(dbt, df, config, telaps):
    if config["bMurilloCondition"]:
        cdx = 0.6
        nmr = 0
        nmur = cdx * np.sqrt(
            2 * dbt["Rh"][:, telaps].values ** (2.0 / 3.0) / (g * config["dx"])
        )
        mask = nmur < df["nmann"]
        df.loc[mask, "nmann1"] = nmur[mask]
        # facmr[mask] = nmur[mask] / df.loc[mask, "nmann"]
        dbt["I1"][mask, telaps] = (
            dbt["I1"][mask, telaps] * nmur[mask] / df.loc[mask, "nmann"]
        )
        vmr = mask
    else:
        vmr, nmur = None, None
    return vmr, nmur


def boundary_conditions(dbt, aux, config, telaps, var_="p"):

    # aux["U" + var_][0, 0] = aux["U" + var_][0, 1]  # Cond. de front. A inicial
    if config["iInitialBoundaryCondition"] == 1:  # Frontera inicial reflejante
        aux["U" + var_][1, 0] = dbt["q"][0, telaps]  # Qaport[0]
    elif config["iInitialBoundaryCondition"] == 2:  # Frontera inicial abierta
        aux["U" + var_][1, 0] = aux["U" + var_][1, 1]

    if config["iFinalBoundaryCondition"] == 1:  # then Cond. de front. Q final
        aux["U" + var_][0, -1] = aux["U" + var_][0, -2]  # Cond. de front. A final
        aux["U" + var_][1, -1] = aux["U" + var_][1, -2]  # Q abierto
    elif config["iFinalBoundaryCondition"] == 2:  # then
        # qff, qver = 0, 0  # Condición de caudal de fijo aguas arriba
        aux["U" + var_][1, -1] = (
            qff + qver
        )  # TODO: caudal que se vierta por un vertedero aguas abajo
        aux["U" + var_][1, -2] = qff + qver  # para que funcione mejor el esquema
    # elif dConfig["iFinalBoundaryCondition"] == 3:  # TODO: chequear esto..
    # aux["U" + var_][0, -1] = aux["mareateln"]  # Marea # TODO: ver que hago con la marea
    # aux["U" + var_][0, -2] = aux["mareatelnm"]
    # aux["U" + var_][0, -1] = aux["U" + var_][0, -2]

    # aux["Un"][0, 0] = aux["Un"][0, 1]  # Cond. de front. A inicial
    # aux["Un"][1, 0] = dbt["q"][0, telaps]  # Qaport[0]
    # if dConfig["iInitialBoundaryCondition"] == 1:  # Frontera inicial reflejante
    #     aux["Un"][1, 0] = dbt["q"][0, telaps]  # Qaport[0]
    # elif dConfig["iInitialBoundaryCondition"] == 2:  # Frontera inicial abierta
    #     aux["Un"][1, 0] = aux["Un"][1, 1]

    # if dConfig["iFinalBoundaryCondition"] == 1:  # Cond. de front. A final y Q final
    #     aux["Un"][0, -1] = aux["Un"][0, -2]  # A abierta
    #     aux["Un"][1, -1] = aux["Un"][1, -2]  # Q abierto
    # elif dConfig["iFinalBoundaryCondition"] == 2:
    #     aux["Un"][0, -1] = aux["Un"][0, -2]  # A fijo
    #     aux["Un"][1, -1] = aux["Un"][1, -2]  # TODO:
    #     aux["Un"][1, -1] = dbt["q"][0, telaps]  # qff + qver  # Q fijo
    #     aux["Un"][1, -2] = dbt["q"][
    #         0, telaps
    #     ]  # qff + qver  # truco para que funcione
    # elif dConfig["iFinalBoundaryCondition"] == 3:
    #     aux["Un"][1, -1] = aux["Un"][1, -2]
    #     aux["Un"][0, -1] = aux["mareateln"]
    #     aux["Un"][0, -2] = aux["mareatelnm"]
    return aux


def savefiles(dConfig, dbt, db, df):
    if dConfig["bLog"]:
        dbt.to_netcdf(dConfig["sOutputFilename"])
        db.to_netcdf(dConfig["auxiliary_file"])
        df.to_csv(dConfig["df_outputFilename"])
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
        df.to_csv(dConfig["df_outputFilename"])
    return


import scipy.special as sp


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
    k = np.sqrt(1 - 1j * r / (2 * np.pi))
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
    k = np.sqrt(1 - 1j * r / (2 * np.pi))
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


def model_eta_u(x, t, constituents, params):
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

    for const in constituents.keys():
        T0, A0, p0, r = (
            constituents[const]["T"],
            constituents[const]["A"],
            constituents[const]["p"],
            constituents[const]["r"],
        )
        g, T = 9.81, T0 * 3600
        lambda_ = np.sqrt(g * params["h0"]) * T

        A0, p0 = A0 / params["h0"], p0 * np.pi / 180.0
        A0 = A0 * np.exp(-1j * p0)

        for k in x:
            ys, yx, nu, m = param_(
                params["xref"],
                lambda_,
                k,
                params["x0"],
                params["xb"],
                params["B0"],
                params["Bb"],
            )
            t = eta_.index - eta_.index[0]
            t = np.array([(j.seconds / 3600 + j.days * 24) / T0 for j in t])
            eta_.loc[:, k] = eta_.loc[:, k] + eta(r, A0, ys, yx, t, nu) * params["h0"]
            u_.loc[:, k] = u_.loc[:, k] + u(r, A0, ys, yx, t, nu, k, m) * lambda_ / T
    return eta_, u_
