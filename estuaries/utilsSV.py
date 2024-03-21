import datetime
import os
import re
import time
from configparser import ConfigParser

import numpy as np
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
            np.round(telap * dConfig["tesp"] / dConfig["tfin"] * 100, decimals=2)
        )
        + " % completed"
    )


def createConfigFile():
    dConfig = ConfigParser()

    dConfig["DEFAULT"] = {
        "idci": 1,  # Identificador de condición inicial: 1=de archivo, 2= Asignado eta y Q=0
        "idtfaa": 2,  # Frontera inicial: 1= reflejante o 2=abierta
        "idmr": 1,  # Algoritmo condicion dx-n
        "idsgm": 0,  # Surface Gradient Method
        "idfl": 0,  # Con esquema McCormack (sin limitador), 1: Con esquema TVD-McCormack (con limitador)
        "idbst": 1,  # Balance de terminos fuente'
        "iddy": 0,  # Algoritmo lecho seco
        "courant": 0.9,  # Número de Courant
        "dtst": 3600,  # Espaciamiento de los datos de salida, en segundos
        "idfi": 4,  # Fórmula de limitador de flujo fi 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
        "idpsi": 1,  # Fórmula para Psi: 1 = García-Navarro, 2 = Tseng
    }
    #     "geometryFilename"
    #    "hydroFilename"

    if dConfig["idpsi"] == 1:
        dConfig["delta"] = 0.2

    with open("config.ini", "w") as configfile:
        dConfig.write(configfile)
    return


def initial_condition(dbt, db, df, dConfig):
    """_summary_

    Args:
        dbt (_type_): _description_
        db (_type_): _description_
        dConfig (_type_): _description_

    Returns:
        _type_: _description_
    """

    if dConfig["idci"] == 2:
        logger.info(
            "Computing initial condition for Q = " + str(dConfig["qci"]) + " m3/s"
        )
        dbt["Q"][:, 0] = dConfig["qci"]
        for i in range(dConfig["nx"]):
            facman = dbt["Q"][i, 0] * df["nmann"][i] / np.sqrt(df["S0"][i])
            facsec = db["A"][i, :] * db["Rh"][i, :] ** (2.0 / 3.0)
            araux = db["A"][i, :]

            dbt["A"][i, 0] = np.interp(facman, facsec, araux)

    elif dConfig["idci"] == 3:
        logger.info(
            "Computing initial condition for Q = 0 and eta = "
            + str(dConfig["etaci"])
            + " m"
        )
        dbt["Q"][
            :, 0
        ] = 40  # dConfig["qci"] # TODO: En el GRE el nivel es muy superior al caudal, meter ambas opciones a la vez
        for i in range(dConfig["nx"]):
            # areava = db["A"][i, :]
            # etava = db["eta"][i, :]
            # etavv = dConfig["etaci"] - df["z"][i]
            dbt["A"][i, 0] = np.interp(
                dConfig["etaci"] - df["z"][i], db["eta"][i, :], db["A"][i, :]
            )

    time.sleep(1)
    return dbt


def read_oldfiles(filename, version_="v48"):
    dConfig = {}
    with open(filename, "r") as file:
        data = file.readlines()

    dConfig["geometryFilename"] = re.split("\s+", data[2][:-1])[0]
    dConfig["hydroFilename"] = re.split("\s+", data[3][:-1])[0]
    dConfig["idci"] = float(
        re.split("\s+", data[4][:-1])[0]
    )  # Identificador de condición inicial: 1=de archivo, 2=Manning, 3= Asignado eta y Q=0

    if dConfig["idci"] == 1:  # lee de archivo
        dConfig["ciFilename"] = re.split("\s+", data[5][:-1])[0]
    elif dConfig["idci"] == 2:  # Q inicial
        dConfig["qci"] = float(re.split("\s+", data[6][:-1])[0])
    else:
        dConfig["etaci"] = float(re.split("\s+", data[7][:-1])[0])

    dConfig["ioLocations"] = re.split("\s+", data[8][:-1])[
        0
    ]  # Archivo con coordenadas X de los puntos para guardar series temporales
    if version_ == "v44":
        k = -1
        dConfig["courant"] = 0.3
    if version_ == "v48":
        k = 0
        dConfig["courant"] = float(re.split("\s+", data[9][:-1])[0])

    dConfig["outputFilename"] = re.split("\s+", data[11 + k][:-1])[0]
    dConfig["tesp"] = float(re.split("\s+", data[12 + k][:-1])[0])

    dConfig["bSedimentTransport"] = float(re.split("\s+", data[20 + k][:-1])[0])
    dConfig["sedimentpropertiesFilename"] = re.split("\s+", data[21 + k][:-1])[0]

    # ------------------------- Opciones del programa: -----------------------
    dConfig["idfl"] = float(
        re.split("\s+", data[23 + k][:-1])[0]
    )  # Sin limitador de flujo (MacCormack)
    dConfig["idfl"] = 1  # TODO: probando, quitar
    dConfig["idsgm"] = float(
        re.split("\s+", data[24 + k][:-1])[0]
    )  # Surface Gradient Method
    dConfig["idbst"] = float(
        re.split("\s+", data[25 + k][:-1])[0]
    )  # Balance de terminos fuente
    dConfig["idbeta"] = float(
        re.split("\s+", data[26 + k][:-1])[0]
    )  # uso de betas, 0=no, 1=s�
    # dConfig["idi2"] = float(
    #     re.split("\s+", data[27][:-1])[0]
    # )  # Para usar I2 de archivo o calcularlo
    dConfig["iddy"] = float(
        re.split("\s+", data[28 + k][:-1])[0]
    )  # Algoritmo lecho seco
    dConfig["idmr"] = float(
        re.split("\s+", data[29 + k][:-1])[0]
    )  # verifica n de Manning para cumplir condici�n de Murillo para dx
    # ------------------------- Opciones de frontera final: ------------------
    dConfig["idqf"] = float(
        re.split("\s+", data[31 + k][:-1])[0]
    )  # Ident. Q en frontera final: 1) Abierta, 2) Fijo, 3) Marea
    dConfig["idqf"] = 3  # TODO: probando la marea aguas abajo ,quitar
    if dConfig["idqf"] == 2:
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

    dConfig["idtfaa"] = float(
        re.split("\s+", data[44 + k][:-1])[0]
    )  # Tipo de frontera aguas arriba: 1=reflejante, 2=abierta
    dConfig["dtmin"] = 1e10

    dConfig["idfi"] = 4  # Fórmula de limitador de flujo fi
    # 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
    dConfig["idpsi"] = 2  # Fórmula para Psi: 1 = Garc�a-Navarro, 2 = Tseng

    if dConfig["idpsi"] == 1:
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


def seccionhid(db, dbt, telaps, pred=True):
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
    indexes_db[mask] -= 1
    facpr = (
        dbt["A"][:, telaps]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    ) / (
        np.take_along_axis(db["A"][:].values, indexes_db + 1, axis=1)[:, 0]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    )

    # Calculate the mean value of variables
    for var in vars:
        if not pred:
            varname = var + "p"
        else:
            varname = var

        dbt[varname][:, telaps] = (
            np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            + (
                np.take_along_axis(db[var][:].values, indexes_db + 1, axis=1)[:, 0]
                - np.take_along_axis(db[var][:].values, indexes_db, axis=1)[:, 0]
            )
            * facpr
        )

    return


def read_geometry(config):
    with open(config["geometryFilename"], "r") as file:
        data = file.readlines()

    nx, nsx = [int(i) for i in re.split("\s+", data[5][:-1])]
    config["nx"], config["nsx"] = nx, nsx

    import pandas as pd

    df = pd.DataFrame(
        index=np.arange(nx),
        columns=["x", "z", "nmann", "xutm", "yutm", "AngMd", "AngMi"],
    )

    eta, A, P, B, Rh, sigma, I1, I2, xleft, xright, beta = (
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
        np.zeros([nx, nsx], dtype=np.float32),
    )
    lines = len(data)
    line = 5
    ix = -1
    is_ = nsx
    while line < lines:  # lines - 1 in Adra
        if is_ == nsx:
            line += 1
            ix += 1
            df.loc[ix, :] = [
                float(i) for i in re.split("\s+", data[line][1:-1])
            ]  # :-1 en Adra, v48
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

    import xarray as xr

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

    t = np.arange(0, config["tfin"] + config["tesp"], config["tesp"])
    t, x = np.meshgrid(t, df["x"])
    vars_ = [
        "A",
        "Ap",
        "Ac",
        "Q",
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

    return df, db, dbt


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
    # hidroaportes = np.zeros([config["nx"], config["nsx"]])
    # hidroaportes1 = np.zeros([config["nx"], config["nsx"]])
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
    import pandas as pd

    hidroaportes = pd.read_csv("hydro.csv", sep=" ", index_col=[0], header=None)
    hidroaportes.columns = ["data"]
    hidroaportes.index.name = "date"
    return hidroaportes / 2


def initialize(config, df):
    if config["idsgm"] == 0:
        df["elev"] = np.zeros(config["nx"])

    if config["idbst"] == 1:
        df["zmedp"] = df["z"].diff(periods=-1)
        df.loc[config["nx"] - 1, "zmedp"] = df.loc[config["nx"] - 2, "zmedp"]

        df["zmedc"] = df["z"].diff()
        df.loc[0, "zmedc"] = df.loc[1, "zmedc"]

    aux = dict()
    aux["Qmedp"], aux["Amedp"], aux["Rmedp"] = (
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
    )
    aux["Qmedc"], aux["Amedc"], aux["Rmedc"] = (
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
        np.zeros(config["nx"]),
    )

    aux["U"], aux["F"], aux["Gv"] = (
        np.zeros([2, config["nx"]]),
        np.zeros([2, config["nx"]]),
        np.zeros([2, config["nx"]]),
    )

    aux["Up"], aux["Fp"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])
    aux["Uc"], aux["Gvp"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])
    aux["Unext"], aux["D"] = np.zeros([2, config["nx"]]), np.zeros([2, config["nx"]])

    return df, aux


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


def fdry(dbt, db, telaps, var_):
    # TODO: cambiar la rutina para que modifique los datos en la variable original
    Adry = 1e-8
    Qdry = 1e-3
    # Qseco = 0.001
    # ndy = 0
    mask = dbt["A"][:, telaps] < Adry
    dbt["A"][mask, telaps] = Adry

    # dbt["Q"][:, telaps]
    # TODO: aunque crea un qseco después no lo usa.
    # TODO: la primera vez que se entra a fdry no tenemos Q
    dbt["Q"][mask, telaps] = Qdry
    # dbt["Q"][mask, telaps] =
    # mask = A < Aseco
    # A2 = A
    # Q2 = Q
    # A2[mask] = Aseco
    # Q2[mask] = Q[mask]
    # vdy = np.unique(np.cumsum(mask))
    return dbt


def limitador(dbt, dConfig, telaps, lambda_):
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
    if dConfig["idsgm"] == 0:  #         ! Sin Surface-Gradient Method
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
                + (-umed + cmed) * (df["elev"][1:] - df["elev"][:-1])
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
                + (-umed - cmed) * (df["elev"][1:] - df["elev"][:-1])
            )
            / (2.0 * cmed)
        )

    # ! C�lculo de Psi, seleccionar opci�n al inicio de la subrutina
    if dConfig["idpsi"] == 1:  #                  ! Garc�a-Navarro
        delta1 = dConfig["delta"]
        delta2 = dConfig["delta"]
    elif dConfig["idpsi"] == 2:  #              ! Tseng
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

    if dConfig["idfi"] == 1:  #               ! MinMod
        fi1med = np.max(0.0, np.min(1.0, r1med))
        fi2med = np.max(0.0, np.min(1.0, r2med))
    elif dConfig["idfi"] == 2:  #          ! Roe's Superbee
        fi1med = np.max(0.0, np.min(2 * r1med, 1.0), np.min(r1med, 2.0))
        fi2med = np.max(0.0, np.min(2 * r2med, 1.0), np.min(r2med, 2.0))
    elif dConfig["idfi"] == 2:  #           ! Van Leer
        fi1med = (np.abs(r1med) + r1med) / (1 + np.abs(r1med))
        fi2med = (np.abs(r2med) + r2med) / (1 + np.abs(r2med))
    elif dConfig["idfi"] == 4:  #         ! Van Albada
        fi1med = (r1med**2.0 + r1med) / (1 + r1med**2.0)
        fi2med = (r2med**2.0 + r2med) / (1 + r2med**2.0)

    # ! C�lculo de D
    # for i in range(dConfig["nx"]-1):
    #     j=i+1
    fac1pr = alfa1med * psi1med * (1 - lambda_ * np.abs(a1med)) * (1 - fi1med)
    fac2pr = alfa2med * psi2med * (1 - lambda_ * np.abs(a2med)) * (1 - fi2med)
    D[0, :-1] = 0.5 * (fac1pr[:-1] * e1med[0, :-1] + fac2pr[:-1] * e2med[0, :-1])
    D[1, :-1] = 0.5 * (fac1pr[:-1] * e1med[1, :-1] + fac2pr[:-1] * e2med[1, :-1])

    D[0, -1] = D[0, -2]
    D[1, -1] = D[1, -2]

    return D


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

    # Added for computing salinity - Diez Minguito et al 2013
    asdif_forward = dbt["A"][1:, telaps].values * (
        dbt["S"][1:, telaps] - dbt["S"][:-1, telaps]
    )
    asdif_forward = np.hstack([asdif_forward[0], asdif_forward])
    # asdif_forward[-1] = 0

    asdif_backward = dbt["A"][:-1, telaps].values * (
        dbt["S"][:-1, telaps] - dbt["S"][1:, telaps]
    )
    asdif_backward = np.hstack([asdif_backward, asdif_backward[-1]])

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
            ausdif[0],
            # (
            #     dbt["Q"][1, telaps].values * dbt["S"][1, telaps]
            #     - dbt["Q"][0, telaps].values * dbt["S"][0, telaps]
            # )
            # * lambda_,  # dConfig["dx"]
            ausdif,
        ]
    )
    ausdif = np.hstack(
        [
            ausdif,
            dbt["Q"][-1, telaps].values * dbt["S"][-1, telaps] * dConfig["lambda"] / 2,
        ]
    )
    ast = (
        dConfig["kh"]
        * dConfig["lambda"] ** 2
        / dConfig["dtmin"]
        * (asdif_forward - asdif_backward)
        - ausdif
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
    direc = np.sign(vel)

    # Cálculo del transporte de fondo POR ARRASTRE: ######################
    c_1 = 18 * np.log((12 * dbt["Rh"][:, telaps].values) / (3 * sedProp["D90"]))
    u_star = ((g**0.5) / (c_1)) * vel

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

    dbt["Qb"][:, telaps] = (direc * gb * dbt["B"][:, telaps]) / (
        sedProp["rhos"] * gamma
    )  # Volum�trico

    # -----------------------------------------------------------------------------------
    # Cálculo del transporte de fondo EN SUSPENSIÓN
    # -----------------------------------------------------------------------------------
    #! Altura del salto
    deltab = sedProp["Dmed"] * 0.3 * sedProp["Diamx"] ** 0.7 * T**0.5

    #! Velocidad al cortante
    Ux = (g * dbt["Rh"][:, telaps].values * df["Sf"].values) ** 0.5

    #! Rugosidad equivalente de Nikurazde
    # TODO: MCB - no estoy del todo convencido de que h deba ser dbt["eta"][:, telaps]
    ks = 12.0 * dbt["eta"][:, telaps] / np.exp(vel / (2.5 * vel))

    #! Nivel de referencia
    a = ks  #         ! Elegir entre    OJO
    #!a=deltab    ! uno y otro
    mask = a < 0.01 * dbt["eta"][:, telaps]
    a[mask] = 0.01 * dbt["eta"][mask, telaps]  #! correci�n

    mask = a > 0.5 * dbt["eta"][:, telaps]
    a[mask] = 0.5 * dbt["eta"][mask, telaps]

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
    beta = 1.0 + 2.0 * (ws / Ux) ** 2.0
    mask = beta >= 2
    beta[mask] = 2.0

    # ! Par�metro de suspensi�n de Rouse
    zr = ws / (beta * kappa * Ux)

    # ! M�xima concentraci�n posible en el nivel de referencia
    C0 = 0.65 * gamma * sedProp["rhos"]

    # ! Factor global de correci�n
    psi = 2.5 * (ws / Ux) ** 0.8 * (Ca / C0) ** 0.4

    # ! Modificaci�n de la z de Rouse
    zp = zr + psi

    # ! Coeficiente adimensional A (Adim)
    Adim = a / dbt["eta"][:, telaps]

    # ! Coeficiente adimensional F
    F = (Adim**zp - Adim**1.2) / ((1.0 - Adim) ** zp * (1.2 - zp))

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
        direc * gbs / (sedProp["rhos"] * gamma) * dbt["B"][:, telaps].values
    )  # ! Volum�trico
    dbt["Qt"][:, telaps] = dbt["Qs"][:, telaps] + dbt["Qb"][:, telaps]
    return dbt
