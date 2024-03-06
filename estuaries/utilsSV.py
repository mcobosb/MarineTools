import datetime
import os
import re
from configparser import ConfigParser

import numpy as np
from loguru import logger
from marinetools.utils import read


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
        + str(telap)
        + " - "
        + str(np.trunc(telap * dConfig["tesp"] / dConfig["tfin"] * 100)).zfill(4)
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
        "idi2": 1,  #
        "courant": 0.9,
        "dtst": 3600,  # Espaciamiento de los datos de salida, en segundos
        "idmarea": 1,  # lee archivo: 1, no hay marea en aguas abajo: 0
        "idfi": 4,  #  F�rmula de limitador de flujo fi 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
        "idpsi": 1,  # �rmula para Psi: 1 = Garc�a-Navarro, 2 = Tseng
    }

    if dConfig["idpsi"] == 1:
        dConfig["delta"] = 0.2

    with open("config.ini", "w") as configfile:
        dConfig.write(configfile)
    return


def read_oldfiles_v48(filename):
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
    dConfig["courant"] = float(re.split("\s+", data[9][:-1])[0])

    dConfig["outputFilename"] = re.split("\s+", data[11][:-1])[0]
    dConfig["tesp"] = float(re.split("\s+", data[12][:-1])[0])

    dConfig["bSedimentTransport"] = float(re.split("\s+", data[20][:-1])[0])
    dConfig["sedimentpropertiesFilename"] = re.split("\s+", data[21][:-1])[0]

    # ------------------------- Opciones del programa: -----------------------
    dConfig["idfl"] = float(
        re.split("\s+", data[23][:-1])[0]
    )  # Sin limitador de flujo (MacCormack)
    dConfig["idfl"] = 1  # TODO: probando, quitar
    dConfig["idsgm"] = float(
        re.split("\s+", data[24][:-1])[0]
    )  # Surface Gradient Method
    dConfig["idbst"] = float(
        re.split("\s+", data[25][:-1])[0]
    )  # Balance de terminos fuente
    dConfig["idbeta"] = float(
        re.split("\s+", data[26][:-1])[0]
    )  # uso de betas, 0=no, 1=s�
    dConfig["idi2"] = float(
        re.split("\s+", data[27][:-1])[0]
    )  # Para usar I2 de archivo o calcularlo
    dConfig["iddy"] = float(re.split("\s+", data[28][:-1])[0])  # Algoritmo lecho seco
    dConfig["idmr"] = float(
        re.split("\s+", data[29][:-1])[0]
    )  # verifica n de Manning para cumplir condici�n de Murillo para dx
    # ------------------------- Opciones de frontera final: ------------------
    dConfig["idqf"] = float(
        re.split("\s+", data[31][:-1])[0]
    )  # Ident. Q en frontera final: 1) Abierta, 2) Fijo, 3) Marea
    dConfig["idqf"] = 3  # TODO: probando la marea aguas abajo ,quitar
    if dConfig["idqf"] == 2:
        dConfig["qfijo"] = float(re.split("\s+", data[32][:-1])[0])

    dConfig["idmarea"] = float(
        re.split("\s+", data[33][:-1])[0]
    )  # orma de introducir la marea (idmarea): 1=senoide, 2=archivo
    if dConfig["idmarea"] == 1:
        dConfig["amplitud"] = float(re.split("\s+", data[34][:-1])[0])
        dConfig["periodo"] = float(re.split("\s+", data[35][:-1])[0])
        dConfig["nivelref"] = float(re.split("\s+", data[36][:-1])[0])
    else:
        dConfig["tideFilename"] = re.split("\s+", data[37][:-1])[0]

    dConfig["idtfaa"] = float(
        re.split("\s+", data[44][:-1])[0]
    )  # Tipo de frontera aguas arriba: 1=reflejante, 2=abierta
    dConfig["dtmin"] = 1e10

    dConfig["idfi"] = 4  # Fórmula de limitador de flujo fi
    # 1 = minmod, 2 = Roe's Superbee, 3 = Van Leer, 4 = Van Albada
    dConfig["idpsi"] = 2  # Fórmula para Psi: 1 = Garc�a-Navarro, 2 = Tseng

    if dConfig["idpsi"] == 1:
        dConfig["delta"] = 0.2  # Para calcular Psi con f�rmula Garc�a-Navarro,
        # "delta = small positive number betwen 0.1 and 0.3"
    return dConfig


def seccionhid(db, dbt, config, telaps, pred=True):
    # if not pred:
    #     vars = ["Rhp", "I1p", "Bp", "etap", "betap"]
    #     A = np.tile(dbt["Ap"][:, telaps].values, [db.sizes["z"], 1]).T
    #     indexes_db = np.argmin(np.abs(A - db["A"].values), axis=1, keepdims=True)
    #     facpr = (
    #         dbt["Ap"][:, telaps]
    #         - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    #     ) / (
    #         np.take_along_axis(db["A"][:].values, indexes_db + 1, axis=1)[:, 0]
    #         - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    #     )
    # else:
    vars = ["Rh", "I1", "B", "eta", "beta"]
    A = np.tile(dbt["A"][:, telaps].values, [db.sizes["z"], 1]).T
    indexes_db = np.argmin(np.abs(A - db["A"].values), axis=1, keepdims=True)
    facpr = (
        dbt["A"][:, telaps]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    ) / (
        np.take_along_axis(db["A"][:].values, indexes_db + 1, axis=1)[:, 0]
        - np.take_along_axis(db["A"][:].values, indexes_db, axis=1)[:, 0]
    )

    # np.take_along_axis(db["A"][:].values, mask, axis=1)
    # dicc = {}
    # dicc["x"] = np.arange(len(jj))
    # dicc["z"] = jj
    # np.stack([np.arange(len(jj)), jj])
    # jj = np.reshape(jj, [1, len(jj)])

    # for ii in range(config["nx"]):
    #     print(ii)
    #     jj = np.argmin(np.abs(dbt["A"][ii, telaps].values - db["A"][ii, :].values))
    #     jj = kk[ii]

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

    # dbt["I1"][:, telaps] = (
    #     np.take_along_axis(db["I1"][:].values, indexes_db, axis=1)[:, 0]
    #     + (
    #         np.take_along_axis(db["I1"][:].values, indexes_db + 1, axis=1)[:, 0]
    #         - np.take_along_axis(db["I1"][:].values, indexes_db, axis=1)[:, 0]
    #     )
    #     * facpr
    # )

    # dbt["B"][:, telaps] = (
    #     np.take_along_axis(db["B"][:].values, indexes_db, axis=1)[:, 0]
    #     + (
    #         np.take_along_axis(db["B"][:].values, indexes_db + 1, axis=1)[:, 0]
    #         - np.take_along_axis(db["B"][:].values, indexes_db, axis=1)[:, 0]
    #     )
    #     * facpr
    # )

    # dbt["eta"][:, telaps] = (
    #     np.take_along_axis(db["eta"][:].values, indexes_db, axis=1)[:, 0]
    #     + (
    #         np.take_along_axis(db["eta"][:].values, indexes_db + 1, axis=1)[:, 0]
    #         - np.take_along_axis(db["eta"][:].values, indexes_db, axis=1)[:, 0]
    #     )
    #     * facpr
    # )

    # dbt["beta"][:, telaps] = (
    #     np.take_along_axis(db["beta"][:].values, indexes_db, axis=1)[:, 0]
    #     + (
    #         np.take_along_axis(db["beta"][:].values, indexes_db + 1, axis=1)[:, 0]
    #         - np.take_along_axis(db["beta"][:].values, indexes_db, axis=1)[:, 0]
    #     )
    #     * facpr
    # )
    if config["idi2"] == 1:
        dbt["I2"][:, telaps] = (
            np.take_along_axis(db["I2"][:].values, indexes_db, axis=1)[:, 0]
            + (
                np.take_along_axis(db["I2"][:].values, indexes_db + 1, axis=1)[:, 0]
                - np.take_along_axis(db["I2"][:].values, indexes_db, axis=1)[:, 0]
            )
            * facpr
        )

    # if config["idi2"] == 2:
    #     dhdx, dI1dx = np.zeros(db.sizes["x"]), np.zeros(db.sizes["x"])
    #     dhdx = (dbt["eta"][1:, telaps] - dbt["eta"][:-1, telaps]) / (
    #         2 * config["dx"]
    #     )
    #     dI1dx = (dbt["I1"][1:, telaps] - dbt["I1"][:-1, telaps]) / (
    #         2 * config["dx"]
    #     )

    #     dhdx[0] = (dbt["eta"][1, telaps] - dbt["eta"][0, telaps]) / config["dx"]
    #     dI1dx[0] = (dbt["I1"][1, telaps] - dbt["I1"][0, telaps]) / config["dx"]

    #     dhdx[-1] = (dbt["eta"][-1, telaps] - dbt["eta"][-2, telaps]) / config["dx"]
    #     dI1dx[-1] = (dbt["I1"][-1, telaps] - dbt["I1"][-2, telaps]) / config["dx"]

    #     dbt["I2"][:, telaps] = dI1dx - db["A"][:, jj] * dhdx

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

    eta, A, P, B, Rh, sigma, I1, I2, xmen, xmas, beta = (
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
    while line < lines - 1:
        if is_ == nsx:
            line += 1
            ix += 1
            df.loc[ix, :] = [float(i) for i in re.split("\s+", data[line][:-1])]
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
                xmen[ix, is_],
                xmas[ix, is_],
                beta[ix, is_],
            ) = [float(i) for i in re.split("\s+", data[line][:-1])]
            is_ += 1
        line += 1

    x, z = np.meshgrid(eta[0, :], df["x"].values)
    vars_ = ["eta", "A", "P", "B", "Rh", "sigma", "I1", "I2", "xmen", "xmas", "beta"]
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
    db["xmen"][:, :] = xmen
    db["xmas"][:, :] = xmas
    db["beta"][:, :] = beta
    del eta, A, P, B, Rh, sigma, I1, I2, xmen, xmas, beta

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
        "vel",
        "c",
    ]
    dict_ = {}
    zeros = np.zeros(np.shape(x))
    for i in vars_:
        dict_[i] = (["x", "t"], zeros.copy())
    dbt = xr.Dataset(
        dict_, coords={"xlocal": (["x", "t"], x), "tlocal": (["x", "t"], t)}
    )

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
        elev = np.zeros(config["nx"])

    if config["idbst"] == 1:
        df["zmedp"] = df["z"].diff(periods=-1)
        df.loc[config["nx"], "zmedp"] = df.loc[config["nx"] - 1, "zmedp"]

        df["zmedc"] = df["z"].diff()
        df.loc[0, "zmedc"] = df.loc[1, "zmedc"]

    return elev, df


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
                + (-umed + cmed) * (elev[1:] - elev[:-1])
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
                + (-umed - cmed) * (elev[1:] - elev[:-1])
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
                a1med - (dbt["vel"][:-1, telaps] + dbt["c"][:-1, telaps]),
                (dbt["vel"][1:, telaps] + dbt["c"][1:, telaps]) - a1med,
            ],
            axis=0,
        )
        delta2 = np.max(
            [
                np.zeros(dConfig["nx"] - 1),
                a2med - (dbt["vel"][:-1, telaps] - dbt["c"][:-1, telaps]),
                (dbt["vel"][1:, telaps] - dbt["c"][1:, telaps]) - a2med,
            ],
            axis=0,
        )

    mask = abs(a1med) >= delta1
    psi1med[mask] = abs(a1med[mask])
    mask = abs(a1med) <= delta1
    psi1med[mask] = delta1[mask]

    mask = abs(a2med) >= delta2
    psi2med[mask] = abs(a2med[mask])
    mask = abs(a2med) <= delta2
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

    # ! C�lculo de fi
    fi1med = np.max(
        [np.zeros(len(r1med)), np.min([np.ones(len(r1med)), r1med], axis=0)], axis=0
    )
    fi2med = np.max(
        [np.zeros(len(r2med)), np.min([np.ones(len(r2med)), r2med], axis=0)], axis=0
    )

    # for i in range(dConfig["nx"]-1):
    if dConfig["idfi"] == 1:  #               ! MinMod
        fi1med = np.max(0.0, np.min(1.0, r1med))
        fi2med = np.max(0.0, np.min(1.0, r2med))
    elif dConfig["idfi"] == 2:  #          ! Roe's Superbee
        fi1med = np.max(0.0, np.min(2 * r1med, 1.0), np.min(r1med, 2.0))
        fi2med = np.max(0.0, np.min(2 * r2med, 1.0), np.min(r2med, 2.0))
    elif dConfig["idfi"] == 2:  #           ! Van Leer
        fi1med = (abs(r1med) + r1med) / (1 + abs(r1med))
        fi2med = (abs(r2med) + r2med) / (1 + abs(r2med))
    elif dConfig["idfi"] == 4:  #         ! Van Albada
        fi1med = (r1med**2.0 + r1med) / (1 + r1med**2.0)
        fi2med = (r2med**2.0 + r2med) / (1 + r2med**2.0)

    # ! C�lculo de D
    # for i in range(dConfig["nx"]-1):
    #     j=i+1
    fac1pr = alfa1med * psi1med * (1 - lambda_ * abs(a1med)) * (1 - fi1med)
    fac2pr = alfa2med * psi2med * (1 - lambda_ * abs(a2med)) * (1 - fi2med)
    D[0, :-1] = 0.5 * (fac1pr[:-1] * e1med[0, :-1] + fac2pr[:-1] * e2med[0, :-1])
    D[1, :-1] = 0.5 * (fac1pr[:-1] * e1med[1, :-1] + fac2pr[:-1] * e2med[1, :-1])

    # D[0, 0] = D[0, 1]
    # D[1, 0] = D[1, 1]
    D[0, -1] = D[0, -2]
    D[1, -1] = D[1, -2]

    return D
