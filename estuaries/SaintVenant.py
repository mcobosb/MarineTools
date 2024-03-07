import datetime
import os
import re
import sys
from configparser import ConfigParser

import numpy as np
import xarray as xr
from loguru import logger
from marinetools.processes import compute

import utilsSV

# config = ConfigParser()
# dConfig = config.read("config.ini")

initial = datetime.datetime.now()

dConfig = utilsSV.read_oldfiles_v48("datosguadv48.txt")


dConfig["tfin"] = 3600 * 24 * 31
g = 9.81

df, db, dbt = utilsSV.read_geometry(dConfig)


dx = (df.x.values[-1] - df.x.values[0]) / (
    dConfig["nx"] - 1
)  # a partir dle número total de perfiles
# Pendientes
df["S0"] = df["z"].diff(periods=-1) / dx
df.loc[dConfig["nx"] - 1, "S0"] = (
    -(df.loc[dConfig["nx"] - 1, "z"] - df.loc[dConfig["nx"] - 2, "z"]) / dx
)
# S0[0] = -(z[1]-z[0])/dx
# df.loc["S0", 1:-1] = (df["z"][:-2] - z[2:])/(2*dx)
# S0[-1] =-(z[-1]-z[-2])/dx

hidroaportes = utilsSV.read_hydro(dConfig)

# def read_sedimentos():
#     # Archivo con las características de los sedimentos, unidades de longitud en metros
#     # 0.00032799899		Diámetro medio, Dmed
#     # 0.00043999989		D90
#     # 0.0004059913		D84
#     # 0.000319		D50
#     # 0.00024899864		D16
#     # 1.6305			Desviación estándar de lamuestra, sigmag
#     # 2.65 			Peso específico relativo, Ss
#     Diamx=D50*((denss-1.)*g/(1.e-6)**2.0)**(1./3.)
# 	return


# armarul = db["A"][-1, :] # área última
# etmarul = db["eta"][-1, :] # eta última
# armarpu = db["A"][-2, :] # área penúltima
# etmarpu = db["eta"][-2, :] # eta penúltima
zmedp = np.zeros(dConfig["nx"])
zmedc = np.zeros(dConfig["nx"])


# Condition inicial calculada
# ------------------------------------
# facsec = np.zeros(dConfig["nsx"])
# araux = np.zeros(dConfig["nsx"])


if dConfig["idci"] == 2:
    logger.info("Computing initial condition for Q = " + str(dConfig["qci"]) + " m3/s")
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
    dbt["Q"][:, 0] = 0.0
    for i in range(dConfig["nx"]):
        # areava = db["A"][i, :]
        # etava = db["eta"][i, :]
        # etavv = dConfig["etaci"] - df["z"][i]
        dbt["A"][i, 0] = np.interp(
            dConfig["etaci"] - df["z"][i], db["eta"][i, :], db["A"][i, :]
        )


telaps = 0  # no tiene por que ser del hidrograma, ya que puede haber caudal exterior#tprhid[0] # tiempo inicial de c�lculo = tiempo inicial de hidrograma de entrada
it = 0

# Calculo
# ! Interpola los caudales de entrada al tiempo de c�lculo
hidronodos = np.zeros(dConfig["nsx"])
dConfig["nxhidros"] = 1
# dConfig["tfinal"] = 1000
tprhid = 0
qhidpr = np.zeros(dConfig["nx"])
Qaport = np.zeros(dConfig["nx"])
qpuntual = np.zeros(dConfig["nx"])

# Qprueba = np.linspace(0, 200, 25)


U, F, Gv = (
    np.zeros([2, dConfig["nx"]]),
    np.zeros([2, dConfig["nx"]]),
    np.zeros([2, dConfig["nx"]]),
)

Up, Fp = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])
Uc, Gvp = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])
Unext, D = np.zeros([2, dConfig["nx"]]), np.zeros([2, dConfig["nx"]])


U[0, :] = dbt["A"][:, 0]
U[1, :] = dbt["Q"][:, 0]
F[0, :] = dbt["Q"][:, 0]
# ---------------------------------------------------------------------------------------
# Initialize the loop in time
# ---------------------------------------------------------------------------------------
while telaps < dConfig["tfin"] / 3600 + 1:  # 500:
    utilsSV.clock(initial, telaps, dConfig)
    for ii in range(dConfig["nxhidros"]):
        qhidpr = hidroaportes.values[:, 0]
        # Caudales por aportes de afluentes (x)
        Qaport[ii] = np.interp(telaps * 3600, hidroaportes.index.values, qhidpr)

        # Qhidit = Qaport[0]
        if dConfig["nxhidros"] >= 2:
            for ii in range(2, dConfig["nxhidros"]):
                qpuntual[hidronodos[ii]] = Qaport[ii] / dx
    #
    # ! para contorno aguas arriba abierto:
    if dConfig["idtfaa"] == 2:
        qpuntual[0] = Qaport[0] / dx

    # Pasa  de elevación de marea a area
    # Solo voy a dar la opción de que insertes la marea o sea cero
    if dConfig["idmarea"] == 0:
        mareateln, mareatelnm = (
            -df.loc[dConfig["nx"] - 1, "z"],
            -df.loc[dConfig["nx"] - 2, "z"],
        )
    else:
        # TODO: crear archivo con la marea
        data = np.loadtxt(dConfig["tideFilename"])  # tiempo y nivel
        tmarfile, emarfile = data[:, 0], data[:, 1]
        mareaelu = np.interp(telaps, tmarfile, emarfile)
        # le sumamos al nivel de marea el valor de la profundidad
        mareateln = np.interp(
            -df.loc[dConfig["nx"] - 1, "z"] + mareaelu, db["eta"][-1, :], db["A"][-1, :]
        )
        mareatelnm = np.interp(
            -df.loc[dConfig["nx"] - 2, "z"] + mareaelu, db["eta"][-2, :], db["A"][-2, :]
        )

    # Para opci�n de revisar lecho seco
    if dConfig["iddy"]:
        dbt = utilsSV.fdry(dbt, db, telaps, "A")
        # dbt = utilsSV.fdry(dbt, db, telaps, "Q")
        # A[:] = A2[iy]
        # Q[:] = Q2[iy]

    #  C�culo de secciones hidr�licas (variables hidr�ulicas de la secci�n en funci�n de A)
    # logger.info(
    #     "Cálculo de secciones hidrálicas (variables hidráulicas de la sección en función de A)"
    # )
    utilsSV.seccionhid(db, dbt, dConfig, telaps)

    # Opci�n dx-n, verifica n de Manning para cumplir condici�n de Murillo para dx
    facmr, ymr, vmr = (
        np.zeros(dConfig["nx"]),
        np.zeros(dConfig["nx"]),
        np.zeros(dConfig["nx"]),
    )
    df["nmann1"] = df["nmann"]
    if dConfig["idmr"]:
        cdx = 0.6
        nmr = 0
        nmur = cdx * np.sqrt(2 * dbt["Rh"][:, telaps].values ** (2.0 / 3.0) / (g * dx))
        mask = nmur < df["nmann"]
        df.loc[mask, "nmann1"] = nmur[mask]
        facmr[mask] = nmur[mask] / df.loc[mask, "nmann"]
        dbt["I1"][mask, telaps] = dbt["I1"][mask, telaps] * facmr[mask]
        # vmr[mask] = mask[mask].index
        vmr = mask
    #     for iy in range(dConfig["nx"]):
    #         nmur = cdx * np.sqrt(2 * dbt["Rh"][:, telaps] ** (2.0 / 3.0) / (g * dx))
    #         if nmur < dbt["nmann"][iy, telaps]:
    #             nmr = nmr + 1
    #             dbt["nmann1"][iy, telaps] = nmur
    #             facmr[iy] = nmur / dbt["nmann"][iy, telaps]
    #             dbt["I1"][iy, telaps] = dbt["I1"][iy, telaps] * facmr[iy]
    #             vmr[nmr] = iy
    #         else:
    #             dbt["nmann1"][iy, telaps] = dbt["nmann"][iy, telaps]

    # else:
    #     dbt["nmann1"][:, telaps] = dbt["nmann"][:, telaps]

    # Variables de ecuaciones (vectores U, F y G) y c�lculo del paso de tiempo (dt)
    # for ii in range(dConfig["nx"]):
    if not dConfig["idbst"]:  #      ! Sin balance de t�rminos fuente
        df["Sf"] = (
            dbt["Q"][:, telaps]
            * abs(dbt["Q"][:, telaps])
            * df["nmann1"] ** 2.0
            / (dbt["A"][:, telaps] ** 2.0 * dbt["Rh"][:, telaps] ** (4.0 / 3.0))
        )
        gAS0 = g * dbt["A"][:, telaps] * df["S0"][:]
        gASf = g * dbt["A"][:, telaps] * df["Sf"][:]
    else:  #   ! Con balance de t�rminos fuente
        # TODO: por verificar
        Qmedp, Amedp, Rmedp = (
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
        )

        Qmedp[:-1] = (dbt["Q"][1:, telaps] + dbt["Q"][:-1, telaps]) / 2
        Amedp[:-1] = (dbt["A"][1:, telaps] + dbt["A"][:-1, telaps]) / 2
        Rmedp[:-1] = (dbt["Rh"][1:, telaps] + dbt["Rh"][:-1, telaps]) / 2

        Qmedp[-1] = dbt["Q"][-1, telaps]
        Amedp[-1] = dbt["A"][-1, telaps]
        Rmedp[-1] = dbt["Rh"][-1, telaps]

        gAS0 = g * Amedp * df["zmedp"]
        gASf = (
            g
            * dbt["nmann1"][:, telaps] ** 2.0
            * Qmedp
            * np.abs(Qmedp)
            / (Amedp * Rmedp ** (4.0 / 3.0))
        )

    if dConfig["idbeta"]:
        F[1, :] = dbt["beta"][:, telaps] * (
            dbt["Q"][:, telaps] ** 2.0 / dbt["A"][:, telaps] + g * dbt["I1"][:, telaps]
        )  # ! s� se usa el coeficiente beta
    else:
        F[1, :] = dbt["Q"][:, telaps] ** 2.0 / (
            dbt["A"][:, telaps] + g * dbt["I1"][:, telaps]
        )  # ! no se usa el coeficiente beta

    Gv[0, :] = qpuntual[:]
    Gv[1, :] = g * dbt["I2"][:, telaps] + gAS0 - gASf
    dbt["vel"][:, telaps] = (
        dbt["Q"][:, telaps] / dbt["A"][:, telaps]
    )  #    ! velocidad media
    dbt["c"][:, telaps] = np.sqrt(
        g * dbt["A"][:, telaps] / dbt["B"][:, telaps]
    )  #  ! celeridad de las perturbaciones
    dtpr = dConfig["courant"] * dx / (abs(dbt["vel"][:, telaps]) + dbt["c"][:, telaps])

    vdy, ndy = np.zeros(dConfig["nx"], dtype=bool), np.zeros(dConfig["nx"], dtype=bool)

    dConfig["dtmin"] = np.min(dtpr).values

    if dConfig["iddy"]:  #   ! Para revisar lecho secho
        # for iy in range(ndy):
        F[0, vdy] = 0
        F[1, vdy] = 0
        Gv[1, vdy] = 0

    dt = dConfig["dtmin"]
    lambda_ = dt / dx
    # telaps = telaps + dt

    # !!!!!!!!!!!!!!!!!!!! Predictor
    # -----------------------------------------------------------------------------------
    Up[0, :-1] = U[0, :-1] - lambda_ * (F[0, 1:] - F[0, :-1]) + dt * Gv[0, :-1]
    Up[1, :-1] = U[1, :-1] - lambda_ * (F[1, 1:] - F[1, :-1]) + dt * Gv[1, :-1]
    if dConfig["idtfaa"] == 1:  #      ! Frontera inicial reflejante
        Up[1, 0] = Qaport[0]
    elif dConfig["idtfaa"] == 2:  #  ! Frontera inicial abierta
        Up[1, 0] = Up[1, 1]

    Up[0, -1] = Up[0, -2]  #  	! Cond. de front. A final
    if dConfig["idqf"] == 1:  # then   	    ! Cond. de front. Q final
        Up[1, -1] = Up[1, -2]  # 	    	    ! Q abierto
    elif dConfig["idqf"] == 2:  # then
        qff, qver = 0, 0  # Condición de caudal de fijo aguas arriba
        Up[1, -1] = qff + qver  #          	! Q fijo
        Up[1, -2] = qff + qver  # 			! para que funcione mejor el esquema
    elif dConfig["idqf"] == 3:
        Up[1, -1] = Up[1, -2]
        Up[0, -1] = mareateln  #           ! Marea
        Up[0, -2] = mareatelnm

    # Recorro en x el rio
    # -------------------------------------------------------------
    dbt["Ap"][:, telaps] = Up[0, :]
    dbt["Qp"][:, telaps] = Up[1, :]

    # ! Si lecho seco
    if dConfig["iddy"] == 1:
        # Por el momento no entro
        # utilsSV.fdry(np, Ap, Qp, A2, Q2, ndy, vdy)
        dbt = utilsSV.fdry(dbt, db, telaps, "Ap")
        # for iy in range(np):
        #     dbt["Ap"][iy, telaps] = A2(iy)
        #     dbt["Qp"][iy, telaps] = Q2(iy)

    #   ! Par�metros hidr�ulicos en secciones transversales para el Predictor
    utilsSV.seccionhid(db, dbt, dConfig, telaps, False)  # aquí debería usar las pes

    # ! Opci�n dx-n Murillo
    if dConfig["idmr"]:
        # for iy in range(nmr):
        dbt["I1p"][vmr, telaps] = dbt["I1p"][vmr, telaps] * facmr[vmr]

    # for ii in range(dConfig["nx"]):
    if not dConfig["idbst"]:  #      ! Sin balance de t�rminos fuente
        df["Sfp"] = (
            dbt["Qp"][:, telaps]
            * abs(dbt["Qp"][:, telaps])
            * df["nmann1"] ** 2.0
            / (dbt["Ap"][:, telaps] ** 2.0 * dbt["Rhp"][:, telaps] ** (4.0 / 3.0))
        )
        gAS0p = g * dbt["Ap"][:, telaps] * df["S0"]
        gASfp = g * dbt["Ap"][:, telaps] * df["Sfp"]
    else:  #   ! Con balance de t�rminos fuente
        Qmedc, Amedc, Rmedc = (
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
            np.zeros(dConfig["nx"]),
        )

        Qmedc[:-1] = (dbt["Qp"][1:, telaps] + dbt["Qp"][:-1, telaps]) / 2
        Amedc[:-1] = (dbt["Ap"][1:, telaps] + dbt["Ap"][:-1, telaps]) / 2
        Rmedc[:-1] = (dbt["Rhp"][1:, telaps] + dbt["Rhp"][:-1, telaps]) / 2

        Qmedc[-1] = dbt["Qp"][-1, telaps]
        Amedc[-1] = dbt["Ap"][-1, telaps]
        Rmedc[-1] = dbt["Rhp"][-1, telaps]

        gAS0p = g * Amedc * df["zmedc"]
        gASfp = (
            g
            * dbt["nmann1"][:, telaps] ** 2.0
            * Qmedc
            * np.abs(Qmedc)
            / (Amedc * Rmedc ** (4.0 / 3.0))
        )
    Fp[0, :] = dbt["Qp"][:, telaps]
    if dConfig["idbeta"]:  #   ! Coeficiente beta
        Fp[1, :] = dbt["betap"][:, telaps] * (
            dbt["Qp"][:, telaps] ** 2.0 / dbt["Ap"][:, telaps]
            + g * dbt["I1p"][:, telaps]
        )  # ! si usar beta
    else:
        Fp[1, :] = (
            dbt["Qp"][:, telaps] ** 2.0 / dbt["Ap"][:, telaps]
            + g * dbt["I1p"][:, telaps]
        )  # ! no usar beta
    Gvp[0, :] = qpuntual[:]
    Gvp[1, :] = g * dbt["I2p"][:, telaps] + gAS0p - gASfp

    # ! Opci�n seco
    if dConfig["iddy"]:
        Fp[0, vdy] = 0.0
        Fp[1, vdy] = 0.0
        Gvp[1, vdy] = 0.0

    # -----------------------------------------------------------------
    # !!!!!!!!!!!!!!!!!!!! Corrector
    # -----------------------------------------------------------------
    Uc[0, 1:] = U[0, 1:] - lambda_ * (Fp[0, 1:] - Fp[0, :-1]) + dt * Gvp[0, 1:]
    Uc[1, 1:] = U[1, 1:] - lambda_ * (Fp[1, 1:] - Fp[1, :-1]) + dt * Gvp[1, 1:]

    Uc[0, 0] = Uc[0, 1]  # 	! Cond. de front. A inicial
    if dConfig["idtfaa"] == 1:  #     ! Frontera inicial reflejante
        Uc[1, 0] = Qaport[0]  # + U[1, 0]
    elif dConfig["idtfaa"] == 2:  #  ! Frontera inicial abierta
        Uc[1, 0] = Uc[1, 1]

    # Uc[1, 0] = Qaport[0]
    if dConfig["idqf"] == 1:  #     ! Cond. de front. A final y Q final
        Uc[0, -1] = Uc[0, -1]  #        	! A abierta
        Uc[1, -1] = Uc[1, -1]  #    	! Q abierto
    elif dConfig["idqf"] == 2:
        Uc[0, -1] = Uc[0, -2]  # 		     ! A gradiente nulo
        Uc[1, -1] = qff + qver  #           ! Q fijo
        Uc[1, -2] = qff + qver  # 		 ! para mejorar el funcionamiento del c�digo
    else:
        Uc[1, -1] = Uc[1, -2]
        Uc[0, -1] = mareateln
        Uc[0, -2] = mareatelnm

    # ! Opci�n seco
    if dConfig["iddy"]:
        # for iy in range[0, nx]:
        dbt["Ac"][:, telaps] = Uc[0, :]
        dbt["Qc"][:, telaps] = Uc[1, :]
        # utilsSV.fdry(np, Ac, Qc, A2, Q2, ndy, vdy)
        dbt = utilsSV.fdry(dbt, db, telaps, "Ac")
        # for iy in range(1,np):
        Uc[0, :] = dbt["Ac"][:, telaps]
        Uc[1, :] = dbt["Qc"][:, telaps]

    # !!!!!!!!!!!!!!!!!!!! Siguiente paso de tiempo
    if not dConfig["idfl"]:  #         ! Sin limitador de flujo (MacCormack)
        Unext[0, :] = 0.5 * (Up[0, :] + Uc[0, :])
        Unext[1, :] = 0.5 * (Up[1, :] + Uc[1, :])
    else:  #     ! Con limitador de flujo (TVD-MacCormack)
        if dConfig["idsgm"] == 0:
            # for j in range(np):
            elev = dbt["eta"][:, telaps] + df["z"]

            D = utilsSV.limitador(dbt, dConfig, telaps, lambda_)
            # np, Q, A, B, lambda_, vel, c, elev, idsgm, D
            # )  #! Calcula t�rmino D (limitador)

        Unext[0, :-1] = 0.5 * (Up[0, :-1] + Uc[0, :-1]) + lambda_ * (D[0, :] - D[0, :])
        Unext[1, :-1] = 0.5 * (Up[1, :-1] + Uc[1, :-1]) + lambda_ * (D[1, :] - D[1, :])

    Unext[0, 0] = Unext[0, 1]  # 		    ! Cond. de front. A inicial
    Unext[1, 0] = Qaport[0]
    if dConfig["idtfaa"] == 1:  #    ! Frontera inicial reflejante
        Unext[1, 0] = Qaport[0]
    elif dConfig["idtfaa"] == 2:  #  ! Frontera inicial abierta
        Unext[1, 0] = Unext[1, 1]

    if dConfig["idqf"] == 1:  #        	! Cond. de front. A final y Q final
        Unext[0, -1] = Unext[0, -2]  # 		! A abierta
        Unext[1, -1] = Unext[1, -2]  #  	! Q abierto
    elif dConfig["idqf"] == 2:
        Unext[0, -1] = Unext[0, -1]  #  		! A fijo
        Unext[1, -1] = qff + qver  #           ! Q fijo
        Unext[1, -2] = qff + qver  # 			! truco para que funcione
    elif dConfig["idqf"] == 3:
        Unext[1, -1] = Unext[1, -2]
        Unext[0, -1] = mareateln
        Unext[0, -2] = mareatelnm

    U[0, :] = Unext[0, :]
    U[1, :] = Unext[1, :]
    F[0, :] = Unext[1, :]

    telaps += 1  # dConfig["tesp"]
    if telaps <= dConfig["tfin"] / 3600:
        dbt["A"][:, telaps] = U[0, :]
        dbt["Q"][:, telaps] = U[1, :]


dbt.to_netcdf("pruebaSaintVenant.nc")
