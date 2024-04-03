import datetime

import numpy as np
import pandas as pd
from marinetools.estuaries import utils
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


def main(sConfigFilename, iVersion=1):
    """Calculate water flux and flooding area (Q, A) using the Saint-Venant equations
    for 1D channels.
    Args:
        sConfigFilename (str): filename of the configuration file
        iVersion (int): 1 or for reading the output from BathymetryGenerator.py or
            reading original Guadalfortran geometry files, respectively.

    The configuration file (sConfigFilename) includes:
        bLog (int): 1 or 0 for run the model writing the log and auxiliary files or not,
            respectively.
        sGeometryFilename (str): name of the estuary geometry file
        sHydroFilename (str):	constant_hydro.csv
        iInitialEstuaryCondition (int): 0 or 1 for fixing Q or eta for the entire
            estuary, respectively.
        sInitialQFilename	-
        fQci	10
        fEtaci	0.02
        fCourantNo	0.80
        sOutputFilename	test_001.nc
        fTimeStep	1 hour
        fFinalTime	250
        bDensity	1
        sSedimentFilename	-
        bMcComarckLimiterFlux	1
        bSurfaceGradientMethod	1
        bSourceTermBalance	1
        bBeta	0
        bDryBed	1
        bMurilloCondition	0
        iInitialBoundaryCondition	1
        iFinalBoundaryCondition	1
        fFixQ	0
        sTideFilename	tides.txt
        fDtMin	10,000,000,000.000000
        iFormulaLimiterFlux	4
        iPsiFormula	2
        fDeltaValue	0.20



    Required functions:
        -

    Improvements:
        - This verions allow the computation of the sediment density (bedload, suspended
        sediment concentration and salinity) coupling the density to the TVD MacCormack
        scheme and updating the salinity with Diez-Minguito et al. (2013).
        - The time step is computed according to fast propagation velocity of fluid,
        perturbations, dispersion (and tides)
        - It will include the tidal currents (v 2.0.0)

    Limitations:
        - This version still requires a constant spatial discretization
        - This version does not include yet the water discharge at the seaward side
    """
    initial = datetime.datetime.now()

    # ----------------------------------------------------------------------------------
    # READING INPUTS AND INITIALIZING VARIABLES
    # ----------------------------------------------------------------------------------
    # Read the configuration file
    if iVersion:
        dConfig = read.xlsx(sConfigFilename, names=["options"])["options"].to_dict()
    else:
        dConfig = utils.read_oldfiles(sConfigFilename)

    utils._configure_time(dConfig)

    # Read the geometrical properties of the estuary
    if iVersion:
        # 1. Sections along the main channel
        df = read.csv(dConfig["sGeometryFilename"], index_col=None)
        # 2. Features in terms of the water level
        db = utils._read_cross_sectional_geometry(dConfig)
    else:
        df, db = utils._read_geometry_oldfiles(dConfig)

    # Initialize vectors and matrix
    dbt = utils._initialize_netcdf(df, dConfig)
    db, df, aux = utils._initialize_auxiliary(db, df, dConfig)

    # Reading hydro files
    hydro = utils._read_hydro(dConfig)

    # Reading tide files
    aux["tidal_level"] = pd.read_csv(
        dConfig["sTideFilename"], sep=" ", index_col=[0], header=0
    )

    if dConfig["bDensity"]:
        # Reading sediment properties
        sedProp = utils._read_sediments(dConfig)

        # Reading initial along-estuary salinity
        dbt["S"][:, 0] = read.csv(dConfig["sSalinityFilename"]).values[:, 0]
        aux["salinity"] = dbt["S"][:, 0].values

    # Fix the initial condition of the estuary
    aux = utils._initial_condition(dbt, db, df, aux, dConfig)

    # Compute the fluvial contribution at every node at the given time steps
    utils._fluvial_contribution(dbt, hydro, dConfig)

    # Inititate the log plots
    if dConfig["bLog"]:
        axs = utils._init_logplots()

    # ----------------------------------------------------------------------------------
    # INITIALIZE THE SCRIPT
    # ----------------------------------------------------------------------------------
    iTime = 0  # time index
    fTelaps = 0  # time elapsed
    it = 0  # number of iteratinos
    # Elapsed time - clock
    utils._clock(initial, it, fTelaps, dConfig)
    while iTime < dConfig["iTime"][-1]:

        # Elapsed time - clock
        utils._clock(initial, it, fTelaps, dConfig)

        # Computing the water level due to tides at the seaward boundary
        aux = utils._tidal_level(db, aux, df, dConfig, iTime)

        # Dry bed algorithm
        if dConfig["bDryBed"]:
            mask_dry = utils._dry_soil(dbt, iTime)

        # Murillo condition for dx (verify Manning number)
        if dConfig["bMurilloCondition"]:
            vmr, nmur = utils._conditionMurillo(dbt, df, dConfig, iTime)

        # Compute the hydraulic parameters as function of A
        utils._hydraulic_parameters(dbt, db, iTime)

        # Compute the time step given the Courant number and velocity of information
        # transference
        dConfig = utils._courant_number(dbt, df, dConfig, iTime)

        # Check the time step for ensuring the save times
        dConfig, fTelaps = utils._check_dt(dConfig, iTime, fTelaps)

        df["Sf"] = (
            dbt["Q"][:, iTime].values
            * np.abs(dbt["Q"][:, iTime].values)
            * df["nmann1"].values ** 2.0
            / (
                dbt["A"][:, iTime].values ** 2.0
                * dbt["Rh"][:, iTime].values ** (4.0 / 3.0)
            )
        )

        # ------------------------------------------------------------------------------
        # Sediment transport
        # ------------------------------------------------------------------------------
        if dConfig["bDensity"]:
            utils._vanRijn(sedProp, dbt, df, iTime)
            utils._density(dbt, dConfig, sedProp, iTime)
        else:
            dbt["rho"][:, iTime] = np.ones(len(dbt["A"][:, iTime])) * 1000

        # ------------------------------------------------------------------------------
        # Compute gAS, F and Gv terms
        # ------------------------------------------------------------------------------
        aux = utils._gAS_terms(dbt, df, aux, dConfig, iTime)
        aux = utils._F_terms(dbt, aux, dConfig, iTime)
        aux = utils._Gv_terms(dbt, aux, iTime)

        # Dry bed algorithm
        if dConfig["bDryBed"]:
            aux["F"][0, mask_dry] = 0
            aux["F"][1, mask_dry] = 0
            aux["Gv"][1, mask_dry] = 0

        # ----------------------------------------------------------------------------------
        # STEP 1. Predictor
        # ----------------------------------------------------------------------------------
        aux["Up"][0, :-1] = (
            aux["U"][0, :-1] * dbt["rho"][:-1, iTime]
            - dConfig["lambda"]
            * (
                aux["F"][0, 1:] * dbt["rho"][1:, iTime]
                - aux["F"][0, :-1] * dbt["rho"][:-1, iTime]
            )
            + dConfig["dtmin"] * aux["Gv"][0, :-1] * dbt["rho"][:-1, iTime]
        ) / dbt["rho"][:-1, iTime]
        aux["Up"][0, -1] = aux["Up"][0, -2]

        aux["Up"][1, :-1] = (
            aux["U"][1, :-1] * dbt["rho"][:-1, iTime]
            - dConfig["lambda"]
            * (
                aux["F"][1, 1:] * dbt["rho"][1:, iTime]
                - aux["F"][1, :-1] * dbt["rho"][:-1, iTime]
            )
            + dConfig["dtmin"] * aux["Gv"][1, :-1] * dbt["rho"][:-1, iTime]
        ) / dbt["rho"][:-1, iTime]
        aux["Up"][1, -1] = aux["Up"][1, -2]
        aux["Up"][1, 0] = dbt["q"][0, iTime].values

        # aux = utils._boundary_conditions(dbt, aux, dConfig, iTime, "p")

        # Update Ap and Qp
        dbt["Ap"][:, iTime] = aux["Up"][0, :]
        dbt["Qp"][:, iTime] = aux["Up"][1, :]

        # Dry bed algorithm
        if dConfig["bDryBed"]:
            mask_dry = utils._dry_soil(dbt, iTime, "p")

        aux["Up"][0, :] = dbt["Ap"][:, iTime].values
        aux["Up"][1, :] = dbt["Qp"][:, iTime].values

        # Compute the hydraulic parameters as function of A
        utils._hydraulic_parameters(dbt, db, iTime, False)

        # Check the Murillo condition for dx
        if dConfig["bMurilloCondition"]:
            dbt["I1p"][vmr, iTime] = (
                dbt["I1p"][vmr, iTime] * nmur[vmr] / df.loc[vmr, "nmann"]
            )

        # -----------------------------------------------------------------------------------
        # STEP 1.1: Compute gAS terms
        # -----------------------------------------------------------------------------------
        aux = utils._gAS_terms(dbt, df, aux, dConfig, iTime, False)
        aux = utils._F_terms(dbt, aux, dConfig, iTime, False)
        aux = utils._Gv_terms(dbt, aux, iTime, False)

        # Dry bed algorithm
        if dConfig["bDryBed"]:
            aux["Fp"][0, mask_dry] = 0.0
            aux["Fp"][1, mask_dry] = 0.0
            aux["Gvp"][1, mask_dry] = 0.0

        # ----------------------------------------------------------------------------------
        # STEP 1.2: Sediment transport
        # ----------------------------------------------------------------------------------
        if dConfig["bDensity"]:
            utils._vanRijn(sedProp, dbt, df, iTime)
            utils._density(dbt, dConfig, sedProp, iTime, False)
        else:
            dbt["rhop"][:, iTime] = np.ones(len(dbt["A"][:, iTime])) * 1000

        # -----------------------------------------------------------------
        # STEP 2: Corrector
        # -----------------------------------------------------------------
        aux["Uc"][0, 1:] = (
            aux["U"][0, 1:] * dbt["rhop"][1:, iTime]
            - dConfig["lambda"]
            * (
                aux["Fp"][0, 1:] * dbt["rhop"][1:, iTime]
                - aux["Fp"][0, :-1] * dbt["rhop"][:-1, iTime]
            )
            + dConfig["dtmin"] * aux["Gvp"][0, 1:] * dbt["rhop"][1:, iTime]
        ) / dbt["rhop"][1:, iTime]
        aux["Uc"][0, 0] = dbt["Ap"][0, iTime]

        aux["Uc"][1, 1:] = (
            aux["U"][1, 1:] * dbt["rhop"][1:, iTime].values
            - dConfig["lambda"]
            * (
                aux["Fp"][1, 1:] * dbt["rhop"][1:, iTime].values
                - aux["Fp"][1, :-1] * dbt["rhop"][:-1, iTime].values
            )
            + dConfig["dtmin"] * aux["Gvp"][1, 1:] * dbt["rhop"][1:, iTime]
        ) / dbt["rhop"][1:, iTime].values
        aux["Uc"][1, 0] = dbt["q"][0, iTime]

        # aux = utils._boundary_conditions(dbt, aux, dConfig, iTime, "c")

        # Update Ac and Qc
        dbt["Ac"][:, iTime] = aux["Uc"][0, :]
        dbt["Qc"][:, iTime] = aux["Uc"][1, :]

        # Dry bed algorithm
        if dConfig["bDryBed"]:
            mask_dry = utils._dry_soil(dbt, iTime, "c")

        # aux["Uc"][0, :] = dbt["Ac"][:, iTime].values
        # aux["Uc"][1, :] = dbt["Qc"][:, iTime].values
        # ------------------------------------------------------------------------------
        # STEP 3: Update the following time step
        # ------------------------------------------------------------------------------
        if not dConfig["bMcComarckLimiterFlux"]:
            aux["Un"][0, :] = 0.5 * (aux["Up"][0, :] + aux["Uc"][0, :])
            aux["Un"][1, :] = 0.5 * (aux["Up"][1, :] + aux["Uc"][1, :])
            # Added to smooth the solution
            # aux["Un"][0, 1:] = 0.5 * (aux["Un"][0, 1:] + aux["Un"][0, :-1])
            # aux["Un"][1, 1:] = 0.5 * (aux["Un"][1, 1:] + aux["Un"][1, :-1])
        else:
            # With flow limiter (TVD-MacCormack)
            if dConfig["bSurfaceGradientMethod"]:
                df["elev"] = dbt["eta"][:, iTime] - df["z"]
            aux["D"] = utils._TVD_MacCormack(dbt, df, dConfig, iTime)

            aux["Un"][0, :] = 0.5 * (aux["Up"][0, :] + aux["Uc"][0, :]) + dConfig[
                "lambda"
            ] * (aux["D"][0, 1:] - aux["D"][0, :-1])
            aux["Un"][1, :] = 0.5 * (aux["Up"][1, :] + aux["Uc"][1, :]) + dConfig[
                "lambda"
            ] * (aux["D"][1, 1:] - aux["D"][1, :-1])

            # aux["Un"][0, 0] = 0.5 * (aux["Up"][0, 0] + aux["Uc"][0, 0])
            # aux["Un"][1, 0] = 0.5 * (aux["Up"][1, 0] + aux["Uc"][1, 0])
            # aux["Un"][0, -1] = 0.5 * (aux["Up"][0, -1] + aux["Uc"][0, -1])
            # aux["Un"][1, -1] = 0.5 * (aux["Up"][1, -1] + aux["Uc"][1, -1])
        # aux["Un"][0, 0] = aux["Un"][0, 1]
        # aux["Un"][1, 0] = dbt["q"][0, iTime].values
        # aux["Un"][0, -1] = aux["Un"][0, -2]
        # aux["Un"][1, -1] = aux["Un"][1, -2]

        # Added to smooth the solution
        aux["Un"][0, 1:] = 0.5 * (aux["Un"][0, 1:] + aux["Un"][0, :-1])
        aux["Un"][1, 1:] = 0.5 * (aux["Un"][1, 1:] + aux["Un"][1, :-1])
        # Update boundary conditions
        # aux = utils._boundary_conditions(dbt, aux, dConfig, iTime, "n")

        if dConfig["bDensity"]:
            # Compute salinity gradient
            ast = utils._salinity_gradient(dbt, dConfig, iTime)

        # ----------------------------------------------------------------------------------
        # Update variables for the following time step
        # ----------------------------------------------------------------------------------
        aux["U"][0, :] = aux["Un"][0, :]
        aux["U"][1, :] = aux["Un"][1, :]
        aux["F"][0, :] = aux["Un"][1, :]

        # Move or not to the following savetime
        if dConfig["next_timestep"]:
            iTime += 1

        if dConfig["bLog"]:
            utils._log_plots(dbt, aux, dConfig, iTime, axs)

        if iTime <= dConfig["iTime"][-1]:
            it += 1
            dbt["A"][:, iTime] = aux["U"][0, :]
            dbt["Q"][:, iTime] = aux["U"][1, :]

            if dConfig["bDensity"]:
                # ----------------------------------------------------------------------
                # TODO: To define which option follow. Two options:
                #   - If Q = 0 m3/s here and it is applied the dry bed algorithm,
                #     the salinity is kept along the estuary
                #   - If not, the salinity is reduced to zero
                if dConfig["bDryBed"]:
                    mask_dry = utils._dry_soil(dbt, iTime)
                ast[mask_dry] = 0
                # ----------------------------------------------------------------------

                # TODO: It should be an input option
                # Seawardside is the ocean
                ast[-1] = 0

                dbt["S"][:, iTime] = ast / dbt["A"][:, iTime].values + aux["salinity"]
                aux["salinity"] = dbt["S"][:, iTime].values

                # Bound the minimum and maximum values of salinity to 0 a 35 psu,
                mask = dbt["S"][:, iTime] < 0
                dbt["S"][mask, iTime] = 0

                mask = dbt["S"][:, iTime] > 35
                dbt["S"][mask, iTime] = 35

        # Elapsed time - clock
        utils._clock(initial, it, fTelaps, dConfig)

    # STEP 4: Save the result to a file
    # ---------------------------------------------------------------------------------------
    utils._savefiles(dbt, db, df, dConfig)
