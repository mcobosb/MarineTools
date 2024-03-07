import os
from datetime import datetime
from pathlib import Path

import glob2
import numpy as np
import pandas as pd
from marinetools.utils import read, save


def folders_routes(params, stretch, str_index, simulation=False):
    """_summary_

    Args:
        params (_type_): _description_
        stretch (_type_): _description_
        str_index (_type_): _description_
        simulation (bool, optional): _description_. Defaults to False.
    """

    # -----------------------------------------------------------------------------------
    # Create folder and file routes
    # -----------------------------------------------------------------------------------
    if simulation:
        # TODO: paths from a file
        params["inputFolder"] = Path(
            "C:/Users/Usuario/Desktop/CoastalME paper/"
            + params["rootFolder"].replace("ñ", "n")
        ) / str_index.replace("_", "")
        params["relativePathtoOutputForIni"] = (
            "/share/synology/results_cme_paper/"
            + stretch
            + "/"
            + str_index.replace("_", "")
        )
    else:
        params["inputFolder"] = Path(params["rootFolder"].replace("ñ", "n")) / str_index
        params["relativePathtoOutputForIni"] = (
            "../../../results_cme_paper/" + stretch + "/" + str_index
        )

    params["relativePathtoInputForIni"] = "data_" + stretch.replace("-", "_") + ".dat"

    # params["cmeData"] = params["inputFolder"] / Path(
    #     "data_" + stretch.replace("-", "_").replace("ñ", "n") + ".dat"
    # )

    os.makedirs(
        str(Path(params["rootFolder"].replace("ñ", "n")))
        + "/"
        + str_index.replace("_", "")
        + "/",
        exist_ok=True,
    )

    params["cmeData"] = (
        str(Path(params["rootFolder"].replace("ñ", "n")))
        + "/"
        + str_index.replace("_", "")
        + "/"
        + "data_"
        + stretch.replace("-", "_").replace("ñ", "n")
        + ".dat"
    )

    params["cmeIni"] = params["inputFolder"] / "cme.ini"
    # params["cmeNonCF"] =  / Path(
    #     params["non_consolidate_fine"] + ".asc"
    # )

    params["seastatesFilename"] = Path(
        "seaStates_" + stretch.replace("-", "_").replace("ñ", "n") + ".dat"
    )
    params["cmeSeastates"] = params["inputFolder"] / params["seastatesFilename"]
    params["cmeLocationSeastates"] = "GIS_files/" + params["seastatesLocation"] + ".shp"
    # params["cmeLocationSeastates"] = params["seastatesLocation"] + ".shp"

    if not simulation:
        str_index = ""

    lonlats = params["lonlatWaveInput"].split(";")
    if isinstance(lonlats, list):
        params["stationsNo"] = 0
        params["seastateDataFile"] = []
        for lonlat in lonlats:
            lon = lonlat.split("_")[0]
            lat = lonlat.split("_")[1]
            params["seastateDataFile"].append(
                Path(params["seastatesFile"])
                / Path(params["UF"])
                / Path(
                    "reconstructed_with_eta_" + lon + "_" + lat + str_index + ".zip.csv"
                )
            )

            params["stationsNo"] += 1
    else:
        params["stationsNo"] = 1
        lon = lonlats.split(",")[0]
        lat = lonlats.split(",")[1]
        params["seastateDataFile"] = (
            Path(params["seastatesFile"])
            / Path(params["UF"])
            / Path("reconstructed_" + lon + "_" + lat + str_index + ".zip.csv")
        )

    params["fname"] = (
        "Seastate_reconstruction/storms_"
        + stretch.replace("-", "_").replace("ñ", "n")
        + str_index
        + ".zip"
    )
    params["cmeBatchServer"] = params["inputFolder"] / "batch"
    params["sealevelFilename"] = params["sealevelFilename"] + ".dat"
    params["cmeSealevel"] = params["inputFolder"] / Path(params["sealevelFilename"])
    # params["cmeSediment"] = params["inputFolder"] / Path(
    #     params["sedimentLocation"] + ".dat"
    # )
    params["cmeSediment"] = str(params["inputFolder"]) / Path(
        params["sedimentLocation"] + ".dat"
    )
    # params["cmeSediment"] = str(params["inputFolder"]) / Path(params["sedimentLocation"] + ".dat")
    params["cmeSedimentLocation"] = (
        "../../../GIS_files/" + params["sedimentLocation"]  # + "_location.shp"
    )

    if not "FLOOD OPTIONS" in params.keys():
        params["cmeFloodLocation"] = ""
    else:
        params["cmeFloodLocation"] = (
            "../../../GIS_files/" + params["floodInputLocation"]
        )

    files_ = params["sedimentSourceFilename"].split(",")
    if len(files_) > 1:
        params["sedimentSourceFile"] = []
        for file_ in files_:
            if "SCS" in file_:
                params["sedimentSourceFile"].append(
                    Path("Rivers_Creeks_sediment_source")
                    / Path(params["UF"])
                    / Path(file_.split(".zip.csv")[0] + str_index + ".zip.csv")
                )
            else:
                params["sedimentSourceFile"].append(
                    Path("Rivers_Creeks_sediment_source")
                    / Path(params["UF"])
                    / Path(file_.split(".csv")[0] + str_index + ".csv")
                )
    else:
        if not "-" in files_[0]:
            if "SCS" in files_[0]:
                params["sedimentSourceFile"] = (
                    Path("Rivers_Creeks_sediment_source")
                    / Path(params["UF"])
                    / Path(files_[0].split(".zip.csv")[0] + str_index + ".zip.csv")
                )
            else:
                params["sedimentSourceFile"] = (
                    Path("Rivers_Creeks_sediment_source")
                    / Path(params["UF"])
                    / Path(files_[0].split(".csv")[0] + str_index + ".csv")
                )
        else:
            params["sedimentSourceFile"] = None

    # params["inputFolderName"] = ""
    params["outputFolder"] = Path("results") / stretch / str_index.replace("_", "")

    params["stormFile"] = params["inputFolder"] / Path(
        "storms_" + stretch + "_observation" + str_index + ".zip"
    )
    params["bathymetryFilename"] = Path("Bathymetry_files") / Path(
        stretch.replace("-", ".")
        + "_interp_"
        + str(params["cellSize"])
        + ".xyz.zip.csv"
    )

    params["cmeStructuresType"] = (
        "../../../GIS_files/" + params["structuresTypesFilename"] + ".asc"
    )
    params["cmeStructuresElevation"] = (
        "../../../GIS_files/" + params["structuresElevationFilename"] + ".asc"
    )
    params["non_consolidate_fine"] = (
        "../../../GIS_files/" + params["non_consolidate_fine"] + ".asc"
    )
    params["non_consolidate_sand"] = (
        "../../../GIS_files/" + params["non_consolidate_sand"] + ".asc"
    )
    params["non_consolidate_coarse"] = (
        "../../../GIS_files/" + params["non_consolidate_coarse"] + ".asc"
    )
    params["consolidate_fine"] = (
        "../../../GIS_files/" + params["consolidate_fine"] + ".asc"
    )
    params["consolidate_sand"] = (
        "../../../GIS_files/" + params["consolidate_sand"] + ".asc"
    )
    params["consolidate_coarse"] = (
        "../../../GIS_files/" + params["consolidate_coarse"] + ".asc"
    )
    params["basementFilename"] = (
        "../../../GIS_files/" + params["basementFilename"] + ".asc"
    )

    # -----------------------------------------------------------------------------------
    # Create folders
    # -----------------------------------------------------------------------------------
    params["inputFolder"].mkdir(parents=True, exist_ok=True)
    params["outputFolder"].mkdir(parents=True, exist_ok=True)

    return


def sediment_source(parameters):
    """[summary]

    Args:
        parameters ([type]): [description]
    """
    if not parameters["sedimentSourceFile"] == None:
        if isinstance(parameters["sedimentSourceFile"], list):
            data = []
            for id_, file_ in enumerate(parameters["sedimentSourceFile"]):
                temp_ = read.csv(file_, ts=True)
                if "SCS" in str(file_):
                    temp_ = temp_.loc[
                        parameters["startDatetime"] : parameters["endDatetime"]
                    ]

                temp_["id"] = str(int(id_ + 1))
                temp_["dates_h"] = (
                    temp_.index - parameters["startDatetime"]
                ).total_seconds().values / 3600
                var_ = "Qmean" if "SCS" in str(file_) else "Qs"
                temp_[var_] = (
                    temp_[var_] if "SCS" in str(file_) else temp_[var_] * 3600
                )  # m3 of sediment per hour
                print("Reduction of " + str(temp_[var_].max() / 50) + " factor")
                temp_[var_] = (
                    temp_[var_] / (temp_[var_].max() / 50)
                    if temp_[var_].max() > 50
                    else temp_[var_]
                )

                if not isinstance(parameters["fineSedimentPortion"], int):
                    fine_sed_ = float(
                        parameters["fineSedimentPortion"]
                        .split(";")[id_]
                        .replace(",", ".")
                    )
                    # temp_["fines"] = fine_sed_ * temp_[var_]

                    sand_sed_ = float(
                        parameters["sandSedimentPortion"]
                        .split(";")[id_]
                        .replace(",", ".")
                    )
                    # temp_["sand"] = sand_sed_ * temp_[var_]

                    coarse_sed_ = float(
                        parameters["coarseSedimentPortion"]
                        .split(";")[id_]
                        .replace(",", ".")
                    )
                    # temp_["coarse"] = coarse_sed_ * temp_[var_]
                else:
                    fine_sed_ = parameters["fineSedimentPortion"]
                    sand_sed_ = parameters["sandSedimentPortion"]
                    coarse_sed_ = parameters["coarseSedimentPortion"]

                temp_["fines"] = fine_sed_ * temp_[var_]
                temp_["sand"] = sand_sed_ * temp_[var_]
                temp_["coarse"] = coarse_sed_ * temp_[var_]

                # Create a kind of plume that depend on type of sediment composition
                A = 5400 * temp_[var_]
                # thick_ = temp_[var_] / A

                zeta = 1 - 0.4 * sand_sed_ - 0.8 * coarse_sed_

                # thickness = thick_ / zeta ** 2
                length = np.sqrt(A) * zeta
                width = length * zeta

                temp_["length"] = length
                # (
                #     np.sqrt(temp_[var_] / 4) * 4
                # )  # It is assigned a rectangle of 4L x L
                temp_["width"] = width  # np.sqrt(temp_[var_] / 4)
                # temp_["thick"] = thickness
                Q_np = temp_.loc[
                    :,
                    [
                        "id",
                        "dates_h",
                        "fines",
                        "sand",
                        "coarse",
                        "length",
                        "width",
                        # "thick",
                    ],
                ]
                Q_np = Q_np.loc[
                    parameters["startDatetime"] : parameters["endDatetime"], :
                ]
                if id_ == 0:
                    data = Q_np
                else:
                    data = data.append(Q_np)
        else:
            file_ = parameters["sedimentSourceFile"]
            temp_ = read.csv(file_, ts=True)
            if "SCS" in str(file_):
                temp_ = temp_.loc[
                    parameters["startDatetime"] : parameters["endDatetime"]
                ]
            temp_["id"] = "1"
            temp_["dates_h"] = (
                temp_.index - parameters["startDatetime"]
            ).total_seconds().values / 3600

            var_ = "Qmean" if "SCS" in str(parameters["sedimentSourceFile"]) else "Qs"
            temp_[var_] = (
                temp_[var_] if "SCS" in str(file_) else temp_[var_] * 3600
            )  # m3 of sediment per hour
            print("Reduction of " + str(temp_[var_].max() / 50) + " factor")
            temp_[var_] = (
                temp_[var_] / (temp_[var_].max() / 50)
                if temp_[var_].max() > 50
                else temp_[var_]
            )

            temp_["fines"] = parameters["fineSedimentPortion"] * temp_[var_]
            temp_["sand"] = parameters["sandSedimentPortion"] * temp_[var_]
            temp_["coarse"] = parameters["coarseSedimentPortion"] * temp_[var_]
            # Create a kind of plume that depend on type of sediment composition
            A = 5400 * temp_[var_]
            # thick_ = temp_[var_] / A

            zeta = (
                1
                - 0.4 * parameters["sandSedimentPortion"]
                - 0.8 * parameters["coarseSedimentPortion"]
            )

            # thickness = thick_ / zeta ** 2
            length = np.sqrt(A) * zeta
            width = length * zeta

            temp_["length"] = length
            # (
            #     np.sqrt(temp_[var_] / 4) * 4
            # )  # It is assigned a rectangle of 4L x L
            temp_["width"] = width  # np.sqrt(temp_[var_] / 4)
            # temp_["thick"] = thickness
            data = temp_.loc[
                :,
                [
                    "id",
                    "dates_h",
                    "fines",
                    "sand",
                    "coarse",
                    "length",
                    "width",
                    # "thick",
                ],
            ]
            # temp_["length"] = (
            #     np.sqrt(temp_[var_].values / 4) * 4
            # )  # It is assigned a rectangle of 4L x L
            # temp_["width"] = np.sqrt(temp_[var_].values / 4)
            # data = temp_.loc[
            #     :, ["id", "dates_h", "fines", "sand", "coarse", "length", "width"]
            # ]
            # data = data.loc[parameters["startDatetime"] : parameters["endDatetime"], :]

        # ------------------------------------------------------------------------------
        # Location of mouths
        # ------------------------------------------------------------------------------
        lon, lat = [], []
        if ";" in parameters["lonlatSedimentInput"]:
            lonlats = str(parameters["lonlatSedimentInput"]).split(";")
            for i in lonlats:
                lon.append(float(i.split("_")[0]))
                lat.append(float(i.split("_")[1]))
        else:
            lon = float(parameters["lonlatSedimentInput"].split("_")[0])
            lat = float(parameters["lonlatSedimentInput"].split("_")[1])
        save.to_shp(str(parameters["cmeSedimentLocation"]), lon, lat)

        # Check if time step different from 1 hour. If so, take with the given step
        data = check_output_timestep_df(data, parameters["output_timestep"])

        fid = open(parameters["cmeSediment"], "w", newline="\n")
        intro_text_sediment_source(fid)
        data.sort_index(inplace=True)
        for ind_, _ in enumerate(data.index):
            fid.write(
                "{}, {}, {}, {}, {}, {}, {}\n".format(
                    data.iloc[ind_, 0],
                    str(int(data.iloc[ind_, 1])) + " hours",
                    np.round(data.iloc[ind_, 2], decimals=4),
                    np.round(data.iloc[ind_, 3], decimals=4),
                    np.round(data.iloc[ind_, 4], decimals=4),
                    np.round(data.iloc[ind_, 5], decimals=4),
                    np.round(data.iloc[ind_, 6], decimals=4),
                    # np.round(data.iloc[ind_, 7], decimals=4),
                )
            )
        fid.close()
    return


def flood_location(parameters):
    # ------------------------------------------------------------------------------
    # Location of floods
    # ------------------------------------------------------------------------------
    lon, lat = [], []
    if ";" in parameters["lonlatFloodInput"]:
        lonlats = str(parameters["lonlatFloodInput"]).split(";")
        for i in lonlats:
            lon.append(float(i.split("_")[0]))
            lat.append(float(i.split("_")[1]))
    else:
        lon = float(parameters["lonlatFloodInput"].split("_")[0])
        lat = float(parameters["lonlatFloodInput"].split("_")[1])
    save.to_shp(str(parameters["cmeFloodLocation"]), lon, lat)
    return


def seastates(parameters: dict, mode_: str = "calibration", var_Hm0="Hm0"):
    """Write the sea state and tidal time series in deep water

    Args:
        data ([type]): [description]
        parameters ([type]): [description]
        rotateDirections (bool, optional): [description]. Defaults to True.
    """

    from marinetools.temporal import analysis

    # Added the angle of the mesh
    # data.DirM = np.remainder(data.DirM + parameters["meshDirection"], 360)

    if isinstance(parameters["seastateDataFile"], list):
        data, lat, lon = [], [], []
        for id_, file_ in enumerate(parameters["seastateDataFile"]):
            lonlats = str(file_).split(".")[0].split("_")[-3:-1]
            lon.append(float(lonlats[0]))
            lat.append(float(lonlats[1]))

            temp_ = read.csv(file_, ts=True)

            temp_.loc[
                temp_.loc[:, var_Hm0] < 0.1, var_Hm0
            ] = 0.1  # Modified values near to zero
            temp_["DirM"] = np.fmod(180 + temp_["DirM"], 360)  # Added 180 due to cshore
            temp_ = temp_.loc[
                parameters["startDatetime"] : parameters["endDatetime"], :
            ]

            # Check if time step different from 1 hour. If so, take with the given step
            temp_ = check_output_timestep_df(temp_, parameters["output_timestep"])

            temp3_ = temp_.loc[:, [var_Hm0, "DirM", "Tp"]].to_numpy()
            if id_ == 0:
                data = temp3_
            else:
                data = np.hstack([data, temp3_])
    else:
        file_ = parameters["seastateDataFile"]
        temp_ = read.csv(file_, ts=True)

        temp_.loc[temp_.Hm0 < 0.1, var_Hm0] = 0.1  # Modified values near to zero
        temp_.loc["DirM"] = np.fmod(
            180 + temp_.loc["DirM"], 360
        )  # Added 180 due to cshore

        # Check if time step different from 1 hour. If so, take with the given step
        temp_ = check_output_timestep_df(temp_, parameters["output_timestep"])

        data = temp_.loc[
            parameters["startDatetime"] : parameters["endDatetime"],
            [var_Hm0, "DirM", "Tp"],
        ].to_numpy()

    parameters["timeStepNo"] = len(data)

    if mode_ == "simulation":
        # ------------------------------------------------------------------------------
        # Create the storm statistics to save file during those if simulations are required
        # ------------------------------------------------------------------------------
        if not os.path.exists(parameters["fname"] + ".csv"):
            vars_ = [var_Hm0, "Tp", "DirM"]
            _ = analysis.storm_properties(temp_, vars_, parameters)

        parameters["timeStepNo"] = len(data)
        # eta = temp_["ma"] + temp_["slr"] + temp_["mm"]
        eta = temp_["eta"]
    else:
        eta = temp_["ma"]

    # Check if time step different from 1 hour. If so, take with the given step
    eta = check_output_timestep_df(eta, parameters["output_timestep"])

    # ----------------------------------------------------------------------------------
    # Write sea level file
    # ----------------------------------------------------------------------------------
    eta.to_csv(parameters["cmeSealevel"], header=False, index=False)

    # ----------------------------------------------------------------------------------
    # Write location seastates file
    # ----------------------------------------------------------------------------------
    save.to_shp(parameters["cmeLocationSeastates"], lon, lat)

    # ----------------------------------------------------------------------------------
    # Write sea states file
    # ----------------------------------------------------------------------------------,
    fid = open(
        parameters["cmeSeastates"],
        "w",
        newline="\n",
    )

    intro_text_seastates(fid, parameters)
    fid.close()
    with open(parameters["cmeSeastates"], "a") as fid:
        np.savetxt(fid, data, fmt="%3.3f", delimiter=",")

    return


def check_output_timestep_df(df: pd.DataFrame, timestep: int = 1):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        timestep (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """

    # Check if time step different from 1 hour. If so, take with the given step
    if timestep != 1:
        remainder = df.index.hour % timestep
        index = np.ones(len(remainder), dtype=bool)
        index[remainder != 0] = False
        df = df.loc[index]

    return df


def data_file(parameters, mode_="simulation", stormStats=False):
    """[summary]

    Args:
        fname ([type]): [description]
        uf ([type]): [description]
    """
    # Just start and end of a storm was save
    if mode_ == "simulation":
        # ------------------------------------------------------------------------------
        # Save the results at every timestep during a storm and once per week
        # ------------------------------------------------------------------------------
        stormStats = read.csv(parameters["fname"] + ".csv", ts=True)
        for ind_, ini_ in enumerate(stormStats["storm_ini"]):
            # saveDates = (
            #     pd.concat([stormStats["storm_ini"], stormStats["storm_end"]])
            #     .sort_values()
            #     .reset_index(drop=True)
            # )
            if ind_ == 0:
                saveDates = pd.date_range(
                    ini_,
                    stormStats.loc[ind_, "storm_end"],
                    freq="6H",  # parameters["time_step"],
                )
            else:
                aux_ = pd.date_range(
                    ini_,
                    stormStats.loc[ind_, "storm_end"],
                    freq="6H",  # parameters["time_step"],
                )
                saveDates = saveDates.union(aux_)
        # saveDates = pd.to_datetime(saveDates)

        if parameters["sediment"]:
            riverCreeksStats = pd.read_csv(
                parameters["cmeSediment"],
                skiprows=38,
                names=["id", "time", "fine", "sand", "coarse", "L", "w"],
            )
            mask = (
                riverCreeksStats.fine + riverCreeksStats.sand + riverCreeksStats.coarse
                > 3.6
            )
            riversaveTimes = np.unique(
                [
                    int(hour.split(" hours")[0])
                    for hour in riverCreeksStats.loc[mask, "time"]
                ]
            )

        saveTimes = (saveDates - parameters["startDatetime"]).total_seconds() / 3600

        # Create monthly save times
        monthlySaveTimes = pd.date_range(
            parameters["startDatetime"] - pd.Timedelta(1, parameters["time_step"]),
            parameters["endDatetime"],
            freq="MS",
        )
        monthlySaveTimes = (
            monthlySaveTimes[1:]
            - parameters["startDatetime"]
            + pd.Timedelta(1, parameters["time_step"])
        ).total_seconds() / 3600

        if parameters["sediment"]:
            saveTimes = np.unique(
                np.sort(np.floor(np.hstack((saveTimes, riversaveTimes))))
            )
        else:
            saveTimes = np.unique(np.sort(np.floor(np.hstack((saveTimes)))))

        # Take savetimes at output timestep cadency
        k, ref = 1, 1
        mask = [True]
        while k < len(saveTimes):
            if (
                saveTimes[k] - saveTimes[k - ref] >= 6
            ):  # imposed by the project, limitation of the number of savetimes
                mask.append(True)
                ref = 1
            else:
                mask.append(False)
                ref += 1
            k += 1

        saveTimes = saveTimes[mask]
        saveTimes = np.sort(np.hstack([saveTimes, monthlySaveTimes]))
        saveTimes = np.unique(np.floor(saveTimes))
    else:
        # ------------------------------------------------------------------------------
        # Read the satellite coastlines and save the results at the same timesteps
        # ------------------------------------------------------------------------------
        strDates = glob2.glob("Calibration_Coastlines/*.gpkg")

        datetimeDates = [
            datetime.strptime(
                date.split("\\")[1].split("_")[1].split(".")[0], "%Y-%m-%d"
            )
            for date in strDates
            if parameters["UF"] in date
        ]

        saveTimes = [
            int((date - parameters["startDatetime"]).total_seconds() / 3600)
            for date in datetimeDates
        ]

        # Take the timestep nearest of saveTimes that fit to the output timestep
        saveTimes = np.asarray(
            [
                savetime + savetime % parameters["output_timestep"]
                for savetime in saveTimes
            ]
        )

    parameters["saveTimes"] = str(parameters["output_timestep"]) + " "
    for i in saveTimes:
        parameters["saveTimes"] += "{} ".format(int(i))
    parameters["saveTimes"] += "hours"

    main_text(parameters)

    return


def intro_text_sediment_source(fid):
    fid.write("; This is the sediment input event data file.\n")
    fid.write(";\n")
    fid.write(
        "; The associated shapefile stores at least one point or line, which is tagged with a unique numeric ID.\n"
    )
    fid.write(";\n")
    fid.write(
        "; If Sediment input type in the main input file is P (Point) mode, then a point shapefile is required. Each point in\n"
    )
    fid.write(
        "; this shapefile is tagged with an ID. Each sediment input event (below) also has a location ID. Unconsolidated sediment\n"
    )
    fid.write(
        "; is deposited at the point with the matching ID tag. In this mode, values for the coast-normal length L and\n"
    )
    fid.write(
        "; along-coast-width W are ignored. Note that more than one sediment input event can occur with the same location ID, at\n"
    )
    fid.write("; different times.\n")
    fid.write(";\n")
    fid.write(
        "; If Sediment input type in the main input file is set to C (Coast) mode, then a point shapefile is required. Each\n"
    )
    fid.write(
        "; point in this shapefile is tagged with an ID. Each sediment input event (below) also has a location ID. Unconsolidated\n"
    )
    fid.write(
        "; sediment is deposited as a block at the point on the coast which is closest to the point with the matching ID tag. This\n"
    )
    fid.write(
        "; is necessary because the coastline moves during the simulation. Therefore it is not possible to exactly specify, before\n"
    )
    fid.write(
        "; the simulation begins, a location which is guaranteed to be on the coastline at a given time. In Coast mode, sediment\n"
    )
    fid.write(
        "; input is assumed to be in the shape of an approximately rectangular block, which has uniform thickness (height),\n"
    )
    fid.write(
        "; extends a length L into the sea (normal to the coastline), and has an along-coastline width W. Note that more than one\n"
    )
    fid.write(
        "; sediment input event can occur with the same location ID, at different times.\n"
    )
    fid.write(";\n")
    fid.write(
        "; If Sediment input type in the main input file is set to L (Line) mode then a line shapefile is required. Each line\n"
    )
    fid.write(
        "; in this shapefile is tagged with an ID. Each sediment input event (below) also has an ID. Unconsolidated sediment is\n"
    )
    fid.write(
        "; deposited at the intersection of the with the matching ID, and a coastline. In this mode, values for the coast-normal\n"
    )
    fid.write(
        "; length L and along-coast-width W are ignored. Note that more than one sediment input event can occur with the same\n"
    )
    fid.write("; line ID, at different times.\n")
    fid.write(";\n")
    fid.write(
        "; Each row below contains data for a single sediment input event, and must include the following comma-separated values:\n"
    )
    fid.write(
        ";  * the location or line ID of the sediment input event (same as the ID in the shapefile. IDs start from 1 and run\n"
    )
    fid.write(";    consecutively\n")
    fid.write(
        ";  * the time of the sediment input event. This can be either relative (i.e. a number of hours or days after the\n"
    )
    fid.write(
        ";    simulation start) or absolute (i.e. a time/date in the format hh-mm-ss dd/mm/yyyy)\n"
    )
    fid.write(";  * the volume (m3) of fine sediment in the sediment event\n")
    fid.write(";  * the volume (m3) of sand sediment in the sediment event\n")
    fid.write(";  * the volume (m3) of coarse sediment in the sediment event\n")
    fid.write(
        ";  * the coast-normal length L (m) of the sediment block (only needed if sediment input type is Coast)\n"
    )
    fid.write(
        ";  * the along-coast width W (m) of the sediment block (only needed if sediment input type is Coast)\n"
    )
    fid.write(";\n")
    fid.write(
        "; Rows should be in time sequence. If more than one sediment input event occurs simultaneously (i.e. events occur at the\n"
    )
    fid.write(
        "; same time at more than one location or line), then list each event on its own line.\n"
    )
    fid.write("\n")
    return


def intro_text_seastates(fid, params):
    fid.write(
        "Start time/date   [hh-mm-ss dd/mm/yyyy]: {}\n".format(params["startTime"])
    )
    fid.write(
        "Timestep              : {}\n".format(str(params["output_timestep"]) + " hours")
    )
    fid.write("Number of Stations    : {}\n".format(params["stationsNo"]))
    fid.write("Number of time steps  : {}\n".format(params["timeStepNo"]))
    fid.write(
        "; Each row below contains the wave height (m), orientation (deg) and period (s) for each time step for all stations\n"
    )
    fid.write(
        "; Starting from the first station (i.e. station with id=1 on the corresponding point shape file)\n"
    )
    fid.write("; Values separated by a COMMA\n")
    fid.write(
        "; Starting on {} at {} time step interval\n".format(
            params["startTime"], params["timeStep"]
        )
    )
    fid.write(
        "; Coordinates of wave point lat-lon: {}\n".format(params["lonlatWaveInput"])
    )
    # fid.write(
    #     "; Coordinates of wave point lat: {} lon: {}\n".format(
    #         params["latWaveInput"], params["lonWaveInput"]
    #     )
    # )
    return


def ini(param):
    """Write the ini file for CoastalME

    Args:
        param ([dict]): parameters that include the relative paths
    """
    fid = open(param["cmeIni"], "w", newline="\n")

    fid.write(
        ";======================================================================================================================\n"
    )
    fid.write(";\n")
    fid.write("; Initialization file for CoastalME\n")
    fid.write(";\n")
    fid.write("; Copyright (C) 2020 David Favis-Mortlock and Andres Payo\n")
    fid.write(";\n")
    fid.write(
        ";=====================================================================================================================\n"
    )
    fid.write(";\n")
    fid.write(
        "; This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public\n"
    )
    fid.write(
        "; License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later\n"
    )
    fid.write("; version.\n")
    fid.write(";\n")
    fid.write(
        "; This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied\n"
    )
    fid.write(
        "; warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
    )
    fid.write(";\n")
    fid.write(
        "; You should have received a copy of the GNU General Public License along with this program; if not, write to the Free\n"
    )
    fid.write("; Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.\n")
    fid.write(";\n")
    fid.write(
        ";======================================================================================================================\n"
    )
    fid.write(
        "Input data file (path and name)                       : {}\n".format(
            param["relativePathtoInputForIni"].replace("ñ", "n")
        )
    )
    fid.write("\n")
    fid.write(
        "Path for output                                      : {}\n".format(
            param["relativePathtoOutputForIni"].replace("ñ", "n") + "/"
        )
    )
    fid.write("\n")
    fid.write(
        "; Uncomment if email notification required. Note that this only works with Linux\n"
    )
    fid.write(
        "; Email address for messages (Linux only)                : {}\n".format(
            param["email"]
        )
    )
    fid.close()

    return


def main_text(param):
    """[summary]

    Args:
        param ([type]): [description]
    """
    # ----------------------------------------------------------------------------------
    # check which sediment layer are presented in the bottom
    # ----------------------------------------------------------------------------------
    fid = open(param["cmeData"], "w", newline="\n")
    fid.write(
        ";======================================================================================================================\n"
    )
    fid.write(";\n")
    fid.write("; Data file for CoastalME\n")
    fid.write(";\n")
    fid.write("; Copyright (C) 2021 David Favis-Mortlock and Andres Payo\n")
    fid.write(";\n")
    fid.write(
        ";=====================================================================================================================\n"
    )
    fid.write(";\n")
    fid.write(
        "; This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software\n"
    )
    fid.write(
        "; Foundation; either version 3 of the License, or (at your option) any later version.\n"
    )
    fid.write(";\n")
    fid.write(
        "; This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS\n"
    )
    fid.write(
        "; FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
    )
    fid.write(";\n")
    fid.write(
        "; You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass\n"
    )
    fid.write("; Ave, Cambridge, MA 02139, USA.\n")
    fid.write(";\n")
    fid.write(
        ";======================================================================================================================\n"
    )
    fid.write(";\n")
    fid.write("; {}\n".format(param["stretchName"]))
    fid.write(";\n")
    fid.write(
        "; Run information -----------------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Main output/log file names                                          [omit path and extension]: {}\n".format(
            param["outputLogFilename"].replace("ñ", "n")
        )
    )
    fid.write(
        "Content of log file                                       [1 = least detail, 3 = most detail]: {}\n".format(
            param["logfileDetail"]
        )
    )
    fid.write("\n")
    fid.write(
        "Simulation start date and time                                          [hh-mm-ss dd/mm/yyyy]: {}\n".format(
            param["startTime"]
        )
    )
    fid.write(
        "Duration of simulation                                           [hours, days, months, years]: {}\n".format(
            param["durationOfsimulation"]
        )
    )
    fid.write(
        "Timestep (hours or days)                                                                     : {}\n".format(
            str(param["output_timestep"]) + " hours"
        )
    )
    fid.write(
        "Save times                                                       [hours, days, months, years]: {}\n".format(
            param["saveTimes"]
        )
    )
    fid.write("\n")
    fid.write(
        "Random number seed(s)                                                                        : 1\n"
    )
    fid.write("\n")
    raster_files = (
        ""
        if (param["outputRasterFiles"] != param["outputRasterFiles"])
        else param["outputRasterFiles"].replace(",", "")
    )  # Feature of nans
    fid.write(
        "GIS raster files to output                                                    [see codes.txt]: {}\n".format(
            raster_files
        )
    )
    fid.write(
        "GIS raster output format                                          [blank = same as DEM input]: {}              ; gdal-config --formats for others\n".format(
            param["outputRasterExt"]
        )
    )
    fid.write(
        "   If needed, also output GIS raster world file?                                        [y/n]: y\n"
    )
    fid.write(
        "   If needed, scale GIS raster output values?                                           [y/n]: y\n"
    )
    fid.write(
        "GIS raster 'slice' elevation(s) (m)                                                          :\n"
    )
    vector_files = (
        ""
        if (param["outputVectorFiles"] != param["outputVectorFiles"])
        else param["outputVectorFiles"].replace(",", "")
    )  # Feature of nans
    fid.write(
        "GIS vector files to output                                                    [see codes.txt]: {}\n".format(
            vector_files
        )
    )
    fid.write(
        "GIS vector output format                                                                     : {}           ; ESRI Shapefile ogrinfo --formats for others\n".format(
            param["outputVectorExt"]
        )
    )
    timeseries_files = (
        ""
        if (param["outputCsvFiles"] != param["outputCsvFiles"])
        else param["outputCsvFiles"].replace(",", "")
    )  # Feature of nans
    fid.write(
        "Time series files to output                                                   [see codes.txt]: {}\n".format(
            timeseries_files
        )
    )
    fid.write("\n")
    fid.write(
        "Coastline smoothing                          [0 = none, 1 = running mean, 2 = Savitsky-Golay]: {}\n".format(
            param["coastlineSmoothing"]
        )
    )

    fid.write(
        "Coastline smoothing window size                                                 [must be odd]: {}\n".format(
            int(param["coastlineWindowSize"])
        )
    )

    fid.write(
        "Polynomial order for Savitsky-Golay coastline smoothing                              [2 or 4]: 4\n"
    )
    fid.write(
        "Grid edge(s) to omit when searching for coastline                                      [NSWE]: {}\n".format(
            param["gridEdgeOmit"]
        )
    )

    fid.write("\n")
    fid.write(
        "Profile slope running-mean smoothing window size                        [must be zero or odd]: {}\n".format(
            int(param["profileSlopeRunningMean"])
        )
    )
    fid.write(
        "Max local slope (m/m)                                                                        : {}\n".format(
            param["maxLocalSlope"]
        )
    )
    fid.write("\n")
    fid.write(
        "Maximum elevation of beach above SWL                                                      [m]: {}\n".format(
            param["maxElevationAboveSWL"]
        )
    )
    fid.write("\n")
    fid.write(
        "; Initial raster GIS layers --------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Number of layers                                                                             : {}\n".format(
            param["noLayers"]
        )
    )
    fid.write("; BASEMENT DEM MUST BE PRESENT\n")
    fid.write(
        "Basement DEM file                                                             [path and name]: {}\n".format(
            str(param["basementFilename"])
        )
    )
    fid.write("   ; LAYER 0 THICKNESS FILES, LAYER 0 IS LOWEST\n")
    fid.write("   ; *** MUST HAVE AT LEAST ONE THICKNESS FILE FOR THE FIRST LAYER\n")
    fid.write(
        "   Initial Unconsolidated fine sediment file                                  [path and name]: {}\n".format(
            str(param["non_consolidate_fine"])
        )
    )
    fid.write(
        "   Initial Unconsolidated sand sediment file                                  [path and name]: {}\n".format(
            str(param["non_consolidate_sand"])
        )
    )
    fid.write(
        "   Initial Unconsolidated coarse sediment file                                [path and name]: {}\n".format(
            str(param["non_consolidate_coarse"])
        )
    )
    fid.write(
        "   Initial Consolidated fine sediment file                                    [path and name]: {}\n".format(
            str(param["consolidate_fine"])
        )
    )
    fid.write(
        "   Initial Consolidated sand sediment file                                    [path and name]: {}\n".format(
            str(param["consolidate_sand"])
        )
    )
    fid.write(
        "   Initial Consolidated coarse sediment file                                  [path and name]: {}\n".format(
            str(param["consolidate_coarse"])
        )
    )
    fid.write("\n")
    fid.write("; OPTIONAL THICKNESS FILE\n")
    fid.write(
        "Initial Suspended sediment file                                               [path and name]:\n"
    )
    fid.write("\n")
    fid.write("; OTHER OPTIONAL RASTER FILES\n")
    fid.write(
        "Initial Landform file                                                         [path and name]:\n"
    )
    fid.write(
        "Intervention class file                                                       [path and name]: {}\n".format(
            (str(param["cmeStructuresType"]) * param["structures"])
        )
    )
    fid.write(
        "Intervention height file                                                      [path and name]: {}\n".format(
            (str(param["cmeStructuresElevation"]) * param["structures"])
        )
    )
    fid.write("\n")
    fid.write(
        "; Hydrology data -------------------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Wave propagation model                                                 [0 = COVE, 1 = CShore]: {}\n".format(
            param["wavePropagationModel"]
        )
    )
    fid.write("\n")
    fid.write(
        "Density of sea water (kg/m3)                                                                 : {}\n".format(
            str(param["waterDensity"])
        )
    )
    fid.write("\n")
    fid.write(
        "Initial still water level (m), or per-timestep SWL file                                      : {}\n".format(
            param["stillWaterLevel"]
        )
    )
    fid.write(
        "Final still water level (m)                                     [blank = same as initial SWL]:\n"
    )
    fid.write("\n")
    fid.write(
        "Deep water wave height (m)                                         [value or point shapefile]: {}\n".format(
            param["cmeLocationSeastates"]
        )
    )
    fid.write(
        "Deep water wave height time series         [omit if deep water wave height is a single value]: {}\n".format(
            param["seastatesFilename"]
        )
    )
    fid.write(
        "Deep water wave orientation (degrees)         [omit if deep water wave height from shapefile]: {}\n".format(
            ""
        )
    )
    fid.write(
        "Wave period (sec)                             [omit if deep water wave height from shapefile]: {}\n".format(
            ""
        )
    )
    fid.write(
        "Tide data file                                                                 [may be blank]: {}\n".format(
            param["sealevelFilename"]
        )
    )
    fid.write(
        "Breaking wave height to water depth ratio (non dimensional)                                  : {}\n".format(
            param["breakingWaveParameter"]
        )
    )
    fid.write("\n")
    fid.write(
        "; Sediment data --------------------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Simulate coast platform erosion?                                                             : {}\n".format(
            param["platformErosion"]
        )
    )
    fid.write(
        "Resistance to erosion of coast platform (m^(9/4)s^(3/2))                                     : {} ; [2-6]e6 to produce 0.6 m/yr recession (Walkden & Hall, 2011), is 1/erodibility\n".format(
            int(param["resistancePlatformErosion"])
        )
    )
    fid.write("\n")
    fid.write(
        "Simulate beach sediment transport?                                                           : {}\n".format(
            param["simulateBeachSedimentTransport"]
        )
    )
    fid.write(
        "Beach sediment transport at grid edges                [0 = closed, 1 = open, 2 = recirculate]: {}\n".format(
            param["sedimentBoundaries"]
        )
    )
    fid.write("\n")
    fid.write(
        "Beach potential erosion/deposition equation                          [0 = CERC, 1 = Kamphuis]: {}\n".format(
            param["sedimentTransportFormula"]
        )
    )
    fid.write("\n")
    fid.write(
        "Median size of fine sediment (mm)                        [0 = default, only for Kamphuis eqn]: {}\n".format(
            param["medianFineSize"]
        )
    )
    fid.write(
        "Median size of sand sediment (mm)                        [0 = default, only for Kamphuis eqn]: {}\n".format(
            param["medianSandSize"]
        )
    )
    fid.write(
        "Median size of coarse sediment (mm)                      [0 = default, only for Kamphuis eqn]: {}\n".format(
            param["medianCoarseSize"]
        )
    )
    fid.write("\n")
    fid.write(
        "Density of beach sediment (kg/m3)                                                            : 2650      ; For quartz\n"
    )
    fid.write(
        "Beach sediment porosity                                                                      : {}\n".format(
            param["sedimentPorosity"]
        )
    )
    fid.write("\n")
    fid.write("; Arbitrary values of relative erosivities\n")
    fid.write(
        "Fine erosivity                                                                               : 1\n"
    )
    fid.write(
        "Sand erosivity                                                                               : 0.7\n"
    )
    fid.write(
        "Coarse erosivity                                                                             : 0.3\n"
    )
    fid.write("\n")
    fid.write(
        "Transport parameter KLS                                                   [only for CERC eqn]: {}\n".format(
            param["sedimentTransportParameter"]
        )
    )
    fid.write(
        "Transport parameter                                                   [only for Kamphuis eqn]: {}\n".format(
            param["sedimentTransportParameter"]
        )
    )
    fid.write("\n")
    fid.write(
        "Berm height i.e. height above SWL of start of depositional Dean profile (m)                  : {}\n".format(
            str(param["bermHeight"])
        )
    )
    fid.write("\n")
    fid.write(
        "Sediment input location                                             [point or line shapefile]: {}\n".format(
            (param["cmeSedimentLocation"] + "_location.shp") * param["sediment"]
        )
    )
    fid.write(
        "Sediment input type        [required if have shapefile, P = Point, C = coast block, L = line]: {}\n".format(
            (param["sedimentInputType"]) * param["sediment"]
        )
    )
    fid.write(
        "Sediment input details file                                      [required if have shapefile]: {}\n".format(
            (param["sedimentLocation"] + ".dat") * param["sediment"]
        )
    )
    fid.write("\n")
    fid.write(
        "; Cliff collapse data --------------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Simulate cliff collapse?                                                                     : {}\n".format(
            param["cliffErosion"]
        )
    )
    fid.write(
        "Resistance to erosion of cliff (m^(9/4)s^(3/2))                                              : {} ; [2-6]e6 to produce 0.6 m/yr recession (Walkden & Hall, 2011),\n".format(
            int(param["resistanceCliffErosion"])
        )
    )
    fid.write(
        "Notch overhang to initiate collapse (m)                                                      : 0.5\n"
    )
    fid.write(
        "Notch base below still water level (m)                                                       : 0.30       ; was 0.3, MUST BE GREATER THAN ZERO\n"
    )
    fid.write(
        "Scale parameter A for cliff deposition (m^(1/3))                                   [0 = auto]: 0         ; For a 0.2 mm D50, will be 0.1\n"
    )
    fid.write(
        "Approximate planview width of cliff deposition talus (m)                                     : 5; \n"
    )
    fid.write(
        "Planview length of cliff deposition talus (m)                                                : 50        ; was 30 15\n"
    )
    fid.write(
        "Height of landward talus end, as fraction of cliff elevation                                 : 0\n"
    )
    fid.write("\n")
    fid.write(
        "; Other data -----------------------------------------------------------------------------------------------------------\n"
    )
    fid.write(
        "Gravitational acceleration (m2/s)                                                            : 9.81\n"
    )
    fid.write("\n")
    fid.write("; TEMPORARY ONLY, CHANGE DYNAMICALLY LATER\n")
    fid.write(
        "Spacing of coastline normals (m)                                        [0 = default minimum]: {}\n".format(
            param["spacingCoastline"]
        )
    )
    fid.write(
        "Random factor for spacing of normals                              [0 to 1, 0 = deterministic]: {}\n".format(
            param["randomFactorSpacing"]
        )
    )
    fid.write(
        "Length of coastline normals (m)                                                              : {}\n".format(
            param["coastlineNormalLength"]
        )
    )
    fid.write(
        "Maximum number of 'cape' normals                                                             : {}\n".format(
            param["maxCapeNormals"]
        )
    )
    fid.write("\n")
    fid.write(
        "Start depth for wave calcs (ratio to deep water wave height)                                 : 10         ; was 10 30 15\n"
    )
    fid.write("\n")
    fid.write(
        "; Flood options --------------------------------------------------------------------------------------------------------\n"
    )
    if not "floodLines" in param.keys():
        param["floodLines"] = ""
        param["runupFormula"] = ""
        param["floodLocation"] = ""
        param["floodInputLocation"] = ""

    fid.write(
        "Flood coastline to output                                                                    : {}\n".format(
            param["floodLines"]
        )
    )
    fid.write(
        "Run-up equation                                                                              : {}\n".format(
            param["runupFormula"]
        )
    )
    fid.write(
        "Characteristic locations of flood                                                            : {}\n".format(
            param["floodLocation"]
        )
    )
    fid.write(
        "Flood input location                                                                         : {}\n".format(
            param["cmeFloodLocation"]
        )
    )
    fid.close()

    return


# def coastalme_basement(param):
#     """[summary]

#     Args:
#         params ([type]): [description]

#     Returns:
#         [type]: [description]
#     """

#     import rasterio

#     sedimentOutputFileNames = {
#         "Sedimentos no consolidados muy finos": "uncons_sed_fine_layer_1_002.tif",
#         "Sedimentos no consolidados finos-medios": "uncons_sed_sands_layer_1_002.tif",
#         "Sedimentos no consolidados medios-gruesos": "uncons_sed_coarse_layer_1_002.tif",
#         "Bolos y/o bloques y/o encostramientos": "cons_sed_sand_layer_1_002.tif",
#         "Afloramientos rocosos masivos": "cons_sed_coarse_layer_1_002.tif",
#     }
#     sedimentInputFileNames = {
#         "Sedimentos no consolidados muy finos": "non_consolidated_fine.asc",
#         "Sedimentos no consolidados finos-medios": "non_consolidated_sand.asc",
#         "Sedimentos no consolidados medios-gruesos": "non_consolidated_coarse.asc",
#         "Bolos y/o bloques y/o encostramientos": "consolidated_sand.asc",
#         "Afloramientos rocosos masivos": "consolidated_coarse.asc",
#     }
#     for sedimentType in sedimentOutputFileNames.keys():
#         if sedimentType in param.keys():
#             bathy = rasterio.open(
#                 os.path.join(
#                     param["outputFolder"], sedimentOutputFileNames[sedimentType]
#                 )
#             ).read(1)
#             save.to_ascii(
#                 bathy,
#                 param["nodesY"],
#                 param["nodesX"],
#                 param["cellSize"],
#                 os.path.join(
#                     param["inputFolder"],
#                     sedimentInputFileNames[sedimentType][:-4]
#                     + "_"
#                     + str(param["loc"]).zfill(4)
#                     + ".asc",
#                 ),
#             )

#     return

# def reactivate_coastalme(data, parameters, stormStats, folder, previous_no):
#     """[summary]

#     Args:
#         data ([type]): [description]
#         parameters ([type]): [description]
#         stormStats ([type]): [description]
#         folder ([type]): [description]
#         previous_no ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     import glob2

#     list_files = glob2.glob(
#         os.path.join(parameters["outputFolder"], "uncons_sed_coarse_*.tif")
#     )

#     no_profile = 1e6
#     if not list_files:
#         print("Not found any previous file.")
#     else:
#         file_ = max(list_files, key=os.path.getctime)
#         no_profile = int(file_.split("_")[-1][:3])

#         file_names = [
#             "uncons_sed_fine_",
#             "uncons_sed_sand_",
#             "uncons_sed_coarse_",
#             "cons_sed_fine_",
#             "cons_sed_sand_",
#             "cons_sed_coarse_",
#         ]
#         for fname in file_names:
#             file_ = file_[:13] + fname + str(no_profile).zfill(3) + ".tif"

#             bathy = read.asci_tiff(file_, type_="matrix")
#             save.to_esriascii(
#                 bathy["z"],
#                 parameters["nodesX"],
#                 parameters["nodesY"],
#                 parameters["cellSize"],
#                 file_,
#                 nodata_value=-9999,
#             )

#         if no_profile == 999:
#             no_profile = previous_no

#         saveDates = (
#             pd.concat([stormStats["storm_ini"], stormStats["storm_end"]])
#             .sort_values()
#             .reset_index(drop=True)
#         )
#         saveTimes = ((saveDates - data.index[0]).dt.total_seconds() / 3600).loc[
#             no_profile:
#         ]
#         parameters["saveTimes"] = ""
#         for i in saveTimes:
#             parameters["saveTimes"] += "{} ".format(int(i))
#         parameters["saveTimes"] += "hours"

#         parameters["inputFolder"] = parameters["inputFolder"][:-4] + str(
#             no_profile
#         ).zfill(4)
#         parameters["outputFolder"] = parameters["outputFolder"][:-4] + str(
#             no_profile
#         ).zfill(4)
#         parameters["non_consolidate_coarse"] = (
#             parameters["non_consolidate_coarse"][:-7]
#             + str(no_profile).zfill(3)
#             + ".asc"
#         )

#         # Files are written
#         auxiliar.mkdir(parameters["inputFolder"])
#         auxiliar.mkdir(parameters["outputFolder"])

#         data.loc[saveDates.loc[no_profile] :, "eta"].to_csv(
#             os.path.join(parameters["inputFolder"], parameters["sealevelFileName"]),
#             header=False,
#             index=False,
#         )
#         coastalme_seastates(data, parameters)
#         coastalme_cme(parameters, data)
#         coastalme_ini(parameters)

#     return no_profile


def cme_info():
    # Files that can be saved if user specifies
    # RASTER_AVG_SEA_DEPTH_NAME = "avg_sea_depth"
    # RASTER_AVG_WAVE_HEIGHT_NAME = "avg_wave_height"
    # RASTER_BEACH_PROTECTION_NAME = "beach_protection"
    # RASTER_BASEMENT_ELEVATION_NAME = "basement_elevation"
    # RASTER_SUSP_SED_NAME = "susp_sed"
    # RASTER_AVG_SUSP_SED_NAME = "avg_susp_sed"
    # RASTER_FINE_UNCONS_NAME = "uncons_sed_fine"
    # RASTER_SAND_UNCONS_NAME = "uncons_sed_sand"
    # RASTER_COARSE_UNCONS_NAME = "uncons_sed_coarse"
    # RASTER_FINE_CONS_NAME = "cons_sed_fine"
    # RASTER_SAND_CONS_NAME = "cons_sed_sand"
    # RASTER_COARSE_CONS_NAME = "cons_sed_coarse"
    # RASTER_COAST_NAME = "rcoast"
    # RASTER_COAST_NORMAL_NAME = "rcoast_normal"
    # RASTER_ACTIVE_ZONE_NAME = "active_zone"
    # RASTER_CLIFF_COLLAPSE_NAME = "cliff_collapse"
    # RASTER_TOTAL_CLIFF_COLLAPSE_NAME = "total_cliff_collapse"
    # RASTER_CLIFF_COLLAPSE_DEPOSITION_NAME = "cliff_collapse_deposition"
    # RASTER_TOTAL_CLIFF_COLLAPSE_DEPOSITION_NAME = "total_cliff_collapse_deposition"
    # RASTER_POLYGON_NAME = "polygon_raster"
    # RASTER_POTENTIAL_PLATFORM_EROSION_MASK_NAME = "potential_platform_erosion_mask"
    # RASTER_INUNDATION_MASK_NAME = "inundation_mask"
    # RASTER_BEACH_MASK_NAME = "beach_mask"
    # RASTER_INTERVENTION_CLASS_NAME = "intervention_class"
    # RASTER_INTERVENTION_HEIGHT_NAME = "intervention_height"
    # SHADOW_ZONE_CODES_NAME = "shadow_zone_codes"

    # VECTOR_NORMALS_CODE = "normals"
    # VECTOR_INVALID_NORMALS_CODE = "invalid_normals"
    # VECTOR_AVG_WAVE_ANGLE_CODE = "avg_wave_angle"
    # VECTOR_COAST_CURVATURE_CODE = "coast_curvature"
    # VECTOR_WAVE_ENERGY_SINCE_COLLAPSE_CODE = "wave_energy"
    # VECTOR_MEAN_WAVE_ENERGY_CODE = "mean_wave_energy"
    # VECTOR_BREAKING_WAVE_HEIGHT_CODE = "breaking_wave_height"
    # VECTOR_POLYGON_NODE_SAVE_CODE = "node"
    # VECTOR_POLYGON_BOUNDARY_SAVE_CODE = "polygon"
    # VECTOR_PLOT_CLIFF_NOTCH_SIZE_CODE = "cliff_notch"
    # VECTOR_PLOT_SHADOW_ZONE_BOUNDARY_CODE = "shadow_boundary"

    # SEAAREATSCODE = "seaarea"
    # STILLWATERLEVELCODE = "waterlevel"
    # EROSIONTSCODE = "erosion"
    # DEPOSITIONTSCODE = "deposition"
    # SEDLOSTFROMGRIDTSCODE = "sedlost"
    # SUSPSEDTSCODE = "suspended"
    return
