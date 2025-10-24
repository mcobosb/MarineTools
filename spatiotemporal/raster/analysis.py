from logging import logger
from pathlib import Path

import numpy as np
import pandas as pd
from glob2 import glob
from marinetools.utils.read import asci_tiff

from .functions import (
    band,
    calculate_grid_angle_and_create_rotated_mesh,
    refinement,
    save_matrix_to_netcdf,
)


def check_inputs(info):
    """
    Validate and prepare input configuration for marine spatial analysis processing.

    Performs comprehensive validation of input parameters, file paths, and configuration
    settings required for the marine tools spatial analysis workflow. Sets up default
    values, creates necessary directory structures, and validates data availability.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing project parameters, file paths, and
        processing settings. Expected keys include:
        - "input_dtm" : Path
            Path to the input Digital Terrain Model file
        - "no_sims" : int
            Number of simulations to process
        - "directories" : dict
            Dictionary with paths for input/output directories
        - "index" : list
            List of indices for processing
        - "region_name" : str
            Name of the region being processed
        - "fld_portion" : int, optional
            Flood portion parameter (default: 3)
        - "seasons" : dict, optional
            Seasonal month definitions

    Returns
    -------
    None
        Function modifies the input dictionary in-place.

    Raises
    ------
    FileNotFoundError
        If required input files or directories do not exist.
    ValueError
        If fld_portion parameter is not a positive integer.

    Notes
    -----
    The function performs the following validations and setup:
    1. Verifies existence of input DTM file
    2. Checks simulation directories and catalogs available TIFF files
    3. Sets default values for optional parameters
    4. Defines seasonal month groupings if not provided
    5. Creates auxiliary output directory structure
    6. Validates flood portion parameter

    TODO items:
    - Check that level series files exist
    - Validate that max_level has data for all months and years
    """
    # Check that paths exist
    if not info["input_dtm"].exists():
        logger.error("Input DTM does not exist: %s", info["input_dtm"])
        raise FileNotFoundError(f"Input DTM does not exist: {info['input_dtm']}")

    info["dtm_filenames"] = []
    for sim in range(1, info["no_sims"]):
        sim_path = info["directories"]["input_dtm"].parent / str(sim).zfill(4)
        if not sim_path.exists():
            logger.error("Simulation %s directory does not exist: %s", sim, sim_path)
            raise FileNotFoundError(
                f"Simulation {sim} directory does not exist: {sim_path}"
            )
        else:
            info["dtm_filenames"][sim] = glob(sim_path)
            if len(info["dtm_filenames"][sim]) == 0:
                logger.warning(
                    "No TIFFs found for sim %s in %s"
                    % (sim, info["directories"]["input_dtm"])
                )
                raise ValueError(
                    f"No TIFFs found for sim {sim} in {info['directories']['input_dtm']}"
                )

    # Check the months for high and low seasons
    if not "seasons" in info:
        info["seasons"] = {
            "AN": "annual",
            "TA": [4, 5, 6, 7, 8, 9],
            "TB": [1, 2, 3, 10, 11, 12],
        }

    for indexes in info["index"]:
        # Create auxiliary output directories
        info["auxiliary_folders"] = {}
        info["auxiliary_folders"][indexes] = {}

        info["auxiliary_folders"][indexes]["matrix"] = (
            Path(info["directories"]["output_path"])
            / info["directories"]["temp_folder"]
            / "matrix"
        )

        # Check and create auxiliary folders
        for key in info["auxiliary_folders"][indexes]:
            info["auxiliary_folders"][indexes][key].mkdir(parents=True, exist_ok=True)

    # Check that dates match between simulations
    for sim in info["project"]["no_sims"]:
        try:
            data = pd.read_csv(
                f"{info['directories']['input_dtm']}/{str(sim).zfill(4)}/levels.csv",
                sep=",",
                header=None,
            )
        except FileNotFoundError:
            logger.error("Level series file not found for simulation %s", sim)
            raise FileNotFoundError(f"Level series file not found for simulation {sim}")

        if sim == 1:
            info["dates"] = data.index.tolist()
        else:
            if np.all(info["dates"] != data.index.tolist()):
                logger.error(
                    "Level dates do not match between simulations 1 and %s.", sim
                )
                raise ValueError(
                    "Level dates do not match between simulations 1 and %s.", sim
                )

    # Check horizon times are in DTM dates
    if "horizon_times" in info:
        for horizon_time in info["horizon_times"]:
            if horizon_time not in info["dates"]:
                logger.error(
                    "Horizon time %s do not match available dates.", horizon_time
                )
                raise ValueError(
                    "Horizon time %s do not match available dates.", horizon_time
                )

    # Check return periods are positive
    if "return_periods" in info:
        for rp in info["return_periods"]:
            if rp <= 0:
                logger.error("Return period %s is not positive.", rp)
                raise ValueError("Return period %s is not positive.", rp)

    # Check mesh size to alert about memory usage
    dem, _ = asci_tiff(info["directories"]["input_dtm"][1][0], type_="matrix")
    nx, ny = dem["x"].size, dem["y"].size
    total_points = nx * ny
    if total_points > 10**7:  # 10 million points
        logger.warning(
            f"Very large mesh size: {nx}x{ny} = {total_points:,} points. "
            f"This may cause memory issues. "
            f"Consider increasing grid_size (current: {info['grid_size']}) to reduce resolution."
        )
    elif total_points > 10**6:  # 1 million points
        logger.info(
            f"Considerable mesh size: {nx}x{ny} = {total_points:,} points. "
            f"Monitor memory usage."
        )

    return


def pretratement(info):
    """
    Pre-treatment steps for marine spatial analysis processing.

    This function performs initial pre-treatment steps required for the marine tools
    spatial analysis workflow. It prepares necessary data structures and variables
    based on the provided configuration.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing project parameters, file paths, and
        processing settings.

    Returns
    -------
    levels : np.ndarray
        Array containing the absolute minimum and maximum values across all simulations.
        Format: [min_value, max_value]

    Notes
    -----
    The function performs the following pre-treatment steps:
    1. Reads and processes level data from all simulations.
    2. Calculates absolute minimum and maximum values across all simulations.
    3. Returns these extreme values for use in subsequent processing steps.

    TODO items:
    - Implement additional pre-treatment steps as required.
    """

    if info["project"]["refinement"]:
        min_values = []
        max_values = []

        for sim in info["project"]["no_sims"]:
            data = pd.read_csv(
                f"{info['directories']['input_dtm']}/{str(sim).zfill(4)}/levels.csv",
                sep=",",
                header=None,
            )

            # Get min and max values from this simulation
            sim_min = data.min().min()  # Get absolute minimum from all columns and rows
            sim_max = data.max().max()  # Get absolute maximum from all columns and rows

            min_values.append(sim_min)
            max_values.append(sim_max)

        # Calculate absolute minimum and maximum across all simulations
        absolute_min = min(min_values)
        absolute_max = max(max_values)

        # Create band_levels array with min and max values
        info["band_levels"] = np.array([absolute_min, absolute_max])

    return info


def binary_matrix(info):
    """_summary_

    Args:
        info (_type_): _description_
    """

    # Banda de refinamiento (una vez)
    initial_file = info["dtm_files"][0]
    da_dem = asci_tiff(initial_file, type_="matrix")

    if info["refinement"]:
        # Create the band for refinement
        band_, coords = band(da_dem, info["band_levels"], info["stretch_orientation"])

        # Create refined grid coordinates X and Y for interpolation
        coords["X"], coords["Y"], _ = calculate_grid_angle_and_create_rotated_mesh(
            coords["X"], coords["Y"], info["refinement_size"]
        )

    # Initialize bin_mask dictionary
    bin_mask = {}
    # Initialize date loop variable to True
    date_loop = True
    for sim_no in info["project"]["no_sims"]:
        # Initialize bin_mask for each simulation
        bin_mask[sim_no] = {}

        if info["index"] == "shoreline":
            level = info["PMVE"] + df_levels_per_date.loc[date, "slr"]
        elif info["index"] == "wave_extend":
            level = df_levels_per_date.loc[date, "total_water_level"]
        elif info["index"] == "flooded_area":
            level = df_levels_per_date.loc[date, "slr"]
        elif info["index"] == "permanent_flood":
            level = info["BMVE"] + df_levels_per_date.loc[date, "slr"]
        elif info["index"] == "mean_level":
            level = df_levels_per_date.loc[date, "slr"]
        elif info["index"] == "flood_RP":
            date_loop = False

        if date_loop:
            # ---------------- Matrix with mask per year ---------------------------------------
            df_levels_per_date = pd.read_csv(
                f"{info['directories']['input_dtm']}/{str(sim_no).zfill(4)}/levels.csv",
                sep=",",
                header=None,
            )

            for file_no, date in enumerate(info["dates"]):
                logger.info("Sim %s - date %s" % (sim_no, date))

                # Get file path for current simulation and date
                file_path = info["dtm_filenames"][sim_no][file_no]

                # Reading DEM
                da_dem = asci_tiff(file_path, type_="matrix")
                if info["refinement"]:
                    # Refinement
                    Z = refinement(da_dem, band_, coords)
                else:
                    Z = da_dem["z"]

                # Mask of data below level
                mask = Z < level

                # Include mask into bin_mask - simulations and year
                bin_mask[sim_no][date] = mask
        else:
            for return_period in info["return_periods"]:
                for horizon_time in info["horizon_times"]:
                    # Load levels for return period and horizon time
                    df_levels_rp = pd.read_csv(
                        f"{info['directories']['input_dtm']}/{str(sim_no).zfill(4)}/levels_rp_tH.csv"
                    )
                    level_values = df_levels_rp.loc[
                        : (horizon_time - info["year_ini"]), str(return_period)
                    ]
                    level = np.max(level_values)

                    # Get file path for current simulation and date
                    file_path = glob.glob(
                        info["dtm_filenames"][sim_no] + f"/**{horizon_time}.tif"
                    )

                    # Reading DEM
                    da_dem = asci_tiff(file_path, type_="matrix")
                    if info["refinement"]:
                        # Refinement
                        Z = refinement(da_dem, band_, coords)
                    else:
                        Z = da_dem["z"]

                    # Mask of data below level
                    mask = Z < level

                    # Include mask into bin_mask - simulations and year
                    bin_mask[sim_no][date] = mask

        output_filename = (
            info["auxiliary_folders"][info["index"]]["matrix"]
            / f"{info["index"]}_sim_{str(sim_no).zfill(2)}.nc"
        )

        save_matrix_to_netcdf(
            bin_mask[sim_no],
            coords,
            info["dates"],
            info,
            sim_no,
            output_filename,
        )
        logger.info("Saving %s", output_filename)
    return


def get_isolines(info):
    # ---------------- Por season ----------------
    for season in temporal_scale:
        # ============== tH (envolvente OR 2020..tH) ==============
        for tH in tHs:
            logger.info("======== tH %s (season %s) ========" % (tH, season))
            data_count = np.zeros((x_ele, y_ele), dtype=np.int32)

            for i in sims:
                years = [y for y in range(year_ini, tH + 1) if y in sim_mask_by_year[i]]
                sim_mask = (
                    np.any(
                        np.stack([sim_mask_by_year[i][y] for y in years], axis=0),
                        axis=0,
                    )
                    if years
                    else np.zeros((x_ele, y_ele), bool)
                )
                data_count += sim_mask.astype(np.int32)

                lines_sim = obtain_lines(X_ref, Y_ref, sim_mask.astype(np.int8), [0.5])
                save_isolines(
                    lines_sim,
                    crs,
                    out_tH_raw / f"l_olas_{season}_tH_{tH}_sim_{str(i).zfill(2)}.gpkg",
                )

            prob = data_count / no_sims
            ci_lo = 0.5 - 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            ci_hi = 0.5 + 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            for line, stat in zip(
                obtain_lines(X_ref, Y_ref, prob, [0.05, ci_lo, 0.5, ci_hi, 0.95]),
                ["5", "lower", "mean", "upper", "95"],
            ):
                save_isolines(
                    line, crs, out_tH_raw / f"l_olas_{season}_tH_{tH}_{stat}.gpkg"
                )

        # ============== Anuales (por año) ==============
        logger.info("======== Exportando líneas ANUALES (season %s) ========" % season)
        for year in range(year_ini, max_tH + 1):
            data_count_year = np.zeros((x_ele, y_ele), dtype=np.int32)
            for i in sims:
                sim_mask = sim_mask_by_year[i].get(year, np.zeros((x_ele, y_ele), bool))
                data_count_year += sim_mask.astype(np.int32)
                lines_sim = obtain_lines(X_ref, Y_ref, sim_mask.astype(np.int8), [0.5])
                save_isolines(
                    lines_sim,
                    crs,
                    out_year_raw
                    / f"l_olas_{season}_year_{year}_sim_{str(i).zfill(2)}.gpkg",
                )

            prob_year = data_count_year / no_sims
            ci_lo = 0.5 - 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            ci_hi = 0.5 + 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            for line, stat in zip(
                obtain_lines(X_ref, Y_ref, prob_year, [0.05, ci_lo, 0.5, ci_hi, 0.95]),
                ["5", "lower", "mean", "upper", "95"],
            ):
                save_isolines(
                    line, crs, out_year_raw / f"l_olas_{season}_year_{year}_{stat}.gpkg"
                )

    return


def main(info):
    check_inputs(info)
    for k, _ in enumerate(info["project"]["index"]):
        logger.info(
            "Inicio del postprocesamiento para el proyecto %s ---",
            info["project"]["index"][k],
        )

        binary_matrix(info)
        get_isolines(info)
        smooth_isolines(info)
        clip_beaches(info)

    return


if __name__ == "__main__":
    main()
