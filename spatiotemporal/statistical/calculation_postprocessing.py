from pathlib import Path

import numpy as np
import pandas as pd
from logging import logger
from glob2 import glob
import xarray as xr


from .functions import (
    boundary_polygon,
    obtain_lines,
    refinement,
    sieve,
    band,
    save_isolines,
    # load_monthly_index,
    pick_first_season_file,
    save_matrix_to_netcdf,
    calculate_grid_angle_and_create_rotated_mesh,
    read_asci_tiff,  # TODO 03: check marinetools.utils.read_asci_tiff
)

def check_inputs(info):
    # Comprobar que las rutas existen
    if not info["input_dtm"].exists():
        logger.error("El DTM de entrada no existe: %s", info["input_dtm"])
        raise FileNotFoundError(f"El DTM de entrada no existe: {info['input_dtm']}")
    
    info["dtm_filenames"] = []
    for sim in range(1, info["no_sims"]):
        sim_path = info["directories"]["input_dtm"].parent / str(sim).zfill(4)
        if not sim_path.exists():
            logger.error("El directorio de la simulación %s no existe: %s", sim, sim_path)
            raise FileNotFoundError(f"El directorio de la simulación {sim} no existe: {sim_path}")
        else:
            info["dtm_filenames"][sim] = glob(sim_path)
            if len(info["dtm_filenames"][sim]) == 0:
                logger.warning("No hay TIFFs para sim %s en %s" % (sim, info["directories"]["input_dtm"]))
                continue
    
    if "fld_portion" not in info:
        info["fld_portion"] = 3  # Valor por defecto si no se proporciona
    elif info["fld_portion"] <= 0:
        logger.error("La porción de FLD debe ser un entero positivo.")
        raise ValueError("La porción de FLD debe ser un entero positivo.")
    
    # Check the months for high and low seasons
    if not "seasons" in info:
        info["seasons"] = {
            "AN": "annual",
            "TA": [4, 5, 6, 7, 8, 9],
            "TB": [1, 2, 3, 10, 11, 12]
        }
    
    info["auxiliary_folders"] = {}
    info["auxiliary_folders"]["series"] = Path(info["directories"]["output_path"]) / info["directories"]["temp_folder"] / "series"

    
    # Crear directorios de salida si no existen
    # info['out_tH_raw'].mkdir(parents=True, exist_ok=True)
    # info['out_year_raw'].mkdir(parents=True, exist_ok=True)

    return

def levels_years_tH(info):

    for season in info["temporal"]["scales"]:
        max_, level_ = 0, 1e10

        df_levels_per_year = pd.DataFrame(
            columns=np.array(info["project"]["no_sims"]),
            index=np.array(
                np.linspace(info["temporal"]["initial_year"], info["temporal"]["final_year"], (info["temporal"]["final_year"] - info["temporal"]["initial_year"]) + 1)
            ).astype("int"),
        )
        df_levels_per_tH = pd.DataFrame(columns=np.array(info["project"]["no_sims"]), index=np.array(info["temporal"]["tHs"]))

        for i in info["project"]["no_sims"]:
            total_level = pd.read_csv(
                f"{info['directories']['input_dtm']}/{str(i).zfill(4)}/levels.csv",
                sep=",",
                header=None,
            )

            # Remove outliers
            # level_threshold = total_level.quantile(info["parameters"]["total_level_perc"])
            # max_level = np.max(total_level)
            # if max_ < max_level:
            #     max_ = max_level

            # if level_ > level_threshold:
            #     level_ = level_threshold

            # total_level = total_level.copy()
            # total_level.loc[total_level > level_threshold] = np.nan

            max_level = total_level.groupby(
                [(total_level.index.year), (total_level.index.month)]
            ).max()

            for tH in info["temporal"]["tHs"]:
                max_years = []
                for _, j in enumerate(range(info["temporal"]["initial_year"], info["temporal"]["final_year"] + 1)):
                    # TODO01: Check that max_level has data for all months and years
                    if season == "TA":
                        df_year = max_level.loc[j][info["seasons"]["TA"]]

                    elif season == "TB":
                        # if j < info["temporal"]["final_year"]-1:
                        df_year = max_level.loc[j][info["seasons"]["TB"]]
                        # np.concatenate(
                        #     (
                        #         np.array(max_level.loc[j][info["seasons"]["TB"]]).flatten(),
                        #         np.array(max_level.loc[j + 1][:2]).flatten(),
                        #     ),
                        #     axis=0,
                        # )

                        # else:
                        #     df_year = max_level.loc[j][9:]

                    # else:
                    #     if j < info["temporal"]["final_year"]-1:
                    #         df_year = np.concatenate(
                    #             (
                    #                 np.array(max_level.loc[j][3:]).flatten(),
                    #                 np.array(max_level.loc[j + 1][:2]).flatten(),
                    #             ),
                    #             axis=0,
                    #         )

                    #     else:
                    #         df_year = max_level.loc[j][9:]

                    max_year = np.max(df_year)
                    df_levels_per_year.loc[j, i] = max_year
                    max_years.append(max_year)

                max_tH = np.max(max_years[: (tH - info["temporal"]["initial_year"] - 2)])
                df_levels_per_tH.loc[tH, i] = max_tH

        # Se guardan los datos
        df_levels_per_year.to_csv(
            Path(info["auxiliary_folders"]["series"]) / f"level_{season}_per_year.csv"
        )

        df_levels_per_tH.to_csv(
            Path(info["auxiliary_folders"]["series"]) / f"level_{season}_per_tH.csv"
        )
        return

def binary_matrix(info):

    # Banda de refinamiento (una vez)
    initial_file = info["dtm_files"][0]
    da_dem = read_asci_tiff(initial_file, type_="matrix")

    levels = pd.read_csv(info["auxiliary_folders"]["series"] / f"level_{season}_per_year.csv", index_col=0)
    band_, coords = band(da_dem, levels, info["refinement_size"], info["stretch_orientation"])
    X, Y, _ = calculate_grid_angle_and_create_rotated_mesh(da_dem, coords["X"], coords["Y"], info["refinement_size"])

    # ---------------- Por season ----------------
    for season in info["temporal"]["scales"]:
        logger.info("---- Season %s ----" % season)

        # Niveles por tH y por año
        # df_levels_per_tH = pd.read_csv(info["auxiliary_folders"]["series"] / f"level_{season}_per_tH.csv", index_col=0)
        df_levels_per_year = pd.read_csv(info["auxiliary_folders"]["series"] / f"level_{season}_per_year.csv", index_col=0)

        sim_mask_by_year = {}
        X_ref = Y_ref = None
    
        for sim_no in info["project"]["no_sims"]:
            logger.info("Prep sim %s"% sim_no)

            sim_mask_by_year[sim_no] = {}

            for year in range(info["temporal"]["initial_year"], max(info["temporal"]["scales"]) + 1):
                logger.info("Season %s - sim %s - year %s"% (season, sim_no, year))
                file_of_year = pick_first_season_file(df_monthly, season, year)
                # if file_of_year is None:
                #     logger.warning("Sin fichero mensual para año %s (sim %s, season %s). 0s." % (year, sim_no, season))
                #     mask = np.zeros((x_ele, y_ele), dtype=bool)
                # else:
                level_year = df_levels_per_year.loc[year, str(sim_no)]
                file_path = Path(dir_results_CME) / f"{str(sim_no).zfill(4)}/{file_of_year}"

                da_dem = read_asci_tiff(file_path, type_="matrix")
                Z = refinement(da_dem, band_, coords)

                # X, Y, Z = refinement(file_path, top_elevation, refinement_size, band_, stretch_orientation)
                Z = boundary_polygon(Z)
                # if info.loc["boundary_polygon", stretch]:
                #     polygon = geopandas.read_file(polygons_path / f"{stretch}.gpkg")
                #     Z = boundary_polygon(X, Y, Z, polygon, info.loc["boundary_polygon_level", stretch])

                data_year = sieve(level_year - Z, info["parameters"]["fld_portion"]).astype(np.int16)
                data_year = (1 - data_year).astype(np.int8)  # invertir
                mask = (data_year > 0)

                if X_ref is None:
                    X_ref, Y_ref = X, Y

                sim_mask_by_year[sim_no][j] = mask
        
        out_nc = dir_matrix_stretch / f"yearly_matrix_l_orilla_{season}_sim_{str(sim_no).zfill(2)}.nc"
        save_matrix_to_netcdf(sim_mask_by_year[sim_no], season, {"x": X_ref, "y": Y_ref}, j, stretch, sim_no, out_nc)
    logger.info("Guardado %s -> %s", season, out_nc)
    return


def get_isolines(stretch):
    # ---------------- Por season ----------------
    for season in temporal_scale:
        # ============== tH (envolvente OR 2020..tH) ==============
        for tH in tHs:
            logger.info("======== tH %s (season %s) ========"% (tH, season))
            data_count = np.zeros((x_ele, y_ele), dtype=np.int32)

            for i in sims:
                years = [y for y in range(year_ini, tH + 1) if y in sim_mask_by_year[i]]
                sim_mask = np.any(np.stack([sim_mask_by_year[i][y] for y in years], axis=0), axis=0) if years else np.zeros((x_ele, y_ele), bool)
                data_count += sim_mask.astype(np.int32)

                lines_sim = obtain_lines(X_ref, Y_ref, sim_mask.astype(np.int8), [0.5])
                save_isolines(lines_sim, crs, out_tH_raw / f"l_olas_{season}_tH_{tH}_sim_{str(i).zfill(2)}.gpkg")

            prob = data_count / no_sims
            ci_lo = 0.5 - 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            ci_hi = 0.5 + 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            for line, stat in zip(obtain_lines(X_ref, Y_ref, prob, [0.05, ci_lo, 0.5, ci_hi, 0.95]),
                                  ["5", "lower", "mean", "upper", "95"]):
                save_isolines(line, crs, out_tH_raw / f"l_olas_{season}_tH_{tH}_{stat}.gpkg")

        # ============== Anuales (por año) ==============
        logger.info("======== Exportando líneas ANUALES (season %s) ========" % season)
        for year in range(year_ini, max_tH + 1):
            data_count_year = np.zeros((x_ele, y_ele), dtype=np.int32)
            for i in sims:
                sim_mask = sim_mask_by_year[i].get(year, np.zeros((x_ele, y_ele), bool))
                data_count_year += sim_mask.astype(np.int32)
                lines_sim = obtain_lines(X_ref, Y_ref, sim_mask.astype(np.int8), [0.5])
                save_isolines(lines_sim, crs, out_year_raw / f"l_olas_{season}_year_{year}_sim_{str(i).zfill(2)}.gpkg")

            prob_year = data_count_year / no_sims
            ci_lo = 0.5 - 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            ci_hi = 0.5 + 1.96 * np.sqrt(0.5 * 0.5 / no_sims)
            for line, stat in zip(obtain_lines(X_ref, Y_ref, prob_year, [0.05, ci_lo, 0.5, ci_hi, 0.95]),
                                  ["5", "lower", "mean", "upper", "95"]):
                save_isolines(line, crs, out_year_raw / f"l_olas_{season}_year_{year}_{stat}.gpkg")


    return

def main(info):
    check_inputs(info)
    for k, _ in enumerate(info["project"]["index"]):
        logger.info("Inicio del postprocesamiento para el proyecto %s ---", info["project"]["index"][k])

        binary_matrix(info)
        get_isolines(info)
        smooth_isolines(info)
        clip_beaches(info)

    return


if __name__ == "__main__":
    main()