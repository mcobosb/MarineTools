import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr
from loguru import logger
from marinetools.processes import load, waves, write
from marinetools.spatial import geospatial as gsp
from marinetools.utils import auxiliar, read, save
from scipy.interpolate import griddata, interp1d
from scipy.special import erfc


def create_db(params, data, mesh="global", time_=None, vars_=None, method="nearest"):
    """Create the project folder with the initializated files for Swan and Copla

    Args:
        * params: the dictionary with the model paramaters
        * data: pd.DataFrame with the time series of the boundary data

    Return:
        A xr.DataSet of th project
    """
    lon, lat = create_mesh(params, mesh)
    if isinstance(vars_, str):
        vars_ = [vars_]

    db = create_xarray(lon, lat, time_=time_, vars_=vars_)
    if time_ is None:
        db["depth"][:, :] = griddata(
            data.loc[:, ["x", "y"]], data.loc[:, "z"].values, (lon, lat), method=method
        )
    else:
        db["depth"][:, :, 0] = griddata(
            data.loc[:, ["x", "y"]],
            data.loc[:, "z"].values,
            (lon, lat),
            method="linear",
        )

    return db


def create_mesh(params, mesh="global"):
    """[summary]

    Args:
        params ([type]): [description]
        mesh (str, optional): [description]. Defaults to 'global'.

    Returns:
        [type]: [description]
    """

    dx = np.linspace(0, params[mesh + "_length_x"], params[mesh + "_nodes_x"])
    dy = np.linspace(0, params[mesh + "_length_y"], params[mesh + "_nodes_y"])
    dx, dy = np.meshgrid(dx, dy)

    x, y = gsp.rotate_coords(dx, dy, params[mesh + "_angle"])
    lon, lat = x + params[mesh + "_coords_x"], y + params[mesh + "_coords_y"]

    return lon, lat


def create_xarray(x, y, time_=None, vars_=None):
    """Create a 2d or 3d xarray with the same fields than vars_

    Args:
        * x:

    """
    dict_ = {}

    if vars_ is None:
        vars_ = "depth", "Hs", "DirM", "U", "DirU", "qc", "ql", "Setup"
    elif vars_ == "full":
        vars_ = (
            "depth",
            "Sb",
            "Hs",
            "Tp",
            "DirM",
            "Setup",
            "cp",
            "L",
            "U",
            "DirU",
            "Dr",
            "Db",
            "Df",
            "Qst",
            "Qbt",
            "qc",
            "ql",
        )

    if time_ is not None:
        if isinstance(time_, int):
            zeros = np.zeros(np.append(np.shape(x), 1))
            time_ = [time_]
        else:
            zeros = np.zeros(np.append(np.shape(x), len(time_)))
        for i in vars_:
            dict_[i] = (["lon", "lat", "time"], zeros.copy())
        db = xr.Dataset(
            dict_,
            coords={"x": (["lon", "lat"], x), "y": (["lon", "lat"], y), "time": time_},
        )
    else:
        zeros = np.zeros(np.shape(x))
        for i in vars_:
            dict_[i] = (["lon", "lat"], zeros.copy())
        db = xr.Dataset(
            dict_, coords={"x": (["lon", "lat"], x), "y": (["lon", "lat"], y)}
        )

    return db


def slopes(data, var_="DirU"):
    """Compute the slopes of the bottom in the direction of var_

    Args:
        * data: pd.DataFrame with the geometry
        * var_: String with the circular variable

    Return:
        A np.Array with the slopes
    """
    xi, yi = (
        (data["x"] + np.cos(data[var_] * np.pi / 180)).reshape(-1),
        (data["y"] + np.sin(data[var_] * np.pi / 180)).reshape(-1),
    )
    Sb = griddata(
        (data["x"].reshape(-1), data["y"].reshape(-1)),
        data["depth"].reshape(-1),
        (xi, yi),
    ) - data["depth"].reshape(-1)
    Sb = Sb.reshape(np.shape(data["x"]))
    Sb[np.isnan(Sb)] = 0
    return Sb


def sediment_transport_Kobayashi(j, data, grid, params):
    """Suspended- and bed-load transport following Kobayashi et al (2008)

    Args:
        * j: pd.DatetimeIndex
        * data: a pd.DataFrame with the input time series
        * grid: a dictionary with required data 2D-np.arrays
        * param: a dictionary with constants

    Return:
        A dictionary with the computation
    """

    G = 9.8091
    grid["D"] = grid["depth"] + grid["Setup"]
    grid["D"][grid["D"] < 0.1] = np.nan  # Total depth
    grid["cp"] = np.sqrt(G * grid["D"])  # Shallow water waves
    grid["sigma"] = data.loc[j, "Hs"] / np.sqrt(8) / grid["D"] * grid["cp"]
    grid["sigma"][np.isnan(grid["sigma"])] = 0

    grid["Rs"] = (2 / params["fb"]) ** (1 / 3) * params["wf"] / grid["sigma"]
    grid["Rs"][np.isnan(grid["Rs"])] = 0
    grid["Ps"] = 0.5 * erfc((grid["Rs"] + grid["U"]) / np.sqrt(2)) + 0.5 * erfc(
        (grid["Rs"] - grid["U"]) / np.sqrt(2)
    )

    grid["Hb"] = (
        0.88
        / grid["kp"]
        * np.tanh(
            params["gamma"]
            * grid["kp"]
            * grid["D"]
            / (0.88 * grid["cp"] * data.loc[j, "Tp"])
        )
    )  # Kobayashi 0.88
    grid["Hb"][grid["Hb"] / grid["D"] > 0.88] = (
        0.88 * grid["D"][grid["Hb"] / grid["D"] > 0.88]
    )
    grid["Hb"][np.isnan(grid["Hb"])] = 0

    #   Suspended load transport
    # --------------------------------------------------------------------------
    grid["Dr"] = (
        G
        * (params["rho"] * 0.9 * grid["Hb"] ** 2 * grid["cp"] / data.loc[j, "Tp"])
        * np.sin(params["br"])
    )  # Roller effect
    grid["Dr"][np.isnan(grid["Dr"])] = 0

    grid["arot"] = 1 / 3 * grid["Sb"] * data.loc[j, "Tp"] * np.sqrt(G / grid["D"])
    grid["arot"][np.isnan(grid["arot"])] = 0
    grid["Db"] = (
        params["rho"]
        * grid["arot"]
        * G
        * grid["Qb"]
        * grid["Hb"] ** 2
        / (4 * data.loc[j, "Tp"])
    )  # Battjes and Stive, 1985
    grid["Df"] = 0.5 * params["rho"] * G * params["fb"] * grid["U"] ** 3
    grid["Vs"] = (
        (params["eb"] * (grid["Db"] - grid["Dr"]) + params["ef"] * grid["Df"])
        * grid["Ps"]
        / (params["rho"] * G * (params["S"] - 1) * params["wf"])
    )
    grid["Qst"] = params["aK"] * grid["U"] * grid["Vs"] * np.sqrt(1 + grid["Sb"] ** 2)

    #    Bed-load transport
    # --------------------------------------------------------------------------
    grid["Rb"] = (
        np.sqrt(
            2 * G * (params["S"] - 1) * params["d50"] * params["Shic"] / params["fb"]
        )
        / grid["sigma"]
    )
    grid["Rb"][np.isnan(grid["Rb"])] = 0
    grid["Pb"] = 0.5 * erfc((grid["Rb"] + grid["U"]) / np.sqrt(2)) + 0.5 * erfc(
        (grid["Rb"] - grid["U"]) / np.sqrt(2)
    )

    grid["Gs"] = (np.tan(params["phi"]) - 2 * grid["Sb"]) / (
        np.tan(params["phi"]) - grid["Sb"]
    )
    grid["Gs"][grid["Sb"] < 0] = np.tan(params["phi"]) / (
        np.tan(params["phi"]) + grid["Sb"][grid["Sb"] < 0]
    )
    grid["Qbt"] = (
        params["bK"]
        * grid["Pb"]
        * grid["Gs"]
        * grid["sigma"] ** 3
        / (G * (params["S"] - 1))
    )  # Formulación de Meyer-Peter-Mueller, modificado

    # Cross-shore and alongshore sediment transport
    # --------------------------------------------------------------------------
    grid["qc"] = (
        (grid["Qbt"] + grid["Qst"])
        * np.sin((grid["DirU"] - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][0])
    grid["ql"] = (
        (grid["Qbt"] + grid["Qst"])
        * np.cos((grid["DirU"] - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][1])

    return grid


def sediment_transport_CERC(data, params, theta_c):
    """[summary]

    Args:
        data ([type]): [description]
            - Hr: breaking wave height (m)
            - thetar: breaking wave angle (deg)
        params ([type]): [description]
            - d50: mean sediment grain size (m)
            - rho: fresh-water density (kg/m3)
            - rho_s: sediment density (kg/m3)
            - n: porosity
            - gamma: breaking parameter


        thetac ([type]): beach angle

    Returns:
        [type]: [description]
    """

    data["alphar"] = np.deg2rad(
        np.min(
            np.asarray(
                [np.abs(data["thetar"] - theta_c), np.abs(theta_c - data["thetar"])]
            ),
            axis=0,
        )
    )
    data.loc[data["alphar"] > np.pi / 2, "alphar"] = (
        0  # el valor absolutio del ángulo entre la normal y el tren de olas no puede ser superior a 90
    )

    G = 9.8091
    K = 1.4 * np.exp(-2.5 * params["d50"])
    Q0 = (
        K
        * (
            params["rho"]
            * np.sqrt(G)
            / (
                16
                * np.sqrt(params["gamma"])
                * (params["rho_s"] - params["rho"])
                * (1 - params["n"])
            )
        )
        * data["Hr"] ** 2.5
    )
    sign_ = np.sign(data["thetar"] - theta_c)
    Q = sign_ * Q0 * np.sin(2 * data["alphar"])
    return sign_, Q


def nesting(k, l, params, data):
    """[summary]

    Args:
        k ([type]): [description]
        l ([type]): [description]
        params ([type]): [description]
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    id_ = str(k + 1).zfill(4)
    write.swan(k, l, id_, data, params, mesh="local")
    current_working_directory = os.path.join(params["cwd"], params["directory"], id_)

    if not os.path.exists(os.path.join(current_working_directory, id_ + ".mat")):
        swan(current_working_directory)

    write.copla(k, l, str(k + 1).zfill(4), data, params, mesh="local")
    copla(current_working_directory)
    # join(current_working_directory)

    # grid = load.copla(params['directory'] + '/' + id_ + '/' + id_+ 'tot.out')
    grid = load.copla(params["directory"] + "/" + id_ + "/" + id_ + "vel.001")
    grid = load.swan(params["directory"] + "/" + id_ + "/" + id_ + ".mat", grid)

    grid["Sb"] = slopes(grid)  # compute the slopes in the wave current direction
    grid = sediment_transport_Kobayashi(
        l, data, grid, params
    )  # compute the sediment transport

    return grid


def swan(current_working_directory):
    """[summary]

    Args:
        current_working_directory ([type]): [description]
    """

    if not "win" in sys.platform:
        swan_path = "/share/apps/swan/4131/swan"
    else:
        swan_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "src", "swan_2020_05.exe"
        )

    subprocess.run(swan_path, cwd=current_working_directory)

    return


def copla(current_working_directory):
    """[summary]

    Args:
        current_working_directory ([type]): [description]
    """
    if not "win" in sys.platform:
        copla_path = "/share/apps/copla/CoPlaSwan_1000_20210129"
    else:
        copla_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "src",
            "CoPlaSwan_1000_20210129.exe",
        )

    subprocess.run(copla_path, cwd=current_working_directory)
    return


def cshore(current_working_directory):
    """[summary]

    Args:
        current_working_directory ([type]): [description]
    """

    if not "win" in sys.platform:
        cshore_path = "/share/apps/cshore/cshore)"
    else:
        cshore_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "src", "cshore_win.exe"
        )

    subprocess.run(cshore_path, cwd=current_working_directory)

    return


def coastalme(current_working_directory):
    """[summary]

    Args:
        current_working_directory ([type]): [description]
    """
    import shutil

    # if 'centos' == sys.platform:
    #     paths = ['/share/apps/coastalme/testing_20210126/cme',
    #              '/share/apps/coastalme/testing_20210126/scape',
    #             '/share/apps/coastalme/testing_20210126/cshore']
    #     for path in paths:
    #         shutil.copy(path, current_working_directory)
    # else:
    #     raise ValueError('CoastalME can only run into a Linux OS.')

    subprocess.run("./cme", cwd=current_working_directory)
    return


def save_db(time_, grid, data, params):
    """Update de xr.DataSet with the computation for time i

    Args:
        * time_: pd.DatetimeIndex
        * db: xr.Dataset
        * grid: dictionary with values obtained for time i

    Return:
        The updated db
    """
    # kw = ['Hs', 'qc', 'ql', 'Setup', 'DirM', 'd', 'cp',
    #     'Dr', 'Db', 'Df', 'L', 'U', 'DirU', 'Qst', 'Qbt']
    kw = ["depth", "Hs", "DirM", "U", "DirU", "qc", "ql", "Setup", "Qb"]

    db = create_db(params, data, "local", vars_=kw, time_=time_)

    for var_ in kw:
        if var_.startswith("Dir"):
            db[var_][:, :, 0] = np.remainder(270 - grid[var_], 360)
        else:
            db[var_][:, :, 0] = grid[var_]

    db.to_netcdf(params["fileout"] + "_" + str(time_).zfill(4) + ".nc")

    return db


def clean(params):
    """Remove the directory tree created to make the computation if 'del' key is True

    Args:
        * params: a dict with the model parameters.

    Return
        None
    """
    if params["delete_folder"]:
        shutil.rmtree(
            os.path.join(params["cwd"], params["directory"]), ignore_errors=True
        )
    return


def equilibrium_plan_shape(params, data):
    """[summary]

    Args:
        params ([type]): [description]
        data ([type]): [description]

    Returns:
        [type]: [description]
    """

    if not "beta_r" in params.keys():
        params["beta_r"] = 2.13

    theta_m = np.remainder(270 - params["theta_m"], 360)

    # compute the angles between the diffraction point and every point along the profile
    thetas = np.arctan2(data.x - params["x"], data.y - params["y"])
    theta_0 = thetas - np.deg2rad(theta_m - 90)
    ds = np.sqrt((data.x - params["x"]) ** 2 + (data.y - params["y"]) ** 2)
    Ys = ds * np.cos(theta_0)

    npoint = np.argmin(
        np.abs(thetas)
    )  # plane that follows the mean energy flux and crosses the diffraction point

    k = waves.wavnum(params["Ts12"], params["h"])
    L = 2 * np.pi / k

    diff, npoints, iter_ = 1, list(), 0
    while diff > 1e-3:
        alpha_min = np.arctan(
            np.sqrt(
                params["beta_r"] ** 4 / 16
                + params["beta_r"] ** 2 * Ys[npoint] / (2 * L)
            )
            / (Ys[npoint] / L)
        )
        npoint = np.argmin(np.abs(alpha_min - theta_0))
        npoints.append(npoint)

        diff = np.abs(alpha_min - theta_0[npoint])
        if any(npoints == npoint):
            break

        iter_ += 1

    beta = np.pi / 2 - theta_0[npoint]
    R0 = ds[npoint]
    R0 = 254.8474172725602  # change this value for convex beaches
    coeffs = read.xlsx("parabolic_coeffs")
    C0 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C0"])
    C1 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C1"])
    C2 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C2"])

    theta = np.deg2rad(np.linspace(np.rad2deg(beta), 180, 100))
    R = R0 * (C0 + C1 * (beta / theta) + C2 * (beta / theta) ** 2)

    theta = np.pi / 2 - theta + np.deg2rad(theta_m - 90)
    x = -R * np.sin(theta)
    y = R * np.cos(theta)

    # print(L, np.rad2deg(theta_0[npoint]))
    return x + params["x"], y + params["y"], theta_0[npoint]


# def flooding(data, points, nx, ny, dxy, angle, coastline=None):
#     """[summary]

#     Args:
#         data ([type]): [description]
#         points ([type]): [description]
#         nx ([type]): [description]
#         ny ([type]): [description]
#         dxy ([type]): [description]
#         angle ([type]): [description]

#     Returns:
#         [type]: [description]
#     """

#     data = data.copy()
#     points = points.copy()

#     ind_ = data.x.argmin()
#     x0, y0 = data.loc[ind_, "x"], data.loc[ind_, "y"]

#     data.x, data.y = gsp.rotate_coords(data.x - x0, data.y - y0, -angle)
#     points.x, points.y = gsp.rotate_coords(points.x - x0, points.y - y0, -angle)
#     points = points.loc[((points.x >= 0) & (points.x <= data.x.max()))]
#     dist = np.sqrt((points.x - x0) ** 2 + (points.y - y0) ** 2)
#     angles = np.arctan2(points.y - y0, points.x - x0)

#     if isinstance(coastline, pd.DataFrame):
#         coastline = coastline.copy()
#         coastline.x, coastline.y = gsp.rotate_coords(
#             coastline.x - x0, coastline.y - y0, -angle
#         )
#         coastline = coastline.loc[((coastline.x >= 0) & (coastline.x <= data.x.max()))]

#     angles = angles - np.deg2rad(angle)
#     pointsx = np.cos(angles) * dist
#     cota_flood = points.loc[pointsx >= 0]
#     cota_flood["projx"] = pointsx.loc[pointsx >= 0]
#     cota_flood["alpha"] = angles.loc[pointsx >= 0]

#     local_data = dict()
#     local_data["z"] = np.reshape(data.z.values, [int(ny), int(nx)])
#     local_data["x"] = np.reshape(data.x.values, [int(ny), int(nx)])
#     local_data["y"] = np.reshape(data.y.values, [int(ny), int(nx)])

#     flood_local = pd.DataFrame(-1, index=np.arange(0, nx), columns=["x", "y", "z"])
#     flood_local["x"] = np.arange(0, nx) * dxy
#     flood_local["z"] = np.interp(
#         flood_local["x"].values, cota_flood["projx"].values, cota_flood["z"].values
#     )

#     y_coast = 0
#     for ind_, value in enumerate(flood_local.index):
#         y_flood = np.interp(
#             flood_local.loc[value, "z"],
#             local_data["z"][:, ind_],
#             local_data["y"][:, ind_],
#         )
#         if isinstance(coastline, pd.DataFrame):
#             y_coast = np.interp(
#                 flood_local.loc[value, "x"], coastline["x"], coastline["y"]
#             )

#         flood_local.loc[value, "y"] = np.max([y_flood, y_coast])

#     flood_local.loc[:, "y"] = auxiliar.smooth_1d(flood_local["y"].values)

#     flood = pd.DataFrame(-1, index=np.arange(0, nx), columns=["x", "y", "z"])
#     flood["z"] = flood_local["z"]
#     flood["x"], flood["y"] = gsp.rotate_coords(
#         flood_local["x"].values, flood_local["y"].values, angle
#     )
#     flood["x"] += x0
#     flood["y"] += y0
#     return flood


def coastline_evolution(coastlines, points):
    """[summary]

    Args:
        coastlines ([type]): [description]
        points ([type]): 2D-points in a list
    """

    no_points = len(points)
    no_steps = len(coastlines)
    evolution = pd.DataFrame(-1, index=range(1, no_steps), columns=range(no_points))

    for ind_, point in enumerate(points):
        index_0 = auxiliar.nearest(coastlines[0], point)
        for time_ in coastlines.keys():
            index_ = auxiliar.nearest(
                coastlines[time_],
                [coastlines[0].loc[index_0, "x"], coastlines[0].loc[index_0, "y"]],
            )
            if (
                coastlines[time_].loc[index_, "y"] - coastlines[0].loc[index_0, "y"]
            ) > 0:
                constant = -1
            else:
                constant = 1
            evolution.loc[time_, ind_] = constant * np.sqrt(
                (coastlines[time_].loc[index_, "x"] - coastlines[0].loc[index_0, "x"])
                ** 2
                + (coastlines[time_].loc[index_, "y"] - coastlines[0].loc[index_0, "y"])
                ** 2
            )

    return evolution


def pr2flow(data, info):
    """[summary]

    Args:
        data ([type]): [description]
        info ([type]): [description]

    Returns:
        [type]: [description]
    """

    if ((not "lon") | (not "slope") | (not "area") | (not "cn2")) in info.keys():
        raise ValueError(
            "Watershed area and mean curve number, main river length and slope are required"
        )

    if not "k_ac" in info.keys():
        info["k_ac"] = 360

    if not "f_abs" in info.keys():
        info["f_abs"] = 1

    if not "rainning_pattern" in info.keys():
        info["rainning_pattern"] = "1_24h"
        logger.info(
            "The rainning pattern was not defined. Assuming a 24-hours evenly distributed pattern."
        )

    if not "events" in info.keys():
        info["events"] = False

    if not "freq_raw_data" in info.keys():
        info["freq_raw_data"] = "D"

    if not "dt" in info.keys():
        info["dt"] = 1

    if not "model" in info.keys():
        raise ValueError(
            "Model of evaluation of concretation times is required. Options are SCS, Temez, Kirpich, and Kirpich-natural_slope"
        )

    info["cn3"] = 23 * info["cn2"] / (10 + 0.13 * info["cn2"])
    info["cn1"] = 4.2 * info["cn2"] / (10 - 0.058 * info["cn2"])

    data = data * info["f_abs"]

    pattern_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "src",
        "patrones_SCS",
    )

    # Read one of the SCS precipitation patterns
    pattern = read.xlsx(pattern_path)[info["rainning_pattern"]].values

    pattern[1:] = pattern[1:] - pattern[:-1]

    # Create the dataframe for the whole period
    ini = datetime(
        data.index[0].year, data.index[0].month, data.index[0].day, data.index[0].hour
    )
    end = datetime(data.index[-1].year, data.index[-1].month, data.index[-1].day, 23)
    index = pd.date_range(ini, end, freq="H")
    df = pd.DataFrame(0, index=index, columns=["pr"])

    # Distribute the precipitation along the day
    df = distribute_precipitation(data, pattern, df, info)

    # Compute the cumulative sum by events
    df["cumulative"] = cumulative_by_events(df)

    window_size = 5 + 1  # 5 hours
    humedo = (
        df["cumulative"]
        .rolling(window_size)
        .apply(wet_soil, args=(info["cn3"],), raw=True)
    )

    window_size = (5 * 24) + 1  # 5 days into hours
    seco = (
        df["cumulative"]
        .rolling(window_size)
        .apply(dry_soil, args=(info["cn1"],), raw=True)
    )

    df["cn"] = 0
    df["cn"] = humedo.fillna(info["cn2"])
    df["cn"].loc[seco.notnull()] = seco

    df["f_a"] = 254.0 * (100 / df["cn"] - 1)  # Fraction of cumulative abstraction
    df["i_a"] = 0.2 * df["f_a"]  # Abstraccion Inicial --> A partir de SCS

    df["effec_cum"] = 0
    df["effec_cum"].loc[df["cumulative"] >= df["i_a"]] = (
        df["cumulative"].loc[df["cumulative"] >= df["i_a"]]
        - df["i_a"].loc[df["cumulative"] >= df["i_a"]]
    ) ** 2 / (df["cumulative"] + 0.8 * df["f_a"].loc[df["cumulative"] >= df["i_a"]])

    df["net_pr"] = 0
    diff = np.diff(df["effec_cum"])
    df["net_pr"].iloc[1:] = np.where(diff > 0, diff, 0)

    df = unit_hydrograph_model(df, info)

    # BASE FLOW
    # Computing the infiltrated flow
    df["infiltration"] = df["pr"] - df["net_pr"]
    df["infil_flow"] = (
        df["infiltration"] * info["area"] * 1e6 / (1000 * 3600)
    )  # mm/h --> m3/s

    df["base_flow"] = 0
    df["base_flow"].acumulado = df["base_flow"].iloc[0]

    window_size = 2
    df["base_flow"] = (
        df["base_flow"]
        .rolling(window_size)
        .apply(base_flow, args=(df, info), raw=False)
    )
    df["base_flow"].iloc[0] = 0

    df["total_flow"] = df["base_flow"] + df["sup_flow"]

    df["total_sup_vol"] = (
        df["sup_flow"].sum() * 3600 / 1e6
    )  # Total volume of runoff (m3)

    df["total_infil_vol"] = (
        df["infil_flow"].sum() * 3600 / 1e6
    )  # Total volume of rain (m3)

    df["total_input_vol"] = (df["pr"] * info["area"] * 1e6 / 1000).sum() / 1e6
    df["mass_balance"] = (
        df["total_sup_vol"] + df["total_infil_vol"] - df["total_input_vol"]
    )
    df["error"] = 100 - (
        (df["total_sup_vol"] + df["total_infil_vol"]) / df["total_input_vol"] * 100
    )
    return df


def wet_soil(window, cn):
    """Estudiamos la dinamica de la humedad del suelo con el tiempo
    mas de 5 horas de lluvia == suelo humedo
    mas de 5 dias sin lluvia == suelo seco
    entre 1 y 5 horas de lluvia == suelo en condiciones medias

    Args:
        window ([type]): [description]

    Returns:
        [type]: [description]
    """
    if np.count_nonzero(~np.isclose(window, 0)) == window.size:
        value = cn
    else:
        value = np.NaN

    return value


def dry_soil(window, cn):
    """[summary]

    Args:
        window ([type]): [description]

    Returns:
        [type]: [description]
    """
    if np.count_nonzero(np.isclose(window, 0)) == window.size:
        value = cn
    else:
        value = np.NaN

    return value


def unit_hydrograph_model(df, info):
    """TRANSFORMACION LLUVIA - CAUDAL. Hidrograma Unitario SCS

    Args:
        info ([type]): [description]

    Returns:
        [type]: [description]
    """
    if info["model"] == "SCS":
        tc = (
            0.071
            * info["lon"] ** (0.8)
            / (info["slope"] ** (1 / 4))
            * (1000 / info["cn2"] - 9) ** (0.7)
        )
    elif info["model"] == "Temez":
        tc = 0.3 * (info["lon"] / info["slope"] ** (1 / 4)) ** (0.76)
    elif info["model"] == "Kirpich":
        tc = 0.066 * (info["lon"] ** (0.77) / info["slope"] ** (0.385))
    elif info["model"] == "Kirpich-natural_slope":
        tc = 0.13 * (info["lon"] ** (0.77) / info["slope"] ** (0.385))
    else:
        raise ValueError(
            "The unit hydrograph model given ({}) is not yet implemented".format(
                info["model"]
            )
        )
    t_p = 0.5 * info["dt"] + 0.6 * tc
    t_b = 2.67 * t_p
    q_p = info["area"] / (1.8 * t_b)

    hu1 = np.array([[0, 0], [t_p, q_p], [t_b, 0]])
    hu2 = interp1d(hu1[:, 0], hu1[:, 1])(np.arange(1, np.ceil(t_b)))

    # Desarrollamos la matriz de convolucion discreta
    df["sup_flow"] = np.convolve(df["net_pr"], hu2)[0 : len(df["net_pr"])]
    return df


def base_flow(window, df, info):
    """[summary]

    Args:
        window ([type]): [description]
        info ([type]): [description]
        q_infil ([type]): [description]

    Returns:
        [type]: [description]
    """
    ex = np.exp(-info["dt"] / info["k_ac"])
    df["base_flow"].acumulado = df["base_flow"].acumulado * ex + df["infil_flow"][
        window.index[1]
    ] * (1 - ex)

    return df["base_flow"].acumulado


def distribute_precipitation(data, pattern, df, info):
    """[summary]

    Args:
        data ([type]): [description]
        patron ([type]): [description]
        df ([type]): [description]
        info ([type]): [description]

    Returns:
        [type]: [description]
    """
    if info["events"]:
        dates = []
        for ind_, _ in enumerate(data.index):
            dates.append(
                data.index[ind_]
                - pd.Timedelta(
                    seconds=data.index[ind_].minute * 60
                    + data.index[ind_].second
                    + data.index[ind_].microsecond / 1e6
                )
            )
        data.index = dates

    if info["freq_raw_data"] == "D":
        df["pr"] = np.outer(data.values, pattern).ravel()
    elif info["freq_raw_data"] == "H":
        df["pr"] = data.values
    else:
        hours = data.index.hour - 1
        k, group = 0, 0
        group_len = int(info["freq_raw_data"].split("H")[0])
        while k < len(hours):
            if group == 0:
                locs = [hours[k] + gr for gr in range(group_len)]
                indexs = np.fmod(locs, 24)
                mult_ = pattern[hours[indexs]] / np.sum(pattern[hours[indexs]])
                date = data.index[k]

            df.loc[date + pd.Timedelta(hours=group)] = data.loc[date] * mult_[group]

            if group == group_len - 1:
                group = 0
                k += 1
            else:
                group += 1

    save.to_csv(df, "distributed_precipitation.zip")
    return df


def cumulative_by_events(df, eps: float = 1e-6):
    """Compute the cumulative sum of a timeseries dividing by events

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    pr_hr = df.values[:, 0]
    cumsum = np.cumsum(pr_hr)

    cumsum_reset = cumsum[pr_hr == 0]
    cumsum_reset = np.insert(cumsum_reset, 0, 0)
    correction = np.diff(cumsum_reset)

    p_horaria_corrected = pr_hr.copy()
    p_horaria_corrected[pr_hr == 0] -= correction

    df_cumulative = pd.Series(p_horaria_corrected.cumsum(), index=df.index)
    df_cumulative[df_cumulative < eps] = 0
    return df_cumulative


def hydraulic_radius(channel_info, type_="rectangular"):

    df = df.copy()

    if type_ == "rectangular":
        df["wet_area"] = channel_info["b"] * y
        df["wet_perimeter"] = channel_info["b"] + 2 * y
        df["rh"] = channel_info["b"] * y / (channel_info["b"] + 2 * y)
        df["water_mirror"] = channel_info["b"]
    elif type_ == "trapezoidal":
        df["wet_area"] = (channel_info["b"] + y) * y
        df["wet_perimeter"] = channel_info["b"] + 2 * y
        df["rh"] = channel_info["b"] * y / (channel_info["b"] + 2 * y)
        df["water_mirror"] = channel_info["b"]
    elif type_ == "triangular":
        df["wet_area"] = channel_info["b"] * y
        df["wet_perimeter"] = channel_info["b"] + 2 * y
        df["rh"] = channel_info["b"] * y / (channel_info["b"] + 2 * y)
        df["water_mirror"] = channel_info["b"]
    elif type_ == "circular":
        df["wet_area"] = channel_info["b"] * y
        df["wet_perimeter"] = channel_info["b"] + 2 * y
        df["rh"] = channel_info["b"] * y / (channel_info["b"] + 2 * y)
        df["water_mirror"] = channel_info["b"]
    elif type_ == "parabolic":
        df["wet_area"] = channel_info["b"] * y
        df["wet_perimeter"] = channel_info["b"] + 2 * y
        df["rh"] = channel_info["b"] * y / (channel_info["b"] + 2 * y)
        df["water_mirror"] = channel_info["b"]
    else:
        raise ValueError("Type of section not yet implemented")

    return df


def water_elevation(type_="rectangular"):
    """[summary]

    Args:
        type_ (str, optional): [description]. Defaults to "rectangular".

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if type_ == "rectangular":

        def rectangular(h, Q, var_):
            return np.abs(
                Q
                - 1
                / var_["n"]
                * (var_["w"] * h)
                * (var_["w"] * h / (var_["w"] + 2 * h)) ** (2 / 3)
                * var_["S"] ** 0.5
            )

        fun = rectangular
    elif type_ == "trapezoidal":

        def trapezoidal(h, Q, var_):
            return np.abs(
                Q
                - 1
                / var_["n"]
                * (var_["w"] * var_["z"] * h)
                * h
                / (var_["w"] + 2 * h * np.sqrt(1 + var_["z"] ** 2)) ** (2 / 3)
                * var_["S"] ** 0.5
            )

        fun = trapezoidal
    elif type_ == "triangular":

        def triangular(h, Q, var_):
            return np.abs(
                Q
                - 1
                / n
                * (var_["z"] * h)
                / (2 * np.sqrt(1 + var_["z"] ** 2)) ** (2 / 3)
                * var_["S"] ** 0.5
            )

        fun = triangular
    elif type_ == "circular":

        def circular(h, Q, var_):
            return np.abs(
                Q
                - 1
                / n
                * (
                    (1 - np.sin(np.rad2deg(var_["theta"])) / np.rad2deg(var_["theta"]))
                    * var_["D"]
                    / 4
                )
                ** (2 / 3)
                * var_["S"] ** 0.5
            )

        fun = circular
    elif type_ == "parabolic":

        def parabolic(h, Q, var_):
            return np.abs(
                Q
                - 1
                / n
                * ((2 * var_["T"] ** 2 * h) / (3 * var_["T"] ** 2 + 8 * h**2))
                ** (2 / 3)
                * var_["S"] ** 0.5
            )

        fun = parabolic
    else:
        raise ValueError("Type of section not yet implemented")

    return fun


def settling_velocity(ds, info, type_="Rubey"):
    g = 9.81

    if type_ == "Rubey":
        F = (2 / 3 + 36 * info["nu"] ** 2 / (g * (info["sg"] - 1) * ds**3)) ** (
            1 / 2
        ) - (36 * info["nu"] ** 2 / (g * (info["sg"] - 1) * ds**3)) ** (1 / 2)
        w_s = F * (g * (info["sg"] - 1) * ds) ** (0.5)
    elif type_ == "Wu_Wang":
        M = 53.5 * np.exp(-0.65 * info["Sp"])
        N = 5.65 * np.exp(-2.5 * info["Sp"])
        n = 0.7 + 0.9 * info["Sp"]
        w_s = (
            M
            * info["nu"]
            / (N * d)
            * (np.sqrt(0.25 + (4 * N / (3 * M**2) * d_ast**3) ** (1 / n)) - 0.5) ** n
        )

    else:
        raise ValueError("Settling velocity formula not implemented.")

    return w_s


def river_sediment_transport(df, info, type_="meyer-peter-muller"):
    """
    MEYER-PETER-MULLER
    EINSTEIN Y BROWN (1942)
    WILCOCK Y CROWE (2003)
    MODELO DE BAGNOLD (1966)
    MODELO DE YANG (1973) PARA ARENAS
    MODELO DE BROWNLIE (1981)

    Args:
        df ([type]): [description]
        info ([type]): [description]
        type_ (str, optional): [description]. Defaults to "meyer-peter-muller".

    Returns:
        [type]: [description]
    """

    g = 9.81

    tau = info["rho_w"] * g * df["h"] * info["S"]
    tau_shi = tau / (g * info["d50"] * info["rho_w"] * (info["sg"] - 1))
    tau_c_shi = 0.047

    if type_ == "meyer-peter-muller":
        logger.info(
            "Based on Meyer-Peter-Müller (1948) for gravel rivers with it uses d50"
            " of the superficial bed layers"
        )
        df["qb"] = 0
        motion = tau_shi >= tau_c_shi

        df["qb"][motion] = (
            8
            * (tau_shi[motion] - tau_c_shi) ** (1.5)
            * (info["d50"] * (g * (info["sg"] - 1) * info["d50"]) ** 0.5)
        )
        df["Qb"] = df["qb"] * 3600 * info["w"]

    elif type_ == "einstein-brown":

        w_s = settling_velocity(info["d50"], info)
        low_motion = tau_shi < 0.18
        mid_motion = (tau_shi > 0.18) & (tau_shi < 0.52)
        high_motion = tau_shi > 0.52
        df["qb"] = 0
        df["qb"][low_motion] = 2.15 * np.exp(-0.391 / tau_shi[low_motion])
        df["qb"][mid_motion] = 40 * tau_shi[mid_motion] ** 3
        df["qb"][high_motion] = 15 * tau_shi[high_motion] ** (1.5)

        df["qb"] = df["qb"] * w_s * info["d50"]
        df["Qb"] = df["qb"] * 3600 * info["w"]  # m3

    elif type_ == "wilcock-crowe":

        dm = np.sum(info["ds_substrate"] * info["frac_substrate"])
        if not "Fs" in info.keys():
            Fs = 0.2  # Porcentaje de arena en la capa de superficie y substrato
            logger.info("Percentage of sands if not given. Using Fs equals to 0.2.")

        tau_rm_shi = 0.021 + 0.015 * np.exp(-20 * Fs)
        tau_rm = tau_rm_shi / (g * info["d50"] * info["rho_w"] * (info["sg"] - 1))
        weight = np.zeros(len(tau))

        df["qb"] = 0

        for i in range(0, len(info["ds_substrate"]) - 1):
            b = 0.67 / (1 + np.exp(1.5 - info["ds_substrate"][i] / dm))  # bug aquí
            tau_ri = tau_rm * (info["ds_substrate"][i] / info["d50"]) ** b
            Fi = tau / tau_ri
            mask = Fi < 1.35
            weight[mask] = 0.002 * Fi[mask] ** (7.5)
            weight[~mask] = 14 * (1 - 0.894 / ((Fi[~mask]) ** 0.5)) ** 4.5

            df["qb"] = df["qb"] + weight * info["frac_substrate"][i] * info["w"] * (
                g * df["h"] * info["S"]
            ) ** (3 / 2) * info["rho_s"] / (
                (info["sg"] - 1) * g
            )  # o sobra un rho_s

        df["Qb"] = df["qb"] * 3600 / info["rho_s"]  # o falta un rho_s

    elif type_ == "bagnold":
        if not "d50" in info.keys():
            raise ValueError("d50 is not given.")

        w_s = settling_velocity(info["d50"], info)

        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        if not "eb" in info.keys():
            info["eb"] = 0.15
            logger.info("Bagnold's effitienty parameter not given. Using 0.15.")

        df["Qb"] = (
            tau
            * U
            / (info["sg"] - 1)
            * (info["eb"] + 0.01 * U / w_s)
            * info["w"]
            / info["rho_s"]
            * 3600
        )

    elif type_ == "yang":

        if not "ds_substrate" in info.keys():
            raise ValueError("ds is required for Yan method.")

        w_s = settling_velocity(info["ds_substrate"], info)

        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        Q = U * info["w"] * df["h"]
        uc = (g * df["h"] * info["S"]) ** 0.5

        landa = np.zeros((len(info["ds_substrate"]), len(h)))
        Cm2 = np.zeros((len(info["ds_substrate"]), len(h)))  # concentración másica
        Qv2 = np.zeros((len(info["ds_substrate"]), len(h)))  # caudal sólido

        for i in range(0, len(info["ds_substrate"]) - 1):
            fac1 = np.zeros(len(U))
            mask = (info["ds_substrate"][i] > 0) & (
                uc * info["ds_substrate"][i] / info["nu"]
                > 1.2 & uc * info["ds_substrate"][i] / info["nu"]
                < 70
            )
            fac1[mask] = (
                2.5 / (np.log(uc[mask] * info["ds_substrate"][i] / info["nu"]) - 0.06)
                + 0.66
            )
            mask = (info["ds_substrate"][i] > 0) & (
                uc * info["ds_substrate"][i] / info["nu"] >= 70
            )
            fac1[mask] = 2.05

            mask = (info["ds_substrate"][i] > 0) & (info["ds_substrate"][i] < 0.002)
            landa[i][mask] = (
                5.435
                - 0.286 * np.log(w_s[i] * info["ds"][i] / info["nu"])
                - 0.457 * np.log(uc[mask] / w_s[i])
            )
            +(
                1.799
                - 0.409 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 0.314 * np.log(uc[mask] / w_s[i])
            ) * np.log(U[mask] * info["S"] / w_s[i] - fac1[mask] * info["S"])

            mask = info["ds_substrate"][i] > 0.002
            landa[i][mask] = (
                6.681
                - 0.633 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 4.816 * np.log(uc[mask] / w_s[i])
            )
            +(
                2.784
                - 0.305 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 0.282 * np.log(uc[mask] / w_s[i])
            ) * np.log(U[mask] * info["S"] / w_s[i] - fac1[mask] * info["S"])

            mask = info["ds_substrate"][i] == 0
            landa[i][mask] = 0

            Cm2[i] = 10 ** landa[i]
            Qv2[i] = Cm2[i] * 0.001 * Q / info["rho_s"]

    elif type_ == "browlie":
        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        Q = U * info["w"] * df["h"]

        d_adim = info["ds_substrate"] * ((info["sg"] - 1) * g / (info["nu"] ** 2)) ** (
            1 / 3
        )

        if "tau_adim" in info:
            if info["tau_adim"] == "Wu_Wang":
                tau_adim_c = np.zeros((len(d_adim)))
                mask = d_adim <= 1.5
                tau_adim_c[mask] = 0.126 * d_adim[mask] ** (-0.44)
                mask = (d_adim > 1.5) & (d_adim <= 10)
                tau_adim_c[mask] = 0.131 * d_adim[mask] ** (-0.55)
                mask = (d_adim > 10) & (d_adim <= 20)
                tau_adim_c[mask] = 0.0685 * d_adim[mask] ** (-0.27)
                mask = (d_adim > 20) & (d_adim <= 40)
                tau_adim_c[mask] = 0.0173 * d_adim[mask] ** (0.19)
                mask = (d_adim > 40) & (d_adim <= 150)
                tau_adim_c[mask] = 0.0115 * d_adim[mask] ** (0.30)
                mask = d_adim > 150
                tau_adim_c[mask] = 0.052
        else:
            logger.info("Browlie's tau adimensional parameter.")
            dens_apa = 1600
            Y_bw = (
                g**0.5
                * (dens_apa / info["rho_w"]) ** 0.5
                * info["d50"] ** (3 / 2)
                / info["nu"]
            ) ** (-0.6)
            tau_adim_c = 0.22 * Y_bw + 0.06 * 10 ** (7.7 * Y_bw)

        df["Cppm"] = 0
        df["Qb"] = 0
        cb = 1.268

        Uc = (
            ((info["sg"] - 1) * g * info["d50"]) ** 0.5
            * 4.596
            * tau_adim_c**0.529
            * info["S"] ** (-0.1405)
            * np.std(info["ds_substrate"]) ** (-0.1606)
        )
        df["Cppm"] = (
            7115
            * cb
            * ((U - Uc) / ((info["sg"] - 1) * g * info["d50"]) ** 0.5) ** 1.978
            * info["S"] ** 0.6601
            * (df["h"] / info["d50"]) ** (-0.3301)
        )
        df["Qb"] = df["Cppm"] * 0.001 * Q * 3600 / info["rho_s"]

    return df["Qb"]


def storm_surge_from_waves(data: pd.DataFrame, location: str, var_name: str = "Hm0"):
    """Compute the storm surge given the significant wave height following the
    methodology of the Atlas de Inundación Español

    Args:
        data (pd.DataFrame): dataframe with only the column of significant wave height
        location (str): location to chose the parameters. Options are Huelva or Malaga
    """

    parametros = {
        "Malaga": {
            "loc": [-0.0334122, 0.08969, -0.0148889],
            "esc": [0.0875315, -0.0238893, 0.0473336, -0.0122261],
        },
        "Huelva": {
            "loc": [-0.0218071, -0.00208886, 0.0286215],
            "esc": [0.0558449, 0.0446334, -0.0106856, -0.000660628],
        },
        "info": "Parametros del Atlas de Inundacion",
    }

    mu = np.polyval(parametros[location]["loc"][::-1], data[var_name])
    esc = np.polyval(parametros[location]["esc"][::-1], data[var_name])

    # Replace negative variance
    esc[esc < 0] = np.random.rand(len(esc[esc < 0])) * 1e-3

    p_Hs = np.random.rand(len(data))
    eta = st.norm.ppf(p_Hs, mu, esc)
    data["mm"] = data[var_name] * 0 + eta
    return data


def floodFill(c, r, mask):
    """
    Crawls a mask array containing only 1 and 0 values from the
    starting point (c=column, r=row - a.k.a. x, y) and returns
    an array with all 1 values connected to the starting cell.
    This algorithm performs a 4-way check non-recursively.
    """
    # cells already filled
    filled = set()
    # cells to fill
    fill = set()
    fill.add((c, r))
    width = mask.shape[1] - 1
    height = mask.shape[0] - 1
    # Our output inundation array
    flood = np.zeros_like(mask, dtype=np.int8)
    # Loop through and modify the cells which need to be checked.
    while fill:
        # Grab a cell
        x, y = fill.pop()
        if (x <= height) & (y <= width) & (x >= 0) & (y >= 0):
            # Don't fill
            # continue
            if (mask[x][y] == 1) & ((x, y) not in filled):
                # Do fill
                flood[x][y] = 1
                filled.add((x, y))
                # Check neighbors for 1 values
                west = (x - 1, y)
                east = (x + 1, y)
                north = (x, y - 1)
                south = (x, y + 1)
                northwest = (x - 1, y - 1)
                northeast = (x + 1, y - 1)
                southwest = (x - 1, y + 1)
                southeast = (x + 1, y + 1)
                if west not in filled:
                    fill.add(west)
                if east not in filled:
                    fill.add(east)
                if north not in filled:
                    fill.add(north)
                if south not in filled:
                    fill.add(south)
                if northwest not in filled:
                    fill.add(northwest)
                if northeast not in filled:
                    fill.add(northeast)
                if southwest not in filled:
                    fill.add(southwest)
                if southeast not in filled:
                    fill.add(southeast)
    return flood


def extractLinesatLevel_3d(x, y, z, level: float = 0, type_: str = "largest"):
    """Obtain the lines from a three dimensional field at a given level

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        level (float, optional): _description_. Defaults to 0.
        lines (str, optional): _description_. Defaults to "largest".

    Returns:
        _type_: _description_
    """
    import matplotlib.pyplot as plt

    cs = plt.contour(x, y, z, levels=[level])

    lines = []
    for collection in cs.collections:
        for path in collection.get_paths():
            if type_ == "all":
                lines.append(np.asarray(path.to_polygons()[0])[:-1, :])
            elif type_ == "largest":
                if len(lines) < len(path.to_polygons()[0]):
                    lines = np.asarray(path.to_polygons()[0])[:-1, :]
    plt.close()

    return lines
