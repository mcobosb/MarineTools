import re

import numpy as np
import pandas as pd
import xarray as xr
from marinetools.utils import read, save
from scipy.io import loadmat as ldm


def create_mesh_dictionary(fname, uf=None):
    """Read the json file with model parameters

    Args:
        * fname: string with the path to the json file
    """
    info = read.xlsx(fname)
    if uf is not None:
        params = info[uf].to_dict()
    else:
        params = info

    return params


def cshore_config():

    # Common configuration file
    props = {}
    props["iline"] = 1  # 1 = single line
    props["iprofl"] = (
        1  # 0 = no morph, 1 = run morph, 1.1 = run morph without initial smoothing
    )
    props["isedav"] = 0  # 0 = unlimited sand, 1 = hard bottom
    props["iperm"] = 0  # 0 = no permeability, 1 = permeable
    props["iover"] = 1  # 0 = no overtopping , 1 = include overtopping
    props["infilt"] = 0  # 1 = include infiltration landward of dune crest
    props["iwtran"] = (
        0  # 0 = no standing water landward of crest, 1 = wave transmission due to overtopping
    )
    props["ipond"] = 0  # 0 = no ponding seaward of SWL
    props["iwcint"] = 1  # 0 = no W & C interaction , 1 = include W & C interaction
    props["iroll"] = 1  # 0 = no roller, 1 = roller
    props["iwind"] = 0  # 0 = no wind effect
    props["itide"] = 0  # 0 = no tidal effect on currents
    props["iclay"] = 0  #
    props["iveg"] = 0  # vegitation effect
    props["veg_Cd"] = 1  # vegitation drag coeff
    props["veg_n"] = 100  # vegitation density
    props["veg_dia"] = 0.01  # vegitation diam
    props["veg_ht"] = 0.20  # vegitation height
    props["veg_rod"] = 0.1  # vegitation erosion limit below sand for failure
    props["veg_extent"] = [
        0.7,
        1,
    ]  # vegitation coverage as fraction of total domain length
    props["gamma"] = 0.8  # shallow water ratio of wave height to water depth
    props["sporo"] = 0.4  # sediment porosity
    props["sg"] = 2.65  # specific gravity
    props["effb"] = 0.005  # suspension efficiency due to breaking eB
    props["efff"] = 0.01  # suspension efficiency due to friction ef
    props["slp"] = 0.5  # susp ed load parameter
    props["slpot"] = 0.1  # overtopping susp ed load parameter
    props["tanphi"] = 0.630  # tangent (sediment friction angle)
    props["blp"] = 0.001  # bedload parameter
    props["rwh"] = 0.02  # numerical rununp wire height
    props["ilab"] = 1  # controls the boundary condition timing. Don't change
    props["fric_fac"] = 0.015  # bottom friction factor

    # boundary conditions and timing
    props["timebc_wave"] = 3600

    props["timebc_surg"] = props["timebc_wave"]
    props["nwave"] = 1  # len(props['timebc_wave'])
    props["nsurg"] = props["nwave"]

    props["Wsetup"] = 0  # wave setup at seaward boundary in meters
    props["swlbc"] = 0.0  # water level at seaward boundary in meters

    return props


def rcshore(file_, path, skiprows=1):
    """[summary]

    Args:
        file ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    header = {
        "bprof": ["z"],
        "bsusl": [r"$P_b$", r"$P_s$", r"$V_s$"],
        "cross": [r"$Q_{b,x}$", r"$Q_{s,x}$", r"$Q_{b,x} + Q_{s,x}$"],
        "crvol": [],
        "energ": [r"Eflux (m3/s)", "Db (m2/s)", "Df (m2/s)"],
        "longs": [r"$Q_{b,y}$", r"$Q_{s,y}$", r"$Q_{b,y} + Q_{s,y}$"],
        "lovol": [],
        "param": ["T (s)", r"$Q_b$ (nondim)", "Sigma* (nondim)"],
        "rolle": ["Rq (m2/s)"],
        "setup": [r"$\eta + S_{tide}$ (m)", "d (m)", r"$\sigma_{eta}$ (m)"],
        "swase": ["de (m)", "Uxe (m/s)", "Qxe (m2/s)"],
        "timse": ["t (id)", "t (s)", "q0 (m2/s)", "qbx,lw (m2/s)", "qsx,lw (m2/s)"],
        "xmome": ["Sxx (m2)", "taubx (m)"],
        "xvelo": [r"$U_x$", r"$U_{x,std}$"],
        "ymome": ["Sxx (m2)", "taubx (m)"],
        "yvelo": ["sin theta (unitary)", r"$U_y$", r"$U_{y,std}$"],
    }

    # TODO: include morphology options
    # EWD: Output exceedance probability 0.015
    # q0: wave overtopping rate, qbx,lw: cross-shore bedload transporte rate at the landward end of the computation domain
    filename = path + "/" + "O" + file_.upper()
    if file_ == "bprof":
        fid = open(filename, "rb")
        properties = fid.readline()
        id_ = int(properties.split()[1])
        df = pd.read_csv(
            filename,
            sep="\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_],
        )
        df = df.iloc[:id_, :]
    else:
        df = pd.read_csv(
            filename,
            sep="\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_],
        )
    # index x distance in meters

    df.columns = df.columns.astype("str")

    return df


def copla(fname, grid=None):
    """[summary]

    Args:
        fname ([type]): [description]

    Returns:
        [type]: [description]
    """
    # data = pd.read_csv(fname,  delim_whitespace=True)
    # fname2 = 'A01_005/0001/0001vel.001'
    data = pd.read_csv(
        fname,
        skiprows=7,
        delim_whitespace=True,
        header=None,
        index_col=0,
        names=["x", "y", "u", "v"],
    )
    _, x = np.meshgrid(data.y.unique(), data.x.unique())

    # vars_ = ['y', 'x', 'U', 'DirU', 'u', 'v']

    if grid is None:
        grid = {}

    # for j, k in enumerate(['y', 'x', '|u|', 'angC', 'u', 'v']):
    #     grid[vars_[j]] = data[k].to_numpy().reshape(np.shape(x))
    #     if k.startswith('Dir'):
    #         grid[vars_[j]] = np.fmod(grid[vars_[j]] + 90, 360)

    grid = dict()
    nx, ny = np.shape(x)
    for var_ in ["u", "v"]:
        grid[var_] = np.zeros([nx + 2, ny + 2])
        grid[var_][1:-1, 1:-1] = data[var_].to_numpy().reshape([nx, ny])

    grid["U"] = np.sqrt(grid["u"] ** 2 + grid["v"] ** 2)
    grid["DirU"] = np.fmod(np.rad2deg(np.arctan2(grid["v"], grid["u"])) + 90, 360)

    return grid


def swan(fname, grid=None, vars_=None):
    """[summary]

    Args:
        fname ([type]): [description]
        vars_ ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not vars_:
        vars_ = ["x", "y", "depth", "Qb", "L", "Setup", "Hs", "DirM"]

    if grid is None:
        grid = {}

    swan_dictionary = ldm(fname)
    for ind_, var_ in enumerate(
        ["Xp", "Yp", "Depth", "Qb", "Wlen", "Setup", "Hsig", "Dir"]
    ):
        grid[vars_[ind_]] = swan_dictionary[var_]
        grid[vars_[ind_]][np.isnan(grid[vars_[ind_]])] = 1e-6

    grid["kp"] = 2 * np.pi / grid["L"]

    return grid


def delft_raw_files_point(
    point, mesh_filename, folder, vars_, nocases, filename="seastates_"
):
    """[summary]

    Args:
        point ([type]): [description]
        mesh_filename ([type]): [description]
        folder ([type]): [description]
        vars_ ([type]): [description]
        nocases ([type]): [description]
        filename ([type]): [description]
    """

    cases = np.arange(1, nocases + 1)

    fid = open(mesh_filename, "r")
    data = fid.readlines()
    readed, kline = [], -1

    for i in range(8, len(data)):
        if data[i].startswith(" ETA=    1 "):
            readed.append(data[i])
            kline += 1
        else:
            readed[kline] += data[i]

    numeric_const_pattern = (
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    x, y = rx.findall(readed[0]), rx.findall(readed[1])

    for i, j in enumerate(x):
        x[i], y[i] = float(x[i]), float(y[i])

    idx = np.where(np.isclose(x, 2))[0][0]
    nlen = int(len(x) / idx)
    idxs = np.arange(0, len(x), idx, dtype=int)

    for i in idxs[::-1]:
        del x[i], y[i]

    x, y = np.reshape(np.array(x), (nlen, idx - 1)), np.reshape(
        np.array(y), (nlen, idx - 1)
    )

    ids = np.where(
        np.min(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2))
        == np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
    )

    if "eta" in vars_:
        datax = xr.open_mfdataset(
            folder + "/case0001/trim-guad.nc", combine="by_coords"
        )
        x = datax.XCOR.compute().data
        y = datax.YCOR.compute().data

        ids_trim = np.where(
            np.min(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2))
            == np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        )

    data = pd.DataFrame(-1, index=cases, columns=[vars_])
    for i in cases:
        fid = open(folder + "/case" + str(i).zfill(4) + "/" + vars_[0] + ".txt", "r")
        info = fid.readlines()
        nodesxt, nodesy, nodest = [int(nodes) for nodes in rx.findall(info[3])]
        nodesx = int(nodesxt / nodest)

        for var_ in vars_:
            if var_ == "eta":
                datax = xr.open_mfdataset(
                    folder + "/case" + str(i).zfill(4) + "/trim-guad.nc",
                    combine="by_coords",
                )
                z = datax.S1.compute().data
                z = z[-1, :, :]
                data.loc[i, "eta"] = z[ids_trim]
            else:
                data.loc[i, var_] = np.loadtxt(
                    folder + "/case" + str(i).zfill(4) + "/" + var_ + ".txt",
                    skiprows=nodesxt - nodesx + 4,
                )[ids[1][0], ids[0][0]]

    save.to_csv(data, filename + str(point[0]) + "_" + str(point[1]) + ".zip")
    return


def delft_raw_files(folder, vars_, case_id_):
    """[summary]

    Args:
        folder ([type]): [description]
        vars_ ([type]): [description]
        filename ([type]): [description]
    """

    numeric_const_pattern = (
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    dic = {}
    for var_ in vars_:
        if var_ == "vars_com_guad":
            fid = open(folder / f"{case_id_}" / f"{vars_['vars_com_guad'][0]}.txt", "r")
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt / nodest)
            for j in vars_["vars_com_guad"]:
                dic[str(j)] = np.loadtxt(
                    folder / f"{case_id_}" / f"{j}.txt", skiprows=nodesxt - nodesx + 4
                )
        else:
            fid = open(folder / f"{case_id_}" / f"{vars_['vars_wavm'][0]}.txt", "r")
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt / nodest)
            for j in vars_["vars_wavm"]:
                dic[str(j)] = np.loadtxt(
                    folder / f"{case_id_}" / f"{j}.txt", skiprows=nodesxt - nodesx + 4
                )

    return dic
