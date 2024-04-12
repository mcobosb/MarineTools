import copy
import math
import os
import sys
from dis import dis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from marinetools.spatial import geospatial as gsp
from marinetools.utils import auxiliar
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree, distance
from sklearn.neighbors import NearestNeighbors


def merge_land_sea(topo, bati, linea0, corners, sea="south"):
    """Given a mesh of topography, another with the bathymetry and zero line with coastline,
    obtain

    Args:
        topo ([type]): [description]
        bati ([type]): [description]
        linea0 ([type]): [description]
        corners ([type]): [description]

    Returns:
        [type]: [description]
    """

    import folium
    import geopandas as gpd
    import shapely.speedups
    from shapely.geometry import Polygon

    shapely.speedups.enable()

    if sea == "south":
        # It is defined the sea area using zero line and the boundary (corners)
        x, y = np.linspace(corners.x[0], corners.x[1], 2), np.tile(corners.y[1], 2)
        coords = np.array([x, y])

        x, y = (corners.x[1], linea0.loc[np.argmax(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(linea0).T]

        x, y = (corners.x[0], linea0.loc[np.argmin(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

    if sea == "east":
        x, y = np.tile(corners.x[0], 2), np.linspace(corners.y[0], corners.y[1], 2)
        coords = np.array([x, y])

        x, y = (linea0.loc[np.argmax(linea0.y), "x"], corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(linea0).T]

        x, y = (linea0.loc[np.argmin(linea0.y), "x"], corners.y[0])
        coords = np.c_[coords, [x, y]]

    if sea == "south-east":

        x, y = (corners.x[0], corners.y[1])
        coords = np.array([x, y])

        xmax = np.max(
            [linea0.loc[np.argmax(linea0.y), "x"], linea0.loc[np.argmax(linea0.x), "x"]]
        )
        x, y = (xmax, corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(linea0).T]

        ymin = np.min(
            [linea0.loc[np.argmin(linea0.y), "y"], linea0.loc[np.argmin(linea0.x), "y"]]
        )
        x, y = (corners.x[0], ymin)
        coords = np.c_[coords, [x, y]]

    # Create the polygon with edges
    polygon_ = Polygon(coords.T)
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon_])

    # Save a map with sea area
    m = folium.Map(location=[np.min(coords[1, :]), np.min(coords[0, :])])
    folium.GeoJson(polygon).add_to(m)

    # os.makedirs("temp", exist_ok=True)
    m.save(
        "malla_bat_"
        + str(corners.x[0])
        + "_"
        + str(corners.y[0])
        + "_"
        + str(corners.x[1])
        + "_"
        + str(corners.y[1])
        + ".html"
    )

    # MDT data in the sea area is removed
    topo_mask = spatial_mask(topo, polygon, op="within")

    # Valid MDT, zero-line and bathymetry are concatenated
    linea0["z"] = 0
    data = pd.concat([topo_mask, bati, linea0], ignore_index=True)

    return data


def spatial_mask(data, polygon, op="within"):
    """[summary]

    Args:
        data ([type]): [description]
        polygon ([type]): [description]

    Returns:
        [type]: [description]
    """
    import geopandas as gpd

    geodata = gpd.GeoDataFrame(
        index=data.index, geometry=gpd.points_from_xy(data.x, data.y)
    )

    # geodata = gpd.GeoDataFrame(
    #     data.loc[:, ["x", "y"]], geometry=gpd.points_from_xy(data.x, data.y)
    # )
    # mask = geodata.within(polygon.loc[0, "geometry"])

    mask = gpd.sjoin(geodata, polygon, op="within", how="left")
    if op == "within":
        data = data.loc[mask.index_right == polygon.index[0]]
    else:
        data = data.loc[mask.index_right != polygon.index[0]]

    return data


def remove_lowland(data, reference_value: float = 0, replace_value: float = 2):
    """Asign above SWL to low areas in land side of a topobathymetry

    Args:
        bathy ([type]): [description]

    Returns:
        [type]: [description]
    """

    ny, nx = np.shape(data)

    bathy = dict()
    bathy["z"] = data.copy()
    bathy["x"], bathy["y"] = np.meshgrid(np.arange(nx), np.arange(ny))

    coastline = auxiliar.isolines(bathy)
    up_side = pd.DataFrame(
        np.asarray([bathy["x"][:, -1], bathy["y"][:, -1]]).T,
        index=np.arange(len(bathy["x"][:, -1])),
        columns=["x", "y"],
    )
    land = create_polygon(up_side, sides=coastline[0])

    bathy = pd.DataFrame(
        np.asarray(
            [np.ravel(bathy["x"]), np.ravel(bathy["y"]), np.ravel(bathy["z"])]
        ).T,
        index=np.arange(bathy["x"].size),
        columns=["x", "y", "z"],
    )
    mask = spatial_mask(bathy, land)

    low_areas = bathy.loc[mask.index, "z"] <= reference_value
    bathy.loc[low_areas.loc[low_areas].index, "z"] = replace_value
    bathy = np.reshape(bathy.z.values, (ny, nx))

    return bathy


def smooth(data, sigma):
    """
    Smooth the transition between intermediate to deep water

    Args:
        - data: coordinates (space, 3D)
        - sigma: standar deviation for Gaussian filter (scipy.ndimage.gaussian_filter)

    Returns:
        - data: smoothed data
    """

    data = gaussian_filter(data, sigma=sigma)
    return data


def merge_sea_sea(tb, bd, corners, sea="south"):
    """[summary]

    Args:
        tb ([type]): topo-bathymetry
        bd ([type]): bathymetry of deep water
        corners ([type]): [description]

    Returns:
        [type]: [description]
    """

    import folium
    import geopandas as gpd
    import shapely.speedups
    from shapely.geometry import Polygon

    shapely.speedups.enable()

    # Nan removed to create the domain using tricontour, later Nan will be replaced
    mask = tb.z.notna()

    # Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
    cs = plt.tricontour(tb.x[mask], tb.y[mask], tb.z[mask], levels=[-40])
    # plt.show()

    datab = [0]
    for collection in cs.collections:
        for path in collection.get_paths():
            if len(datab) < len(path.to_polygons()[0]):
                datab = np.asarray(path.to_polygons()[0])[:-1, :]

    plt.close()
    # plt.plot(datab[:, 0], datab[:, 1])
    # plt.show()
    # TODO: crear la función para los dominios
    if sea == "south":
        x, y = np.linspace(corners.x[0], corners.x[1], 2), np.tile(corners.y[1], 2)
        coords = np.array([x, y])

        x, y = (
            np.tile(corners.x[1], 2),
            np.linspace(corners.y[1], datab[np.argmax(datab[:, 0]), 1], 2),
        )
        coords = np.c_[coords, [x, y]]
        coords = np.c_[datab[::-1].T, coords]
    if sea == "east":
        x, y = np.tile(corners.x[0], 2), np.linspace(corners.y[0], corners.y[1], 2)
        coords = np.array([x, y])

        x, y = (datab[np.argmax(datab[:, 1]), 0], corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(datab).T]

        x, y = (datab[np.argmin(datab[:, 1]), 0], corners.y[0])
        coords = np.c_[coords, [x, y]]

    if sea == "south-east":

        x, y = (corners.x[0], corners.y[1])
        coords = np.array([x, y])

        xmax = np.max(
            [datab[np.argmax(datab[:, 0]), 0], datab[np.argmax(datab[:, 1]), 0]]
        )
        x, y = (xmax, corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(datab).T]

        ymin = np.min(
            [datab[np.argmin(datab[:, 0]), 1], datab[np.argmin(datab[:, 1]), 1]]
        )
        x, y = (corners.x[0], ymin)
        coords = np.c_[coords, [x, y]]

    polygon = Polygon(coords.T)
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])

    m = folium.Map(
        location=[np.min(coords[1, :]), np.min(coords[0, :])]
    )  # , crs='Simple')
    folium.GeoJson(polygon).add_to(m)

    # auxiliar.mkdir("temp")
    m.save(
        "malla_topobat_"
        + str(corners.x[0])
        + "_"
        + str(corners.y[0])
        + "_"
        + str(corners.x[1])
        + "_"
        + str(corners.y[1])
        + ".html"
    )

    topoxy = gpd.GeoDataFrame(
        tb.loc[:, ["x", "y"]], geometry=gpd.points_from_xy(tb.x, tb.y)
    )

    mask = topoxy.within(polygon.loc[0, "geometry"])
    tb.loc[mask[~mask].index, "z"] = bd.loc[mask[~mask].index, "z"]

    return tb


def transform_data(data, projs, by_columns=False):
    """Transform data to the given projection

    Args:
        data ([type]): [description]
        projs ([type]): [description]
        by_columns (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    import pyproj

    # clockzone = str(int(math.floor(long/6) + 31))
    proj = {
        "wgs84": "epsg:4326",
        "ed50": "epsg:23030",
        "utm": "epsg:25830",
        "epsg:25829": "epsg:25829",
        "wgs84_seudo_mercator": "epsg:3857",
    }  # TODO: coger el uso correcto para utm +clockzone}

    dataout = data.copy()
    if projs[0] != projs[1]:
        p1 = pyproj.Proj(proj[projs[0]])
        p2 = pyproj.Proj(proj[projs[1]])

        if not by_columns:
            if projs[0] != "wgs84":
                dataout.y, dataout.x = pyproj.transform(
                    p1, p2, dataout.x.values, dataout.y.values
                )  # han cambiado con el pyproj nuevo la salida de datos
            elif projs[0] == "wgs84":
                dataout.x, dataout.y = pyproj.transform(
                    p1, p2, dataout.y.values, dataout.x.values
                )  # han cambiado con el pyproj nuevo la salida de datos

        else:
            if isinstance(dataout, list):
                dataout = pyproj.transform(p1, p2, dataout[1], dataout[0])
            elif isinstance(dataout, np.ndarray):
                if dataout.ndim == 1:
                    dataout[0], dataout[1] = pyproj.transform(
                        p1, p2, dataout[0], dataout[1]
                    )
                else:
                    dataout[:, 0], dataout[:, 1] = pyproj.transform(
                        p1, p2, dataout[:, 0], dataout[:, 1]
                    )
            else:
                dataout[:, 0], dataout[:, 1] = pyproj.transform(
                    p1, p2, dataout[:, 0], dataout[:, 1]
                )

    return dataout


def select_data(data, corners):
    """Select data within the boundaries

    Args:
        data ([type]): [description]
        corners ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(corners, dict):
        data = data.loc[(data["x"] > corners["x"][0]) & (data["x"] < corners["x"][1])]
        data = data.loc[(data["y"] > corners["y"][0]) & (data["y"] < corners["y"][1])]
    elif isinstance(corners, pd.DataFrame):
        data = data.loc[(data["x"] > corners.x[0]) & (data["x"] < corners.x[1])]
        data = data.loc[(data["y"] > corners.y[0]) & (data["y"] < corners.y[1])]
    else:
        raise ValueError("Corners should be given in dictionary or DataFrame.")

    return data


def interp(base, data, dist=100, method="linear", fill_values=np.nan):
    """[summary]

    Args:
        base ([type]): Base map for interpolating
        data ([type]): Points where interpolate
        dist (int, optional): [description]. Defaults to 100.
        method (str, optional): [description]. Defaults to 'linear'.

    Returns:
        [type]: [description]
    """
    if isinstance(data, pd.DataFrame):
        nx, ny = (
            int((data.x[1] - data.x[0]) / dist),
            int((data.y[1] - data.y[0]) / dist),
        )
        x, y = np.meshgrid(
            np.linspace(data.x[0] + dist / 2, cordataners.x[1] - dist / 2, nx),
            np.linspace(data.y[0] + dist / 2, data.y[1] - dist / 2, ny),
        )
    else:
        x, y = data["x"], data["y"]
        if len(np.shape(x)) == 1:
            nx, ny = len(x), 1
        else:
            nx, ny = np.shape(x)
    z = griddata(
        base.loc[:, ["x", "y"]].values,
        base.loc[:, "z"].values,
        (x, y),
        method=method,
        fill_value=fill_values,
    )  # TODO: update pandas
    df = pd.DataFrame(
        np.asarray([np.ravel(x), np.ravel(y), np.ravel(z)]).T,
        index=np.arange(int(nx * ny)),
        columns=["x", "y", "z"],
    )

    return x, y, z, df


def fillna(data, var_="z", method="nearest"):
    """Fill nan values with spatial data using any method"""
    mask = pd.notna(data.loc[:, var_])
    z = griddata(
        data.loc[mask, ["x", "y"]].values,
        data.loc[mask, "z"].values,
        (data.x, data.y),
        method=method,
    )
    data.z = z
    return data


def normal_profiles(topobat, info):
    """[summary]

    Args:
        topobat ([type]): [description]
        info ([type]): [description]

    Returns:
        [type]: [description]
    """

    # x = topobat.loc[((topobat.loc[:, 'z'] == 0) & (topobat.loc[:, 'xt'] >= 0) & (topobat.loc[:, 'xt'] <= distx)), 'xt'].values
    # y = topobat.loc[((topobat.loc[:, 'z'] == 0) & (topobat.loc[:, 'xt'] >= 0) & (topobat.loc[:, 'xt'] <= distx)), 'yt'].values
    # part_dist = np.cumsum(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))
    # ang = pd.DataFrame(np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1]))
    # ang[ang < 0] = ang[ang < 0] + 2*np.pi
    # El agua siempre se encuentra al sur
    # norm_ang = ang.rolling(25, center=True, min_periods=1).mean() - np.pi/2

    # x, y = (x[1:] + x[:-1])/2, (y[1:] + y[:-1])/2

    # loc_profiles = np.arange(100, distx - 500, 250)
    # idd = []
    # for i in loc_profiles:
    #     idd.append(np.abs(part_dist - i).argmin())

    # d_prof = np.arange(-100, 2500, 5) # genero una malla con paso de 5 metros.
    # x_prof, y_prof = [], []
    # for i in idd:
    #     x_prof.append(x[i] + d_prof*np.cos(norm_ang.loc[i].values))
    #     y_prof.append(y[i] + d_prof*np.sin(norm_ang.loc[i].values))
    x, y, z = [], [], []
    for i, j in enumerate(info["x"]):
        x.append(np.linspace(info["x"][i][0], info["x"][i][1], 1000))
        y.append(np.linspace(info["y"][i][0], info["y"][i][1], 1000))
        z.append(
            griddata(
                topobat.loc[:, ["x", "y"]].values,
                topobat.loc[:, "z"].values,
                np.vstack([x[i], y[i]]).T,
                method="linear",
                fill_value=np.nan,
            )
        )
    return x, y, z


# def normal_profiles(topobat, p0, p1):
#     """[summary]

#     Args:
#         topobat ([type]): [description]
#         p0 ([type]): [description]
#         p1 ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     x, y = np.linspace(p0[0], p1[0], 1000), np.linspace(p0[1], p1[1], 1000)
#     d = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
#     z = griddata(
#         topobat.loc[:, ["x", "y"]],
#         topobat.loc[:, "z"],
#         (x, y),
#         method="linear",
#         fill_value=np.nan,
#     )
#     data = pd.DataFrame(np.array([x, y, z, d]).T, columns=["x", "y", "z", "d"])
#     return data


def rotate_coords(x, y, angle):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        angle ([type]): [description]
    """
    d = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    dx, dy = d * np.cos(angles), d * np.sin(angles)
    x = dx * np.cos(np.deg2rad(angle)) - dy * np.sin(np.deg2rad(angle))
    y = dx * np.sin(np.deg2rad(angle)) + dy * np.cos(np.deg2rad(angle))
    return x, y


def globales_2_locales(x_glob, y_glob, alpha, lon_0_glob, lat_0_glob):
    """[summary]

    Args:
        x_glob ([type]): [description]
        y_glob ([type]): [description]
        alpha ([type]): [description]
        lon_0_glob ([type]): [description]
        lat_0_glob ([type]): [description]

    Returns:
        [type]: [description]
    """

    x_loc = (x_glob - lon_0_glob) * np.cos(math.radians(alpha)) + (
        y_glob - lat_0_glob
    ) * np.sin(math.radians(alpha))
    y_loc = (y_glob - lat_0_glob) * np.cos(math.radians(alpha)) - (
        x_glob - lon_0_glob
    ) * np.sin(math.radians(alpha))

    return x_loc, y_loc


def locales_2_globales(x_loc, y_loc, alpha, lon_0_glob, lat_0_glob):
    """[summary]

    Args:
        x_loc ([type]): [description]
        y_loc ([type]): [description]
        alpha ([type]): [description]
        lon_0_glob ([type]): [description]
        lat_0_glob ([type]): [description]

    Returns:
        [type]: [description]
    """

    x_glob = (
        ((x_loc) / (np.cos(math.radians(alpha))))
        / (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        - (
            ((y_loc * np.sin(math.radians(alpha))))
            / (np.cos(math.radians(alpha)) * np.cos(math.radians(alpha)))
        )
        / (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        + lon_0_glob
    )
    y_glob = (
        (y_loc) / (np.cos(math.radians(alpha)))
        + (
            (np.sin(math.radians(alpha)))
            * (x_loc - y_loc * np.tan(math.radians(alpha)))
        )
        / (
            (np.cos(math.radians(alpha)))
            * (np.cos(math.radians(alpha)))
            * (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        )
        + lat_0_glob
    )

    return x_glob, y_glob


def continuous_line(data: pd.DataFrame, limiting_distance: float = 1e9):
    """Create a continuous line from a cloud of data through the distance each other.
    Two dimensions x and y. The algorithm begins at the left edge.

    Args:
        data (pd.DataFrame): coordinates x and y of data
        limiting_distance (float): using to break the loop is distance is bigger. The same units as the input data.

    Returns:
        [type]: [description]
    """

    cols = data.columns
    cont_line = pd.DataFrame(-1, index=np.arange(len(data.index)), columns=cols)
    index0 = data.index[data.x.argmin()]
    cont_line.loc[0] = data.loc[index0]
    data.drop(index0, inplace=True)
    cont_line["dist"] = 0

    k = 0
    while not data.empty:
        distance = np.sqrt(
            (cont_line.loc[k, "x"] - data["x"]) ** 2
            + (cont_line.loc[k, "y"] - data["y"]) ** 2
        )

        idx = data.index[np.argmin(distance)]
        if np.min(distance) > limiting_distance:
            break
        cont_line.loc[k + 1, cols] = data.loc[idx]
        cont_line.loc[k + 1, "dist"] = distance[idx]
        data.drop(idx, inplace=True)
        k += 1

    # ind_ = cont_line['dist'].argmax() TODO: check when a jump is produced in the last points
    # if cont_line.loc[ind_, 'dist'] > cont_line['dist'].quantile(0.999):

    # if not removed.empty:
    #     for ind_ in removed.index:
    #         print('Element ' + str(ind_) + ' was removed (' + str(removed.loc[ind, 'x']) + ', ' + str(removed.loc[ind, 'y']))

    # cont_line = cont_line.loc[cont_line['dist'] <= cont_line['dist'].quantile(0.99)]
    cont_line.drop(columns="dist", inplace=True)
    cont_line.drop(cont_line[cont_line["x"] == -1].index, inplace=True)

    return cont_line


def create_polygon(data, crs="epsg:25830", sides=[]):
    """[summary]

    Args:
        data ([type]): [description]
        crs (str, optional): [description]. Defaults to 'epsg:25830'.
        sides (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """

    import geopandas as gpd
    import shapely.speedups
    from shapely.geometry import Polygon

    if not isinstance(sides, list):
        sides = [sides]

    coords = data.loc[:, ["x", "y"]].values
    for side in sides:
        coords = np.vstack([coords, side.loc[:, ["x", "y"]].values])

    polygon = Polygon(coords)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon])

    return polygon


def fillna(data, var_="z", method="nearest"):
    """Fill nan values with using the data given"""

    mask = pd.notna(data[var_])
    z = griddata(
        data.loc[mask, ["x", "y"]].values,
        data.loc[mask, "z"].values,
        (data.loc[~mask, "x"], data.loc[~mask, "y"]),
        method=method,
    )
    data.loc[~mask, var_] = z
    return data


def ci_2D(data, loc0=None, angle=None):
    """[summary] Same minimum and maximum positions for all the data is required

    Args:
        data ([type]): [description]
        x0 ([type]): [description]
        y0 ([type]): [description]
        angle ([type]): [description]
    """

    rot = data.copy()
    if loc0 == None:
        ind_min = data[1].x.argmin()
        loc0 = [data[1].loc[ind_min, "x"], data[1].loc[ind_min, "y"]]

    if angle == None:
        angle = np.arctan2(data.y, data.x)

    y = np.zeros([len(data), 1000])
    for ind_, key in enumerate(data.keys()):
        data[key].sort_values(by="x", inplace=True)
        rot[key].x, rot[key].y = gsp.rotate_coords(
            data[key].x - loc0[0], data[key].y - loc0[1], -angle
        )

        if ind_ == 0:
            x = np.linspace(rot[key].x.min(), rot[key].x.max(), 1000)
            init_key = key

        if ind_ == len(data) - 1:
            end_key = key

        y[ind_, :] = np.interp(x, rot[key].x, data[key].y)

    mean_ = np.mean(y, axis=0)
    std_ = np.std(y, axis=0)
    min_ = np.min(y, axis=0)
    max_ = np.max(y, axis=0)
    stats = pd.DataFrame(
        np.vstack([mean_, std_, min_, max_]).T,
        index=x,
        columns=["mean", "std", "min", "max"],
    )

    for cols_ in ["mean", "min", "max"]:
        stats["x"], stats[cols_] = gsp.rotate_coords(x, stats[cols_].values, angle)
        stats["x"], stats[cols_] = stats["x"] + loc0[0], stats[cols_] + loc0[1]

    _, stats["ini"] = gsp.rotate_coords(x, y[0, :], angle)
    stats["ini"] = stats["ini"] + loc0[1]
    _, stats["end"] = gsp.rotate_coords(x, y[-1, :], angle)
    stats["end"] = stats["end"] + loc0[1]
    return stats


def generate_CVD(points, iterations, bounding_box):
    """Performs x iterations of loyd's algorithm to calculate a Centroidal Voronoi Diagram
    following https://www.py4u.net/discuss/21901

    Args:
        points ([type]): [description]
        iterations ([type]): [description]
        bounding_box ([type]): [description]

    Returns:
        [type]: [description]
    """

    p = copy.copy(points)

    for i in range(iterations):
        vor = bounded_voronoi(p, bounding_box)
        centroids = []

        for region in vor.filtered_regions:
            vertices = vor.vertices[
                region + [region[0]], :
            ]  # grabs vertices for the region and adds a duplicate of the first one to the end
            centroid = centroid_region(vertices)
            centroids.append(list(centroid[0, :]))

        p = np.array(centroids)

    return bounded_voronoi(p, bounding_box)


eps = sys.float_info.epsilon


def in_box(towers, bounding_box):
    # Returns a new np.array of towers that within the bounding_box

    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]
        ),
        np.logical_and(
            bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]
        ),
    )


def bounded_voronoi(towers, bounding_box):
    # Generates a bounded vornoi diagram with finite regions
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)

    # Mirror points left, right, above, and under to provide finite regions for the edge regions of the bounding box
    points_center = towers[i, :]

    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])

    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])

    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])

    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )

    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)

    vor.filtered_points = points_center  # creates a new attibute for points that form the diagram within the region
    vor.filtered_regions = np.array(vor.regions)[
        vor.point_region[: vor.npoints // 5]
    ]  # grabs the first fifth of the regions, which are the original regions

    return vor


def centroid_region(vertices):
    # Finds the centroid of a region. First and last point should be the same.

    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1]
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])


def triangulation(x, y):

    import matplotlib.tri as tri

    # # Create the Triangulation; no triangles so Delaunay triangulation created.
    triangles = tri.Triangulation(x, y)
    xmid = x.values[triangles.triangles].mean(axis=1)

    triangles2remove = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        17,
        18,
        19,
        20,
        21,
        24,
        25,
        30,
        31,
        32,
        33,
        34,
        37,
        39,
        40,
        41,
        42,
        47,
        48,
        49,
        50,
        52,
        53,
        55,
        56,
        60,
        61,
        62,
        63,
        64,
        68,
        69,
        70,
        75,
        76,
        77,
        78,
        79,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        100,
        101,
        103,
        104,
        105,
        106,
        107,
        118,
        120,
        122,
        124,
        125,
        126,
        127,
        128,
        131,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        144,
        145,
        146,
        147,
        148,
        156,
        157,
        159,
        160,
        163,
        164,
        165,
        167,
    ]
    mask = np.ones(len(xmid))

    for i in range(len(xmid)):
        if i not in triangles2remove:
            mask[i] = False

    triangles.set_mask(mask)
    return triangles
