import io
from urllib.request import Request, urlopen

import cartopy
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.io.img_tiles as cimgt
import cmocean
import matplotlib.pyplot as plt
import numpy as np
from marinetools.graphics.utils import handle_axis, labels, show
from marinetools.spatial import geospatial as gsp
from marinetools.utils import read
from PIL import Image

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)


def plot_interps(x, y, z, zoff, niveles=np.arange(-50, 20, 2), fname=None):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        z ([type]): [description]
        zoff ([type]): [description]
        niveles ([type], optional): [description]. Defaults to np.arange(-50, 20, 2).
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cb = axs[0].contour(x, y, zoff, levels=niveles)
    cbar = fig.colorbar(cb, ax=axs[0])
    axs[0].set_title("GEBCO Bathymetry")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    cbar.ax.set_ylabel("z (m)")

    cb = axs[1].contour(x, y, z, levels=niveles)
    cbar = fig.colorbar(cb, ax=axs[1])
    axs[1].set_title("IGN/MITECO Bathymetry")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    cbar.ax.set_ylabel("z (m)")
    show(fname)
    return


def plot_mesh(
    data,
    levels=[-10, -1, 0, 1, 2, 5, 10, 20, 50, 100],
    var_="z",
    title=None,
    ax=None,
    fname=None,
    regular=False,
    cmap=cmocean.cm.deep_r,
    bar_label=r"\textbf{z (m)}",
    xlabel=r"\textbf{x (m)}",
    ylabel=r"\textbf{y (m)}",
    centercolormap=False,
    hide_colorbar=False,
    alpha=1,
):
    """[summary]

    Args:
        data ([type]): [description]
        levels (list, optional): [description]. Defaults to [-10, -1, 0, 1, 2, 5, 10, 20, 50, 100].
        var_ (str, optional): [description]. Defaults to "z".
        title ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        regular (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    ax = handle_axis(ax, figsize=(10, 6))
    from matplotlib import colors

    if centercolormap:
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min(data[var_]), vcenter=0.0, vmax=np.max(data[var_])
        )
    else:
        divnorm = None

    # ax = ax[0]
    if regular:
        cbf = ax.contourf(
            data["x"], data["y"], data[var_], 100, cmap=cmap, norm=divnorm, alpha=alpha
        )
        if levels != None:
            cb = ax.contour(
                data["x"], data["y"], data[var_], levels=levels, colors="white"
            )
            ax.clabel(cb, inline=True, fontsize=8)
    else:
        cbf = ax.tricontourf(
            data["x"].values,
            data["y"].values,
            data[var_].values,
            cmap=cmap,
            alpha=alpha,
        )
        if levels != None:
            cb = ax.tricontour(
                data["x"].values,
                data["y"].values,
                data[var_].values,
                levels=levels,
                cmap=cmap,
            )
            ax.clabel(cb, inline=True, fontsize=8)

    ax.grid("+")
    fig = plt.gcf()
    if not hide_colorbar:
        cbar = fig.colorbar(cbf, ax=ax)
        cbar.ax.set_ylabel(bar_label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.set_aspect("equal", "box")
    show(fname)

    return ax


def plot_profiles(x, y, z, idx, nos=np.arange(0, 1000, 100)):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        z ([type]): [description]
        idx ([type]): [description]
        nos ([type], optional): [description]. Defaults to np.arange(0, 1000, 100).
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cb = axs[0].contourf(x, y, z)
    cbar = fig.colorbar(cb, ax=axs[0])
    cbar.ax.set_ylabel("z (m)")
    axs[0].plot(x[0, :], y[idx, 0], "k", lw=2, label="merge location")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    for i in nos:
        axs[0].plot(x[:, i], y[:, i])
    axs[0].legend()

    for i in nos:
        axs[1].plot(y[:, idx[i]], z[:, i])
        axs[1].plot(y[idx[i], i], z[idx[i], i], ".")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")

    plt.show()

    return


def onclick(event):
    """[summary]

    Args:
        event ([type]): [description]
    """
    print(
        "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
        % (
            "double" if event.dblclick else "single",
            event.button,
            event.x,
            event.y,
            event.xdata,
            event.ydata,
        )
    )


def plot_preview(data):
    """[summary]

    Args:
        data ([type]): [description]
    """

    fig = plt.figure(figsize=(12, 5))
    plt.scatter(data["x"], data["y"], c=data["z"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return


# def plot_nprofiles(
#     d,
#     z,
#     ax=None,
#     label: str = None,
#     show_legend: bool = False,
#     fname: str = None,
# ):
#     """[summary]

#     Args:
#         topobat ([type]): [description]
#         x ([type]): [description]
#         y ([type]): [description]
#         z ([type]): [description]
#         info ([type]): [description]
#         fname ([type]): [description]
#     """
#     ax = handle_axis(ax)

#     ax.plot(d, z, label=label)

#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("z (m)")

#     ax.set_ylim([-40, 10])
#     if show_legend:
#         ax.legend()

#     show(fname)
#     return


# def plot_nprofiles(topobat, x, y, z, fname):
#     """[summary]

#     Args:
#         topobat ([type]): [description]
#         x ([type]): [description]
#         y ([type]): [description]
#         z ([type]): [description]
#         fname ([type]): [description]
#     """

#     # l0 = read_kmz(fname, 'utm') update
#     _, ax = plt.subplots(2, 1)
#     ax[0].plot(topobat["x"], topobat["y"], ".", label="Cota cero")
#     ax[0].plot(x, y, label="profile")
#     ax[0].set_xlabel("x (m)")
#     ax[0].set_ylabel("y (m)")
#     ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

#     d = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
#     ax[1].plot(d, z, label="profile")
#     ax[1].set_xlabel("x (m)")
#     ax[1].set_ylabel("z (m)")
#     ax[1].legend()
#     plt.show()
#     return


def plot_db(
    xarr,
    var_,
    coords=["lon", "lat"],
    ind_=0,
    levels=[0, 1, 2, 5, 10, 40],
    title=None,
    ax=None,
    fname=None,
):
    """[summary]

    Args:
        xarr ([type]): [description]
        var_ ([type]): [description]
        coords (list, optional): [description]. Defaults to ['lon', 'lat'].
    """
    ax = handle_axis(ax)
    cbf = ax.contourf(
        xarr[coords[0]].data,
        xarr[coords[1]].data,
        xarr[var_][:, :, ind_].data,
        cmap=cmocean.cm.deep,
    )
    try:
        cb = ax.contour(
            xarr[coords[0]].data,
            xarr[coords[1]].data,
            xarr["depth"][:, :, ind_].data,
            levels=levels,
            colors="white",
        )
        plt.clabel(cb, inline=True, fontsize=8)
    except:
        pass

    ax.grid("+")
    fig = plt.gcf()
    cbar = fig.colorbar(cbf, ax=ax)

    if title is not None:
        ax.set_title(title, loc="right")

    ax.set_xlabel(labels(coords[0]))
    ax.set_ylabel(labels(coords[1]))
    plt.gca().set_aspect("equal", adjustable="box")
    cbar.ax.set_ylabel(labels(var_))
    show(fname)

    return ax


def plot_ascifiles(data, title=None, fname=None, ax=None):
    """[summary]

    Args:
        data ([type]): [description]
        levels (list, optional): [description]. Defaults to [0, -1, -2, -5, -10, -40].
        title ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    ax = handle_axis(ax)
    cbar = ax.imshow(
        data["z"],
        interpolation="none",
        extent=[data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()],
    )
    ax.contour(data["x"], data["y"], data["z"], colors="white")

    fig = plt.gcf()
    cbar = fig.colorbar(cbar, ax=ax)

    ax.set_xlabel(labels(list(data.keys())[0]))
    ax.set_ylabel(labels(list(data.keys())[1]))
    cbar.ax.set_ylabel(labels(list(data.keys())[2]))

    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    show(fname)
    return ax


def planta_2DH(topobat, isolines, fname):
    """[summary]

    Args:
        topobat ([type]): [description]
        isolines (bool): [description]
        fname ([type]): [description]
    """
    plt.figure()
    cs = plt.tricontour(
        topobat.loc[:, "x"],
        topobat.loc[:, "y"],
        topobat.loc[:, "z"],
        levels=isolines,
        colors="k",
    )
    plt.clabel(cs, fontsize=9, inline=1)
    plt.xlabel(r"x$_{UTM}$ (m)")
    plt.ylabel(r"y$_{UTM}$ (m)")
    plt.show()
    return


def folium_map(data, more=[], fname="folium_map"):
    """[summary]

    Args:
        data ([type]): [description]
        more (list, optional): [description]. Defaults to [].
    """

    import folium
    import geopandas as gpd
    import shapely.speedups
    from IPython.display import display
    from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                                  Polygon)

    data = gsp.transform_data(data, ["utm", "wgs84"], by_columns=True)
    polygon = LineString(data[:, ::-1])
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])

    coords = data.min(axis=0)
    map_ = folium.Map(
        location=[coords[0], coords[1]], zoom_start=12
    )  # , tiles="Stamen Terrain")
    folium.GeoJson(polygon).add_to(map_)

    for element in more:
        folium.GeoJson(element).add_to(map_)

    map_.save(fname + ".html")
    return


def flood_map(
    data, coast_line, flood_line, points, flood_polygon, title=None, fname=None, ax=None
):
    """[summary]

    Args:
        data ([type]): [description]
        coast_line ([type]): [description]
        flood_line ([type]): [description]
        points ([type]): [description]
        flood_polygon ([type]): [description]
        title ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    ax = handle_axis(ax)

    # ax = plot_mesh(data, levels=None, ax=ax, fname="to_axes")
    ax.plot(
        coast_line.x,
        coast_line.y,
        "--",
        color="k",
        label="linea de costa inicial",
        ms=2,
    )
    ax.plot(
        flood_line["x"],
        flood_line["y"],
        "cyan",
        lw=2,
        label="inundación (A ="
        + str(np.round(flood_polygon.area[0], decimals=2))
        + r" m$^2$)",
    )
    ax.plot(points.x, points.y, "+", color="k", label="puntos")
    ax.set_xlabel("x UTM (m)", fontweight="bold")
    ax.set_ylabel("y UTM (m)", fontweight="bold")

    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    ax.legend()
    show(fname)
    return ax


def plot_quiver(
    db,
    is_db=True,
    vars_=["U", "DirU"],
    cadency=1,
    title=None,
    scale=1,
    label_="U",
    ax=None,
    fname=None,
):
    """[summary]

    Args:
        db ([type]): [description]
        is_db (bool, optional): [description]. Defaults to True.
        title ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    U, V = (
        db[vars_[0]] * np.cos(np.deg2rad(270 - db[vars_[1]])),
        db[vars_[0]] * np.sin(np.deg2rad(270 - db[vars_[1]])),
    )
    M = np.hypot(U, V)

    ax = plot_db(
        db, "depth", coords=["x", "y"], ind_=0, levels=[0], fname="to_axes", ax=ax
    )

    if is_db:
        cbar = ax.quiver(
            db["x"].data[::cadency, ::cadency],
            db["y"].data[::cadency, ::cadency],
            U[:, :, 0].data[::cadency, ::cadency],
            V[:, :, 0].data[::cadency, ::cadency],
            M[:, :, 0].data[::cadency, ::cadency],
            units="x",
            cmap=cmocean.cm.thermal,
            pivot="tip",
            scale=scale,
            width=10,
        )
    else:
        cbar = ax.quiver(
            db["x"][::cadency, ::cadency],
            db["y"][::cadency, ::cadency],
            U[::cadency, ::cadency],
            V[::cadency, ::cadency],
            M[::cadency, ::cadency],
            units="x",
            cmap=cmocean.cm.thermal,
            scale=scale,
            pivot="tip",
            width=10,
        )

    cbar = fig.colorbar(cbar, ax=ax)
    cbar.ax.set_ylabel(labels(label_))

    ax.set_xlabel(labels("x"), fontweight="bold")
    ax.set_ylabel(labels("y"), fontweight="bold")

    ax.grid(True)
    if title is not None:
        ax.set_title(title, loc="right")

    show(fname)
    return ax


def coastline_ci(coast_lines, title=None, fname=None, ax=None):
    """[summary]

    Args:
        data ([type]): [description]
        coast_line ([type]): [description]
        flood_line ([type]): [description]
        points ([type]): [description]
        flood_polygon ([type]): [description]
        title ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    ax = handle_axis(ax)
    colors = {"mean": "k", "min": "gray", "max": "k", "ini": "red", "end": "purple"}
    linestyles = {"mean": "--", "min": "--", "max": "--", "ini": "-", "end": "-"}
    lws = {"mean": 1, "min": 1, "max": 1, "ini": 2, "end": 2}
    names = {
        "mean": "Perfil medio",
        "min": 1,
        "max": 1,
        "ini": "Perfil inicial",
        "end": "Perfil final",
    }

    for name in ["mean", "ini", "end"]:
        ax.plot(
            coast_lines.x.values,
            coast_lines[name].values,
            label=names[name],
            color=colors[name],
            linestyle=linestyles[name],
            lw=lws[name],
        )

    ax.fill_between(
        coast_lines.x.values,
        coast_lines["max"].values,
        coast_lines["min"].values,
        label="Envolvente",
        color="darkblue",
        alpha=0.25,
    )

    ax.set_xlabel(labels("x"), fontweight="bold")
    ax.set_ylabel(labels("y"), fontweight="bold")
    ax.set_ylim(coast_lines.loc[:, "min"].min() - 100, coast_lines.loc[:, "max"].max())

    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="bold", loc="right", color="gray")

    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    show(fname)
    return ax


def plot_voronoi_diagram(vor, bounding_box, fname=False):
    """[summary]

    Args:
        vor ([type]): [description]
        bounding_box ([type]): [description]
        fname (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # Initializes pyplot stuff
    fig = plt.figure(figsize=(12, 11))
    ax = fig.gca()

    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], "b.")

    # Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], "go")

    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], "k-")

    # stores references to numbers for setting axes limits
    margin_percent = 0.1
    width = bounding_box[1] - bounding_box[0]
    height = bounding_box[3] - bounding_box[2]

    ax.set_xlim(
        [
            bounding_box[0] - width * margin_percent,
            bounding_box[1] + width * margin_percent,
        ]
    )
    ax.set_ylim(
        [
            bounding_box[2] - height * margin_percent,
            bounding_box[3] + height * margin_percent,
        ]
    )

    show(fname)

    return fig, ax


def osm_image(
    lons, lats, style="satellite", epsg=None, title=None, ax=None, fname=None
):
    """This function makes OpenStreetMap satellite or map image with circle and random points.
    Change np.random.seed() number to produce different (reproducable) random patterns of points.
    Also review 'scale' variable.
    Based on "Mathew Lipson" code (m.lipson@unsw.edu.au).

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        style (str, optional): style of Open Street Map (satellite or map). Defaults to "satellite".
        epsg (int, optional): the European Petroleum Survey Group code of lon, lat
    """
    bEpsg = False
    if str(epsg).startswith("258") and (
        str(epsg).endswith("30") | str(epsg).endswith("29")
    ):
        radius = np.max([np.abs(lons[0] - lons[1]), np.abs(lats[1] - lats[0])])
        # The limits should be included in epsg:3857
        data_CRS = ccrs.epsg(epsg)

        x0, x1 = lons
        y0, y1 = lats

        geodetic_CRS = ccrs.Geodetic()
        lons[0], lats[0] = geodetic_CRS.transform_point(x0, y0, data_CRS)
        lons[1], lats[1] = geodetic_CRS.transform_point(x1, y1, data_CRS)

    elif (epsg == 4326) | (epsg == 4328):
        # The EPSG code must correspond to a “projected coordinate system”, so EPSG codes
        # such as 4326 (WGS-84) which define a “geodetic coordinate system” will not work.

        data_CRS = ccrs.Geodetic()
        utm_proj = ccrs.epsg(25830)
        x0, y0 = utm_proj.transform_point(lons[0], lats[0], data_CRS)
        x1, y1 = utm_proj.transform_point(lons[1], lats[1], data_CRS)

        radius = np.max([np.abs(x0 - x1), np.abs(y1 - y0)])

        bEpsg = True
    else:
        R = 6371  # km
        dLat = np.deg2rad(lats[1] - lats[0])
        dLon = np.deg2rad(lons[1] - lons[0])
        a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.deg2rad(lats[0])) * np.cos(
            np.deg2rad(lats[1])
        ) * np.sin(dLon / 2) * np.sin(dLon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        radius = R * c / 2 * 1000

        geodetic_CRS = ccrs.Geodetic()
        map_proj = ccrs.epsg(25830)
        x0, y0 = map_proj.transform_point(lons[0], lats[0], geodetic_CRS)
        x1, y1 = map_proj.transform_point(lons[1], lats[1], geodetic_CRS)

    lon, lat = (lons[0] + lons[1]) / 2, (lats[0] + lats[1]) / 2

    if style == "map":
        ## MAP STYLE
        cimgt.OSM.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        raise ValueError('No valid style. Choose "satellite" or "map".')

    ############################################################################

    ax = handle_axis(ax, dim=0, projection=img.crs)

    # auto-calculate scale
    scale = int(120 / np.log(radius))
    scale = (scale < 20) and scale or 19

    # or change scale manually
    # NOTE: scale specifications should be selected based on radius
    # but be careful not have both large scale (>16) and large radius (>1000),
    #  it is forbidden under [OSM policies](https://operations.osmfoundation.org/policies/tiles/)
    # -- 2     = coarse image, select for worldwide or continental scales
    # -- 4-6   = medium coarseness, select for countries and larger states
    # -- 6-10  = medium fineness, select for smaller states, regions, and cities
    # -- 10-12 = fine image, select for city boundaries and zip codes
    # -- 14+   = extremely fine image, select for roads, blocks, buildings

    extent = calc_extent(lon, lat, radius * 1.1)
    ax.set_extent(extent)  # set extents

    def label_grid(x0, x1, y0, y1, utm=False):
        """Warning: should only use with small area UTM maps"""
        # It is changed the xticks to the desired epsg

        ax = plt.gca()
        nx, ny, k_ = 7, 5, 0  # len(ax.get_xticklabels()), len(ax.get_yticklabels()), 0

        decimals = 3 if not utm else 0

        xtickini, xtickend, ytickini, ytickend = (
            ax.get_xticks()[0],
            ax.get_xticks()[-1],
            ax.get_yticks()[0],
            ax.get_yticks()[-1],
        )
        xlabels_, xticks_ = [], []
        for k_ in range(nx):
            values_ = np.round(x0 + (x1 - x0) * k_ / (nx - 1), decimals=decimals)
            xticks_.append(
                np.round(
                    xtickini + (xtickend - xtickini) * k_ / (nx - 1), decimals=decimals
                )
            )
            values_ = values_ if not utm else int(values_)
            xlabels_.append(str(values_))
        plt.xticks(xticks_, xlabels_)

        ylabels_, yticks_ = [], []
        for k_ in range(ny):
            values_ = np.round(y0 + (y1 - y0) * k_ / (ny - 1), decimals=decimals)
            yticks_.append(
                np.round(
                    ytickini + (ytickend - ytickini) * k_ / (ny - 1), decimals=decimals
                )
            )
            values_ = values_ if not utm else int(values_)
            ylabels_.append(str(values_))
        plt.yticks(yticks_, ylabels_)

        for xaxis_ in ax.get_xgridlines():
            xaxis_.set_color("black")

        for yaxis_ in ax.get_ygridlines():
            yaxis_.set_color("black")

        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.grid(True)

    ax.add_image(img, int(scale))  # add OSM with zoom specification

    ax.set_extent([lons[0], lons[1], lats[0], lats[1]])
    if not bEpsg:
        label_grid(x0, x1, y0, y1, utm=True)
    else:
        label_grid(lons[0], lons[1], lats[0], lats[1])

    if title is not None:
        ax.set_title(title)

    if not bEpsg:
        ax.set_xlabel(r"\textbf{x (m)}")
        ax.set_ylabel(r"\textbf{y (m)}")
    else:
        ax.set_xlabel(r"\textbf{latitude ($\mathbf{^o}$)")
        ax.set_ylabel(r"\textbf{longitude ($\mathbf{^o}$)")

    show(fname)

    return ax


def calc_extent(lon, lat, dist):
    """This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    """

    dist_cnr = np.sqrt(2 * dist ** 2)
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-45, distances=dist_cnr
    )[:, 0:2][0]
    bot_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=135, distances=dist_cnr
    )[:, 0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent


def image_spoof(self, tile):
    """this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy"""

    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header("User-agent", "Anaconda 3")  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), "lower"  # reformat for cartopy


def include_Andalusian_coast(path, ax):
    data = read.shp(path)
    costa = gsp.transform_data(data[0], ["utm", "wgs84"])
    ax.plot(costa.x, costa.y, "dimgrey")
    costa = gsp.transform_data(data[1], ["utm", "wgs84"])
    ax.plot(costa.x, costa.y, "dimgrey")
    return


def include_coastal_Andalusian_cities(ax):
    cities = {
        "Malaga": {"coords": [4064894.21433146, 373065.536614759], "proj": "utm"},
        "Cadiz": {"coords": [4045830.06891307, 742762.266396567], "proj": "epsg:25829"},
        "Huelva": {
            "coords": [4125852.79525744, 682253.299985719],
            "proj": "epsg:25829",
        },
        "Almeria": {"coords": [4076597.42632942, 547820.672473857], "proj": "utm"},
        "Ceuta": {"coords": [3974169.33152635, 290472.395171953], "proj": "utm"},
        "Melilla": {"coords": [3905458.34473556, 505628.319059769], "proj": "utm"},
        "Motril": {"coords": [4067337.8742406, 453769.08587931], "proj": "utm"},
        "Algeciras": {"coords": [4001510.6438788, 279492.87333613], "proj": "utm"},
        # "Garrucha": {"coords": [4116184.67, 604626.25], "proj": "utm"},
    }

    for city in cities.keys():
        location = gsp.transform_data(
            np.array(cities[city]["coords"][::-1]),
            [cities[city]["proj"], "wgs84"],
            by_columns=True,
        )
        ax.plot(location[1], location[0], "o", ms=2, color="black")
        rotation = 45
        valign = "bottom"
        halign = "left"
        if city == "Huelva":
            rotation = 0
            valign = "center"
            location[1] = location[1] + 0.05
        elif city == "Ceuta":
            valign = "top"
            halign = "right"
        elif city == "Algeciras":
            halign = "center"

        ax.text(
            location[1],
            location[0],
            city,
            rotation=rotation,
            verticalalignment=valign,
            horizontalalignment=halign,
            color="black",
        )

    return


def include_seas(ax):
    seas = {
        "Atlantic Ocean": {
            "coords": [3974169.33152635, 660253.299985719],
            "proj": "epsg:25829",
        },
        "Mediterranean Sea": {
            "coords": [3974169.33152635, 440769.08587931],
            "proj": "utm",
        },
    }

    for sea in seas.keys():
        location = gsp.transform_data(
            np.array(seas[sea]["coords"][::-1]),
            [seas[sea]["proj"], "wgs84"],
            by_columns=True,
        )

        label = r"\textbf{" + sea + "}"
        ax.text(
            location[1],
            location[0],
            label,
            rotation=0,
            horizontalalignment="left",
            color="blue",
            fontsize=12,
        )
    return
