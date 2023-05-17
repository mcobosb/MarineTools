import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marinetools.graphics.utils import handle_axis, labels, show
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

cmp_g = cmocean.cm.haline_r
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)
params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}


def pr2flow(
    df: pd.DataFrame,
    ax=None,
    file_name: str = None,
    title: str = "",
):
    """Plot a graph that includes precipitation, infiltration, and base flow

    Args:
        df (pd.DataFrame): raw timeseries
        ax (matplotlib.axis, optional): Axis where include the new plot. Defaults to None.
        file_name ([type], optional): [description]. Defaults to None.
        legend (bool, optional): [description]. Defaults to True.
        title (str, optional): Title name.

    Returns:
        The figure
    """

    ax = handle_axis(ax, figsize=(14, 5))

    ax.plot(df["total_flow"], color="blue", label="Caudal total")
    ax.plot(df["base_flow"], color="cyan", label="Caudal base")

    ax.set_ylim([0, df["total_flow"].max() * 1.5])
    ax2 = ax.twinx()
    ax2.set_ylim([0, df["pr"].max() * 1.5])

    ax2.plot(df["pr"], color="black", label="Precipitación")
    ax2.plot(df["pr"] - df["net_pr"], color="red", label="Infiltración")

    ax2.invert_yaxis()

    ax.set_ylabel("Caudal Simulado\n(m$^3$/s)")
    ax2.set_ylabel("P-I (mm)")

    ax.legend(loc=4)
    ax2.legend(loc=1)
    ax.xaxis_date()
    ax2.xaxis_date()
    ax.grid()

    if isinstance(title, str):
        ax.set_title(title, color="k", fontweight="bold")

    show(file_name)

    return ax


def transport_mode(
    df: pd.DataFrame, ax=None, file_name: str = None, type_: str = "hist"
):
    """Plot a graph that show the transport mode

    Args:
        df (pd.DataFrame): [description]
        ax (matplotlib.axis, optional): An axis where plot the new figure. Defaults to None.
        file_name (str, optional): [description]. Defaults to None.
        type_ (str, optional): [description]. Defaults to "hist".

    Returns:
        [type]: [description]
    """

    ax = handle_axis(ax, figsize=(9, 4))

    no_motion = df < 0.2
    bed_load = (df >= 0.2) & (df < 0.5)
    mixed_load = (df >= 0.5) & (df < 2)
    suspended_load = (df >= 2) & (df < 5)
    washing_load = df > 5

    if type_ == "series":

        ax.plot(df[no_motion], ".", color="k", label="No motion")
        ax.plot(df[bed_load], ".", color="blue", label="Bed-load")
        ax.plot(df[mixed_load], ".", color="brown", label="Mixed-load")
        ax.plot(df[suspended_load], ".", color="red", label="Suspended-load")
        ax.plot(df[washing_load], ".", color="purple", label="Washing-load")
        ax.set_ylabel("u$^*$/w$_s$")
        ax.legend()

    else:
        data = df.copy()
        data.loc[:] = np.nan
        data[bed_load] = 1
        data[mixed_load] = 2
        data[suspended_load] = 3
        data[washing_load] = 4

        data.dropna(inplace=True)

        ax.hist(data, bins=np.append(data.unique(), data.max() + 1), density=True)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    ax.grid()

    show(file_name)

    return ax


def cme_calibration(
    cme_coastlines: dict,
    satellite_coastlines: dict,
    polygons: dict,
    nDate2date: dict,
    selected: str,
    ax=None,
    file_name: str = None,
    title: str = None,
):
    """Draw the calibration plot of Coastal Modelling Environment

    Args:
        cme_coastlines (dict): The cme coastlines for every date
        satellite_coastlines (dict): The satellite coastlines for every date
        polygons (dict): The polygons created between cme and satellite coastlines
        nDate2date (dict): conversion beetween dates and number of coastlines outputs
        selected (str): the selected test
        ax (matplotlib.ax, optional): An ax where plot the new figure. Defaults to None.
        file_name (str, optional): The name to be saved. Defaults to None.
        title (str, optional): Title name. Defaults to None.
    """

    import matplotlib.colors as mcolors

    colors = mcolors.CSS4_COLORS

    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
        for name, color in colors.items()
    )
    colorname = [name for hsv, name in by_hsv][:12]
    colorname = [
        "dimgrey",
        "indianred",
        "lightsalmon",
        "saddlebrown",
        "darkorange",
        "gold",
        "olive",
        "greenyellow",
        "palegreen",
        "seagreen",
        "aquamarine",
        "darkcyan",
        "cadetblue",
        "deepskyblue",
        "steelblue",
        "dodgerblue",
        "royalblue",
        "navy",
        "slateblue",
    ]

    n_plots = len(cme_coastlines.keys())

    # axs = handle_axis(ax,row_plots=n_plots,figsize=(6, 13),kwargs={"sharex": "all", "sharey": "all"})
    key_ = list(cme_coastlines.keys())[0]
    start_x = cme_coastlines[key_]["Test_001"].x.min()
    start_y = cme_coastlines[key_]["Test_001"].y.min()
    end_x = cme_coastlines[key_]["Test_001"].x.max()
    end_y = cme_coastlines[key_]["Test_001"].y.max()
    absdy = np.abs(end_y - start_y)
    absdx = np.abs(end_x - start_x)

    is_layout_horizontal = False
    if absdy / absdx > 0.5:
        is_layout_horizontal = True

    if is_layout_horizontal:
        axs = handle_axis(
            ax,
            col_plots=n_plots,
            figsize=(n_plots * 3, 8),
            kwargs={"sharex": "all", "sharey": "all"},
        )
    else:
        axs = handle_axis(
            ax,
            row_plots=n_plots,
            figsize=(8, n_plots * 3),
            kwargs={"sharex": "all", "sharey": "all"},
        )
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    nTests = polygons[0].keys()
    # fig, axs = plt.subplots(len(cme_coastlines.keys()), 1, sharex="all", sharey="all")
    # axs = axs.flatten()
    for id_, nDate in enumerate(cme_coastlines.keys()):
        pmarks = []
        for k, nTest in enumerate(nTests):

            axs[id_] = polygons[id_][nTest].plot(
                ax=axs[id_], color=colorname[k], alpha=0.5
            )

            pmarks.append(
                Patch(
                    facecolor=colorname[k],
                    label=nTest.replace("_", " "),
                )
            )
            if selected == nTest:
                axs[id_].plot(
                    cme_coastlines[nDate][selected].x,
                    cme_coastlines[nDate][selected].y,
                    label="CoastalME coastline",
                )

                axs[id_].plot(
                    satellite_coastlines[nDate].x,
                    satellite_coastlines[nDate].y,
                    label="Satellite coastline",
                )

            if is_layout_horizontal:
                axs[0].set_ylabel("Y UTM (m)", fontweight="bold")
                axs[id_].set_xlabel("X UTM (m)", fontweight="bold")
                fig = plt.gcf()
                fig.suptitle(title)
                if (id_ == 0) & (k == 0):
                    axs[id_].set_title(
                        "Fecha de obtencion: "
                        + nDate2date[nDate].split("_")[1].split(".")[0]
                    )
                elif k == 0:
                    axs[id_].set_title(
                        nDate2date[nDate].split("_")[1].split(".")[0]
                    )

            else:
                if (id_ == 0) & (k == 0):
                    # fig.suptitle()  # , ha="right")
                    axs[id_].text(
                        0.95,
                        1.05,
                        title
                        + "Fecha de obtencion: "
                        + nDate2date[nDate].split("_")[1].split(".")[0],
                        horizontalalignment="right",
                        verticalalignment="bottom",
                        transform=axs[id_].transAxes,
                    )
                elif k == 0:
                    if n_plots > 8:
                        text_yloc = 0.7
                    else:
                        text_yloc = 0.9
                    axs[id_].text(
                        0.95,
                        text_yloc,
                        nDate2date[nDate].split("_")[1],
                        horizontalalignment="right",
                        transform=axs[id_].transAxes,
                    )
                if not is_layout_horizontal:
                    if n_plots > 8:
                        if id_ % 2:
                            axs[id_].set_ylabel("Y UTM (m)", fontweight="bold")
                    else:
                        axs[id_].set_ylabel("Y UTM (m)", fontweight="bold")
                axs[-1].set_xlabel("X UTM (m)", fontweight="bold")
        # axs[id_].set_axis_off()
        axs[id_].grid()
        # axs[id_].set_aspect("equal", "box")
        axs[id_].get_yaxis().get_major_formatter().set_useOffset(False)
        axs[id_].get_yaxis().get_major_formatter().set_scientific(False)
        # axs[id_].set_xticklabels(axs[id_].get_xticks(), rotation=45)

    handles, _ = axs[id_].get_legend_handles_labels()
    box = axs[id_].get_position()

    if is_layout_horizontal:
        ax_num = int(id_ / 2)
        add = 0.5
        if id_ == 0:
            ax_num = 0
        elif (id_ + 1) % 2 == 0:
            add = 0
        else:
            add = 1

        axs[ax_num].legend(
            loc="upper center",
            bbox_to_anchor=(add, -0.25),
            fancybox=True,
            shadow=True,
            handles=[*handles, *pmarks],
            ncol=4,
        )
        axs[id_].set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        plt.subplots_adjust(bottom=0.5)

    else:
        ax_num = int(id_ / 2)
        add = 0.5
        if id_ == 0:
            ax_num = 0
        elif (id_ + 1) % 2 == 0:
            add = 0
        else:
            add = 1
        xadd = np.min([1.0 + n_plots * 0.06, 1.4])
        axs[ax_num].legend(
            loc="upper center",
            bbox_to_anchor=(xadd, add),
            fancybox=True,
            shadow=True,
            handles=[*handles, *pmarks],
            ncol=1,
        )
        axs[id_].set_position(
            [box.x0 + box.width * 0.1, box.y0, box.width * 0.9, box.height]
        )
        # set the spacing between subplots
        # plt.subplot_tool()
        plt.subplots_adjust(hspace=0.2)

    # axs[id_].legend(handles=[*handles, *pmarks], ncol=4)
    # fig = plt.gcf()
    # fig.tight_layout()
    show(file_name)
    return
