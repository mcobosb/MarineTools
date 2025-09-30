import matplotlib.pyplot as plt


def show(file_name: str = None, res: int = 600):
    """Saves into a file or displays on the screen the result of a plot

    Args:
        * fname (None or string): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * res (int): resolution in dpi of the saved figure

    Returns:
        * Displays on the screen the figure or saves it into a file
    """

    if not file_name:
        plt.show()
    elif file_name == ("to_axes"):
        pass
    else:
        if "png" or "pdf" in file_name:
            plt.savefig(f"{file_name}", dpi=res, bbox_inches="tight")
        else:
            plt.savefig(f"{file_name}" + ".png", dpi=res, bbox_inches="tight")
        plt.close()
    return


def handle_axis(
    ax,
    row_plots: int = 1,
    col_plots: int = 1,
    dim: int = 2,
    figsize: tuple = (5, 5),
    projection=None,
    kwargs: dict = {},
):
    """Creates the matplotlib.axis of the figure if it is required or nothing if it is given

    Args:
        * ax (matplotlib.axis): axis for the plot.
        * row_plots (int, optional): no. of row axis. Defaults to 1.
        * col_plots (int, optional): no. of row axis. Defaults to 1.
        * dim (int, optional): no. of dimensions. Defaults to 2.
        * figsize: as matplotlib
        * projection: to include projections in axis
        * kwargs: as matplotlib

    Returns:
        * Same as the arguments
    """

    fig = None
    if not ax:
        if dim == 2:
            fig, ax = plt.subplots(row_plots, col_plots, figsize=figsize, **kwargs)
        elif not projection is None:
            ax = plt.axes(
                projection=projection
            )  # project using coordinate reference system (CRS) of street map
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        if row_plots + col_plots > 2:
            ax = ax.flatten()

    return fig, ax


def labels(variable):
    """Gives the labels and units for plots

    Args:
        * variable (string): name of the variable

    Returns:
        * units (string): label with the name of the variable and the units
    """

    units = {
        "calms": r"$\delta$ (h)",
        "depth": "Depth (m)",
        "dm": r"$\theta_m$ (deg)",
        "dv": r"$\theta_v$ (deg)",
        "DirM": r"$\theta_m$ (deg)",
        "DirU": r"$\theta_U$ (deg)",
        "DirV": r"$\theta_v$ (deg)",
        "Dmd": r"$\theta_m$ (deg)",
        "Dmv": r"$\theta_v$ (deg)",
        "dur": r"d (hr)",
        "dur_calms": r"$\Delta_0$ (hr)",
        "dur_storm": r"$d_0$ (hr)",
        "eta": r"$\eta$ (m)",
        "hs": r"$H_{s}$ (m)",
        "Hm0": r"$H_{m0}$ (m)",
        "Hs": r"$H_{s}$ (m)",
        "lat": "Latitud (deg)",
        "lon": "Longitude (deg)",
        "ma": r"$M_{ast}$ (m)",
        "mm": r"$M_{met}$ (m)",
        "pr": r"P (mm/day)",
        "Q": r"Q (m$^3$/s)",
        "Qd": r"$Q_d$ (m$^3$/s)",
        "S": r"S (psu)",
        "slr": r"$\Delta\eta$ (m)",
        "storm": r"d (h)",
        "surge": r"$\eta_s$ (m)",
        "swh": r"$H_{s}$ (m)",
        "t": "t (s)",
        "Tm0": r"$T_{m0}$ (s)",
        "tp": r"$T_p$ (s)",
        "Tp": r"$T_p$ (s)",
        "U": r"U (m/s)",
        "V": r"V (m/s)",
        "VelV": "[m/s]",
        "vv": r"$V_v$ (m/s)",
        "Vv": r"$V_v$ (m/s)",
        "Wd": r"$W_d$ (deg)",
        "Wv": r"$W_v$ (m/s)",
        "x": "x (m)",
        "y": "y (m)",
        "z": "z (m)",
        "None": "None",
    }

    if isinstance(variable, str):
        if not variable in units.keys():
            labels_ = ""
        else:
            labels_ = units[variable]
    elif isinstance(variable, list):
        labels_ = list()
        for var_ in variable:
            if not var_ in units.keys():
                labels_.append("")
            else:
                labels_.append(units[var_])
    else:
        raise ValueError
    return labels_
