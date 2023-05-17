import datetime
from itertools import product
from pathlib import Path

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.graphics.utils import handle_axis, labels, show
from marinetools.temporal.analysis import storm_properties
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.temporal.fdist.copula import Copula
from marinetools.utils import auxiliar
from pandas.plotting import register_matplotlib_converters

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

register_matplotlib_converters()

cmp_g = cmocean.cm.haline_r
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# matplotlib.rcParams["text.latex.preamble"] = [r"\boldmath", r"]


def plot_pot_lmom(
    info,
    nvar,
    file_name: str = None,
):
    """Funcion que dibuja los resultados del analisis de peaks sobre umbral con el metodos de los momentos.

    Args:
        info ([type]): [description]
        nvar ([type]): [description]
    """
    selected_value = info["thresholds"][np.argmin(info["au2pv_lmom"])]

    xlim = (np.floor(np.min(info["thresholds"])), np.ceil(np.max(info["thresholds"])))
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(left=0.2, wspace=0.4)

    ax = []
    ax.append(plt.subplot2grid((3, 3), (0, 0)))
    ax.append(plt.subplot2grid((3, 3), (1, 0)))
    ax.append(plt.subplot2grid((3, 3), (2, 0)))

    # MRLP
    ax[0].set_title(r"\textbf{PARAMETERS USING L-MOMENTS}", fontsize=10)
    ax[0].plot(info["thresholds"], info["mean_value_lmom"][:, 3], "k-", lw=2)
    ax[0].plot(
        info["thresholds"],
        info["upper_lim"][:, 3],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 3],
        "k--",
    )
    ax[0].axvline(x=selected_value, color="r")
    ax[0].set_ylabel(r"\textbf{Location parameters ($\mathbf{\nu}$)}")
    ax[0].grid()
    ax[0].set_xlim(xlim)
    ax[0].get_xaxis().set_ticklabels([])
    ax[0].yaxis.set_label_coords(-0.2, 0.5)

    # shape
    ax[1].plot(info["thresholds"], info["mean_value_lmom"][:, 0], "k-", lw=2)
    ax[1].plot(
        info["thresholds"],
        info["upper_lim"][:, 0],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 0],
        "k--",
    )
    ax[1].axvline(x=selected_value, color="r")
    ax[1].set_ylabel(r"\textbf{Shape parameter (k)}")
    ax[1].grid()
    ax[1].set_xlim(xlim)
    ax[1].get_xaxis().set_ticklabels([])
    ax[1].yaxis.set_label_coords(-0.2, 0.5)

    # scale modif
    ax[2].plot(info["thresholds"], info["mean_value_lmom"][:, 4], "k-", lw=2)
    ax[2].plot(
        info["thresholds"],
        info["upper_lim"][:, 4],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 4],
        "k--",
    )
    ax[2].axvline(x=selected_value, color="r")
    ax[2].set_ylabel(r"\textbf{Modified scale parameter ($\mathbf{\sigma^*}$)}")
    ax[2].grid()
    ax[2].set_xlabel(r"\textbf{Threshold}")
    ax[2].set_xlim(xlim)
    ax[2].yaxis.set_label_coords(-0.2, 0.5)

    # Gráficas QTR
    for i in range(0, np.size(info["tr_eval"])):
        ax.append(plt.subplot2grid((3, 3), (0 + i, 1)))
        ax[3 + i].plot(
            info["thresholds"], info["mean_value_lmom"][:, 6 + i], "k-", lw=2
        )
        # plt.hold(True)
        ax[3 + i].plot(
            info["thresholds"],
            info["upper_lim"][:, 6 + i],
            "k--",
            info["thresholds"],
            info["lower_lim"][:, 6 + i],
            "k--",
        )
        ax[3 + i].axvline(x=selected_value, color="r")
        ax[3 + i].set_ylabel(
            r"\textbf{Return period of " + str(int(info["tr_eval"][i])) + " yr}"
        )
        ax[3 + i].grid()
        ax[3 + i].set_xlim(xlim)
        ax[3 + i].set_ylim(
            (
                np.floor(np.min(info["lower_lim"][:, 6 + i])),
                np.ceil(np.max(info["upper_lim"][:, 6 + i])),
            )
        )
        if i+1 < np.size(info["tr_eval"]):
            ax[3 + i].get_xaxis().set_ticklabels([])

    ax[3].set_title(r"\textbf{VALUES FOR SEVERAL RETURN PERIODS}", fontsize=10)
    ind_ = 3 + np.size(info["tr_eval"])
    ax[ind_ - 1].set_xlabel(r"\textbf{Threshold}")

    # Gráficas Estadístico y p-valor
    ax.append(plt.subplot2grid((3, 3), (0, 2), rowspan=2))
    ax[ind_].plot(info["thresholds"], info["au2_lmom"], "k-", lw=2)
    ax[ind_].axvline(x=selected_value, color="r")
    ax[ind_].grid()
    ax[ind_].set_ylabel(r"\textbf{$\mathbf{A_R^2}$ statistic}")
    # ax[ind_].yaxis.set_label_coords(-0.2, 0.5)
    # ax[ind_].get_yaxis().set_ticklabels([])

    ax.append(ax[ind_].twinx())
    ax[ind_ + 1].plot(info["thresholds"], info["au2pv_lmom"], "k--")
    ax[ind_ + 1].axvline(x=selected_value, color="r")
    ax[ind_ + 1].plot(
        selected_value,
        info["au2pv_lmom"][np.argmin(info["au2pv_lmom"])],
        "or",
        label="Selected",
    )
    ax[ind_ + 1].set_ylabel(r"\textbf{$\mathbf{1-p_{value}}$}")
    ax[ind_ + 1].grid()
    ax[ind_ + 1].set_xlim(xlim)
    ax[ind_ + 1].legend()

    # Eventos/año
    ax.append(plt.subplot2grid((3, 3), (2, 2)))
    ax[ind_ + 2].plot(info["thresholds"], info["mean_value_lmom"][:, 5], "k-", lw=2)
    ax[ind_ + 2].axvline(x=selected_value, color="r")
    ax[ind_ + 2].set_ylabel(r"\textbf{No. of annual maxima}", fontweight="bold")
    ax[ind_ + 2].set_xlabel(r"\textbf{Threshold}", fontweight="bold")
    ax[ind_ + 2].grid()
    ax[ind_ + 2].set_xlim(xlim)
    # ax[ind_ + 2].yaxis.set_label_coords(-0.2, 0.5)

    ax.append(ax[ind_ + 2].twinx())
    nl = len(ax[ind_ + 2].get_yticks())
    nmax = np.ceil(
        np.ceil(np.max(info["mean_value_lmom"][:, 5])) * info["nyears"] / (nl - 1)
    )
    n2l = np.linspace(0, nmax * (nl - 1), nl)

    ax[ind_ + 3].set_ylabel(r"\textbf{No. of peaks}", fontweight="bold")
    ax[ind_ + 3].set_ylim(
        [0, np.ceil(np.max(info["mean_value_lmom"][:, 5])) * info["nyears"]]
    )
    ax[ind_ + 3].set_xlim(xlim)
    ax[ind_ + 3].set_yticks(n2l)
    show(file_name)
    return


def plot_annual_maxima_analysis(
    boot,
    orig,
    ci,
    tr,
    peaks,
    npeaks,
    eventanu,
    func,
    flabel=r"H$_{m0}$ (m)",
    tr_plot=100,
):
    """[summary]

    Args:
        boot (dict): [description]
        orig (dict): [description]
        ci (dict): [description]
        tr (type): [description]
        peaks ([type]): [description]
        npeaks ([type]): [description]
        eventanu ([type]): [description]
        func ([type]): [description]
        flabel (regexp, optional): [description]. Defaults to r'H{m0}$ (m)'.
        tr_plot (int, optional): Return period for zoom in. Defaults to 100 years.
    """

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.2, wspace=0.4)

    # Histograma de k
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    n0, c0 = np.histogram(boot[0][0][:, 0], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    if boot[1]:
        plt.plot(np.array([0, 0]), ax1.get_ylim(), "r--")

    plt.plot(orig[0][0][0] * np.array([1, 1]), ax1.get_ylim(), "b--")
    plt.xlabel(" Shape (k)")
    plt.ylabel("Frequency")
    ax1.get_yaxis().set_ticks([])

    # Histograma de sigma
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    n0, c0 = np.histogram(boot[0][0][:, 2], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )

    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, 2], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][2] * np.array([1, 1]), ax2.get_ylim(), "r--")

    plt.plot(orig[0][0][2] * np.array([1, 1]), ax2.get_ylim(), "b--")
    plt.xlabel(r"Scale ($\sigma$)")
    plt.ylabel("Frequency")
    ax2.get_yaxis().set_ticks([])

    # Histograma de mu
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    n0, c0 = np.histogram(boot[0][0][:, 1], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, 1], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][1] * np.array([1, 1]), ax3.get_ylim(), "r--")

    plt.plot(orig[0][0][1] * np.array([1, 1]), ax3.get_ylim(), "b--")
    plt.xlabel(r"Position ($\mu$)")
    plt.ylabel("Frequency")
    ax3.get_yaxis().set_ticks([])

    # Histograma de Tr plot

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.68, 0.05),
        0.25,
        0.42,
        fill=False,
        color="r",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    ax4 = plt.subplot2grid((2, 3), (1, 2))

    tri = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
    tri = np.hstack((np.arange(1.1, 2 + 1e-6, 0.1), tri[tri > 2]))
    idx = np.where(np.abs(tri - tr_plot) == np.min(np.abs(tri - tr_plot)))[0][0]

    n0, c0 = np.histogram(boot[0][0][:, idx + 3], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    plt.plot(orig[0][0][idx + 3] * np.array([1, 1]), ax4.get_ylim(), "b--")
    plt.xlabel(flabel + r"$_{Tr " + str(int(tri[idx])) + " yr}$")
    plt.ylabel("Frequency")
    ax4.get_yaxis().set_ticks([])

    plt.axvline(x=ci[0][0][1, idx + 3], color="gray")
    plt.axvline(x=ci[0][0][0, idx + 3], color="gray")

    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, idx], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][idx] * np.array([1, 1]), ax4.get_ylim(), "r--")

    # Gráfico Tr-Xtr
    ax5 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax5.fill(
        np.hstack((tr, tr[::-1])),
        np.hstack((ci[0][0][0, 4:], ci[0][0][1, ::-1][:-4])),
        color="gray",
        alpha=0.5,
        lw=0,
        label="confidence band",
    )
    if func == "genpareto":
        ax5.plot(
            1.0 / ((1.0 - np.arange(1.0, npeaks + 1.0) / (npeaks + 1.0)) * eventanu),
            np.sort(peaks),
            "g.",
            markersize=8,
            label="data",
        )
    else:
        ax5.plot(
            1.0 / npeaks[::-1], np.sort(eventanu), "g.", markersize=8, label="data"
        )

    ax5.plot(tr, orig[0][0][4:], "b--", label=func)
    ax5.axvline(x=tri[idx], color="r")

    if boot[1]:
        label = "gumbel_r"
        if func == "genpareto":
            label = "expon"
        ax5.plot(tr, orig[1][0][-len(tr) :], "r--", label=label)
        ax5.plot(tr, ci[1][0][0, -len(tr) :], "r-.")
        ax5.plot(tr, ci[1][0][1, -len(tr) :], "r-.")

    ax5.legend()

    ax5.set_xlim([1, 1000])
    ax5.set_xscale("log", nonposx="clip")
    ax5.set_ylabel(flabel)
    ax5.set_xlabel("Return period (yr)")
    return


def plot_serie_peaks(df_serie, df_peaks, ylab, nombre):
    """Funcion que dibuja la serie temporal y peaks de la variable"""

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(df_serie.index, df_serie.iloc[:, 0], color="b")
    plt.plot(df_peaks.index, df_peaks.iloc[:, 0], "ro", color="orange")
    plt.ylabel(ylab)
    plt.xticks(rotation=30)
    plt.legend(("Timeseries", "peaks"), loc="best")
    plt.title(nombre)


def plot_serie_peaks_umbral(df_serie, df_peaks, ylab, umbral, nombre):
    """Funcion que dibuja la serie temporal y los valores sobre umbral de la variable"""

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(df_serie.index, df_serie.iloc[:, 0], color="b")
    plt.plot(df_peaks.index, df_peaks.iloc[:, 0], "ro", color="orange")
    plt.ylabel(ylab)
    plt.xticks(rotation=30)
    plt.legend(
        ("Timeseries", "peaks (threshold = {:.2f}".format(umbral) + ")"), loc="best"
    )

    plt.title(nombre)
