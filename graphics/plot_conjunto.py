# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from marinetools.graphics.utils import handle_axis, labels, show
from matplotlib import gridspec
from scipy.stats import genextreme, lognorm, norm

# register_matplotlib_converters()

# cmp_g = cmocean.cm.haline_r
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)
params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}


def dscatter(x, y, *varargin):
    """ """
    lambda_ = 20

    minx, maxx, miny, maxy = np.min(x), np.max(x), np.min(y), np.max(y)

    nbins = np.array(
        [np.min((np.size(np.unique(x)), 200)), np.min((np.size(np.unique(y)), 200))]
    )

    edges1 = np.linspace(minx, maxx, nbins[0] + 1)
    edges1 = np.hstack((-np.inf, edges1[1:-1], np.inf))
    edges2 = np.linspace(miny, maxy, nbins[0] + 1)
    edges2 = np.hstack((-np.inf, edges2[1:-1], np.inf))

    n = nbins[0] + 1
    bin_ = np.zeros([n, 2])
    # Reverse the columns to put the first column of X along the horizontal
    # axis, the second along the vertical.
    dum, bin_[:, 1] = np.histogram(x, bins=edges1)
    dum, bin_[:, 0] = np.histogram(y, bins=edges2)
    h = np.bincount(bin_, weights=nbins([1, 0])) / n
    g = smooth1D(h, nbins[1] / lambda_)
    f = smooth1D(g.T, nbins[0] / lambda_).T

    f = f / np.max(f)
    ind = np.ravel_multi_index(np.shape(f), bin_[:, 0], bin[:, 1], order="F")
    col = f[ind]
    h = plt.scatter(x, y, col, "s", "filled")


def smooth1D(y, lambda_):
    m, n = np.shape(y)
    e = np.eye(m)
    d1 = np.diff(e)
    d2 = np.diff(d1)
    p = lambda_ ** 2 * d2.T * d2 + 2 * lambda_ * d1.T * d1
    z = (e + p) / y
    return z


def condicional(
    xpar,
    xparaux,
    xlim,
    ylim,
    alpha,
    dist,
    param,
    yest,
    reg,
    xg,
    yg,
    pdfg,
    ci,
    x,
    y,
    xlab,
    ylab,
    yrlab,
    valmed,
):
    """Funcion que dibuja los resultados del regimen conjunto a partir de la funcion condicional

    Args:
        - xpar:
        - xlim:
        - ylim:
        - alpha:
        - dist:
        - param:
        - yest:
        - reg:
        - xg:
        - yg:
        - pdfg:
        - ci:
        - x:
        - y:

    Returns:
        La grafica

    """

    from marinetools.temporal.fdist.regimen_conjunto import poli1, poli2, pote1, pote2

    xpar2 = np.asarray(xparaux)

    ndat = len(yest)

    ndat_max = 10000
    if ndat > ndat_max:
        # ids_r = np.random.choice(len(yest), ndat_max, replace=False)
        ids_r = np.linspace(0, ndat - 1, ndat_max, dtype=int)
        xpar = xpar[ids_r, :]
        xpar2 = xpar2[ids_r, :]
        yest = yest[ids_r, :]
        x = x[ids_r]
        y = y[ids_r]

    plt.style.use("ggplot")
    xaux = np.linspace(np.min(x), np.max(x), 1000)
    yaux = np.linspace(np.min(y), np.min(y), 1000)
    d_dist, d_reg = dict(), dict()
    d_dist["lognormal"], d_dist["normal"], d_dist["gev"] = lognorm, norm, genextreme
    d_reg["poli1"], d_reg["poli2"], d_reg["pote1"], d_reg["pote2"] = (
        poli1,
        poli2,
        pote1,
        pote2,
    )

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6)

    ax1 = fig.add_subplot(gs[0, 0:2])
    id1 = np.where(xpar[:, -2] >= alpha)
    id2 = np.where(xpar[:, -2] < alpha)
    ax1.plot(xpar[id1, 0], xpar[id1, -2], "b.")
    ax1.plot(xpar[id2, 0], xpar[id2, -2], "r.")
    ax1.plot((np.min(xpar[:, 0]), np.max(xpar[:, 0])), (alpha, alpha), "-r")
    # ax1.set_ylim([0,1])
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("p-valor")
    ax1.tick_params(axis="x", labelsize=10)
    #    plt.legend((r'$p-valor > \alpha$', r'$p-valor < \alpha$', r'$\alpha$'), loc='best', )
    #     plt.title('chi2test - ajuste con dist. '+dist)

    npar = len(xparaux[0])

    if npar == 2:
        ini = 0
        for i in range(npar):
            ax2 = fig.add_subplot(gs[1, ini : ini + 3])
            ax2.plot(xpar[:, 0], xpar2[:, i], ".", color="gray")
            ax2.plot(xaux, param[i], "-r", lw=2)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(dist + [" - localizacion", " - escala"][i])
            ax2.tick_params(axis="x", labelsize=8)
            ini += 3

    elif npar == 3:
        ini = 0
        for i in range(npar):
            ax2 = fig.add_subplot(gs[1, ini : ini + 2])
            ax2.plot(xpar[:, 0], xpar2[:, i], ".", color="gray")
            ax2.plot(xaux, param[i], "-r", lw=2)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(dist + [" - forma", " - localizacion", " - escala"][i])
            ax2.tick_params(axis="x", labelsize=8)
            ini += 2

    ax3 = fig.add_subplot(gs[0, 2:-2])
    id_ = np.where(np.isnan(yest) | np.isinf(yest))[0]
    if any(id_):
        print("Hay valores reconstruidos de la variable que son nan o inf")
    id_ = np.where(~np.isnan(yest) | ~np.isinf(yest))[0]
    #    dscatter(xpar[id_, 1], yest[id_])
    ax3.scatter(xpar[id_, 1], yest[id_])
    ax3.plot(xpar[id_, 1], xpar[id_, 1], "k", lw=2)
    ax3.set_xlabel(ylab)
    ax3.set_ylabel(yrlab)
    ax3.tick_params(axis="x", labelsize=8)
    # plt.title('Y original - Y reconstruida')

    ax4 = fig.add_subplot(gs[0, 4:])
    #    dscatter(x, y)
    ax4.contour(xg, yg, pdfg)
    ax4.scatter(x, y, 10, "gray", label="datos")
    ax4.plot(xaux, valmed, "k", lw=2, label="ajuste")
    ax4.plot(xaux, ci[0, :], "--r", lw=2, label="ic")
    ax4.plot(xaux, ci[1, :], "--r", lw=2)
    ax4.legend(loc="best", scatterpoints=1, fontsize=8)
    ax4.set_xlim([np.min(x), np.max(x)])
    # plt.tight_layout(pad=0.5)
    ax4.set_xlabel(xlab)
    ax4.set_ylabel(ylab)
    ax4.tick_params(axis="x", labelsize=8)

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.4)


def plot_copula(copula, ax=None, labels=[], file_name: str = None, log: bool = False):
    """Plots the copula function

    Args:
        * df (pd.DataFrame): raw time series
        * variables (list): names of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        The plot
    """

    nlen, nlent = 1000, 1000

    fig, ax = handle_axis(ax)

    data1 = copula.X
    data2 = copula.Y

    x, y = [], []
    xt = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlent)
    u, v = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlen), np.linspace(
        1 / len(data1), 1 - 1 / len(data1), nlent
    )
    copula.generate_C(u, v)
    for j in xt:
        copula.U = xt
        copula.V = np.ones(nlent) * j
        copula.generate_xy()
        if copula.X1.size == 0:
            x.append(copula.U * 0)
        else:
            x.append(copula.X1)

        if copula.Y1.size == 0:
            y.append(copula.U * 0)
        else:
            y.append(copula.Y1)

    if log:
        x, y = np.log10(x), np.log10(y)
    cs = ax.contour(np.asarray(x), np.asarray(y), copula.C, 8, linestyles="dashed")

    f, xedges, yedges = np.histogram2d(data1, data2, bins=nlen)
    Fe = np.cumsum(np.cumsum(f, axis=0), axis=1) / (np.sum(f) + 1)
    xmid, ymid = (xedges[0:-1] + xedges[1:]) / 2, (yedges[0:-1] + yedges[1:]) / 2
    xe, ye = np.meshgrid(xmid, ymid)
    if log:
        xe, ye = np.log10(xe), np.log10(ye)
    cs = ax.contour(xe, ye, np.flipud(np.rot90(Fe)), 8, linestyles="solid")

    ax.clabel(cs, cs.levels, inline=True, fontsize=10)
    ax.text(
        0.6,
        0.8,
        r"$\theta$ = " + str(np.round(copula.theta, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.6,
        0.75,
        r"$\tau$ = " + str(np.round(copula.tau, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    show(file_name)

    return ax
