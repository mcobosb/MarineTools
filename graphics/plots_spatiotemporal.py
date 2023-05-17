import matplotlib.pyplot as plt
import numpy as np
from marinetools.spatiotemporal import covST
from matplotlib.pyplot import cm


def covExpTh(covDistanceS, covDistanceT, covEmpST, tLag, res, family, type):
    """[summary]

    Args:
        covDistanceS ([type]): [description]
        covDistanceT ([type]): [description]
        covEmpST ([type]): [description]
        tLag ([type]): [description]
        res ([type]): [description]
        family ([type]): [description]
        type ([type]): [description]
    """
    if type == "3d":
        fig = plt.figure(figsize=(5, 4))
        ax = fig.gca(projection="3d")

        x, y = np.meshgrid(
            np.linspace(0, np.amax(covDistanceS), 20),
            np.linspace(0, np.amax(covDistanceT), 20),
        )
        covTh = covST.covariance(family, [x, y], res.x)
        ax.plot_wireframe(x, y, covTh, rstride=1, cstride=1, color="k", lw=0.5)
        ax.plot_surface(
            covDistanceS, covDistanceT, covEmpST, cmap=cm.autumn_r, alpha=0.6
        )

        ax.scatter(covDistanceS, covDistanceT, covEmpST, marker="o", color="k")

        ax.set_xlabel(r"$\mathbf{S_{lag}}$")
        ax.set_ylabel(r"$\mathbf{T_{lag}}$")
        ax.set_ylim(np.max(tLag), np.min(tLag))
        ax.set_zlabel("cov", fontweight="bold")

        covTh = covST.covariance(family, [covDistanceS, covDistanceT], res.x)
        plt.figure(figsize=(5, 4))
        error = covEmpST - covTh
        # e_mda = np.sum(np.abs(error))/np.size(covEmpST)
        e_mse = np.sqrt(np.sum(error ** 2)) / np.size(covEmpST)
        CS3 = plt.contour(covDistanceS, covDistanceT, error, 5, colors="k")
        # cbar = plt.colorbar()
        plt.clabel(CS3, inline=1, fontsize=8)
        plt.ylim([np.max(tLag), np.min(tLag)])
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        textstr = (
            r"$\mathbf{\varepsilon_{RMSE}}$="
            + " "
            + str(np.round(e_mse, 2))
            + "\n"
            + r"$\mathbf{\varepsilon_{max}}$="
            + " "
            + str(np.round(np.max(error), 2))
            + "\n"
            + r"$\mathbf{\varepsilon_{min}}$="
            + " "
            + str(np.round(np.min(error), 2))
        )
        plt.text(
            0.75,
            0.25,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
    else:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        x, y = np.meshgrid(
            np.linspace(0, np.amax(covDistanceS), 20),
            np.linspace(0, np.amax(covDistanceT), 20),
        )
        covTh = covST.covariance(family, [x, y], res.x)
        CS1 = plt.contour(
            x, y, covTh, 10, cmap=cm.autumn_r, alpha=0.6, label=r"$c_{th}$"
        )
        plt.clabel(CS1, inline=1, fontsize=10)
        CS2 = plt.contour(
            covDistanceS,
            covDistanceT,
            covEmpST,
            10,
            cmap=cm.autumn_r,
            alpha=0.6,
            label=r"$c_{emp}$",
        )
        for c in CS2.collections:
            c.set_dashes([(0, (2.0, 2.0))])
        plt.clabel(CS2, inline=1, fontsize=10)

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
        plt.ylim([np.max(tLag), np.min(tLag)])
        plt.legend([r"$c_{th}$", r"$c_{emp}$"], loc="best")

        covTh = covST.covariance(family, [covDistanceS, covDistanceT], res.x)
        plt.subplot(1, 2, 2)
        CS3 = plt.contour(covDistanceS, covDistanceT, np.abs(covEmpST - covTh), 5)
        # cbar = plt.colorbar()
        plt.clabel(CS3, inline=1, fontsize=10)
        plt.ylim([np.max(tLag), np.min(tLag)])

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
    plt.show()


def covSTang(covdist, covdistd, covdistt, empcovang, slag, type):
    """[summary]

    Args:
        covdist ([type]): [description]
        covdistd ([type]): [description]
        covdistt ([type]): [description]
        empcovang ([type]): [description]
        slag ([type]): [description]
        type ([type]): [description]
    """
    covdistd = np.radians(np.vstack((covdistd, covdistd + 180, covdistd[0, :] + 360)))
    covdist = np.vstack((covdist, covdist, covdist[0, :]))

    if type[0] == "polar":
        # -- Plot... ------------------------------------------------
        for i in range(np.shape(empcovang)[2]):
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
            if type[1] == "variogram":
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
                empcov = empcov[0, 0] - empcov
            else:
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
            CS = ax.contourf(covdistd, covdist, empcov)
            ax.contour(covdistd, covdist, empcov, color="k")
            plt.colorbar(CS)
            ax.grid(True)
            ax.set_title("t = " + str(covdistt[i]))

    elif type[0] == "3d":
        covdistx, covdisty = slag * np.cos(covdistd), slag * np.sin(covdistd)

        for i in range(np.shape(empcovang)[2]):
            if type[1] == "variogram":
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
                empcov = empcov[0, 0] - empcov
                title = r"$\mathbf{\gamma}$"
            else:
                title = r"$\mathbf{c}$"
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
            fig = plt.figure(figsize=(5, 4))
            ax = fig.gca(projection="3d")
            ax.plot_surface(covdistx, covdisty, empcov, alpha=0.6, cmap=cm.autumn_r)

            ax.set_xlabel(r"$\mathbf{S_{x}}$", fontweight="bold", labelpad=20)
            axis = np.hstack((-covdistx[0, ::-1], covdistx[0, :]))
            ax.set_xticks(axis)
            ax.set_xticklabels(np.round(np.abs(axis), decimals=2), rotation=45)
            ax.set_ylabel(r"$\mathbf{S_{y}}$", fontweight="bold", labelpad=20)
            ax.set_yticks(axis)
            ax.set_yticklabels(np.round(np.abs(axis), decimals=2), rotation=-45)
            ax.set_title(
                r"$\mathbf{T}$ =  " + str(np.round(covdistt[i], decimals=2)),
                fontweight="bold",
            )
            ax.set_zlabel(title)
            ax.set_zlim([0, np.nanmax(empcovang)])

    plt.show()
    return
