import warnings as warn

import numpy as np
from marinetools.spatiotemporal.bme import pairsindex, stmatrix


def cov(dfh, dfs, slag, tlag):
    """
    Compute the ST covariance.

    Args:
        - dfh: dataframe of hard data
        - dfs: dataframe of soft data
        - slag: vector with the spatial distances where compute the covariance
        - tlag: vector with the temporal distances where compute the covariance

    Returns:
        - empcovst: matrix with ST covariance
        - pairsnost: no of pairs at every ST location
        - covdists: spatial distances at every location
        - covdistt: temporal distances at every location
    """
    slagtol = np.hstack((0, np.diff(slag)))
    tlagtol = np.hstack((0, np.diff(tlag)))

    dfz, coups, coupt = stmatrix(dfh, dfs)

    ns, nt = len(slag), len(tlag)
    idxpairs = pairsindex(coups, slag, slagtol)
    idtpairs = pairsindex(coupt, tlag, tlagtol)

    # -----------------------------------------
    # computing covariance
    # -----------------------------------------
    empcovst = np.ones((ns, nt)) * np.nan
    pairsnost = np.zeros((ns, nt))
    xmean, ymean, xymean = (
        np.ones((ns, nt)) * np.nan,
        np.ones((ns, nt)) * np.nan,
        np.ones((ns, nt)) * np.nan,
    )
    for inds in range(ns):
        # Test if we have data for this slag
        if np.shape(idxpairs[inds])[1] != 0:
            for indt in range(0, nt):
                # Test if we have data for this tlag
                if np.shape(idtpairs[indt])[1] != 0:
                    xcol = dfz.loc[idtpairs[indt][0], idxpairs[inds][0]].values
                    xrow = dfz.loc[idtpairs[indt][1], idxpairs[inds][1]].values
                    idxValid = ~np.isnan(xcol) & ~np.isnan(xrow)
                    pairsnost[inds, indt] = np.sum(idxValid)
                    if pairsnost[inds, indt] != 0:
                        xcol = xcol[idxValid]
                        xrow = xrow[idxValid]
                        xmean[inds, indt] = np.mean(xcol)
                        ymean[inds, indt] = np.mean(xrow)
                        xymean[inds, indt] = np.mean(xcol * xrow)
                        empcovst[inds, indt] = (
                            xymean[inds, indt] - xmean[inds, indt] * ymean[inds, indt]
                        )
    # -------------------------------------------------------------------

    empcovst[0, 0] = dfz.stack().var()
    covdists = np.tile(slag, (nt, 1)).T
    covdistt = np.tile(tlag, (ns, 1))
    return empcovst, pairsnost, covdists, covdistt


def covang(dfh, dfs, slag, tlag, dinfo):
    """
    Compute the ST covariance for several directions.

    Args:
        - dfh: dataframe of hard data
        - dfs: dataframe of soft data
        - slag: vector with the spatial distances where compute the covariance
        - tlag: vector with the temporal distances where compute the covariance
        - dinfo: vector with the information about directions along to compute the covariance

    Returns:
        - empcovst: 3d array with STD covariance
        - pairsnost: 3d array with the no of pairs at every STD location
        - covdistx: matrix with spatial location x
        - covdisty: matrix with spatial location y
        - covdistt: vector with temporal distances
    """
    slagtol = np.hstack((0, np.diff(slag)))
    tlagtol = np.hstack((0, np.diff(tlag)))
    dlag, dlagtol = dinfo[0], dinfo[1]

    dfz, coups, coupt = stmatrix(dfh, dfs)

    ns, nt, nd = len(slag), len(tlag), len(dlag)
    idxpairs = pairsindex(coups, slag, slagtol, [dlag, dlagtol])
    idtpairs = pairsindex(coupt, tlag, tlagtol)

    # -----------------------------------------
    # computing covariance
    # -----------------------------------------
    empcovst = np.ones((ns, nd, nt)) * np.nan
    pairsnost = np.zeros((ns, nt, nd))
    xmean, ymean, xymean = (
        np.ones((ns, nt, nd)) * np.nan,
        np.ones((ns, nt, nd)) * np.nan,
        np.ones((ns, nt, nd)) * np.nan,
    )
    for indd in range(nd):
        print("indd: " + str(indd))
        for inds in range(ns):
            print("inds: " + str(inds))
            # Test if we have data for this slag
            if np.shape(idxpairs[inds])[1] > 1:
                for indt in range(nt):
                    print("indt: " + str(indt))
                    # Test if we have data for this tlag
                    if np.shape(idtpairs[indt])[1] > 1:
                        xcol = dfz.loc[
                            idtpairs[indt][0], idxpairs[inds][indd][0]
                        ].values
                        xrow = dfz.loc[
                            idtpairs[indt][1], idxpairs[inds][indd][1]
                        ].values
                        idxValid = ~np.isnan(xcol) & ~np.isnan(xrow)
                        pairsnost[inds, indt, indd] = np.sum(idxValid)
                        if pairsnost[inds, indt, indd] != 0:
                            xcol = xcol[idxValid]
                            xrow = xrow[idxValid]
                            xmean[inds, indt, indd] = np.mean(xcol)
                            ymean[inds, indt, indd] = np.mean(xrow)
                            xymean[inds, indt, indd] = np.mean(xcol * xrow)
                            empcovst[inds, indd, indt] = (
                                xymean[inds, indt, indd]
                                - xmean[inds, indt, indd] * ymean[inds, indt, indd]
                            )
    # -------------------------------------------------------------------

    empcovst[0, 0, 0] = dfz.stack().var()
    covdist, covdistd, covdistt = np.tile(slag, (nd, 1)), np.tile(dlag, (ns, 1)).T, tlag
    return empcovst, pairsnost, covdist, covdistd, covdistt


def covariance(name, D, param):
    """
    Compute the theoretical covariance.

    Args:
        - name: family of theoretical ST covariances shapes
        - D: list with two elements. First is the spatial distance and the second is the temporal distance
        - param: list with the parameters of the model

    Returns:
        - res: covariance at D points.
    """
    d, t = np.asarray(D[0]), np.asarray(D[1])
    if name == "exponentialST":
        res = param[0] * np.exp(-d / param[1] - t / param[2]) - param[3]
    elif name == "exponentialSTC":
        res = (
            param[0]
            * np.exp(-(d ** 0.5) / param[1] - t / param[2] - d ** 0.5 * t / param[3])
            - param[4]
        )
    else:
        warn.warn("Sorry! This model is not implemented yet.")
    return res


def fit(param, empcovst, dist, name):
    """
    Compute the minimum square error between the theorical and empirical covariance.

    Args:
        - param: parameters of the model
        - empcovst: empirical covariance at spatiotemporal distance
        - dist: ST distance
        - name: family of the theoretical covariance model

    Returns:
        - square estimation error
    """
    cov = covariance(name, dist, param)
    return np.sum(np.sum(((empcovst - cov) ** 2), axis=1), axis=0)


# w0, w1 = np.meshgrid(np.linspace(0, 1, len(D[0])), np.linspace(0, 1, len(D[1])))
# weight = np.fliplr(w0) + np.flipud(w1)
#    if fit:
#        cov = param[0] * np.exp(-D[0]/param[1] - D[1]/param[2]) + param[3]
# out = np.sum(np.sum(weight*((expCovST - cov) ** 2), axis=1), axis=0)
