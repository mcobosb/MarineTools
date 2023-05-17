import os

import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.spatiotemporal import covST


def mom(dfk, dfh, dfs, covmodel, covparam, nmax, dmax, order, options, path, name):
    """
    Compute the bme moments.

    Args:
        - dfk: dataframe of estimation points
        - dfh: dataframe of hard-data
        - dfs: dataframe of soft-data
        - covmodel: covariance model
        - covparam: parameters of the covariance model
        - nmax: list with the max no. of data allowed for hard and soft data
        - dmax: list with the max range in S, T and the S/T rate
        - order: list with the order of the bme local regression in space and time
        - options: a list with the max no of intervals to compute the integral, no of moments and the percentile

    Returns:
        - moments: vector with the estimated moments at estimation points
    """
    if os.path.isfile(os.path.join(path, name + ".npy")):
        moments = np.load(os.path.join(path, name + ".npy"))
    else:
        # no of elements
        nk = np.shape(dfk)[0]

        moments = np.zeros([nk, int(options[1]) + 1]) * np.nan

        # Main loop starts here
        for i in range(len(dfk)):
            print(
                "Point no. "
                + str(i + 1).zfill(5)
                + " of "
                + str(len(dfk))
                + ". A "
                + str(np.round((i + 1) / float(len(dfk)) * 100, decimals=2)).zfill(5)
                + "% completed."
            )

            ck0 = pd.DataFrame([dfk.values[i]], index=[0], columns=["x", "y", "t"])
            # print(ck0)

            # Select the local neighbourhood for all variables
            chl, zhl, dh, sumnh, _ = neighbours(
                ck0, dfh.loc[:, ["x", "y", "t"]], dfh.loc[:, ["h"]], nmax[0], dmax
            )
            csl, zsl, ds, sumns, indexs = neighbours(
                ck0, dfs.loc[:, ["x", "y", "t"]], dfs.loc[:, ["h"]], nmax[1], dmax
            )

            # Calculate the covariance matrices
            k_kk = covST.covariance(covmodel, [0, 0], covparam)
            k_hh = covST.covariance(
                covmodel,
                [
                    coord2dist(chl.loc[:, ["x", "y"]].values),
                    coord2dist(chl.loc[:, "t"].values),
                ],
                covparam,
            )
            k_ss = covST.covariance(
                covmodel,
                [
                    coord2dist(csl.loc[:, ["x", "y"]].values),
                    coord2dist(csl.loc[:, "t"].values),
                ],
                covparam,
            )
            k_kh = covST.covariance(covmodel, [dh[0], dh[1]], covparam)
            k_ks = covST.covariance(covmodel, [ds[0], ds[1]], covparam)
            k_sh = covST.covariance(
                covmodel,
                [
                    coord2dist(
                        csl.loc[:, ["x", "y"]].values, chl.loc[:, ["x", "y"]].values
                    ),
                    coord2dist(csl.loc[:, "t"].values, chl.loc[:, "t"].values),
                ],
                covparam,
            )

            if k_hh.size == 0:
                khs = k_ss
            else:
                khs = np.vstack((np.hstack((k_hh, k_sh)), np.hstack((k_sh.T, k_ss))))

            vsl = pd.DataFrame(dfs.iloc[indexs].s)
            mkest, mhest, msest, _ = localmeanbme(
                ck0, chl, csl, zhl, zsl, vsl, khs, order
            )
            if sumnh > 0:  # substract the mean from the hard data
                zhl.h = zhl.h - mhest
            if sumns > 0:  # substract the mean from the soft data
                zsl.h = zsl.h - msest

            # Calculate the statistical moments
            BsIFh = np.dot(k_sh.T, np.linalg.inv(k_hh))
            KsIFh = k_ss - np.dot(BsIFh, k_sh)

            kk_hs = np.hstack((k_kh, k_ks))
            BkIFhs = np.dot(kk_hs, np.linalg.inv(khs))
            KkIFhs = k_kk - np.dot(BkIFhs, kk_hs.T)
            moments[i, 0], moments[i, 1], moments[i, 2], moments[i, 3] = momentsfun(
                zhl.h.values,
                zsl.h.values,
                vsl.values,
                [sumnh, sumns],
                BsIFh,
                KsIFh,
                BkIFhs,
                KkIFhs,
                options,
            )

            moments[i, 1] = moments[i, 1]
            # + mkest
            moments[i, 2] = moments[i, 2] ** 2
            # print(moments[i, :])
            # print(mkest)
        np.save(os.path.join(path, name + ".npy"), moments)

    return moments


def localmeanbme(ck, ch, cs, zh, ms, vs, khs, order):
    """
    Compute the local mean estimation for bme

    Args:
        - ck, ch, cs: coordinates of estimation point, hard and soft-data
        - zh, ms, vs: values of hard data, mean and variance of soft-data
        - k_hh, k_sh, k_ss: covariance matrix for hard data, soft-hard data and soft-data
        - order: polinomial order or regression

    Returns:
        - mkest: estimate of the mean at the estimation point
        - mhest: vector of estimate of the mean for hard data locations
        - msest: vector of estimate of the mean for soft data locations
        - vkest: variance of the estimated mean at zk
    """
    nh = np.shape(zh)[0]
    c = pd.concat([ch, cs], sort=False)

    ind_diag = np.arange(nh, len(c))
    khs[ind_diag, ind_diag] = np.diag(vs)
    z = np.vstack((zh, ms))

    best, vbest, mbest = bmeregression(c, z, order, khs)
    mhest = mbest[:nh, 0]
    msest = mbest[nh:, 0]

    ck = np.vstack((ck.x.values, ck.y.values, ck.t.values)).T
    x = designmatrix(ck, order)

    mkest = np.dot(x, best)
    vkest = np.dot(np.dot(x, vbest), x.T)

    return mkest, mhest, msest, vkest


def momentsfun(zh, zs, vs, nhs, BsIFh, KsIFh, BkIFhs, KkIFhs, options):
    """
    Compute the moments of the bme posterior pdf

    Args:
        - zh:
        - nhs:
        - vsest:
        - BsIFh:
        - KsIFh:
        - BkIFhs:
        - KkIFhs:
        - options:

    Returns:
        - moments: estimated moments
    """
    nh, ns = nhs[0], nhs[1]
    maxpts, nmom, perc = options[0], options[1], options[2]

    Bs = np.ones(2)
    moments = np.zeros(4)
    if zh.size == 0:
        msIFh = zs
    else:
        msIFh = np.dot(BsIFh, zh)
    Bs[1] = np.dot(BkIFhs[:nh], zh)

    per = np.tile([1 - perc, perc], (ns, 1))
    zper = []
    for i, j in enumerate(zs):
        zper.append(st.norm.ppf(per[i], loc=j, scale=np.sqrt(vs[i])))
    zper = np.asarray(zper)

    As = np.zeros([ns, 2])
    As[:, 1] = BkIFhs[nh:]
    p = np.array([1, 1])
    val = mvmomvec(msIFh, vs, KsIFh, As, Bs, p, maxpts, zper)

    moments[0] = val[0]
    moments[1] = val[1] / val[0]
    p = np.array([2, 3])
    As[:, 0] = BkIFhs[nh:]
    Bs[0] = Bs[1] - moments[1]
    val = mvmomvec(msIFh, vs, KsIFh, As, Bs, p, maxpts, zper)

    moments[2] = np.sqrt(KkIFhs + val[0] / moments[0])
    moments[3] = (val[1] / moments[0]) / moments[2] ** 3

    return moments


def mvmomvec(ms, vs, ks, As, Bs, p, maxpts, zp):
    """
    Compute the integrals arising from calculation of moments using the trapezoidal rule

    Args:
        - ms: mean of soft data
        - vs: variance of soft data
        - As:
        - Bs:
        - p: order of the moments
        - maxpts: max no of intervals to compute the integration

    Returns:
        - results of the numerical integration
    """
    maxpts = int(maxpts)
    xs = np.asarray([np.linspace(zp[i, 0], zp[i, 1], maxpts) for i in range(len(zp))])

    # def is_pos_def(x):
    #     return np.all(np.linalg.eigvals(x) > 0)

    # if is_pos_def(ks):
    stn = st.multivariate_normal(cov=ks)
    # else:
    #     stn = st.multivariate_normal(cov=ks*0+np.diag(ks))
    nms_c = stn.pdf(xs.T)
    Bs = np.tile(Bs, (maxpts, 1)).T
    if np.all(nms_c == 0):
        nms_c = 1
        if p[0]:
            Bs[:, 1] = Bs[:, 1] * 0
        elif p[0] == 2:
            Bs = np.zeros(np.shape(Bs))

    stn = st.norm(loc=ms, scale=np.sqrt(np.ravel(vs)))
    ns = stn.pdf(xs.T)

    p = np.tile(p, (maxpts, 1)).T
    dxs = np.abs(xs[:, 1] - xs[:, 0])
    I = np.zeros([2, maxpts])
    for i in range(len(zp)):
        I += (
            (np.dot(As.T, xs) + Bs) ** p
            * np.tile(ns[:, i], (2, 1))
            * np.tile(nms_c, (2, 1))
            * dxs[i]
        )
    results = np.sum(I, axis=1)

    return results


def smoothing(dfh, dfs, dfk, nmax, dmax, path):

    if os.path.isfile(path + "/Smooth_h.npy"):
        zh = np.load(path + "/Smooth_h.npy")
        zs = np.load(path + "/Smooth_s.npy")
        zk = np.load(path + "/Smooth_k.npy")
    else:
        zh = smooth(
            dfh.loc[:, ["x", "y", "t"]],
            dfh.loc[:, ["x", "y", "t"]],
            dfh.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_h.npy", zh)
        zs = smooth(
            dfs.loc[:, ["x", "y", "t"]],
            dfs.loc[:, ["x", "y", "t"]],
            dfs.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_s.npy", zs)

        dfhs = pd.concat([dfh, dfs], sort=False)
        dfhs.reset_index(drop=True, inplace=True)

        zk = smooth(
            dfk.loc[:, ["x", "y", "t"]],
            dfhs.loc[:, ["x", "y", "t"]],
            dfhs.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_k.npy", zk)

    dfh.h = (dfh.h - zh) / (dfh.h - zh).std()
    dfs.h = (dfs.h - zs) / (dfs.h - zs).std()
    dfhs = pd.concat([dfh, dfs], sort=False)
    dfhs.reset_index(drop=True, inplace=True)
    # figures.norm(dfhs)

    return zh, zs, zk, dfh, dfs


import itertools as it
import os


def stmatrix(dfh, dfs):
    """
    Create a dataframe of values where columns have the length of space different points and index has the time
    length.

    Args:
        - dfh: dataframe with hard data (x, y, t, value)
        - dfs: dataframe with soft data (x, y, t, mean, var)

    Returns:
        - dfz: the dataframe with the values
        - coups: vector with unique space points
        - coupt: vector with unique time points
    """

    cPt = dfh.append(dfs)
    cPt.sort_index(sort=False, inplace=True)
    coupt = cPt.t.unique()
    coups = np.unique((cPt.x, cPt.y), axis=1).T
    dfz = pd.DataFrame(np.nan, index=coupt, columns=(np.arange(len(coups))))

    for i, j in enumerate(coups[:, 0]):
        for t in coupt:
            dfz.loc[t, i] = cPt[
                ((cPt.t == t) & (cPt.x == j) & (cPt.y == coups[i, 1]))
            ].h.values[0]

    return dfz, coups, coupt


def coord2dist(pi, *args):
    """
    Compute the distance between all the elements in pi in space (two columns) or time (one column), or the distance
    between every element in pi and args (just one position).

    Args:
        - pi: vector or matrix of coordinates
        - *args (optional): one coordinate (two dimensional in space or one dimensional in time)

    Returns:
        - dist: dataframe with distances between elements
    """
    if args:
        ncoor = len(np.shape(pi))
        pi0 = args[0]
        nl = np.shape(pi0)[0]
        if nl == 1:
            if ncoor == 1:
                dist = np.abs(pi - pi0)
            else:
                dist = np.sqrt(
                    (pi[:, 0] - pi0[0, 0]) ** 2 + (pi[:, 1] - pi0[0, 1]) ** 2
                )
        else:
            dist = []
            if ncoor == 1:
                for m in range(nl):
                    dist.append(np.abs(pi - pi0[m]))
            else:
                for m in range(nl):
                    dist.append(
                        np.sqrt(
                            (pi[:, 0] - pi0[m, 0]) ** 2 + (pi[:, 1] - pi0[m, 1]) ** 2
                        )
                    )
    else:
        ncoor = len(np.shape(pi))
        df = [[] for ii in range(ncoor)]
        if ncoor == 1:
            pi = pi[:, np.newaxis]

        for m in range(ncoor):
            df[m] = pd.DataFrame(
                0, index=np.arange(len(pi)), columns=np.arange(len(pi))
            )
            dx = []
            for a, b in it.combinations_with_replacement(pi[:, m], 2):
                dx.append(a - b)
            dx = np.array(dx)

            k, l = len(pi), 0
            for i, j in enumerate(df[m].index):
                df[m].loc[j, i:] = dx[l : l + k]
                df[m].loc[i:, j] = dx[l : l + k]
                l += k
                k -= 1

        if ncoor == 1:
            dist = pd.DataFrame(np.abs(df[0]))
        elif ncoor == 2:
            dist = pd.DataFrame(np.sqrt(df[0] ** 2.0 + df[1] ** 2.0))
        else:
            dist = 0

    return dist


def coord2distang(pi):
    """
    Compute the distance and angle between all the elements in pi in space (two columns).

    Args:
        - pi: vector or matrix of coordinates

    Returns:
        - dist: dataframe with distances between elements
        - ang: dataframe with angles betweeen elements
    """
    df = [[] for ii in range(2)]

    for m in range(2):
        df[m] = pd.DataFrame(0, index=np.arange(len(pi)), columns=np.arange(len(pi)))
        dx = []
        for a, b in it.combinations_with_replacement(pi[:, m], 2):
            dx.append(a - b)
        dx = np.array(dx)

        k, l = len(pi), 0
        for i, j in enumerate(df[m].index):
            df[m].loc[j, i:] = dx[l : l + k]
            df[m].loc[i:, j] = dx[l : l + k]
            l += k
            k -= 1

    dist = pd.DataFrame(np.sqrt(df[0] ** 2.0 + df[1] ** 2.0))
    ang = pd.DataFrame(np.angle(df[0] + 1j * df[1], deg=True))
    ang[ang < 0] += 180
    return dist, ang


def pairsindex(pi, plag, plagtol, *args):
    """
    Find pairs of points separated by a given distance interval

    Args:
        - pi: coordinates (space, 2D) or temporal (1D) matrix (or vector)
        - plag: array with spatial or temporal distances
        - plagtol: array with spatial or temporal lag distances
        - args(optional): a list with: dlag, array with mean angles; and dlagtol, angle lags.

    Returns:
        - index: a list where every element is a mask where spatial or temporal distances are fulfilled
    """
    if not args:
        nr = len(plag)
        dist = coord2dist(pi)

        idxpairs = [[] for i in range(0, nr)]
        for ir in range(0, nr):
            idxpairs[ir] = np.where(
                (plag[ir] - plagtol[ir] <= dist) & (dist <= plag[ir] + plagtol[ir])
            )
    else:
        dlag, dlagtol = args[0][0], args[0][1]
        nr, nd = len(plag), len(dlag)

        dist, ang = coord2distang(pi)
        idxpairs = [[[] for j in range(nd)] for i in range(nr)]
        db = 2 * plag[1] * np.sin(dlagtol * np.pi / 360)
        daI2 = np.arctan(db / (2 * dist)) * 180 / np.pi
        for i in range(nd):
            idxpairs[0][i] = np.where(
                (plag[0] - plagtol[0] <= dist) & (dist <= plag[0] + plagtol[0])
            )
            idxpairs[1][i] = np.where(
                (plag[1] - plagtol[1] <= dist)
                & (dist <= plag[1] + plagtol[1])
                & (dlag[1] - dlagtol / 2 <= ang)
                & (ang <= dlag[1] + dlagtol / 2)
            )

        for id in range(nd):
            for ir in range(2, nr):
                idxpairs[ir][id] = np.where(
                    (plag[ir] - plagtol[ir] <= dist)
                    & (dist <= plag[ir] + plagtol[ir])
                    & (dlag[id] - daI2 <= ang)
                    & (ang <= dlag[id] + daI2)
                )

    return idxpairs


def neighbours(c0, c, z, nmax, dmax):
    """
    Compute the neighbours to c0.

    Args:
        - c0: estimation point ST coordinates
        - c: ST coordinates of data
        - z: values of data
        - nmax: max no of data
        - dmax: max distance

    Returns:
        - csub: coordinates of the subset
        - zsub: values of the subset
        - dsub: distances of the subset
        - nsub: lenght of subset vector
        - index: order of the subset matrix
    """
    # computing distances
    # nd = np.shape(c)[1]
    ds = coord2dist(c.loc[:, ["x", "y"]].values, c0.loc[:, ["x", "y"]].values)
    dt = coord2dist(c.loc[:, "t"].values, c0.loc[:, "t"].values)
    index = np.where((ds <= dmax[0]) & (dt <= dmax[1]))[0]

    # check the number of data that fulfill the conditions: if it is more, take which ones are nearest to estimation
    # points
    nsub = len(index)
    dp = ds + dmax[2] * dt
    if nsub > nmax:
        di = dp[index]
        dis, indexi = np.sort(di), np.argsort(di)
        indexi = indexi[:nmax]
        nsub = nmax
        index = index[indexi]

    dsub = [ds[index], dt[index], dp[index]]

    if np.shape(z)[1] == 1:
        zsub = z.iloc[index]
    else:
        zsub = z.iloc[index, :]

    csub = c.iloc[index, :]

    return csub, pd.DataFrame(zsub), dsub, nsub, index


def bmeregression(c, z, order, k):
    """
    Compute the parameters estimation in a linear regression

    Args:
        - c: matrix of spatiotemporal coordinates of all data
        - z: vector of values at c
        - order: order of the polinomial in space and time
        - k: covariance matrix of data

    Returns:
        - best: vector of parameter estimates
        - vbest: covariance for the best parameter estimates
        - zest: vector of estimated regression values at c
    """
    c = np.vstack((c.x.values, c.y.values, c.t.values)).T
    x = designmatrix(c, order)

    xtinvk = np.dot(x.T, np.linalg.inv(k))
    vbest = np.linalg.inv(np.dot(xtinvk, x))
    best = np.dot(np.dot(vbest, xtinvk), z)
    zest = np.dot(x, best)

    return best, vbest, zest


def designmatrix(c, order):
    """
    Designing matrix in a linear regression model

    Args:
        - c: matrix of coordinates for locations
        - order: list of ST order

    Returns:
        - x: matrix where each column corresponds to one of the polynomial term sorted by spatial order and then
             temporal order
    """
    n, nd = np.shape(c)
    x = np.ones([n, 1 + order[0] * 2 + order[1]])

    for i in range(order[0]):
        x[:, 1 + i * 2 : i * 2 + 3] = c[:, 0:2] ** (i + 1)

    for i in range(order[1]):
        x[:, 1 + order[0] * 2 + i] = c[:, 2] ** (i + 1)

    return x


def smooth(ck, chs, zhs, v, nmax, dmax):
    """
    Smoothing data
    """

    zk = np.zeros(len(ck))
    for i in range(len(ck)):
        print(
            "Point no. "
            + str(i + 1).zfill(5)
            + " of "
            + str(len(ck))
            + ". A "
            + str(np.round((i + 1) / float(len(ck)) * 100, decimals=2)).zfill(5)
            + "% completed."
        )
        ck0 = ck[ck.index == i]
        chl, zhl, dhl, sumnh, _ = neighbours(ck0, chs, zhs, nmax, dmax)
        if sumnh > 0:
            lam = np.exp(-dhl[2] / v)
            lam = lam / np.sum(lam)
            zk[i] = np.dot(lam.T, zhl)
    return zk


def cross_validation(
    dfh, dfs, zh, covmodel, covparam, nmax, dmax, order, option, path, name, k
):
    if os.path.isfile(os.path.join(path, "e_mda.npy")):
        e_mda = np.load(os.path.join(path, "e_mda.npy"))
        e_mse = np.load(os.path.join(path, "e_mse.npy"))
    else:
        lh = len(dfh)
        npoints = int(lh / k)
        e_mda, e_mse = [], []
        for i in range(k):
            indx = np.arange(lh)
            ind = np.random.choice(lh, npoints, replace=False)
            dfk = dfh.loc[ind, ["x", "y", "t"]]
            indx = [ii for ii in indx if ii not in ind]
            dfhm = dfh.loc[indx]
            namei = name + str(k) + "_" + str(i)
            moments = mom(
                dfk,
                dfhm,
                dfs,
                covmodel,
                covparam,
                nmax,
                dmax,
                order,
                option,
                path,
                namei,
            )
            moments[:, 1] = moments[:, 1] * moments[:, 2] + zh[ind]
            e_mda.append(np.sum(np.abs(zh[ind] - moments[:, 1])) / npoints)
            e_mse.append(np.sqrt(np.sum((zh[ind] - moments[:, 1]) ** 2) / npoints))
        np.save(os.path.join(path, "e_mda.npy"), e_mda)
        np.save(os.path.join(path, "e_mse.npy"), e_mse)
    return e_mda, e_mse

