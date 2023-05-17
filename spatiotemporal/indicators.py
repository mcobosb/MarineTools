import numpy as np
from marinetools.graphics import plots_spatiotemporal as figures


def raeh(df, thresh=[]):
    """
    Fractional area exceeding the threshold
    """
    ndat = len(df)
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        res[i] = np.sum(df >= j) / ndat

    return thresh, res


def mew(df, thresh=[]):
    """
    Average exceeding the threshold over the total region
    """
    ndat = len(df)
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        res[i] = np.sum(df[df >= j]) / ndat

    return thresh, res


def medw(df, thresh=[]):
    """
    Average difference exceeding the threshold over the total region
    """
    ndat = len(df)
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        res[i] = np.sum(df[df >= j] - j) / ndat

    return thresh, res


def wmew(df, thresh=[]):
    """
    Average exceeding the threshold over the exceeding region
    """
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        res[i] = np.mean(df[df >= j])

    return thresh, res


def wmdw(df, thresh=[]):
    """
    Average difference exceeding the threshold over the exceeding region
    """
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        res[i] = np.mean(df[df >= j] - j)

    return thresh, res


def aean(df, thresh=[]):
    """
    Ratio between area exceeding the threshold and area non-exceeding
    """
    ndat = len(df)
    if np.size(thresh) == 0:
        thresh = np.linspace(0, np.max(df), 100)
    res = np.zeros(len(thresh))
    for i, j in enumerate(thresh):
        area = np.sum(df >= j) / ndat
        res[i] = area / (1.0 - area)

    return thresh, res


def one_point(moments):
    labels = ["RAEH", "MEW", "MEDW", "WMEW", "WMDW"]
    thresh = [[] for i in range(len(labels))]
    indic = [[] for i in range(len(labels))]

    thresh[0], indic[0] = raeh(moments[:, 1])
    thresh[1], indic[1] = mew(moments[:, 1])
    # thresh[2], indic[2] = indicators.aean(moments[:, 1])
    thresh[2], indic[2] = medw(moments[:, 1])
    thresh[3], indic[3] = wmew(moments[:, 1])
    thresh[4], indic[4] = wmdw(moments[:, 1])

    figures.indicators(thresh, indic, labels)
    return
