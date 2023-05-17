import marinetools.graphics.plots_spatiotemporal as figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marinetools.spatiotemporal import bme, indicators
from scipy.io import loadmat as ldm

tstr = ["20_Jan_2017", "26_Jan_2017"]
# tstr = ['19_Feb_2017', '24_Feb_2017']
path = "./ProcessedData/" + tstr[0] + "-" + tstr[1] + "/"

nmax = [40, 10]
dmax = np.array([10, 50, 0.2])
order = [1, 1]
option = np.array([100, 3, 0.9])
var = "s"

covmodel, covparam = load(["family_" + var, "param_" + var], path)
print(covparam)

dfh = pd.read_csv(path + "/Hard_data.txt", sep=" ", names=["x", "y", "t", "h"])
dfs = pd.read_csv(path + "/Soft_data.txt", sep=" ", names=["x", "y", "t", "h", "s"])
dfk = pd.read_csv(path + "/Output_mesh.txt", sep=" ", names=["x", "y", "t"])

# Smoothing signals
zh, zs, zk, dfh, dfs = bme.smoothing(dfh, dfs, dfk, nmax, dmax, path)
# ut.save([zh, zs, zk, dfh, dfs], ['zh', 'zs', 'zk', 'hardn', 'softn'], path, '.txt')

# Computing moments
name = "moments"
moments = bme.mom(
    dfk, dfh, dfs, covmodel, covparam, nmax, dmax, order, option, path, name
)
moments[:, 1] = moments[:, 1] * moments[:, 2] + zk
mx = np.ma.masked_invalid(moments)
moments[mx.mask] = 0

name = "cross_"
e_mda, e_mse = bme.cross_validation(
    dfh, dfs, zh, covmodel, covparam, nmax, dmax, order, option, path, name, 10
)
print(e_mda, e_mse)
figures.map(
    dfk,
    moments[:, 1:3],
    [2, 1, 2],
    [r"$\mu$ (m)", r"$\sigma \quad (m^2)$"],
    [-5.5, 35, 10, 44],
    path,
)
indicators.one_point(moments)
