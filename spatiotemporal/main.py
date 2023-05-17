import os

import numpy as np
import pandas as pd
from marinetools.graphics import plots_spatiotemporal as figures
from marinetools.spatiotemporal import covST
from scipy.optimize import minimize

tstr = ["20_Jan_2017", "26_Jan_2017"]
# tstr = ['19_Feb_2017', '24_Feb_2017']

if tstr[0] == "20_Jan_2017":
    smax, ns, tmax, nt = 9, 6, 72, 8
else:
    smax, ns, tmax, nt = 5, 6, 72, 8

path = "../ProcessedData/" + tstr[0] + "-" + tstr[1] + "/"
var = "s"
slag = smax * (1 - np.log(ns + 1 - (np.arange(ns) + 1)) / np.log(ns))
tlag = tmax * (1 - np.log(nt + 1 - (np.arange(nt) + 1)) / np.log(nt))

if not os.path.isfile(os.path.join(path, "empcov" + var + ".npy")):
    dfh = pd.read_csv(path + "Hard_data.txt", sep=" ", names=["x", "y", "t", "h"])
    dfs = pd.read_csv(path + "Soft_data.txt", sep=" ", names=["x", "y", "t", "h", "s"])

    if var == "h":
        empcovst, covpairs, covdists, covdistt = covST.cov(dfh, dfh, slag, tlag)
    else:
        empcovst, covpairs, covdists, covdistt = covST.cov(dfs, dfs, slag, tlag)
    ut.save(
        [empcovst, covdists, covdistt, covpairs],
        ["empcov" + var, "covdists" + var, "covdistt" + var, "covpairs" + var],
        path,
    )
else:
    empcovst, covdists, covdistt, covpairs = ut.load(
        ["empcov" + var, "covdists" + var, "covdistt" + var, "covpairs" + var], path
    )


D = [covdists, covdistt]
# par0 = [np.amax(empcovst), 0.1, 40, -0.2] # hard-jan
par0 = [np.amax(empcovst), 15.0, 20, -0.2]  # soft-jan
family = "exponentialST"
res = minimize(
    covST.fit, par0, method="SLSQP", args=(empcovst, D, family), options={"disp": True}
)
print(res.x)

fig.covExpTh(covdists, covdistt, empcovst, tlag, res, family, "3d")
ut.save([family, res.x], ["family_" + var, "param_" + var], path)
