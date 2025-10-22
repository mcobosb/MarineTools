import os

import numpy as np
import pandas as pd
import utils as ut
from marinetools.graphics import plots_spatiotemporal as figures
from marinetools.spatiotemporal import covST
from scipy.optimize import minimize

tstr = ["20_Jan_2017", "26_Jan_2017"]
# tstr = ['19_Feb_2017', '24_Feb_2017']

if tstr[0] == "20_Jan_2017":
    smax, ns, tmax, nt, nd = 9, 5, 72, 8, 6
else:
    smax, ns, tmax, nt, nd = 4, 5, 72, 8, 4

path = os.path.join("..", "ProcessedData", tstr[0] + "-" + tstr[1])

if not os.path.isfile(os.path.join(path, "empcovang.npy")):
    dfh = pd.read_csv(
        os.path.join(path, "Hard_data.txt"), sep=" ", names=["x", "y", "t", "h"]
    )
    dfs = pd.read_csv(
        os.path.join(path, "Soft_data.txt"), sep=" ", names=["x", "y", "t", "h", "s"]
    )

    slag = smax * (1 - np.log(ns + 1 - (np.arange(ns) + 1)) / np.log(ns))
    tlag = tmax * (1 - np.log(nt + 1 - (np.arange(nt) + 1)) / np.log(nt))
    dlag, dlagtol = np.arange(nd) * 180.0 / nd, 90.0 / nd

    empcovang, pairsnostd, covdist, covdistd, covdistt = covST.covang(
        dfh, dfs, slag, tlag, [dlag, dlagtol]
    )
    ut.save(
        [empcovang, pairsnostd, covdist, covdistd, covdistt],
        ["empcovang", "covangpairs", "covdist", "covdistd", "covdistt"],
        path,
    )

else:
    empcovang, pairsnostd, covdist, covdistd, covdistt = ut.load(
        ["empcovang", "covangpairs", "covdist", "covdistd", "covdistt"], path
    )


# D = [covdistx, covdisty]
# par0 = [np.amax(empcovst), 50, 50, -0.2]
# family = 'exponentialST'
# res = minimize(covST.fit, par0, method='SLSQP', args=(expcovst, D, family), options = {'disp': True})
#
# fig.covExpTh(covdists, covdistt, expcovst, tlag, res, family)
# ut.save([family, res], ['family', 'param'], path)
slag = smax * (1 - np.log(ns + 1 - (np.arange(ns) + 1)) / np.log(ns))
figures.covSTang(covdist, covdistd, covdistt, empcovang, slag, ["polar", "covariance"])
