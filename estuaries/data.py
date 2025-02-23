import os
import pickle
from datetime import timedelta

import marinetools.estuaries.utils as ut
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marinetools.spectral import analysis

# from modules.ttide.t_tide import t_predic, t_tide


def run_average(var, num):
    """Computed the moving average of the data

    Args:
        x: data
        num: width of the moving window

    Returns:
        The averaged signal
    """
    cont = 0
    average = var.copy()
    for i in range(0, int(num / 2)):
        average.iloc[i] = var.iloc[cont : i + int(num / 2)].mean()
    for i in range(int(num / 2), len(var) - int(num / 2)):
        average.iloc[i] = var.iloc[i - int(num / 2) : i + int(num / 2)].mean()
    for i in range(len(var) - int(num / 2), len(var)):
        average.iloc[i] = var.iloc[i - int(num / 2) :].mean()
    return average


def calc_rhoe(x, xboxes, B, h, rho, rhos, dt):
    """Compute the density in every box

    Args:
        x: location of the box (km)
        B: width at different locations
        h: mean depth of every box (m)
        rho: densities

    Returns:
        The density gradient (drhoe), the volumen of the boxes (Vol) and the mean width (Am)
    """

    # rho.index = rho.index.to_datetime(rho.index)
    colnames = [str(i) + " km" for i in xboxes]
    rhoe = pd.DataFrame(-1, index=rho.index, columns=colnames)
    drhoe = pd.DataFrame(0, index=rho.index, columns=colnames)
    rhos = rhos.tz_localize(None)
    if any("0.0 km" in s for s in rho.columns):
        rhoe["sea"] = rhos.loc[:, "sea"] * 0.25 + rho.loc[:, "0.0 km"] * 0.75
    else:
        rhoe["sea"] = rhos.loc[:, "sea"] * 0.25 + rho.loc[:, 0] * 0.75
    drhoe["sea"] = np.append(0.0, np.diff(rhoe.loc[:, "sea"]) / dt)
    Vol = pd.DataFrame(0, index=["hm3"], columns=colnames)
    for i, j in enumerate(colnames):
        print(i, j)
        rho_c, Bc, xc, hc = (
            rho.iloc[:, i + 1 : i + 3],
            B[i : i + 2],
            x[i : i + 2],
            h.iloc[0, i : i + 2].values,
        )
        f1 = Bc * hc * rho_c
        f2 = Bc * hc
        rhoe.loc[:, j] = np.trapz(f1, xc, 2) / np.trapz(f2, xc)
        drhoe.loc[1:, j] = np.diff(rhoe.loc[:, j]) / dt
        drhoe.loc[drhoe.index[0], j] = drhoe.loc[drhoe.index[1], j]
        Vol[j] = np.trapz(f2, xc[::-1] * 1000)

    return rhoe, drhoe, Vol


def calc_phi(rhoe, h, box):
    """Compute the potential energy anomaly

    Args:
        rhoe: estimated density of every box
        h: depth of boxes
        rho_s: density of the salted water

    Returns:
        Averaged box potential energy anomaly
    """

    g = 9.806
    # rhoe.insert(0, 'river', rhoe.iloc[:, 0]*0 + rhor)
    if box:
        Phi_m = pd.DataFrame(-1, index=rhoe.index, columns=rhoe.columns[:-1])
        for i, j in enumerate(Phi_m.columns):
            Phi_m.loc[:, j] = -g * h[i] * (rhoe.iloc[:, i] - rhoe.iloc[:, i + 1]) / 2
    else:
        Phi_m = pd.DataFrame(-1, index=rhoe.index, columns=rhoe.columns[1:-1])
        for i, j in enumerate(Phi_m.columns):
            Phi_m.loc[:, j] = (
                -g * h.iloc[0, i] * (rhoe.iloc[:, i + 1] - rhoe.iloc[:, i + 2]) / 2
            )

    return Phi_m


def spatial_interpolation(xedges, M, V):
    """Computed the spatial interpolation of data to the mean and edge locations of every box

    Args:
        xboxes: new location of data query
        M: tides
        V: velocities at surface in the river axis
        Rho: densities

    Returns:
        Three dataframes with tides, velocities at the surface and densities in the new points
    """

    nam_sl = ["Bonanza", "5.3 km", "26.8 km", "36.45 km", "51.8 km", "62.55 km"]
    xsl = np.array([0, 5.3, 26.8, 36.45, 51.8, 62.55])
    # nam_ctds = ['0 km', '17.3 km', '26.2 km', '35.3 km', '47.1 km', '57.6 km', '84.3 km']
    # xctds = np.array([0, 17.3, 26.2, 35.3, 47.1, 57.6, 84.3])
    nam_us = ["14.3 km", "20.8 km", "31.8 km", "39.8 km", "49.3 km", "63.8 km"]
    xus = np.array([14.3, 20.8, 31.8, 39.8, 49.3, 63.8])
    M, V = M.reindex(columns=nam_sl), V.reindex(columns=nam_us)
    # Rho = Rho.reindex(columns=nam_ctds)

    # colnames = [str(i)+' km' for i in xboxes]
    colnames = [str(i) + " km" for i in xedges]
    Mx, Vx = pd.DataFrame(-1, index=M.index, columns=colnames), pd.DataFrame(
        -1, index=M.index, columns=colnames
    )
    for j in Vx.index:
        Mx.loc[j, colnames] = np.interp(xedges, xsl, M.loc[j, nam_sl])
        Vx.loc[j, colnames] = np.interp(xedges, xus, V.loc[j, nam_us])
        # Rhox.iloc[j, :] = np.interp(xedges, xctds, Rho.iloc[j])

    return Mx, Vx


def tidal_average(signal, tm, ind_):
    """Computed the tidal average of the signal

    Args:
        ref: tidal signal of reference
        signal: series to interporlate

    Returns:
        The averaged signal
        :rtype: object
    """
    if tm is None:
        signalm = signal - np.mean(signal)
        sg = np.sign(signalm.values)
        ps = sg[0:-1] * sg[1:]

        # Sign change
        ind_ = np.where(ps < 0)[0]
        sc = sg[ind_]

        if sc[0] > 0:
            ind_ = ind_[1:]

        if np.remainder(len(ind_), 2) == 0:
            ind_ = ind_[0:-1]

        nid = len(ind_)
        val_ = np.zeros(int((nid - 1) / 2.0))
        tm = []
        for j in range(0, nid - 1, 2):
            val_[int(j / 2)] = np.max(
                signal.iloc[ind_[j] : ind_[j + 2]], axis=0
            ) - np.min(signal.iloc[ind_[j] : ind_[j + 2]], axis=0)
            dt = timedelta(
                seconds=(signal.index[ind_[j + 2]] - signal.index[ind_[j]]).seconds
                / 2.0
            )
            tm.append(signal.index[ind_[j]] + dt)
    else:
        nid = len(ind_)
        val_ = np.zeros([int((nid - 1) / 2.0), len(np.atleast_1d(signal.iloc[0]))])
        for j in range(0, nid - 1, 2):
            val_[int(j / 2), :] = np.mean(signal.iloc[ind_[j] : ind_[j + 2]])

    average = pd.DataFrame(val_, index=tm, columns=signal.columns)
    return average, ind_


def setting_up_boxes(M_, V_, Rho_, Q_, Qs_, W_, D_, C_, loc, aux_info):
    """Compute the main characteristics of the boxes. These characteristics are tidal-averaged and given in xboxes
    location.

    Args:
        rho: density measured at each sensor
        xboxes: location of the boxes (km)

    """

    B_box = 795.15 * np.exp(-loc["xbox"] / 65.5)
    h_box = 5839.4 * np.exp(-loc["xbox"] / 60.26) / B_box

    B_edges = 795.15 * np.exp(-loc["xedg"] / 65.5)
    h_edges = pd.DataFrame(
        [5839.4 * np.exp(-loc["xedg"] / 60.26) / B_edges],
        index=["m"],
        columns=Rho_.columns[::-1],
    )
    h_edges.rename(columns={"0 km": "0.0 km"}, inplace=True)
    A_edges = B_edges * h_edges

    Rho_["river"] = Rho_["river"]
    if any(Rho_.columns) == "23.6 km":
        Rho_.drop("23.6 km", 1, inplace=True)

    Rho_.rename(columns={"0 km": "0.0 km"}, inplace=True)
    Rhox = Rho_[Rho_.columns[::-1]].copy()
    Mx, Vx = spatial_interpolation(loc["xedg"], M_, V_)
    # Mx.insert(np.shape(Mx)[1], 'sea', M_['Bonanza'])
    # Vx.insert(np.shape(Vx)[1], 'sea', Vx.iloc[:, -1])
    rho["sea"] = rho["sea"].tz_localize(None)
    Rhox = Rhox.tz_localize(None)
    Rhox = Rhox.assign(sea=rho["sea"].values)

    Ac, ind_ = tidal_average(M_["Bonanza"].to_frame(), None, 0)
    Q, _ = tidal_average(Q_, Ac.index, ind_)
    Q.rename(columns={"flow (m3/s)": "river"}, inplace=True)

    # Rhox, _ = tidal_average(Rhox, Ac.index, ind_)
    dt = 1.0
    dt1 = np.asarray(
        [(Rhox.index[i + 1] - Rhox.index[i]).seconds for i in range(len(Rhox) - 1)]
    )
    #    dt1 = np.asarray(dt1)
    #    dt1 = (Rhox.index[1:] - Rhox.index[0:-1]).seconds
    rhoe, drhoe, Vol = calc_rhoe(
        loc["xedg"], loc["xbox"], B_edges, h_edges, Rhox, rho["sea"], dt1
    )
    phie = calc_phi(rhoe, h_box, True)
    phi = calc_phi(Rhox, h_edges, False)

    # Se multiplica por el ancho y se incluye el bombeo mareal
    # ttes_adv = ttes_advectivos(phi, Mx, Vx, Ac, h_edges, B_edges, loc['xedg'], ind_, dt, aux_info['lat'])
    # ttes_adv['stokes'] = ttes_adv['stokes']*(1+aux_info['coefs_stokes'])*B_edges

    # ttes_dif = ttes_difusivos(Rhox, B_edges, loc['xedg'], rhoe.columns, Ac, ind_)

    ter_fuente = terminos_fuente(
        loc["xedg"], B_edges, C_, Vx, W_.to_frame(), Qs_, Ac, ind_, rhoe.columns
    )

    # promedios mareales
    Mx, _ = tidal_average(Mx, Ac.index, ind_)
    Vx, _ = tidal_average(Vx, Ac.index, ind_)
    W, _ = tidal_average(W_.to_frame(), Ac.index, ind_)
    D, _ = tidal_average(D_.to_frame(), Ac.index, ind_)
    # Vl, _ = tidal_average(Vl, Ac.index, ind_)
    Qs, _ = tidal_average(Qs_, Ac.index, ind_)
    phi, _ = tidal_average(phie, Ac.index, ind_)
    rhoe, _ = tidal_average(rhoe, Ac.index, ind_)
    drhoe, _ = tidal_average(drhoe, Ac.index, ind_)
    Rhox, _ = tidal_average(Rhox, Ac.index, ind_)
    h_box = pd.DataFrame([h_box], index=["m"], columns=rhoe.columns[:-1])

    var_names = [
        "rhoe",
        "drhoe",
        "ttes_adv",
        "ttes_dif",
        "Phi",
        "Q",
        "Qs",
        "Ac",
        "Vol",
        "h_box",
        "h_edges",
        "Mx",
        "Vx",
        "W",
        "D",
        "Rhox",
        "A_edges",
    ]
    ut.saving_data(
        [
            rhoe,
            drhoe,
            ttes_adv,
            ttes_dif,
            phie,
            Q,
            Qs,
            Ac,
            Vol,
            h_box,
            h_edges,
            Mx,
            Vx,
            W,
            D,
            Rhox,
            A_edges,
        ],
        var_names,
    )
    ut.saving_data([ind_], ["ind_"], path, "npy")
    hs = {"box": h_box, "edges": h_edges}

    return (
        rhoe,
        Rhox,
        drhoe,
        ttes_adv,
        ttes_dif,
        phi,
        Q,
        Qs,
        Ac,
        Vol,
        A_edges,
        hs["edges"],
        Vx,
        W,
        D,
    )


def preprocessing_data(time_it, ang_thalw, root, path, graph="n"):
    """Computed the linear interpolation every 10 minutes of the in-dataframes and the density and parallel velocity

    Args:
        c: turbidity (FNU)
        t: temperature (C)
        s: salinity (psu)
        ue0, un0: east- and north-velocities at the surface
        m0, m1: bonanza and icman tides
        r: radiation
        w: nean wind velocity
        d: mean wind direction
        q: discharge
        time_it: list of strings with the start and end dates
        time_int: list of datetimes with the start and end dates

    Returns:
        Interpolated and new dataframes with the density
    """
    if not os.path.exists(root):
        c, t, s, ue0, un0, m0, m1, r, w, d, q, o2, fl = load_data(time_it, graph)

        loc = ["0 km", "17.3 km", "26.2 km", "35.3 km", "47.1 km", "57.6 km"]
        T, S, C, O2, Fl = {}, {}, {}, {}, {}
        for j in loc:
            T[j] = pd.DataFrame({i[1]: t[i] for i in t.columns if j in i}).mean(axis=1)
            S[j] = pd.DataFrame({i[1]: s[i] for i in s.columns if j in i}).mean(axis=1)
            C[j] = pd.DataFrame({i[1]: c[i] for i in c.columns if j in i}).mean(axis=1)
            O2[j] = pd.DataFrame({i[1]: o2[i] for i in o2.columns if j in i}).mean(
                axis=1
            )
            Fl[j] = pd.DataFrame({i[1]: fl[i] for i in fl.columns if j in i}).mean(
                axis=1
            )

        C, T, S, O2, Fl = (
            pd.DataFrame(C),
            pd.DataFrame(T),
            pd.DataFrame(S),
            pd.DataFrame(O2),
            pd.DataFrame(Fl),
        )
        C, T, S, Ve, Vn, M0, M1, Q, w, d, O2, Fl = (
            C.resample("H").interpolate(),
            T.resample("H").interpolate(),
            S.resample("H").interpolate(),
            ue0.resample("H").interpolate(limits=36),
            un0.resample("H").interpolate(limits=36),
            m0.resample("H").interpolate(limits=36),
            m1.resample("H").interpolate(limits=36),
            q.resample("H").interpolate(limits=36),
            w.resample("H").interpolate(limits=36),
            d.resample("H").interpolate(limits=36),
            O2.resample("H").interpolate(limits=36),
            Fl.resample("H").interpolate(limits=36),
        )
        # r.resample('H').interpolate(limits=36).to_frame(), w.resample('H').interpolate(limits=36).to_frame(), d.resample('H').interpolate(limits=36).to_frame()
        Qs = r.resample("H").interpolate(limits=36).to_frame()  # radiacion
        Q = Q.ix[time_it[0] : time_it[1]]
        Q[Q == 0] = 1e-1
        Rho_hp, Rho = bulk_fluid_density(T, S, C)
        # Rho.drop('84.3 km', 1, inplace=True)

        V = Ve * np.cos(ang_thalw) + Vn * np.sin(ang_thalw)
        # V = Ve*np.cos(ang_thalw) + Vn*np.sin(ang_thalw)
        # V = np.sqrt(Ve**2 + Vn**2)

        M = pd.concat([M0, M1], axis=1)
        M.rename(columns={0: "Bonanza"}, inplace=True)
        M.index.tz_localize(None)

        os.makedirs(path)
        var_names = ["C", "T", "S", "V", "M", "Q", "Rho", "W", "D", "O2", "Fl", "Qs"]
        ut.saving_data([C, T, S, V, M, Q, Rho, w, d, O2, Fl, Qs], var_names, path)
    else:
        var_names = ["M", "Q", "V", "Rho", "Qs", "W", "D", "C"]
        data = ut.loading_data(var_names, path)
        M, Q, V, Rho, Qs, w, d, C = (
            data["M"],
            data["Q"],
            data["V"],
            data["Rho"],
            data["Qs"],
            data["W"],
            data["D"],
            data["C"],
        )

    return M, Q, V, Rho, Qs, w, d, C


def load_data(time_str, time_it, graph="n"):
    """Load data and choose the measurements in the locations and the time interval given

    Args:
        time_it: list of strings with the start and end dates (i.e. '23-Jul-2008 11:00')
        time_int: list of datetimes with the start and end dates

    Returns:
        Multiple dataframes containing: c (concentration), t (temperature), s (salinity), ue (vel-east),
        un (vel-north), m (tides in Bonanza and from Icman), r (radiation), w (vel-wind), d (dir-wind),
        q (water discharge)
    """

    c = pickle.load(
        open(
            "../Data/"
            + time_str[0]
            + "-"
            + time_str[1]
            + "/raw/Turbidez_Normalizada_FNU.p",
            "rb",
        )
    )
    t = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/Temperatura_C.p", "rb"
        )
    )
    s = pickle.load(
        open("../Data/" + time_str[0] + "-" + time_str[1] + "/raw/Salinidad.p", "rb")
    )
    o2 = pickle.load(
        open(
            "../Data/"
            + time_str[0]
            + "-"
            + time_str[1]
            + "/raw/Oxigeno_Disuelto_mg_l.p",
            "rb",
        )
    )
    fl = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/Fluorescencia_V.p",
            "rb",
        )
    )

    q = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/discharge_Alcala.p",
            "rb",
        )
    )
    #    q[q < 5.] = 5.

    ue = pickle.load(
        open("../Data/" + time_str[0] + "-" + time_str[1] + "/raw/eastADCP.p", "rb")
    )
    un = pickle.load(
        open("../Data/" + time_str[0] + "-" + time_str[1] + "/raw/northADCP.p", "rb")
    )
    # x = ['14.3 km', '20.8 km', '31.8 km', '39.8 km', '49.3 km', '63.8 km']
    # d = ['0 m', '-1 m', '-2 m', '-3 m', '-4 m', '-5 m', '-6 m']
    # u = ['e', 'n', 'u']
    # uel, unl = list(), list()
    # for i, k in enumerate(x):
    #     uel.append(pd.DataFrame(index=us.index, columns=d))
    #     unl.append(pd.DataFrame(index=us.index, columns=d))
    #     for j in d:
    #         uel[i].loc[:, j] =  us[(k, j, u[0])]
    #         unl[i].loc[:, j] =  us[(k, j, u[1])]

    # ue, un = pd.DataFrame(-1, index=us.index, columns=x), pd.DataFrame(-1, index=us.index, columns=x)
    # for i, j in enumerate(x):
    #     ue.loc[:, j] = uel[i].mean(axis=1)
    #     un.loc[:, j] = unl[i].mean(axis=1)

    m0 = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/sea_level_Bonanza.p",
            "rb",
        )
    )
    m1 = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/sea_level_Icman.p",
            "rb",
        )
    )

    if graph == "y":
        plt.figure()
        cont = 1
        for i in t.columns:
            if "1-cell" in i:
                if cont == 1:
                    ax1 = plt.subplot(9, 3, cont)
                else:
                    plt.subplot(9, 3, cont, sharex=ax1)
                plt.plot(t[i].ix["2008-02-01":"2010-03-01"])
                plt.xticks(visible=False)
                cont += 3
        plt.subplot(9, 3, cont, sharex=ax1)
        plt.plot(q.ix["2008-02-01":"2010-03-01"])
        plt.ylim([0, 60])
        plt.xticks(rotation=30)
        cont = 2
        for i in m1.columns:
            plt.subplot(9, 3, cont, sharex=ax1)
            plt.plot(m1[i].ix["2008-02-01":"2010-03-01"])
            plt.xticks(visible=False)
            cont += 3
        plt.subplot(9, 3, cont, sharex=ax1)
        plt.plot(m0["Astro-tide"].ix["2008-02-01":"2010-03-01"])
        plt.xticks(rotation=30)
        cont = 3
        for i in ue.columns:
            plt.subplot(9, 3, cont, sharex=ax1)
            plt.plot(ue[i].ix["2008-02-01":"2010-03-01"])
            plt.xticks(visible=False)
            cont += 3
    plt.show()

    c, t, s = (
        c.ix[time_it[0] : time_it[1]],
        t.ix[time_it[0] : time_it[1]],
        s.ix[time_it[0] : time_it[1]],
    )
    ue0, un0 = ue.ix[time_it[0] : time_it[1]], un.ix[time_it[0] : time_it[1]]
    m0 = m0["Astro-tide"].ix["2008-06-15":"2008-09-29"]
    m0 = m0.resample("10Min").interpolate(limits=36)
    n0 = dates.date2num(m0.index[0])
    # n_eta, fu_eta, tidecon_eta, out = t_tide(
    #     m0.values, stime=n0, dt=10 / 60.0, lat=36.0
    # )
    # TODO: actualizar el análisis armónico - 20240606
    res = analysis.harmonic(m0.values, stime=n0, dt=10 / 60.0, lat=36.0)
    n_ref = m0.mean()
    nd = pd.date_range(time_it[0], time_it[1], freq="10Min")
    ndt = nd.map(dates.date2num)
    ndt = np.array([np.float(i) for i in ndt])
    eta_s = analysis.reconstruction_tidal_level(df, tidalConstituents)
    # eta_s = t_predic(ndt, n_eta, fu_eta, tidecon_eta, lat=36.0)
    mB = pd.DataFrame(eta_s, index=nd) + n_ref

    m1 = m1.ix["2008-06-15":"2008-09-15"]
    m1 = m1.resample("10Min").interpolate(limits=36)
    m1.drop("76.0 km", 1, inplace=True)
    mL = pd.DataFrame(-1, index=nd, columns=m1.columns)
    for i in m1.columns:
        n1 = dates.date2num(m1[i].index[0])
        # n_eta, fu_eta, tidecon_eta, out = t_tide(
        #     m1[i].values[:], stime=n1, dt=10.0 / 60, lat=36.0
        # )
        # TODO: actualizar el análisis armónico - 20240606
        res = analysis.harmonic(m0.values, stime=n0, dt=10 / 60.0, lat=36.0)
        n_ref = m1[i].mean()
        eta_s = analysis.reconstruction_tidal_level(df, tidalConstituents)
        # eta_s = t_predic(ndt, n_eta, fu_eta, tidecon_eta, lat=36.0)
        mL[i] = eta_s + n_ref
    rwd = pickle.load(
        open(
            "../Data/" + time_str[0] + "-" + time_str[1] + "/raw/qual_Salmedina.p", "rb"
        )
    )
    r, w, d = (
        rwd.ix[time_it[0] : time_it[1], "R_mean (W/m2)"],
        rwd.ix[time_it[0] : time_it[1], "V_mean (m/s)"],
        rwd.ix[time_it[0] : time_it[1], "D_mean (proc, grados)"],
    )

    q = q.ix[time_it[0] - timedelta(days=1) : time_it[1] + timedelta(days=1)]

    return c, t, s, ue0, un0, mB, mL, r, w, d, q, o2, fl


def EOS_sea_water(T, S):
    """Compute the density of the standard mean ocean water taken as pure water reference

    Args:
        T: dataframe with the temperature (C)
        S: dataframe with the salinity (psu)

    Returns:
        A dataframe with rho
    """
    p = 0.10073  # bar at -1m
    rho_w = (
        999.842594
        + 6.793952 * 1e-2 * T
        - 9.095290 * 1e-3 * T**2
        + 1.001685 * 1e-4 * T**3
        - 1.120083 * 1e-6 * T**4
        + 6.536332 * 1e-9 * T**5
    )

    # Density of sea water at one standard atmosphere (p=0)
    rho_S_T_0 = (
        rho_w
        + (
            8.24493e-1
            - 4.0899e-3 * T
            + 7.6438e-5 * T**2
            - 8.2467e-7 * T**3
            + 5.3875e-9 * T**4
        )
        * S
        + (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2) * (S ** (3 / 2))
        + 4.8314e-4 * S**2
    )

    # Density of sea water at high pressure is: rho_S_T_p
    Kw = (
        19652.21
        + 148.4206 * T
        - 2.327105 * T**2
        + 1.360477e-2 * T**3
        - 5.155288e-5 * T**4
    )
    K_S_T_0 = (
        Kw
        + (54.6746 - 0.603459 * T + 1.09987e-2 * T**2 - 6.1670e-5 * T**3) * S
        + (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T**2) * S ** (3 / 2)
    )

    Aw = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T**2 - 5.77905e-7 * T**3
    A = (
        Aw
        + (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T**2) * S
        + 1.91075e-4 * S ** (3 / 2)
    )
    Bw = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T**2
    B = Bw + (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T**2) * S
    K_S_T_p = K_S_T_0 + A * p + B * p**2
    rho = rho_S_T_0 / (1 - p / (K_S_T_p))
    return rho


def bulk_fluid_density(T, S, C, rhos=2650):
    """Compute the density from concentration, temperature and salinity given.
    Multiply c by 1.6015e-3 to get g/l/FNU

    Args:
        T: dataframe with the temperature (C)
        S: dataframe with the salinity (psu)
        C: dataframe with the turbidity (FNU)
        rhos: sediment density (kg/m3)

    Returns:
        Two dataframes with rho and rho0_prof_1
    """
    rho = EOS_sea_water(T, S)
    rho0_prof_1 = rho + (1 - (rho / rhos)) * C * 1.6015 * 1e-3
    return rho, rho0_prof_1
