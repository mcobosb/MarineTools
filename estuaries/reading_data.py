# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:03:30 2016

@author: m_cob
"""

from scipy.io import loadmat as ldm
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd
import numpy as np


def ctds():
    data = ldm('../Data/measured/ctds_hasta_nov2010.mat')
    var_name = {1: 'Fecha', 2: 'Temperatura_C', 3: 'Conductividad_S_m',  4: 'Oxigeno_V',  5: 'Turbidez_V',
                6: 'Fluorescencia_V', 7: 'Salinidad', 8: 'Oxigeno_Disuelto_mg_l', 9: 'Saturacion_oxigeno_mg_l',
                10: 'Porcentage_Saturacion_oxigeno', 11: 'Fluorescencia_Normalizada_V',
                12: 'Turbidez_Normalizada_FNU'}
    varnames_ds = {1: 'fecha', 2: 'temp', 3: 'cond',  4: 'oxig',  5: 'turb_V',
                6: 'fluo_V', 7: 'sali', 8: 'oxig_dis', 9: 'sat_oxig',
                10: 'porc_sat_oxig', 11: 'fluo_N',
                12: 'turb_N'}
    x = {0: '0 km', 1: '17.3 km', 2: '23.6 km', 3: '26.2 km', 4: '35.3 km', 5: '47.1 km', 6: '57.6 km', 7: '84.3 km'}
    d = {1: '1-cell', 2: '2-cell', 3: '3-cell', 4: '4-cell'}

    lkeys = len(data.keys())
    col = [data.keys()[i] for i in range(0, lkeys) if data.keys()[i].startswith('CTD')]

    dc = dict()
    aux = np.zeros([8, 4, 35271])
    dateini, datefin, ndate = list(), list(), list()
    for n in range(len(var_name)-1):
        for j in col:
            if data[j].any():
                dateini.append(data[j][0, 0])
                datefin.append(data[j][-1, 0])
                ndate.append(len(data[j][:, 0]))
    mindate, maxdate, maxnodate = np.min(dateini), np.max(dateini), np.max(ndate)
    dist = np.arange(0, 8)
    prof = np.arange(1, 5)


    for n in range(len(var_name)-1):
        cont = 0
        colname = list()
        for j in col:
            if data[j].any():
                date = [dates.num2date(i-366) for i in data[j][:, 0]]
                if cont == 0:
                    df = pd.DataFrame({j: data[j][:, n+1]}, index=date)
                    cont += 1
                else:
                    df2 = pd.DataFrame({j: data[j][:, n+1]}, index=date)
                    df = pd.concat([df, df2], axis=1)
                colname.append((x[int(j[3])], d[int(j[-1])]))

        df.columns = colname
        df.to_csv('../Data/processed/'+var_name[n+2]+'.zip', compression='.zip')
        # df.index = df.index.strftime('%m/%d/%Y %H:%M')
        # writer = pd.ExcelWriter('../Data/processed_data/ctd/excel/'+var_name[n+2]+'.xlsx')
        # df.to_excel(writer)
        # writer.save()



def sea_level_Bon():
    data = ldm('../Data/measured/Bonanza_nuevo.mat')
    # print data['info']

    colnames = ['Meteo-tide', 'Astro-tide', 'Residual-tide']
    date = [dates.num2date(i-366) for i in data['Bon'][:, 0]]

    r = 0
    for j in colnames:
        if r == 0:
            rng_1h = pd.date_range(start=np.min(date), end=np.max(date), freq='1H')
            df1 = pd.DataFrame({j: data['Bon'][:, r+1]}, index=rng_1h)
            rng = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='10Min')
            df = df1.reindex(index=rng, copy='True').interpolate()
        else:
            rng_1h = pd.date_range(start=np.min(date), end=np.max(date), freq='1H')
            df1 = pd.DataFrame({j: data['Bon'][:, r+1]}, index=rng_1h)
            rng = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='10Min')
            df2 = df1.reindex(index=rng, copy='True').interpolate()
            df = pd.concat([df, df2], axis=1)
        r += 1

    df.columns = colnames
    df.rename(columns={'Astro-tide': '5.3 km'}, inplace=True)
    df['5.3 km'].to_csv('../Data/processed/sea_level_Bonanza.csv', compression='zip')
    # df.index = df.index.strftime('%m/%d/%Y %H:%M')
    # writer = pd.ExcelWriter('../Data/processed_data/sea_level/excel/sea_level_Bonanza.xlsx')
    # df.to_excel(writer)
    # writer.save()


def sea_level_Icm():
    data = ldm('../Data/measured/mareografosICMAN_03_03_10.mat')
    x = {1: '21.55 km', 2: '26.8 km', 3: '36.45 km', 4: '51.8 km', 5: '62.55 km', 6: '76.0 km', 7: '93.73 km'}

    col = [i for i in data.keys() if i.startswith('TG')]

    cont = 0
    colname = list()
    for j in col:
        if data[j].any():
            date = [dates.num2date(i-366) for i in data[j][:, 0]]
            if cont == 0:
                df1 = pd.DataFrame({j: data[j][:, 2]}, index=date)
                rng = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='10Min')
                df = df1.reindex(index=rng, copy='True', method='nearest')
                cont += 1
            else:
                df1 = pd.DataFrame({j: data[j][:, 2]}, index=date)
                rng = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='10Min')
                df1 = df1.reindex(index=rng, copy='True', method='nearest')
                df = pd.concat([df, df1], axis=1)
            colname.append((x[int(j[2])]))

    df.columns = colname
    df.to_csv('../Data/processed/sea_level_Icman.csv', compression='zip')
    # df.index = df.index.strftime('%m/%d/%Y %H:%M')
    # writer = pd.ExcelWriter('../Data/processed_data/sea_level/excel/sea_level_Icman.xlsx')
    # df.to_excel(writer)
    # writer.save()


def discharge():
    data = ldm('../Data/measured/QAlcala_27abr2016.mat')
    date = [dates.num2date(i-366) for i in data['x'][:, 0]]

    df = pd.DataFrame({'flow (m3/s)': data['x'][:, 1]}, index=date)

    df.to_csv('../Data/processed/discharge_Alcala.csv', compression='zip')
    # df.index = df.index.strftime('%m/%d/%Y %H:%M')
    # writer = pd.ExcelWriter('../Data/processed_data/discharge/excel/discharge_Alcala.xlsx')
    # df.to_excel(writer)
    # writer.save()


def quality():
    data = ldm('../Data/measured/Salmedina_hasta2010.mat')
    # print data['info_Salmedina']

    col = ['Radiacion', 'Humedad', 'Direccion', 'Viento', 'Presion', 'Temperatura']
    var_ = [{1: 'R_mean (W/m2)', 2: 'R_max (W/m2)'}, {1: 'Rel_humidity (%)'}, {1: 'D_mean (proc, grados)',
            2: 'D_max (proc, grados)', 3: 'D_sig (proc, grados)'}, {1: 'V_mean (m/s)', 2: 'V_max (m/s)',
            3: 'V_sig (m/s)'}, {1: 'P_atm_mean (mbar)'}, {1: 'T_mean (C)', 2: 'T_max (C)', 3: 'T_sig (C)'}]

    cont = 0
    colname = list()
    for k, j in enumerate(col):
        for n in range(int(np.shape(data[j])[1]-1)):
            date = [dates.num2date(i-366) for i in data[j][:, 0]]
            if cont == 0:
                df = pd.DataFrame({j: data[j][:, n+1]}, index=date)
                cont += 1
            else:
                df2 = pd.DataFrame({j: data[j][:, n+1]}, index=date)
                df = pd.concat([df, df2], axis=1)
            colname.append(var_[k][n+1])

    df.columns = colname
    df.to_pickle('../Data/processed/qual_Salmedina.zip', compression='.zip')
    # df.index = df.index.strftime('%m/%d/%Y %H:%M')
    # writer = pd.ExcelWriter('../Data/processed_data/quality/excel/qual_Salmedina.xlsx')
    # df.to_excel(writer)
    # writer.save()


def flow_velocities():
    data = ldm('../Data/measured/velocs_al_9marzo2010.mat')
    x = {0: '14.3 km', 1: '20.8 km', 2: '31.8 km', 3: '39.8 km', 4: '49.3 km', 5: '63.8 km'}
    d = {0: '0 m', 1: '-1 m', 2: '-2 m', 3: '-3 m', 4: '-4 m', 5: '-5 m', 6: '-6 m'}
    u = {0: 'e', 1: 'n', 2: 'u'}

    col = ['eastADCP', 'northADCP', 'upADCP']

    cont = 0
    lkeys = len(x.keys())
    for j in range(lkeys):
        for n in range(int(np.shape(data[col[0]+str(j+1)])[1])):
            date = [dates.num2date(i-366) for i in data['tADCP'+str(j+1)][:, 0]]
            for k in range(len(col)):
                if cont == 0:
                    df = pd.DataFrame({(x[j], d[n], u[k]): data[col[k]+str(j+1)][:, n]}, index=date)
                    df = df.groupby(df.index).first()
                    cont += 1
                else:
                    df2 = pd.DataFrame({(x[j], d[n], u[k]): data[col[k]+str(j+1)][:, n]}, index=date)
                    df2 = df2.groupby(df2.index).first()
                    df = pd.concat([df, df2], axis=1)

    df.to_pickle('../Data/processed/velocities.zip', compression='.zip')
    # df.index = df.index.strftime('%m/%d/%Y %H:%M')
    # writer = pd.ExcelWriter('../Data/processed_data/water_velocity/excel/velocities.xlsx')
    # df.to_excel(writer)
    # writer.save()
