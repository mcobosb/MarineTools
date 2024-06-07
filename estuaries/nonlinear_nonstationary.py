import os

import marinetools.estuaries.utils as ut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def exchange_volumes(Vol, Q_, Qs, Vl, rho, rhoe, A, L, h, Vx, Wx, Dx, path):
    """ Compute the exchange volumes (Q_i) and densities (rhom_i) in the estuary and the blockeage location
    """
    if not os.path.exists(path):

        # Renombramos stokes (y bombeo) y la densidad
        Vl.rename(columns={'0.0 km':0, '17.3 km':1, '26.2 km':2, '35.3 km':3, '47.1 km':4, '57.6 km':5}, inplace=True)
        Vx.rename(columns={'0.0 km':0, '17.3 km':1, '26.2 km':2, '35.3 km':3, '47.1 km':4, '57.6 km':5}, inplace=True)
        rho.rename(columns={'0.0 km':0, '17.3 km':1, '26.2 km':2, '35.3 km':3, '47.1 km':4, '57.6 km':5}, inplace=True)
        A.rename(columns={'0 km':0, '17.3 km':1, '26.2 km':2, '35.3 km':3, '47.1 km':4, '57.6 km':5}, inplace=True)
        h.rename(columns={'0 km':0, '17.3 km':1, '26.2 km':2, '35.3 km':3, '47.1 km':4, '57.6 km':5}, inplace=True)
        Vol.rename(columns={'8.65 km':1, '21.75 km':2, '30.75 km':3, '41.2 km':4, '52.35 km':5}, inplace=True)
        rhoe.rename(columns={'8.65 km':0, '21.75 km':1, '30.75 km':2, '41.2 km':3, '52.35 km':4}, inplace=True)

        # Preparamos el df de densidades de mezcla
        rhom = pd.DataFrame(-1, index=rho.index, columns=['s', 0, 1, 2, 3, 4, 5])
        rhom[5] = 1000#rho[5] #rho['river']
        # plt.figure()
        # plt.plot(rho[5])
        # plt.show()
        
        rhom['s'] = rho[0].copy()#rho[0]*0.75 + rho['sea']*0.25
        rhom.iloc[0, [1, 2, 3, 4, 5]] = np.array([1020, 1015, 1010, 1005, 1000])#rhoe.iloc[0:2, [0, 1, 2, 3, 4]].copy()
        # plt.figure(figsize=(11, 6))
        # plt.plot(rhom['s'])
        # plt.plot(rho[0])
        # plt.plot(rho['sea'])
        # plt.show()

        # Preparamos el df de caudales de intercambio
        Q = pd.DataFrame(-1, index=rho.index, columns=[0, 1, 2, 3, 4, 5])
        Q[5] = Q_['river'].copy() # 40 #48.22071358
        Q.loc[:, [0, 1, 2, 3, 4]] = Vl.loc[:, [0, 1, 2, 3, 4]].copy()
        # Vl.loc[:, 2] = Vl.loc[:, 2]*0.1
        # Vl.loc[:, 3] = Vl.loc[:, 3]*0.5

        

        # Calculamos los coeficientes
        V = Vol.values[:][0][::-1]
        Ae = 2*V/(h.values[0, 1:] + h.values[0, 0:-1])
        Ae = np.insert(Ae, -1, Ae[-1])

        inc_t = (rhom.index[1:] - rhom.index[0:-1]).seconds
        alpha = pd.DataFrame(-1, index=['ij'], columns=['01', '11', '12', '22', '23', '33', '34', '44', '45', '55'])
        for i in alpha.columns:
            # alphav.loc['ij', i] = A.loc['m', int(i[0])]/(A.loc['m', int(i[1])-1] + A.loc['m', int(i[1])])
            alpha.loc['ij', i] = Ae[int(i[0])]/(Ae[int(i[1])-1] + Ae[int(i[1])])

        print(alpha)
        beta = pd.DataFrame(0, index=rhom.index[1:], columns=[1, 2, 3, 4, 5])
        for i, j in enumerate(beta.columns):
            beta.loc[:, j] = 9.81*h.values[0, i]*V[i]/(inc_t)
        
        delta = pd.DataFrame(0, index=rhom.index[1:], columns=[1, 2, 3, 4, 5])
        for i, j in enumerate(beta.columns):
            delta.loc[:, j] = V[i]/inc_t

        coefs = pd.DataFrame(0, index=['alpha', 'gamma'], columns=[0, 1, 2, 3, 4, 5])
        alph, gamm = .5, .33
        #259 #14593 # 0.01, 0.0001
        for i, j in enumerate(coefs.columns):
            coefs.loc['alpha', j] = alph#*np.exp(np.sum(L[0:i])/(65500))
            coefs.loc['gamma', j] = gamm#*np.exp(np.sum(L[0:i])/(65500))
        
        # Q, Vl, rhom = blockage_opt(Vol, h, Q, Vl, rhoe, drhoe, rhom, coefs)
        Q, rhom, block = blockage(V, A, h, Q, Qs, Vl, rhom, rhoe, np.abs(Vx), np.abs(Wx), Dx, alpha, beta, delta, coefs)

        os.makedirs(path)
        var_names = ['Q', 'rhom', 'Vl', 'block']
        ut.saving_data([Q, rhom, Vl, block], var_names, path)
    return


def blockage(V, A, h, Q, Qs, Vl, rhom, rhoe, Vx, Wx, Dx, alpha, beta, delta, coefs):
    """ Compute the blockage between boxes in the estuary

    Args:
        Vol: volumen of the box
        h: depth of the boxes
        Q: discharge flow
        Vl: stokes flow
        rhoe: estimated density
        dhroe: diff of estimated density

    Retunrs:
        The location of the blockage and the gradient of density computed
    """
    block = pd.DataFrame(0, index=rhom.index, columns=[0, 1, 2, 3, 4])
    newton = True
    Ae = 2*V/(h.values[0, 1:] + h.values[0, 0:-1])

    if newton:
        # Propiedades para el bucle del metodo de Newton
        told, tolf, maxiter = 1e-7, 1e-5, 1000
        for i, j in enumerate(rhom.index[1:]):
            k, inc_delta, eps, F0, t_1 = 0, 1, 1, 1, rhom.index[i]

            Xn_0 = np.array([Q.loc[t_1, 0], rhom.loc[t_1, 0], Q.loc[t_1, 1], rhom.loc[t_1, 1], Q.loc[t_1, 2], rhom.loc[t_1, 2], Q.loc[t_1, 3],
                rhom.loc[t_1, 3], Q.loc[t_1, 4], rhom.loc[t_1, 4]])
            
            Xn_1 = np.array([Q.loc[t_1, 0], rhom.loc[t_1, 0], Q.loc[t_1, 1], rhom.loc[t_1, 1], Q.loc[t_1, 2], rhom.loc[t_1, 2], Q.loc[t_1, 3],
                rhom.loc[t_1, 3], Q.loc[t_1, 4], rhom.loc[t_1, 4]])
            
            (Q.loc[j, 0], rhom.loc[j, 0], Q.loc[j, 1], rhom.loc[j, 1], Q.loc[j, 2], rhom.loc[j, 2], Q.loc[j, 3],
                rhom.loc[j, 3], Q.loc[j, 4], rhom.loc[j, 4]) = (Q.loc[t_1, 0], rhom.loc[t_1, 0], Q.loc[t_1, 1], rhom.loc[t_1, 1], Q.loc[t_1, 2], rhom.loc[t_1, 2], Q.loc[t_1, 3],
                rhom.loc[t_1, 3], Q.loc[t_1, 4], rhom.loc[t_1, 4])
            # print Xn_1
            while ((inc_delta > told) | (eps > tolf)):
                
                F = func(rhom, Q, Qs, Vl, Vx, Wx, Dx, A, Ae, h, alpha, beta, delta, coefs, j, t_1)
                J = jac_func(rhom, Q, Vl, alpha, beta, delta, coefs, j)
                
                Xn = Xn_1 - np.dot(np.linalg.inv(J), F)
                
                inc_delta = np.sqrt(np.sum((Xn - Xn_1)**2))
                eps = np.sqrt(np.sum(F**2))

                Xn_1, F0 = Xn, F
                k += 1
            
                (Q.loc[j, 0], rhom.loc[j, 0], Q.loc[j, 1], rhom.loc[j, 1], Q.loc[j, 2], rhom.loc[j, 2], Q.loc[j, 3],
                rhom.loc[j, 3], Q.loc[j, 4], rhom.loc[j, 4]) = Xn

            print(str(rhom.index[i]) + '   Iteracion:  ' + str(k) + ' delta:    ' + str(inc_delta) + ' eps:    ' + str(eps))
            if k == maxiter:
                print('k maxiter - changing')
                (Q.loc[j, 0], rhom.loc[j, 0], Q.loc[j, 1], rhom.loc[j, 1], Q.loc[j, 2], rhom.loc[j, 2], Q.loc[j, 3],
                rhom.loc[j, 3], Q.loc[j, 4], rhom.loc[j, 4]) = Xn_0
            print(Xn)
            if np.mod(i+1, 150) == 0:
                plt.figure(figsize=(6, 10))
                plt.subplot(311)
                plt.plot(Q.loc['24/07/2008':j])
                plt.legend(Q.columns, loc='best')
                plt.subplot(312)
                plt.plot(rhom.loc['24/07/2008':j, [0, 1, 2, 3, 4]])
                plt.plot(rhoe.loc['24/07/2008':j, [0, 1, 2, 3, 4]], '--')
                plt.legend([0, 1, 2, 3, 4, 5], loc='best')
                plt.subplot(313)
                plt.plot(Vl.loc['24/07/2008':j])
                plt.legend(Vl.columns, loc='best')
                plt.show()
                plt.close()

    else:
        bnds = [[] for i in range(10)]
        for i in range(0, 10, 2):
            bnds[i] = None, None
        
        for i, j in enumerate(rhom.index[1:]):
            t_1 = rhom.index[i]
            # ib = 0
            for k in np.arange(1, 10, 2):
                bnds[k] = 999, rhom.loc[j, 's']
                # if ib == 0:
                #     bnds[i] = rhoe.loc[j, ib], rhom.loc[j, 's']
                # else:
                #     bnds[i] = rhoe.loc[j, ib], rhoe.loc[j, ib-1]
                # ib += 1

            Xn_1 = np.array([Q.loc[t_1, 0], rhom.loc[t_1, 0], Q.loc[t_1, 1], rhom.loc[t_1, 1], Q.loc[t_1, 2], rhom.loc[t_1, 2], Q.loc[t_1, 3],
                rhom.loc[t_1, 3], Q.loc[t_1, 4], rhom.loc[t_1, 4]])


            # print func_rmse(Xn_1, rhom, Q, Vl, alpha, beta, coefs, j, t_1)
            # cons = {{'type': 'ineq',
            #          'fun': lambda x: np.array([x[1] - x[3]])},
            #          {'type': 'ineq',
            #          'fun': lambda x: np.array([x[3] - x[5]])},
            #          {'type': 'ineq',
            #          'fun': lambda x: np.array([x[5] - x[7]])},
            #          {'type': 'ineq',
            #          'fun': lambda x: np.array([x[7] - x[9]])}}
            res = minimize(func_rmse, Xn_1, method='SLSQP', bounds=bnds, args=(
                rhom, Q, Qs, Vl, Vx, A, Ae, h, alpha, beta, coefs, j, t_1), options = {'disp': True})
            (Q.loc[j, 0], rhom.loc[j, 0], Q.loc[j, 1], rhom.loc[j, 1], Q.loc[j, 2], rhom.loc[j, 2], Q.loc[j, 3],
                 rhom.loc[j, 3], Q.loc[j, 4], rhom.loc[j, 4]) = res.x
            # Q = si/rhom
            print(res.x)
            if np.mod(i+1, 200) == 0:
                plt.figure(figsize=(6, 10))
                plt.subplot(311)
                plt.plot(Q.loc['24/07/2008':j])
                plt.subplot(312)
                plt.plot(rhom.loc['24/07/2008':j])
                plt.plot(rhoe.loc['24/07/2008':j], '--')
                plt.subplot(313)
                plt.plot(Vl.loc['24/07/2008':j])
                plt.show()
                plt.close()
        # Q = si/rhom


    for i in block.columns:
        if i == 0:
            mask = ((rhom.loc[:, i] > rhom.loc[:, 's']) | (Q.loc[:, i+1] > Q.loc[:, i]) | (Q.loc[:, i] < 0))
        else:
            mask = ((rhom.loc[:, i] > rhoe.loc[:, i-1]) | (Q.loc[:, i+1] > Q.loc[:, i]) | (Q.loc[:, i] < 0))
      
        
        # mask_waterdensity = (rhom.iloc[:, i+1].values < 1000)
        # if any(mask_waterdensity):
        #     rhom.ix[mask_waterdensity, i+1] = 1000
        #     # Q.ix[mask, i+1] = 0
        #     Q.iloc[:, i+1] = ji.iloc[:, i+1]/rhom.iloc[:, i+1]
        #     # block.ix[mask, i] = 1
        #     # ji.ix[mask, i+1] = 0

        # mask = rhom.iloc[:, i+1].values < rhoe.iloc[:, i].values
        # if any(mask):
        #     # error.ix[mask, i] = rhom.ix[mask, i+1].values-rhoe.ix[mask, i].values
        #     # rhom.ix[mask, i+1] = rhoe.ix[mask, i].values
        #     # Q.ix[mask, i+1] = ji.ix[mask, i+1]/rhom.ix[mask, i+1]
        #     block.ix[mask, i] = -1

    return Q, rhom, block


def func(rhom, Q, Qs, Vl, Vx, Wx, Dx, A, Ae, h, alpha, beta, delta, coefs, t, t_1):
    """ Balance de anomalia de la energia potencial y de masa para las cinco cajas
    """
    
    F = np.zeros(10)
    G = 9.81
    coef_mix = 0.0038*4/(3*np.pi)*0.0025
    Dx = Dx*np.pi/180.
    coef_sol = 1.7e-4*G/(2*4200)
    coef_wind = -0.039*6.4e-5*1.225*1.25
    F[0] = (rhom.loc[t, 's']-rhom.loc[t, 0]*alpha.loc['ij', '01'] - rhom.loc[t, 1]*alpha.loc['ij', '11'] -
            (rhom.loc[t_1, 's'] - rhom.loc[t_1, 0]*alpha.loc['ij', '01'] - rhom.loc[t_1, 1]*alpha.loc['ij', '11']))*G*beta.loc[t, 1]*0.5 \
        + coefs.loc['gamma', 1]*G*(rhom.loc[t, 's'] - (rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11']))*Vl.loc[t, 1] \
        + coefs.loc['alpha', 0]*G*(rhom.loc[t, 's'] - rhom.loc[t, 0])*Q.loc[t, 0] \
        - coefs.loc['alpha', 1]*G*(rhom.loc[t, 's'] - rhom.loc[t, 1])*Q.loc[t, 1] \
        + coef_mix*rhom.loc[t, 's']*0.378**3*Ae[0]/h.loc['m', 0] \
        - coef_sol*Qs.loc[t]*Ae[0] \
        + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[0]/h.loc['m', 0]
        # + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[0]/h.loc['m', 0]

    F[2] = (rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t, 1]*alpha.loc['ij', '12'] - rhom.loc[t, 2]*alpha.loc['ij', '22'] -
            (rhom.loc[t_1, 0]*alpha.loc['ij', '01'] + rhom.loc[t_1, 1]*alpha.loc['ij', '11'] - rhom.loc[t_1, 1]*alpha.loc['ij', '12'] - rhom.loc[t_1, 2]*alpha.loc['ij', '22']))*G*beta.loc[t, 2]*0.5 \
        + coefs.loc['gamma', 2]*G*(rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - (rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22']))*Vl.loc[t, 2] \
        + coefs.loc['alpha', 1]*G*(rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t, 1])*Q.loc[t, 1] \
        - coefs.loc['alpha', 2]*G*(rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t, 2])*Q.loc[t, 2] \
        + coef_mix*rhom.loc[t, 0]*0.555**3*Ae[1]/h.loc['m', 1] \
        - coef_sol*Qs.loc[t]*Ae[1] \
        + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[1]/h.loc['m', 1]
        # + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[1]/h.loc['m', 0]

    
    F[4] = (rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t, 2]*alpha.loc['ij', '23'] - rhom.loc[t, 3]*alpha.loc['ij', '33'] -
            (rhom.loc[t_1, 1]*alpha.loc['ij', '12'] + rhom.loc[t_1, 2]*alpha.loc['ij', '22'] - rhom.loc[t_1, 2]*alpha.loc['ij', '23'] - rhom.loc[t_1, 3]*alpha.loc['ij', '33']))*G*beta.loc[t, 3]*0.5 \
        + coefs.loc['gamma', 3]*G*(rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - (rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33']))*Vl.loc[t, 3] \
        + coefs.loc['alpha', 2]*G*(rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t, 2])*Q.loc[t, 2] \
        - coefs.loc['alpha', 3]*G*(rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t, 3])*Q.loc[t, 3] \
        + coef_mix*rhom.loc[t, 1]*0.555**3*Ae[2]/h.loc['m', 2] \
        - coef_sol*Qs.loc[t]*Ae[2] \
        + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[2]/h.loc['m', 2]
        # + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[2]/h.loc['m', 0]
    
    F[6] = (rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t, 3]*alpha.loc['ij', '34'] - rhom.loc[t, 4]*alpha.loc['ij', '44'] - 
            (rhom.loc[t_1, 2]*alpha.loc['ij', '23'] + rhom.loc[t_1, 3]*alpha.loc['ij', '33'] - rhom.loc[t_1, 3]*alpha.loc['ij', '34'] - rhom.loc[t_1, 4]*alpha.loc['ij', '44']))*G*beta.loc[t, 4]*0.5 \
        + coefs.loc['gamma', 4]*G*(rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - (rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44']))*Vl.loc[t, 4] \
        + coefs.loc['alpha', 3]*G*(rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t, 3])*Q.loc[t, 3] \
        - coefs.loc['alpha', 4]*G*(rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t, 4])*Q.loc[t, 4] \
        + coef_mix*rhom.loc[t, 2]*0.376**3*Ae[3]/h.loc['m', 3] \
        - coef_sol*Qs.loc[t]*Ae[3] \
        + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[3]/h.loc['m', 3]
        # + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[3]/h.loc['m', 0]

    
    F[8] = (rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - rhom.loc[t, 4]*alpha.loc['ij', '45'] - rhom.loc[t, 5]*alpha.loc['ij', '55'] -
            (rhom.loc[t_1, 3]*alpha.loc['ij', '34'] + rhom.loc[t_1, 4]*alpha.loc['ij', '44'] - rhom.loc[t_1, 4]*alpha.loc['ij', '45'] - rhom.loc[t_1, 5]*alpha.loc['ij', '55']))*G*beta.loc[t, 5]*0.5 \
        + coefs.loc['gamma', 5]*G*(rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - (rhom.loc[t, 4]*alpha.loc['ij', '45'] + rhom.loc[t, 5]*alpha.loc['ij', '55']))*Vl.loc[t, 5] \
        + coefs.loc['alpha', 4]*G*(rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - rhom.loc[t, 4])*Q.loc[t, 4] \
        - coefs.loc['alpha', 5]*G*(rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - rhom.loc[t, 5])*Q.loc[t, 5] \
        + coef_mix*rhom.loc[t, 3]*0.297**3*Ae[4]/h.loc['m', 4] \
        - coef_sol*Qs.loc[t]*Ae[4] \
        + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[4]/h.loc['m', 4]
        # + coef_wind*Wx.loc[t, 'V_mean (m/s)']**3*Ae[4]/h.loc['m', 0]
    
    F[1] = (rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t_1, 0]*alpha.loc['ij', '01'] - rhom.loc[t_1, 1]*alpha.loc['ij', '11'])*delta.loc[t, 1] \
        + Q.loc[t, 0]*rhom.loc[t, 0] \
        + (rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'])*Vl.loc[t, 1] \
        - Q.loc[t, 1]*rhom.loc[t, 1] \
        - rhom.loc[t, 's']*Vl.loc[t, 0]
    
    F[3] = (rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t_1, 1]*alpha.loc['ij', '12'] - rhom.loc[t_1, 2]*alpha.loc['ij', '22'])*delta.loc[t, 2] \
        + Q.loc[t, 1]*rhom.loc[t, 1] \
        + (rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'])*Vl.loc[t, 2] \
        - Q.loc[t, 2]*rhom.loc[t, 2] \
        - (rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'])*Vl.loc[t, 1]
    
    F[5] = (rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t_1, 2]*alpha.loc['ij', '23'] - rhom.loc[t_1, 3]*alpha.loc['ij', '33'])*delta.loc[t, 3] \
        + Q.loc[t, 2]*rhom.loc[t, 2] \
        + (rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'])*Vl.loc[t, 3] \
        - Q.loc[t, 3]*rhom.loc[t, 3] \
        - (rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'])*Vl.loc[t, 2]
        
    F[7] = (rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - rhom.loc[t_1, 3]*alpha.loc['ij', '34'] - rhom.loc[t_1, 4]*alpha.loc['ij', '44'])*delta.loc[t, 4] \
        + Q.loc[t, 3]*rhom.loc[t, 3] \
        + (rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'])*Vl.loc[t, 4] \
        - Q.loc[t, 4]*rhom.loc[t, 4] \
        - (rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'])*Vl.loc[t, 3]
    
    F[9] = (rhom.loc[t, 4]*alpha.loc['ij', '45'] + rhom.loc[t, 5]*alpha.loc['ij', '55'] - rhom.loc[t_1, 4]*alpha.loc['ij', '45'] - rhom.loc[t_1, 5]*alpha.loc['ij', '55'])*delta.loc[t, 5] \
        + Q.loc[t, 4]*rhom.loc[t, 4] \
        + (rhom.loc[t, 4]*alpha.loc['ij', '45'] + rhom.loc[t, 5]*alpha.loc['ij', '55'])*Vl.loc[t, 5] \
        - rhom.loc[t, 5]*Q.loc[t, 5] \
        - (rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'])*Vl.loc[t, 4]
        
    return F


def jac_func(rhom, Q, Vl, alpha, beta, delta, coefs, t):
    """ Jacobiano de la funcion objetivo
    """
    J = np.zeros([10, 10])
    G = 9.81
    J[0, 0] = coefs.loc['alpha', 0]*G*(rhom.loc[t, 's'] - rhom.loc[t, 0])
    J[0, 1] = -alpha.loc['ij', '01']*G*beta.loc[t, 1]*0.5 - coefs.loc['gamma', 0]*G*alpha.loc['ij', '01']*Vl.loc[t, 1] - coefs.loc['alpha', 0]*G*Q.loc[t, 0]
    J[0, 2] = -coefs.loc['alpha', 1]*G*(rhom.loc[t, 's'] - rhom.loc[t, 1])
    J[0, 3] = -alpha.loc['ij', '11']*G*beta.loc[t, 1]*0.5 - coefs.loc['gamma', 0]*G*alpha.loc['ij', '11']*Vl.loc[t, 1] + coefs.loc['alpha', 0]*G*Q.loc[t, 1] 

    J[2, 1] = alpha.loc['ij', '01']*G*beta.loc[t, 2]*0.5 + coefs.loc['gamma', 1]*G*alpha.loc['ij', '01']*Vl.loc[t, 2] + coefs.loc['alpha', 1]*G*alpha.loc['ij', '01']*Q.loc[t, 1] \
        - coefs.loc['alpha', 2]*G*alpha.loc['ij', '01']*Q.loc[t, 2]
    J[2, 2] = coefs.loc['alpha', 1]*G*(rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t, 1])
    J[2, 3] = (alpha.loc['ij', '11'] - alpha.loc['ij', '12'])*G*beta.loc[t, 2]*0.5 \
        + coefs.loc['gamma', 2]*G*(alpha.loc['ij', '11'] - alpha.loc['ij', '12'])*Vl.loc[t, 2] \
        + coefs.loc['alpha', 1]*G*(alpha.loc['ij', '11'] - 1)*Q.loc[t, 1] \
        - coefs.loc['alpha', 2]*G*alpha.loc['ij', '11']*Q.loc[t, 2]
    J[2, 4] = -coefs.loc['alpha', 2]*G*(rhom.loc[t, 0]*alpha.loc['ij', '01'] + rhom.loc[t, 1]*alpha.loc['ij', '11'] - rhom.loc[t, 2])
    J[2, 5] = -alpha.loc['ij', '22']*G*beta.loc[t, 2]*0.5 - coefs.loc['gamma', 1]*G*alpha.loc['ij', '22']*Vl.loc[t, 2] + coefs.loc['alpha', 2]*G*Q.loc[t, 2]

    J[4, 3] = alpha.loc['ij', '12']*G*beta.loc[t, 3]*0.5 + coefs.loc['gamma', 2]*G*alpha.loc['ij', '12']*Vl.loc[t, 3] + coefs.loc['alpha', 2]*G*alpha.loc['ij', '12']*Q.loc[t, 2] \
        - coefs.loc['alpha', 3]*G*alpha.loc['ij', '12']*Q.loc[t, 3]
    J[4, 4] = coefs.loc['alpha', 2]*G*(rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t, 2])
    J[4, 5] = (alpha.loc['ij', '22'] - alpha.loc['ij', '23'])*G*beta.loc[t, 3]*0.5 \
        + coefs.loc['gamma', 3]*G*(alpha.loc['ij', '22'] - alpha.loc['ij', '23'])*Vl.loc[t, 3] \
        + coefs.loc['alpha', 2]*G*(alpha.loc['ij', '22'] - 1)*Q.loc[t, 2] \
        - coefs.loc['alpha', 3]*G*alpha.loc['ij', '22']*Q.loc[t, 3]
    J[4, 6] = -coefs.loc['alpha', 3]*G*(rhom.loc[t, 1]*alpha.loc['ij', '12'] + rhom.loc[t, 2]*alpha.loc['ij', '22'] - rhom.loc[t, 3])
    J[4, 7] = -alpha.loc['ij', '33']*G*beta.loc[t, 3]*0.5 - coefs.loc['gamma', 2]*G*alpha.loc['ij', '33']*Vl.loc[t, 3] + coefs.loc['alpha', 3]*G*Q.loc[t, 3]

    J[6, 5] = alpha.loc['ij', '23']*G*beta.loc[t, 4]*0.5 + coefs.loc['gamma', 3]*G*alpha.loc['ij', '23']*Vl.loc[t, 4] + coefs.loc['alpha', 3]*G*alpha.loc['ij', '23']*Q.loc[t, 3] \
        - coefs.loc['alpha', 4]*G*alpha.loc['ij', '23']*Q.loc[t, 4]
    J[6, 6] = coefs.loc['alpha', 3]*G*(rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t, 3])
    J[6, 7] = (alpha.loc['ij', '33'] - alpha.loc['ij', '34'])*G*beta.loc[t, 4]*0.5 \
        + coefs.loc['gamma', 4]*G*(alpha.loc['ij', '33'] - alpha.loc['ij', '34'])*Vl.loc[t, 4] \
        + coefs.loc['alpha', 3]*G*(alpha.loc['ij', '33'] - 1)*Q.loc[t, 3] \
        - coefs.loc['alpha', 4]*G*alpha.loc['ij', '33']*Q.loc[t, 4]
    J[6, 8] = -coefs.loc['alpha', 4]*G*(rhom.loc[t, 2]*alpha.loc['ij', '23'] + rhom.loc[t, 3]*alpha.loc['ij', '33'] - rhom.loc[t, 4])
    J[6, 9] = -alpha.loc['ij', '44']*G*beta.loc[t, 4]*0.5 - coefs.loc['gamma', 3]*G*alpha.loc['ij', '44']*Vl.loc[t, 4] + coefs.loc['alpha', 4]*G*Q.loc[t, 4]


    J[8, 7] = alpha.loc['ij', '34']*G*beta.loc[t, 5]*0.5 + coefs.loc['gamma', 4]*G*alpha.loc['ij', '34']*Vl.loc[t, 5] + coefs.loc['alpha', 4]*G*alpha.loc['ij', '34']*Q.loc[t, 4] \
        - coefs.loc['alpha', 5]*G*alpha.loc['ij', '34']*Q.loc[t, 5]
    J[8, 8] = coefs.loc['alpha', 4]*G*(rhom.loc[t, 3]*alpha.loc['ij', '34'] + rhom.loc[t, 4]*alpha.loc['ij', '44'] - rhom.loc[t, 4])
    J[8, 9] = (alpha.loc['ij', '44'] - alpha.loc['ij', '45'])*G*beta.loc[t, 5]*0.5 \
        + coefs.loc['gamma', 5]*G*(alpha.loc['ij', '44'] - alpha.loc['ij', '45'])*Vl.loc[t, 5] \
        + coefs.loc['alpha', 4]*G*(alpha.loc['ij', '44'] - 1)*Q.loc[t, 4] \
        - coefs.loc['alpha', 5]*G*alpha.loc['ij', '44']*Q.loc[t, 5]
    
    J[1, 0] = rhom.loc[t, 0]
    J[1, 1] = alpha.loc['ij', '01']*delta.loc[t, 1] + alpha.loc['ij', '01']*Vl.loc[t, 1] + Q.loc[t, 0]
    J[1, 2] = -rhom.loc[t, 1]
    J[1, 3] = alpha.loc['ij', '11']*delta.loc[t, 1] + alpha.loc['ij', '11']*Vl.loc[t, 1] - Q.loc[t, 1]

    J[3, 1] = -alpha.loc['ij', '01']*Vl.loc[t, 1]
    J[3, 2] = rhom.loc[t, 1]
    J[3, 3] = alpha.loc['ij', '12']*delta.loc[t, 2] + Q.loc[t, 1] + alpha.loc['ij', '12']*Vl.loc[t, 2] -  alpha.loc['ij', '11']*Vl.loc[t, 1]
    J[3, 4] = -rhom.loc[t, 2]
    J[3, 5] = alpha.loc['ij', '22']*delta.loc[t, 2] + alpha.loc['ij', '22']*Vl.loc[t, 2] - Q.loc[t, 2]

    J[5, 3] = - alpha.loc['ij', '12']*Vl.loc[t, 2]
    J[5, 4] = rhom.loc[t, 2]
    J[5, 5] = alpha.loc['ij', '23']*delta.loc[t, 3] + Q.loc[t, 2] + alpha.loc['ij', '23']*Vl.loc[t, 3] -  alpha.loc['ij', '22']*Vl.loc[t, 2]
    J[5, 6] = -rhom.loc[t, 3]
    J[5, 7] = alpha.loc['ij', '33']*delta.loc[t, 3] + alpha.loc['ij', '33']*Vl.loc[t, 3] - Q.loc[t, 3]

    J[7, 5] = - alpha.loc['ij', '23']*Vl.loc[t, 3]
    J[7, 6] = rhom.loc[t, 3]
    J[7, 7] = alpha.loc['ij', '34']*delta.loc[t, 4] + Q.loc[t, 3] + alpha.loc['ij', '34']*Vl.loc[t, 4] -  alpha.loc['ij', '33']*Vl.loc[t, 3]
    J[7, 8] = -rhom.loc[t, 4]
    J[7, 9] = alpha.loc['ij', '44']*delta.loc[t, 4] + alpha.loc['ij', '44']*Vl.loc[t, 4] - Q.loc[t, 4]

    J[9, 7] = -alpha.loc['ij', '34']*Vl.loc[t, 4]
    J[9, 8] = rhom.loc[t, 4]
    J[9, 9] = alpha.loc['ij', '45']*delta.loc[t, 5] + Q.loc[t, 4] + alpha.loc['ij', '45']*Vl.loc[t, 5] -  alpha.loc['ij', '44']*Vl.loc[t, 4]

    return J
