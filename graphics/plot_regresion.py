# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def picos_sobre_umbral(event, df, umb):
    """ Funcion que dibuja los graficos de dispersion dos-a-dos de las variables contenidas en df y los eventos
    extremos anuales y que superan los umbrales

    Args:
        - event: dataframe con los picos anuales de la variable principal y el valor de las variables acompañantes
        - df: dataframe con todos los datos de la variable principal y acompañantes
        - umb: serie con los umbrales a dibujar

    Returns:
        Devuelve la figura con los graficos de dispersion

    """

    plt.style.use('ggplot')
    df = df.dropna()
    naux, numb = np.shape(df)[1]-1, len(umb)
    ncol, nfil = 1, 1
    nven = ncol*nfil
    nomb = df.columns
    event, df = event.values, df.values
    while nven < naux:
        if ncol == nfil:
            ncol += 1
        else:
            nfil += 1
        nven = ncol*nfil

    col = ['k', 'b', 'g', 'purple', 'brown']
    fig = plt.figure()
    for i in range(1, naux+1):
        fig.add_subplot(nfil, ncol, i)
        plt.plot(df[:, 0], df[:, i], '.', color='gray')

        for j in range(0, numb):
            id_ = np.where(event[:, 0] > umb[j])[0]
            plt.plot(event[id_, 0], event[id_, i], '.', color=col[j], label='umbral '+str(umb[j]))

        plt.xlim(np.min(df[:, 0]), np.max(df[:, 0]))
        plt.ylim(np.min(df[:, i]), np.max(df[:, i]))
        plt.xlabel(nomb[0])
        plt.ylabel(nomb[i])
#    plt.legend(loc='best', numpoints=1)


def ajustes_sobre_umbral(event, df, ajuste, umb, comb, fun, ejes):
    """ Funcion que dibuja los graficos de dispersion dos a dos y los ajustes sobre estos

    Args:
        - event: lista con los valores que superan cada uno de los umbrales (longitud igual al numero de umbrales)
        - df: dataframe con todas las series
        - ajuste: lista con los valores del ajuste (x, ci-sup, medio, ci-inf) para cada uno de los umbrales (longitud
            igual al numero de umbrales)

    Returns:

    """

    plt.style.use('ggplot')
    numb = len(umb)
    nfun = len(fun)
    nombres = df.columns
    ncomb = len(comb)
    col = ['b', 'y', 'purple', 'brown']
    for k in range(ncomb):
        plt.figure(figsize=(3*nfun + 2, 3))
        plt.title(comb[k])
        for i in range(0, nfun):
            ax = plt.subplot(1, nfun, i+1)
            ax.set_title('Ajuste con '+fun[i])
            plt.plot(df[nombres[0]], df[nombres[k+1]], '.', color='gray', markersize=7)
            for j in range(numb):
                plt.plot(event[j][nombres[0]], event[j][nombres[k+1]], '.', color=col[j], markersize=7)

                if ((comb[k] == 'dh') | (comb[k] == 'dv')):

                    if ajuste['dirN' + comb[k - 1] + fun[i] + str(umb[j])][0]:
                        for kk in range(ajuste['ndirp' + comb[k] + fun[i] + str(umb[j])][0]):
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['y' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360,
                                     color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['ysup' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360, '.',
                                     markersize=0.5, color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['yinf' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360, '.',
                                     markersize=0.5, color=col[j])
                    else:
                        for kk in range(ajuste['ndirp' + comb[k] + fun[i] + str(umb[j])][0]):
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['y' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['ysup' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)], '--',
                                     markersize=0.8, color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['yinf' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)], '--',
                                     markersize=0.8, color=col[j])

                    plt.xlim((0, np.max(ajuste['x' + comb[k] + fun[i] + str(umb[j])  + 'dir0'])))
                    plt.ylim((0, 360))

                else:
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['y'+comb[k]+fun[i]+str(umb[j])], color=col[j])
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['ysup'+comb[k]+fun[i]+str(umb[j])], '--',
                             color=col[j])
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['yinf'+comb[k]+fun[i]+str(umb[j])], '--',
                             color=col[j])

                    plt.xlim((0, np.max(ajuste['x' + comb[k] + fun[i] + str(umb[j])])))
                    plt.ylim((0, 1.5 * np.max(df[nombres[k + 1]])))

            if i > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.ylabel(ejes[k+1])


            plt.xlabel(ejes[0])
        plt.gcf().subplots_adjust(bottom=0.2, left=0.15)


def ajustes_adim(event, df, ajuste, fun, ejes):
    """ Funcion que dibuja los graficos de dispersion precisos para ajustar hs y tp, limitando los valores al dado
	por el oleaje totalmente desarrollado.

    Args:
        - event: lista con los valores que superan cada uno de los umbrales (longitud igual al numero de umbrales)
        - df: dataframe con todas las series
        - ajuste: lista con los valores del ajuste (x, ci-sup, medio, ci-inf) para cada uno de los umbrales (longitud
            igual al numero de umbrales)

    Returns:

    """

    plt.style.use('ggplot')
    nombres = ['hs', 'tp', 'vv']
    nomb = ['tp', 'vv', 'adim']
    col = ['b', 'y', 'purple']
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(df[nombres[0]], df[nombres[1]], '.', color='gray', markersize=7)
    plt.plot(event[0][nombres[0]], event[0][nombres[1]], '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['y'+nomb[0]+str(fun)], color=col[0])
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['ysup'+nomb[0]+str(fun)], '--', color=col[0])
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['yinf'+nomb[0]+str(fun)], '--', color=col[0])
    plt.plot(ajuste['hs_aux'], ajuste['t_aux'], 'k', lw=2)
    plt.xlabel(ejes[0])
    plt.ylabel(ejes[1])

    plt.subplot(1, 3, 2)
    plt.plot(df[nombres[0]]*9.81/(1.1*df[nombres[2]]**2), df[nombres[1]]*9.81/(1.1*df[nombres[2]]), '.', color='gray',
             markersize=7)
    plt.plot(event[0][nombres[0]]*9.81/(1.1*event[0][nombres[2]]**2), event[0][nombres[1]]*9.81/(
        1.1*event[0][nombres[2]]), '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['y'+nomb[2]+str(fun)], color='b')
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['ysup'+nomb[2]+str(fun)], '--', color='b')
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['yinf'+nomb[2]+str(fun)], '--', color='b')
    plt.plot(np.array([0.2541, 0.2541, 0]), np.array([0, 7.944, 7.944]), '--k')
    plt.xlim([0, 0.3])
    plt.ylim([0, 10])
    plt.xlabel(r'$\frac{H_sg}{v^2}$')
    plt.ylabel(r'$\frac{T_pg}{v}$')

    plt.subplot(1, 3, 3)
    plt.plot(df[nombres[0]], df[nombres[2]], '.', color='gray', markersize=7)
    plt.plot(event[0][nombres[0]], event[0][nombres[2]], '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['y'+nomb[1]+str(fun)], color='b')
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['ysup'+nomb[1]+str(fun)], '--', color='b')
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['yinf'+nomb[1]+str(fun)], '--', color='b')
    plt.xlabel(ejes[0])
    plt.ylabel(ejes[2])
    plt.tight_layout(pad=0.5)
