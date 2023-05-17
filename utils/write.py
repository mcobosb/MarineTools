import numpy as np
from marinetools.utils import auxiliar, save
from scipy.io import loadmat as ldm


def cshore(properties: dict, folder: str):
    """[summary]

    Args:
        properties (dict): [description]
        folder (str): [description]
    """

    fid = open(folder + "/infile", "w")
    fid.write("3 \n")
    fid.write("{} \n".format(str(properties["header"])))
    fid.write(
        "{}                                               ->ILINE\n".format(
            str(properties["iline"])
        )
    )
    fid.write(
        "{}                                               ->IPROFL\n".format(
            str(properties["iprofl"])
        )
    )
    fid.write(
        "{}                                               ->ISEDAV\n".format(
            str(properties["isedav"])
        )
    )
    fid.write(
        "{}                                               ->IPERM\n".format(
            str(properties["iperm"])
        )
    )
    fid.write(
        "{}                                               ->IOVER\n".format(
            str(properties["iover"])
        )
    )
    fid.write(
        "{}                                               ->IWTRAN\n".format(
            str(properties["iwtran"])
        )
    )
    fid.write(
        "{}                                               ->IPOND\n".format(
            str(properties["ipond"])
        )
    )
    fid.write(
        "{}                                               ->INFILT\n".format(
            str(properties["infilt"])
        )
    )
    fid.write(
        "{}                                               ->IWCINT\n".format(
            str(properties["iwcint"])
        )
    )
    fid.write(
        "{}                                               ->IROLL \n".format(
            str(properties["iroll"])
        )
    )
    fid.write(
        "{}                                               ->IWIND \n".format(
            str(properties["iwind"])
        )
    )
    fid.write(
        "{}                                               ->ITIDE \n".format(
            str(properties["itide"])
        )
    )
    fid.write(
        "{}                                               ->IVEG  \n".format(
            str(properties["iveg"])
        )
    )
    # fid.write('{}                                               ->ICLAY  \n'.format(str(properties['iclay'])))
    fid.write(
        "{:11.4f}                                     ->DX\n".format(properties["dx"])
    )
    fid.write(
        "{:11.4f}                                     ->GAMMA \n".format(
            properties["gamma"]
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}{:11.4f}               ->D50 WF SG\n".format(
            properties["d50"], properties["wf"], properties["sg"]
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}{:11.4f}{:11.4f}               ->EFFB EFFF SLP\n".format(
            properties["effb"],
            properties["efff"],
            properties["slp"],
            properties["slpot"],
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}                          ->TANPHI BLP\n".format(
            properties["tanphi"], properties["blp"]
        )
    )
    fid.write(
        "{:11.4f}                                     ->RWH \n".format(
            properties["rwh"]
        )
    )
    fid.write(
        "{}                                               ->ILAB\n".format(
            str(properties["ilab"])
        )
    )
    fid.write(
        "{}                                               ->NWAVE \n".format(
            str(properties["nwave"])
        )
    )
    fid.write(
        "{}                                               ->NSURGE \n".format(
            str(properties["nsurg"])
        )
    )
    fid.write(
        "{:11.2f}{:11.4f}{:11.4f}{:11.4f}{:11.4f}{:11.4f}\n".format(
            properties["timebc_wave"],
            properties["Tp"],
            properties["Hrms"],
            properties["Wsetup"],
            properties["swlbc"],
            properties["angle"],
        )
    )
    fid.write(
        "{}                             ->NBINP \n".format(str(len(properties["x"])))
    )

    # if properties.iperm==1|properties.isedav >= 1:
    #     fid.write('{}                             ->NPINP \n',length(properties.x_p))
    fid.close()

    fid = open(folder + "/infile", "a")
    dum = np.vstack([properties["x"], properties["zb"], properties["fw"]])
    for line in range(dum.shape[1]):
        fid.write("{:11.4f}{:11.4f}{:11.4f}\n".format(*dum[:, line]))

    # if properties.iperm == 1 | properties.isedav >= 1:
    #     dum = [properties.x_p(:) properties.zb_p(:)]
    #     fid.write('#11.4f#11.4f\n', dum)

    if properties["iwind"]:
        fid.write("1 \n")
        fid.write(
            "{:11.1f}{:11.4f}{:11.4f}\n".format(
                0, properties["VelV"], properties["DirV"]
            )
        )
        fid.write(
            "{:11.1f}{:11.4f}{:11.4f}\n".format(
                properties["timebc_wave"], properties["VelV"], properties["DirV"]
            )
        )

    if properties["itide"]:
        fid.write("1 \n")
        fid.write("{:11.1f}{:13.8f}\n".format(0, properties["slgradient"]))
        fid.write(
            "{:11.1f}{:13.8f}\n".format(
                properties["timebc_wave"], properties["slgradient"]
            )
        )

    # if properties.iveg==1
    # fid.write('#5.3f                                ->VEGCD\n',properties.veg_Cd )
    # dum = zeros(length(properties.x(:)),4)
    # ind = find(properties.x>=max(properties.x)*properties.veg_extent(1)&properties.x<=max(properties.x)*properties.veg_extent(2))
    # dum(ind,:) = repmat([properties.veg_n properties.veg_dia properties.veg_ht properties.veg_rod],length(ind),1)
    # fid.write('#11.3f#11.3f#11.3f#11.3f\n',dum')
    fid.close()

    return


def swan(i, index_, id_, data, params, mesh="global", local=False, nested=False):
    """[summary]

    Args:
        i ([type]): [description]
        index_ ([type]): [description]
        id_ ([type]): [description]
        data ([type]): [description]
        param ([type]): [description]
        mesh (str, optional): [description]. Defaults to 'global'.
        local (bool, optional): [description]. Defaults to False.
        nested (bool, optional): [description]. Defaults to False.
    """

    swfile = open(params["directory"] + "/" + id_ + "/swaninit", "w")
    swfile.write("4                                   version of initialisation file\n")
    swfile.write("GDFA                                name of institute\n")
    swfile.write("3                                   command file ref. number\n")
    swfile.write("input_" + mesh + "_" + id_ + ".swn\n")
    swfile.write("4                                   print file ref. number\n")
    swfile.write("print_" + mesh + "_" + id_ + ".prt\n")
    swfile.write("4                                   test file ref. number\n")
    swfile.write("                                    test file name\n")
    swfile.write("6                                   screen ref. number\n")
    swfile.write("99                                  highest file ref. number\n")
    swfile.write("$                                   comment identifier\n")
    swfile.write(" 	                                TAB character\n")
    swfile.write("/                                   dir sep char in input file\n")
    swfile.write(
        "/                                   dir sep char replacing previous one\n"
    )
    swfile.write("1                                   default time coding option\n")
    swfile.close()

    imswfile = open(
        params["directory"] + "/" + id_ + "/input_" + mesh + "_" + id_ + ".swn", "w"
    )
    imswfile.write(
        "$*******************************HEADING******************************************\n"
    )
    imswfile.write("$\n")
    imswfile.write("PROJ '" + params["project_name"] + "' '" + id_ + "' \n")
    imswfile.write("$Caso " + mesh + "_" + id_ + "\n")
    imswfile.write(
        "$***************************** MODEL INPUT ****************************************\n"
    )
    imswfile.write(
        "SET LEVEL " + str(np.round(data.loc[index_, "eta"], decimals=2)) + "\n"
    )
    imswfile.write("$                xpc		ypc		alpc		xlenc		ylenc		mxc		myc \n")
    imswfile.write("$\n")
    imswfile.write(
        "CGRID REGULAR   "
        + str(params[mesh + "_coords_x"])
        + "   "
        + str(params[mesh + "_coords_y"])
        + "   "
        + str(np.round(np.remainder(params[mesh + "_angle"], 360), decimals=2))
        + "    "
        + str(params[mesh + "_length_x"])
        + "   "
        + str(params[mesh + "_length_y"])
        + "    "
        + str(params[mesh + "_nodes_x"] - 1)
        + "   "
        + str(params[mesh + "_nodes_y"] - 1)
        + "  CIRCLE 36 0.05 0.50\n"
    )
    imswfile.write("$\n")
    imswfile.write("$xpinp		ypinp		alpinp		mxinp		myinp		dxinp		dyinp\n")
    imswfile.write(
        "INPGRID BOTTOM     "
        + str(params[mesh + "_coords_x"])
        + "   "
        + str(params[mesh + "_coords_y"])
        + "   "
        + str(np.round(np.remainder(params[mesh + "_angle"], 360), decimals=2))
        + "    "
        + str(params[mesh + "_nodes_x"] - 1)
        + "   "
        + str(params[mesh + "_nodes_y"] - 1)
        + "   "
        + str(params[mesh + "_inc_x"])
        + "  "
        + str(params[mesh + "_inc_y"])
        + "\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "$              fac    fname       idla    nhedf     formato  ((mxinp+1)FN.d)\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "READINP BOTTOM -1. '"
        + id_
        + "_"
        + mesh
        + ".dat' 3 0 FORMAT '("
        + str(params[mesh + "_nodes_x"])
        + "F9.3)'\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "WIND  "
        + str(np.round(data.loc[index_, "Vv"], decimals=2))
        + " "
        + str(np.round(np.remainder(270 - data.loc[index_, "DirV"], 360), decimals=2))
        + "\n"
    )
    imswfile.write("$\n")

    if mesh == "global":
        # Here, direction has PdE convention (waves coming from N: 0º, E: 90º)
        if data.loc[index_, "DirM"] < 90:
            side = ["N", "E"]
        elif (data.loc[index_, "DirM"] < 180) & (data.loc[index_, "DirM"] >= 90):
            side = ["S", "E"]
        elif (data.loc[index_, "DirM"] < 270) & (data.loc[index_, "DirM"] >= 180):
            side = ["S", "W"]
        else:
            side = ["N", "W"]
        imswfile.write("BOUN SHAPESPEC JONSWAP PEAK DSPR POWER\n")
        imswfile.write("$                             Hs	Tp	Dir	dd(spreading power)\n")
        imswfile.write(
            "BOUN SIDE "
            + side[0]
            + " CONSTANT PAR  "
            + str(np.round(data.loc[index_, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[index_, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[index_, "DirM"], 360), decimals=2)
            )
            + " 2.00\n"
        )
        imswfile.write(
            "BOUN SIDE "
            + side[1]
            + " VARIABLE PAR  200 "
            + str(np.round(data.loc[index_, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[index_, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[index_, "DirM"], 360), decimals=2)
            )
            + " 2.00\n"
        )
    else:
        imswfile.write("BOUNdnest1 NEST '" + id_ + ".bnd' CLOSED\n")

    imswfile.write("$*************************WIND GROWTH**************************\n")
    imswfile.write("GEN3 AGROW\n")
    imswfile.write(
        "$*************************WAVE-WAVE INTERACTION**************************\n"
    )
    imswfile.write("TRIAD\n")
    imswfile.write("QUAD\n")
    imswfile.write("$*************************DISSIPATION**************************\n")
    imswfile.write("BREAKING\n")
    imswfile.write("WCAP\n")
    imswfile.write("$*************************************************************\n")
    imswfile.write("SETUP\n")
    imswfile.write("DIFFRACTION\n")
    imswfile.write("NUM ACCUR 0.005 0.01 0.005 99.5 STAT 50 0.1\n")
    if nested:
        imswfile.write(
            "$*************************OUPUT LOCATIONS**************************\n"
        )
        imswfile.write(
            "NGRID '"
            + id_
            + ".dat"
            + "'  "
            + str(params["local_coords_x"])
            + "   "
            + str(params["local_coords_y"])
            + "   "
            + str(np.round(np.remainder(params["local_angle"], 360), decimals=2))
            + "    "
            + str(params["local_length_x"])
            + "   "
            + str(params["local_length_y"])
            + "   "
            + str(params["local_nodes_x"] - 1)
            + "  "
            + str(params["local_nodes_y"] - 1)
            + "   \n"
        )
        imswfile.write(
            "$***************************** MODEL OUTPUT LOCATIONS***************************************\n"
        )
        imswfile.write("NESTOUT '" + id_ + ".dat" + "' '" + id_ + ".bnd' \n")
    else:
        imswfile.write(
            "$***************************** MODEL COMPUTACIONAL GRID OUTPUT ***************************************\n"
        )
        imswfile.write(
            "BLOCK 'COMPGRID' NOHEAD '"
            + id_
            + ".mat' LAY 3  XP YP HSIGN TPS DIR QB WLEN DEPTH SETUP\n"
        )
    imswfile.write(
        "$***************************** COMPUTATIONS ***************************************\n"
    )
    imswfile.write("COMPUTE\n")
    imswfile.write("STOP\n")
    imswfile.close()
    return


def copla(i, index_, id_, data, params, mesh="local"):
    """[summary]

    Args:
        i ([type]): [description]
        index_ ([type]): [description]
        id_ ([type]): [description]
        data ([type]): [description]
        param ([type]): [description]
        mesh (str, optional): [description]. Defaults to 'local'.
    """

    fSwan = ldm(params["directory"] + "/" + id_ + "/" + id_ + ".mat")

    Hsig, h_CoPla = fSwan["Hsig"], fSwan["Depth"]
    Dir = (
        fSwan["Dir"] - 90
    )  # lo giro para adapatarlo al eje de Copla, los ejes de copla son Xc = -Ys, Yc = Xs (c: copla, s:swan)
    nr, mr = params[mesh + "_nodes_x"], params[mesh + "_nodes_y"]
    xp, yp = (
        np.arange(0, mr) * params["local_inc_x"],
        np.arange(0, nr) * params["local_inc_y"],
    )  # Confirmar que es correcto
    Hsig[np.isnan(Hsig)], Dir[np.isnan(Dir)], h_CoPla[np.isnan(h_CoPla)] = (
        1e-6,
        1e-6,
        1e-6,
    )

    fp = open(params["directory"] + "/" + id_ + "/" + id_ + "out.dat", "w")
    fp.write(str(nr) + " " + str(mr) + " 1\n")

    for col in range(0, nr - 1):
        fp.write(str(yp[col]) + " ")
    fp.write(str(yp[-1]) + "\n")

    for row in range(0, mr):
        fp.write(str(xp[row]) + "\n")
        for col in range(0, nr - 1):
            fp.write("%8.3f" % h_CoPla[row, col] + " ")
        fp.write("%8.3f" % h_CoPla[row, -1] + "\n")

        for col in range(0, nr - 1):
            fp.write("%8.3f" % (Hsig[row, col] / 2) + " ")
        fp.write("%8.3f" % (Hsig[row, -1] / 2) + "\n")

        if row != 0:
            for col in range(0, nr - 1):
                fp.write("%8.3f" % (Dir[row, col]) + " ")
            fp.write("%8.3f" % (Dir[row, -1]) + "\n")
    fp.close()

    fp = open(params["directory"] + "/" + id_ + "/CLAVE.DAT", "w")
    fp.write(id_ + "\n")
    fp.close()

    # Escritura de fichero clavein
    fp = open(params["directory"] + "/" + id_ + "/" + id_ + "in.dat", "w")
    htol, nd = 50, 1
    fp.write("nx ny\n")
    fp.write(str(nr) + " " + str(mr) + " 1\n")
    fp.write("iu ntype icur ibc\n")
    fp.write("  1   1   0   1\n")
    fp.write("dx dy htol\n")
    fp.write(
        str(params["local_inc_y"])
        + " "
        + str(params["local_inc_x"])
        + " "
        + str(htol)
        + "\n"
    )
    fp.write("nd\n")
    fp.write(str(nd) + "\n")
    fp.write("if1 if2 if3\n")
    fp.write("1   0   0\n")
    fp.write("iinput ioutput\n")
    fp.write("1  1\n")
    fp.write("T marea\n")
    fp.write(str(data.loc[index_, "Tp"]) + " " + str(data.loc[index_, "eta"]) + "\n")
    fp.write("amp dir(grados)\n")
    fp.write(
        str(data.loc[index_, "Hs"] / 2)
        + " "
        + str(180 - data.loc[index_, "DirM"])
        + " \n"
    )
    fp.close()

    fp = open(params["directory"] + "/" + id_ + "/" + id_ + "dat", "w")
    fp.write("*\n")
    fp.write("*\n")
    fp.write("*        FICHERO DE DATOS PARA CALIBRACION\n")
    fp.write("*\n")
    fp.write("F(2F10.3,3I5)\n")
    fp.write("*        IT      = INTERVALO DE TIEMPO\n")
    fp.write("*        ROZA    = RUGOSIDAD DE CHEZY --> 1/Mannig\n")
    fp.write("*        NT      = NUMERO DE ESCRITURAS EN FICHERO\n")
    fp.write("*        REPE    = NUMERO DE ITERACIONES ENTRE LAS ESCRITURAS\n")
    fp.write("*        IESDAO = NUMERO DE REPEs HASTA LA PRIMERA ESCRITURA\n")
    fp.write("*\n")
    fp.write("*        EN TOTAL LAS ITERACIONES SON --> ((NT-1)*REPE + IESDAO*REPE)\n")
    fp.write(
        "*        HAY QUE CUMPLIR LA CONDICION --> (NN >= (NT-1)*REPE + IESDAO*REPE)\n"
    )
    fp.write("*\n")
    fp.write("*      IT      ROZA    NT REPE IESDAO\n")
    fp.write("******.***######.###*****#####*****\n")
    fp.write("     0.500    15.000    1 1000    1\n")
    fp.write("*\n")
    fp.write("*      EDDY = FACTOR EDDY VISCOSITY\n")
    fp.write("*      CORI = FACTOR DE CORIOLIS\n")
    fp.write("*      NINTER= NUMERO ITERACIONES EN TERMINOS NO LINEALES\n")
    fp.write("F(2F10.3,I5)\n")
    fp.write("*     EDDY     CORI   NINTER\n")
    fp.write("    30.000     0.000    3\n")
    fp.write("* \n")
    fp.write("*       IANL  = TERMINOS NO LINEALES   (SI = 1)\n")
    fp.write("*       IAGUA = INUNDACION DE CELDAS   (SI = 1)\n")
    fp.write("*       ISLIP = CONTORNOS SIN FRICCION (SI = 1)\n")
    fp.write("F(3I5)\n")
    fp.write("* IANL IAGUA ISLIP\n")
    fp.write("    1    0    0\n")
    fp.write("*\n")
    fp.write("*\n")
    fp.write(
        "*      COORDENADAS DE PUNTOS DONDE SE DESEE TENER UN FICHERO EN EL TIEMPO     \n"
    )
    fp.write("*      DE SUPERFICIE LIBRE (ETA), VELOCIDAD (U), VELOCIDAD (V). \n")
    fp.write("F(I5)\n")
    fp.write("*NUMERO DE PUNTOS (MAXIMO 30 PUNTOS)\n")
    fp.write("    0\n")

    return


def directory(params, data, global_db, local_db):
    """Create the project folder with the initializated files for Swan and Copla

    Args:
        * params: the dictionary with the model paramaters
        * data: pd.DataFrame with the time series of the boundary data

    Return:
        A xr.DataSet of th project
    """
    auxiliar.mkdir(params["directory"])
    for ind_, time in enumerate(data.index):
        id_ = str(ind_ + 1).zfill(4)
        auxiliar.mkdir(params["directory"] + "/" + id_)
        save.to_txt(
            params["directory"] + "/" + id_ + "/" + id_ + "_global.dat",
            global_db["depth"].data[:, :],
            format="%9.3f",
        )
        save.to_txt(
            params["directory"] + "/" + id_ + "/" + id_ + "_local.dat",
            local_db["depth"].data[:, :],
            format="%9.3f",
        )
        swan(ind_, time, id_, data, params, mesh="global", nested=True)

    return
