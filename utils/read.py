import json
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import dates


def keys_as_int(obj: dict):
    """Convert the keys at reading json file into a dictionary of integers

    Args:
        obj (dict):input dictionary

    Returns:
        out: the dictionary
    """
    try:
        out = {int(k): v for k, v in obj.items()}
    except:
        out = {k: v for k, v in obj.items()}
    return out


def keys_as_nparray(obj: dict):
    """Convert the keys at reading json file into a dictionary of np.arrays

    Args:
        obj (dict): input dictionary

    Returns:
        out: the dictionary
    """
    if isinstance(obj, dict):
        out = {}
        for item0, level0 in obj.items():
            if isinstance(level0, dict):
                out[item0] = {}
                for item1, level1 in level0.items():
                    if isinstance(level1, dict):
                        out[item0][item1] = {}
                        for item2, level2 in level1.items():
                            try:
                                out[item0][item1][item2] = np.asarray(level2)
                            except:
                                out[item0][item1][item2] = level2

                    else:
                        try:
                            out[item0][item1] = np.asarray(level1)
                        except:
                            out[item0][item1] = level1
            else:
                try:
                    out[item0] = np.asarray(level0)
                except:
                    out[item0] = level0

    return out


def rjson(file_name: str, parType: str = None):
    """Reads data from json files

    Args:
        - fname (string): filename of data
        - parType (None or string): defines the type of data to be read.
        Defaults to read param and "td" from temporal dependency.

    Returns:
        - data (pd.DataFrame): the read data
    """
    if parType == "td":
        params = json.load(open(file_name, "r"), object_hook=keys_as_nparray)
    else:
        params = json.load(open(file_name, "r"), object_hook=keys_as_int)
    return params


def PdE(file_name: str, new: bool = False):
    """Reads data from PdE files

    Args:
        - file_name (string): filename of data
        - new (bool, False): to take into account the new files from PdE which headers
        differ from previous versions

    Returns:
        - data (pd.DataFrame): the read data
    """
    if new:
        data = pd.read_table(
            file_name,
            delimiter="\s+",
            parse_dates={"date": [0, 1, 2, 3]},
            index_col="date",
            skiprows=2,
            header=None,
            engine="python",
        )
        data.set_axis(
            [
                "Hs",
                "Tm",
                "Tp",
                "DirM",
                "Hswind",
                "DirMwind",
                "Hsswell1",
                "Tmswell1",
                "DirMswell1",
                "Hsswell2",
                "Tmswell2",
                "DirMswell2",
            ],
            axis=1,
            inplace=True,
        )
    else:
        with open(file_name) as file_:
            content = file_.readlines()

        for ind_, line_ in enumerate(content):
            if "LISTADO DE DATOS" in line_:
                skiprows = ind_ + 2

        data = pd.read_table(
            file_name,
            delimiter="\s+",
            parse_dates={"date": [0, 1, 2, 3]},
            index_col="date",
            skiprows=skiprows,
            engine="python",
        )

    data.replace(-100, np.nan, inplace=True)
    data.replace(-99.9, np.nan, inplace=True)
    data.replace(-99.99, np.nan, inplace=True)
    data.replace(-9999, np.nan, inplace=True)
    data.replace(-9999.9, np.nan, inplace=True)
    return data


def csv(
    file_name: str,
    ts: bool = False,
    date_format=None,
    sep: str = ",",
    encoding: str = "utf-8",
    non_natural_date: bool = False,
    no_data_values: int = -999,
):
    """Reads a csv file

    Args:
        - file_name (string): filename of data
        - ts (boolean, optional): stands for a time series
        (the index is a datetime index) or not
        - date_parser: required for special datetime formats
        - sep: separator
        - encoding: type of encoding
        - non_natural days: some models return 30 days/month which generate problems with real timeseries
        - no_data_values: integer with values that will be considered as nan

    Returns:
        - data (pd.DataFrame): the read data
    """
    from loguru import logger

    if not any(item in str(file_name) for item in ["dat", "txt", "csv", "zip"]):
        filename = str(file_name) + ".csv"
    else:
        filename = str(file_name)

    if non_natural_date:
        ts = False

    if not ts:
        if "zip" in filename:
            data = pd.read_csv(
                filename,
                sep=sep,
                index_col=[0],
                compression="zip",
                engine="python",
            )
        else:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    index_col=[0],
                    encoding=encoding,
                )
            except:
                data = pd.read_csv(
                    filename, sep=sep, engine="python", encoding=encoding
                )

        if non_natural_date:
            start = pd.to_datetime(data.index[0])
            # days = timedelta(np.arange(len(data)))
            index_ = [
                start + timedelta(nodays)
                for nodays in np.arange(len(data), dtype=np.float64)
            ]
            data.index = index_
    else:
        if "zip" in filename:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    compression="zip",
                    date_format=date_format,
                )
            except:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    date_format=date_format,
                )
                logger.info("{}, It is not a zip file.".format(str(filename) + ".csv"))
        else:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=["date"],
                    index_col=["date"],
                    date_format=date_format,
                )
            except:
                if date_format == None:
                    data = pd.read_csv(
                        filename,
                        sep=sep,
                        parse_dates=[0],
                        index_col=[0],
                    )
                else:
                    data = pd.read_csv(
                        filename,
                        sep=sep,
                        parse_dates=[0],
                        index_col=[0],
                        date_format=date_format,
                    )
    data = data[data != no_data_values]
    return data


def npy(file_name: str):
    """Reads data from a numpy file

    Args:
        - file_name (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    try:
        data = np.load(f"{file_name}.npy")
    except:
        data = np.load(f"{file_name}.npy", allow_pickle=True)
        if not isinstance(data, pd.DataFrame):
            data = {i: data.item().get(i) for i in data.item()}

    return data


def xlsx(file_name: str, sheet_name: str = 0):
    """Reads xlsx files

    Args:
        - file_name (string): filename of data
        - sheet_name (string): name of sheet

    Returns:
        - data (pd.DataFrame): the read data
    """
    xlsx = pd.ExcelFile(file_name + ".xlsx")
    data = pd.read_excel(xlsx, sheet_name=sheet_name, index_col=0)
    return data


def netcdf(
    file_name: str,
    variables: str = None,
    latlon: list = None,
    depth: float = None,
    time_series: bool = True,
    glob: bool = False,
):
    """Reads netCDF4 files

    Args:
        - fname (string): filename of data or directory where glob will
        be applied (required command glob True)
        - variables (list): strings with the name of the objective variables

    Returns:
        - df (pd.DataFrame): the read data
    """

    import xarray as xr

    # data = Dataset(fname + '.nc', mode='r', format='NETCDF4')
    if not glob:
        data = xr.open_dataset(file_name + ".nc")
    else:
        try:
            data = xr.open_mfdataset(file_name)
        except:
            # try:
            #     files_ = glob2.glob(fname)
            raise ValueError(
                "NetCDF files are not consistents."
                "Some variables are not adequately saved."
                "Glob version cannot be used for this dataset."
            )

    if isinstance(latlon, list):
        try:
            import xrutils as xru
        except:
            raise ValueError("xrutils package is required. Clone it from gdfa.ugr.es")

        if not depth:
            data = xru.nearest(data, latlon[0], latlon[1])
            # data = data.sel(
            #     rlon=lonlat[0], rlat=lonlat[1], method="nearest"
            # ).to_dataframe()
            data = data.to_dataframe()

            try:
                data = data.loc[0]
                data.index = pd.to_datetime(data.index)
            except:
                print("Data has not bounds.")

            nearestLonLat = data.longitude.values[0], data.latitude.values[0]
        else:
            data = data.sel(
                depth=0.494025,
                longitude=latlon[0],
                latitude=latlon[1],
                method="nearest",
            ).to_dataframe()
        print("Nearest lon-lat point: ", nearestLonLat)
        if variables is not None:
            if len(variables) == 1:
                data = data[[variables]]
            else:
                data = data[variables]
        # data.index = data.to_datetimeindex(unsafe=False)
    else:
        # TODO: hacer el nearest en otra funcion
        # df = data.to_dataframe()  # TODO: change with more than one variable
        if time_series:
            if not data.indexes["time"].dtype.name == "datetime64[ns]":
                times, goodPoints = [], []
                for index, time in enumerate(data.indexes["time"]):
                    try:
                        times.append(time._to_real_datetime())
                        goodPoints.append(index)
                    except:
                        continue
                if variables is not None:
                    data = pd.DataFrame(
                        data.to_dataframe()[variables].values[goodPoints],
                        index=times,
                        columns=[variables],
                    )
                else:
                    pd.DataFrame(data.to_dataframe().values[goodPoints], index=times)
            else:
                if isinstance(data, xr.core.dataset.Dataset):
                    values_ = np.squeeze(data[variables].values)
                    data = pd.DataFrame(
                        values_,
                        index=data.indexes["time"],
                        columns=[variables],
                    )
                else:
                    data = pd.DataFrame(
                        data[variables].data,
                        index=data.indexes["time"],
                        columns=[variables],
                    )

    return data


def asci_tiff(fname, type_="row"):
    """Reads ASCII or tiff files

    Args:
        - fname (string): filename of data or directory where glob will be applied (required command glob True)
        - variables (list): strings with the name of the objective variables

    Returns:
        - dout (pd.DataFrame or dict): the read data
    """
    import rasterio

    data = rasterio.open(fname)
    z = data.read()[0, :, :]

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    T0 = data.transform
    x, y = T0 * (cols, rows)

    if type_ == "row":
        dout = pd.DataFrame(
            np.vstack(
                [
                    np.asarray(x).flatten(),
                    np.asarray(y).flatten(),
                    np.asarray(z).flatten(),
                ]
            ).T,
            columns=["x", "y", "z"],
        ).drop_duplicates(subset=["x", "y"])
        # dout.replace(data._nodatavals[0], np.nan, inplace=True)
        # dout.dropna(inplace=True)
    else:
        dout = {"x": x, "y": y, "z": z}
        # dout["z"][z == data._nodatavals[0]] = np.nan

    return dout


def kmz(fname, joint=False):
    """[summary]

    Args:
        fname ([type]): [description]

    Returns:
        [type]: [description]
    """

    from zipfile import ZipFile

    from lxml import etree

    if fname.endswith("kmz"):
        kmz = ZipFile(fname, "r")
        kml_file = kmz.open("doc.kml", "r")
    else:
        kml_file = open(fname, "r")

    try:
        # Se procesa el fichero kml
        tree = etree.parse(kml_file)
        places = tree.xpath(
            ".//kml:Placemark", namespaces={"kml": "http://www.opengis.net/kml/2.2"}
        )

        # Se intenta encontrar donde se encuentra almacenada la cota
        place = places[0]

        try:
            level_detection_method = 1
            float(
                place.xpath(
                    ".//kml:SimpleData[@name='ELEVATION']",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
            )
        except:
            try:
                level_detection_method = 2
                cdata = place.xpath(
                    ".//kml:description",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
                cdata_root = etree.HTML(cdata)
                float(cdata_root.xpath("/html/body/table/tr/td[2]")[0].text)
            except:
                try:
                    level_detection_method = 3
                    float(
                        place.xpath(
                            ".//kml:SimpleData[@name='Elevation']",
                            namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                        )[0].text
                    )
                except:
                    level_detection_method = -1
                    pass

        x, y, z = [], [], []
        # Se genera cada uno de los sitios del kmz
        for id_, place in enumerate(places):
            if level_detection_method == 1:
                c = float(
                    place.xpath(
                        ".//kml:SimpleData[@name='ELEVATION']",
                        namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                    )[0].text
                )
            elif level_detection_method == 2:
                cdata = place.xpath(
                    ".//kml:description",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
                cdata_root = etree.HTML(cdata)
                c = float(cdata_root.xpath("/html/body/table/tr/td[2]")[0].text)
            elif level_detection_method == 3:
                c = float(
                    place.xpath(
                        ".//kml:SimpleData[@name='Elevation']",
                        namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                    )[0].text
                )
            else:
                c = "0"

            latlong = place.xpath(
                ".//kml:LineString/kml:coordinates",
                namespaces={"kml": "http://www.opengis.net/kml/2.2"},
            )
            coordinates = latlong[0].text.strip().split(" ")
            if joint:
                z.extend(np.ones(len(coordinates)) * float(c))
            else:
                z.append([])
                z[id_] = np.ones(len(coordinates)) * float(c)

            # Se crea una entrada para cada una de las coordenadas de la polilinea
            if joint:
                for coordinate in coordinates:
                    c = coordinate.split(",")
                    x.append(float(c[0]))
                    y.append(float(c[1]))
            else:
                # Se crea una entrada para cada una de las coordenadas de la polilinea
                x.append([])
                y.append([])
                for coordinate in coordinates:
                    c = coordinate.split(",")
                    x[id_].append(float(c[0]))
                    y[id_].append(float(c[1]))

    except:
        sys.stderr.write("ERROR")
        sys.exit(-1)

    if joint:
        data = pd.DataFrame(
            np.vstack([np.asarray(x), np.asarray(y), np.asarray(z)]).T,
            columns=["x", "y", "z"],
        ).drop_duplicates(subset=["x", "y"])
    else:
        data = x, y, z

    return data


def shp(fname: str, joint: bool = False, var_: str = None):
    """Read a shapefile using geopandas. Return a list with all the elements on shapefile.

    Args:
        - fname (str): file name
        - joint (bool, optional): reduce all shapes readed to a one dataframe
        - var_ (str, optional): look for variable name and the location coordinates
    """
    import geopandas as gpd

    shape_file = gpd.read_file(str(fname))

    no_elements = len(shape_file)

    if var_ is not None:
        extra_var = []

    xy, data = [], []

    k, element = 0, 0
    while k < no_elements:
        if not shape_file.geometry[k] == None:
            type_ = shape_file.geometry[k].geom_type
            if type_ == "Point":
                xy.append(np.asarray(shape_file["geometry"][k].coords.xy).T)
            elif type_ == "Polygon":
                xy.append(np.asarray(shape_file["geometry"][k].exterior.coords.xy).T)
            elif type_ == "Multipoints":
                xy.append(
                    np.asarray(
                        shape_file.apply(
                            lambda x: [y for y in x["geometry"].exterior.coords], axis=0
                        )[k].T
                    )
                )
            elif type_ == "LineString":
                xy.append(np.asarray(shape_file["geometry"][k].coords.xy).T)
                # xy.append(
                #     np.asarray(
                #         shape_file.apply(
                #             lambda x: [y for y in x["geometry"][k].coords.xy], axis=1
                #         )
                #     )
                # )
                if var_ is not None:
                    extra_var.append(shape_file[var_][k])

            elif type_ == "MultiLineString":
                for linestring_ in shape_file["geometry"][k]:
                    xy.append(np.asarray(linestring_.coords.xy).T)

            elif type_ == "MultiPolygon":
                for k_, polygon_ in enumerate(shape_file["geometry"][k]):
                    if not polygon_ == None:
                        if polygon_ == "Polygon":
                            xy.append(np.asarray(polygon_.exterior.coords.xy).T)
                        elif polygon_.geom_type == "linearRing":
                            xy.append(
                                np.asarray(
                                    shape_file.apply(
                                        lambda x: [y for y in polygon_.coords],
                                        axis=1,
                                    )[k_]
                                )
                            )
            else:
                raise ValueError(
                    "Shapefile can not be readed with methods coords or exterior.coords or exterior.coords.xy"
                )

            element += 1
        k += 1

    for k, element in enumerate(xy):
        if var_ is not None:
            data.append(
                pd.DataFrame(
                    np.vstack(
                        [
                            element[:, 0],
                            element[:, 1],
                            np.ones(len(element)) * extra_var[k],
                        ]
                    ).T,
                    columns=["x", "y", var_],
                )
            )
        else:
            data.append(
                pd.DataFrame(
                    np.vstack([element[:, 0], element[:, 1]]).T,
                    columns=["x", "y"],
                )
            )

    if joint:
        data = pd.concat([element for element in data], ignore_index=True)

    if isinstance(data, list) & (len(data) == 1):
        data = data[0]

    return data


def mat(file_name: str, var_: str = "x", julian: bool = False):
    """[summary]

    Args:
        file_name ([type]): [description]
        var_ (str, optional): [description]. Defaults to "x".
        julian (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    from scipy.io import loadmat as ldm

    data = ldm(file_name)
    if not julian:
        date = data[var_][:, 0] + dates.date2num(
            np.datetime64("0000-12-31")
        )  # Added in matplotlib 3.3
        date = [dates.num2date(i - 366, tz=None) for i in date]
    else:
        date = data[var_][:, 0]

    df = pd.DataFrame({"Q": data[var_][:, 1]}, index=date)
    df.index = df.index.tz_localize(None)
    return df


def pdf(fileName, encoding="latin-1", table=False, guess=False, area=None):
    """_summary_

    Args:
        fileName (_type_): _description_
        encoding (str, optional): _description_. Defaults to "latin-1".
        table (bool, optional): _description_. Defaults to False.
        area (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if not table:
        import PyPDF2

        reader = PyPDF2.PdfFileReader(fileName)
        n_pages = len(pdf_reader.pages)
        # TODO: habilitar la lectura multipágina
        logger.info(
            "The file contains " + str(n_pages) + ". Obtaining just the first page."
        )
        page = reader.getPage(0)
        data = page.extractText()
    else:
        from tabula import read_pdf

        data = read_pdf(
            fileName,
            guess=guess,
            pages=1,
            stream=True,
            encoding=encoding,
            area=area,
        )

    return data
