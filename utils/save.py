import json

import numpy as np
import pandas as pd


def npy2json(params: dict):
    """Convert a dictionary with numpy ndarray into json dictionary and save the file

    Args:
        params (dict): parameters to be transformed into json
    """
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            params[key] = list(params[key])

    for loc, mode in enumerate(params["mode"]):
        params["mode"][loc] = int(mode)

    if "all" in params.keys():
        for loc, mode in enumerate(params["all"]):
            params["all"][loc] = [str(mode[0]), float(mode[1]), mode[2].tolist()]

    for loc, fun in enumerate(params["fun"]):
        if not isinstance(fun, str):
            params["fun"][loc] = params["fun"][loc].name

    to_json(params, params["fname"])

    return


def to_json(params: dict, file_name: str, npArraySerialization: bool = False):
    """Saves to a json file

    Args:
        - params (dict): data to be saved
        - file_name (string): path of the file
        - npArraySerialization (bool): applied a serialization. True or False.

    Return:
        - None
    """

    with open(f"{str(file_name)}.json", "w") as f:
        if npArraySerialization:
            for key in params.keys():
                if isinstance(params[key], dict):
                    for subkey in params[key].keys():
                        try:
                            params[key][subkey] = params[key][subkey].tolist()
                        except:
                            None
                else:
                    try:
                        params[key] = params[key].tolist()
                    except:
                        None
        json.dump(params, f, ensure_ascii=False, indent=4)

    return


def to_csv(data: pd.DataFrame, file_name: str, compression: str = "infer"):
    """Saves to a kind of csv file

    Args:
        - data (pd.DataFrame): data to be saved
        - file_name (str): path of the file
        - compression (str, opt): define the type of compression if required

    Return:
        - None
    """
    if ".zip" in file_name:
        data.to_csv(file_name, compression="zip")
    else:
        data.to_csv(file_name, compression=compression)

    return


def to_npy(data: np.ndarray, file_name: str):
    """Saves to a numpy file

    Args:
        - params (dict): data to be saved
        - fname (string): path of the file

    Return:
        - None
    """
    np.save(f"{str(file_name)}.npy", data)
    return


def to_xlsx(data: pd.DataFrame, file_name: str):
    """Saves to an excel file

    Args:
        - params (dict): data to be saved
        - file_name (string): path of the file

    Return:
        - None
    """

    wbook, wsheet = cwriter(str(file_name))

    # Writting the header
    if data.index.name is not None:
        wsheet.write(0, 0, data.index.name, formats(wbook, "header"))
    else:
        wsheet.write(0, 0, "Index", formats(wbook, "header"))

    for col_num, value in enumerate(data.columns.values):
        wsheet.write(0, col_num + 1, value, formats(wbook, "header"))

    # Adding data
    k = 1
    for i in data.index:
        if k % 2 == 0:
            fmt = "even"
        else:
            fmt = "odd"
        wsheet.write_row(k, 0, np.append(i, data.loc[i, :]), formats(wbook, fmt))
        k += 1

    wbook.close()
    return


def cwriter(file_out: str):
    """Create a new file with the book and sheet for excel

    Args:
        - file_out (string): path of the file

    Returns:
        - wbook (objects): excel book
        - wsheet (objects): excel sheet
    """
    writer = pd.ExcelWriter(
        file_out,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}},
    )
    df = pd.DataFrame([0])
    df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=1, header=False)
    wsheet = writer.sheets["Sheet1"]
    wbook = writer.book
    return wbook, wsheet


def formats(wbook, style):
    """Gives some formats to excel file

    Args:
        - wbook (object): excel book
        - style (string): name of the style

    Returns:
        - Adds the format to the woorkbook
    """
    fmt = {
        "header": {
            "bold": True,
            "text_wrap": True,
            "valign": "center",
            "font_color": "#ffffff",
            "fg_color": "#5983B0",
            "border": 1,
        },
        "even": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#DEE6EF",
            "border": 1,
        },
        "odd": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#FFFFFF",
            "border": 1,
        },
    }

    return wbook.add_format(fmt[style])


def to_esriascii(data, ncols, nrows, cellsize, fname, x0=0, y0=0, nodata_value=-9999):
    """[summary]

    Args:
        param ([type]): [description]
        fname ([type]): [description]
        data ([type]): [description]
    """
    fid = open(str(fname), "w")
    fid.write("ncols    {}\n".format(ncols))
    fid.write("nrows    {}\n".format(nrows))
    fid.write("xllcorner    {}\n".format(x0))
    fid.write("yllcorner    {}\n".format(y0))
    fid.write("cellsize    {}\n".format(cellsize))
    fid.write("NODATA_value    {}\n".format(nodata_value))
    fid.close()

    with open(str(fname), "ab") as file:
        np.savetxt(file, data, fmt="%8.3f", newline="\n")
    fid.close()
    return


def as_float_bool(obj: dict):
    """Checks the value of each key of the dict passed
    Args:
        * obj (dict): The object to decode

    Returns:
        * dict: The new dictionary with changes if necessary
    """
    for keys in obj.keys():
        try:
            obj[keys] = float(obj[keys])
            if obj[keys] == np.round(obj[keys]):
                obj[keys] = int(obj[keys])
        except:
            pass

        if obj[keys] == "True":
            obj[keys] = True
        elif obj[keys] == "False":
            obj[keys] = False

    return obj


def to_geotiff(data, fname, profile, transform=None, auxiliar=None):
    """[summary]

    Args:
        data ([type]): [description]
        profile ([type]): [description]
        fname ([type]): [description]
    """
    import pyproj
    import rasterio
    from affine import Affine
    from pyproj import CRS
    from rasterio.transform import from_origin

    if profile is None:
        transform = (
            Affine.translation(auxiliar["corners"][0], auxiliar["corners"][1])
            * Affine.scale(auxiliar["dx"], auxiliar["dy"])
            * Affine.rotation(auxiliar["angle"])
        )

        profile = {
            "driver": auxiliar["driver"],
            "dtype": auxiliar["dtype"],
            "nodata": auxiliar["nodata"],
            "width": auxiliar["nodesy"],
            "height": auxiliar["nodesx"],
            "count": auxiliar["count"],
            "crs": CRS.from_epsg(auxiliar["crsno"]),
            "transform": transform,
            "tiled": False,
            "interleave": "band",
        }

    with rasterio.Env():
        profile.update(dtype=rasterio.float32, count=1, compress="lzw")

        with rasterio.open(str(fname), "w", **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
    return


def to_txt(data: pd.DataFrame, file_name: str, format: str = "%9.3f"):
    """Save data to a txt

    Args:
        * file_name (str): path where save the file
        * data (pd.DataFrame): raw data
        * format: save format
    """
    np.savetxt(str(file_name), data, delimiter="", fmt=format)
    return


def to_shp(
    fname: str,
    lon: pd.Series,
    lat: pd.Series,
    type_: str = "point",
    values: pd.Series = None,
):
    """Save data to a shapefile.

    Args:
        fname (str): file name
        lon (pd.Series or list): longitude or x position of data
        lat (pd.Series or list): latitude or y position of data
    """
    import shapefile

    iofile = shapefile.Writer(str(fname))
    iofile.field("id")

    if type_ == "point":
        if isinstance(lon, list):
            for i, j in enumerate(lon):
                iofile.point(j, lat[i])
                iofile.record(str(int(i + 1)))
        else:
            iofile.point(lon, lat)
            iofile.record("1")
    elif type_ == "multi-point":
        for ind_, lon_key in enumerate(lon):
            iofile.point(lon_key, lat[ind_])
            iofile.record(str(ind_))
    elif type_ == "line":
        coords = [[]]
        for ind_, lon_key in enumerate(lon):
            coords[0].append([lon_key, lat[ind_]])
        iofile.line(coords)
        iofile.record("1")
    elif type_ == "multi-line":
        unique_values = values.unique()
        coords = [[] for _ in unique_values]
        for k_index, k in enumerate(unique_values):
            mask = values == k
            for ind_, lon_key in enumerate(lon[mask]):
                coords[k_index].append([lon_key, lat[ind_]])
            iofile.line([coords[k_index]])
            iofile.record(str(k))
    else:
        raise ValueError(
            "Type {} not implemented. Options are point, multi-point, line or multi-line.".format(
                type_
            )
        )

    iofile.close()
    return


def to_netcdf(data: pd.DataFrame, path: str):
    """Save data to netcdf file

    Args:
        data (pd.DataFrame): raw timeseries
        path (str): path where save the file
    """
    data.to_netcdf(str(path) + ".nc")
    return
