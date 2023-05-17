import shlex
import subprocess


def cmems(
    user: str,
    pwd: str,
    variables: str,
    lat: list,
    lon: list,
    depths: list,
    period: list,
    server: str = "https://my.cmems-du.eu/motu-web/Motu",
    service_id: str = "IBI_MULTIYEAR_PHY_005_002-TDS",
    product_id: str = "cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m",
    folder: str = None,
    file_name: str = None,
):
    """Download data from Marine Copernicus Service (https://resources.marine.copernicus.eu/products)
    Recommendation: go to the web view of the Marine Copernicus Service and clik on
    subset and download, and then click on Show Api request. Copy the main properties and
    pass to this function.

    Args:
        user (str): Username of Marine Copernicus Service
        pwd (str): Password of Marine Copernicus Servic4e
        variables (str): Name of the variables
        lat (list): _description_
        lon (list): _description_
        depths (list): _description_
        period (list): _description_
        server (str, optional): _description_. Defaults to "https://my.cmems-du.eu/motu-web/Motu".
        service_id (str, optional): _description_. Defaults to "IBI_MULTIYEAR_PHY_005_002-TDS".
        product_id (str, optional): _description_. Defaults to "cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m".
        folder (str, optional): _description_. Defaults to None.
        file_name (str, optional): _description_. Defaults to None.
    """

    if not file_name:
        file_name = ""
        for var_ in variables:
            file_name += var_ + "_"

        file_name += (
            lat[0]
            + "_"
            + lon[0]
            + "_"
            + lat[1]
            + "_"
            + lon[1]
            + "_"
            + service_id
            + "_"
            + product_id
            + "_"
            + depths[0]
            + "_"
            + depths[1]
            + "_"
            + str(period[0][:4])
            + "_"
            + str(period[1][:4])
            + ".nc"
        )

    if not folder:
        folder = "."

    command_line = (
        "python -m motuclient --motu {}".format(server)
        + " --service-id {} --product-id {}".format(service_id, product_id)
        + " --longitude-min {} --longitude-max {}".format(lon[0], lon[1])
        + " --latitude-min {} --latitude-max {}".format(lat[0], lat[1])
        + " --date-min {} --date-max {}".format(period[0], period[1])
        + " --depth-min {} --depth-max {}".format(depths[0], depths[1])
    )

    for var in variables:
        command_line += " --variable {}".format(var)

    command_line += ' --out-dir {} --out-name {} --user "{}" --pwd "{}"'.format(
        folder,
        file_name,
        user,
        pwd,
    )
    args = shlex.split(command_line)
    print(command_line)

    subprocess.call(command_line, shell=True)
