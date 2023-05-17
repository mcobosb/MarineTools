from loguru import logger
import os


def andalusian_coast_analysis(
    path_: str,
    var_: str,
    analysis: str,
    model: str,
    lat: float,
    lon: float,
    period: str = "1",
):
    """Return the parameters of the Andalusian wave climate analysis showed in "Parametric Characterization of Wave Climate along the Andalusian Coast for Non-Stationary Stochastic Simulation"
    Cobos et al. (2022)

    Args:
        path_ (str): _description_
        var_ (str): _description_
        analysis (str): _description_
        model (str): _description_
        lat (float): _description_
        lon (float): _description_
        period (str, optional): _description_. Defaults to "1".
    """

    # Check the path is correct
    if not os.path.exists(path_):
        logger.info(
            "1. Library of results from Zenodo (www.) was not properly download or linked. Ensure the path from the current working folder ({}) to the library ({}) is correct.".format(
                os.getcwd(), path_
            )
        )
    else:
        logger.info("1. Library properly download and link.")
        info = read.xlsx(facade + "/info_" + var_ + "_" + facade)

    # Check latitude
    if (lat > -7.5) & (lat < -5.7):
        logger.info("1. Location at Atlantic Andalusian facade selected.")
        facade = "atlantic"
    elif (lat > -5.7) & (lat < -1.416):
        logger.info("1. Location at Mediterranean Andalusian facade selected.")
        facade = "mediterranean"
    else:
        logger.info("1. Latitude is outside of the study area (7.5W - 1.416W).")

    # Check longitude
    if (lon < 35.16) | (lon > 37.5):
        logger.info("2. Longitude is outside of the study area (35.16N - 37.5N).")
    else:
        logger.info("2. Longitude inside the study area.")

    # Check if model is available
    if facade == "atlantic":
        if model is not ("ACCE", "CMCC", "CNRM", "GFDL", "HADG", "IPSL", "MIRO"):
            logger.info(
                "3. Model selected is not available. Available models are: ACCE, CMCC, CNRM, GFDL, HADG, IPSL or MIRO."
            )
        else:
            logger.info("3. Model selected is available.")
    elif facade == "mediterranean":
        if model is not (
            "EART",
            "ESM2",
            "CNRM",
            "HADG",
            "IPSL",
            "MECD",
            "MIRO",
            "MPIE",
        ):
            logger.info(
                "3. Model selected is not available. Available models are: EART, ESM2, CNRM, HADG, IPSL, MECD, MIRO or MPIE."
            )
        else:
            logger.info("3. Model selected is available.")

    # Check period
    if period is not ("1", "2"):
        logger.info(
            "4. Period selected is not available. Periods available are: 1 (2026 - 2045), or 2 (2081 - 2100)."
        )
    else:
        logger.info("4. Period selected is available.")

    # Check the type of analysis
    if analysis is not ("marginal_fit", "dependency"):
        logger.info(
            "5. Parameters of that analysis are not available. Parameters available are: marginal_fit or dependency."
        )
    else:
        logger.info("5. Parameters for {} analysis are available.".format(analysis))

    # Check variable
    if var_ is not ("Hs"):
        logger.info(
            "6. Variable selected is not available. Variables available are: Hs (significant wave height)."
        )
    else:
        logger.info("6. Variable selected is available.")

    return parameters
