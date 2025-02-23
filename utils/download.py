import asyncio
import math
import os
import urllib.request

import numpy as np
from marinetools.utils import auxiliar
from PIL import Image


def wget2queries(fname, path=""):
    with open(fname, "r") as file_:
        data = file_.read()
        data = data.split("EOF--dataset.file.url.chksum_type.chksum")[1]
        data = data.split("\n")[1:-1]

    data = [line_.split(" ")[0].replace("'", "").split("_")[:-1] for line_ in data]
    separator = "_"
    data = [separator.join(line_) for line_ in data]
    filenames_ = list(dict.fromkeys(data))

    vars_ = [line_.split("_")[0] for line_ in filenames_]
    domains_ = [line_.split("_")[1] for line_ in filenames_]
    models_ = [line_.split("_")[2] for line_ in filenames_]
    experiments_ = [line_.split("_")[3] for line_ in filenames_]
    ensembles_ = [line_.split("_")[4] for line_ in filenames_]
    rcms_ = [line_.split("_")[5] for line_ in filenames_]
    downscaling_ = [line_.split("_")[6] for line_ in filenames_]
    freqs_ = [line_.split("_")[7] for line_ in filenames_]

    queries_ = {}
    for ind_, filename_ in enumerate(filenames_):
        queries_[ind_] = {
            "filename": path + "/" + filename_,
            "query": {
                "project": "CORDEX",
                "variable": vars_[ind_],
                "time_frequency": freqs_[ind_],
                "domain": domains_[ind_],
                "experiment": experiments_[ind_],
                "ensemble": ensembles_[ind_],
                "rcm_version": downscaling_[ind_],
                "driving_model": models_[ind_],
                "institute": rcms_[ind_].split("-")[0],
            },
        }
    return queries_


async def server_query(query_, credentials, point=None, region=None):
    import xrutils as xru

    try:
        ds, _ = xru.cordex(
            query_["query"],
            openid="https://esg-dn1.nsc.liu.se/esgf-idp/openid/Manuel",
            password="0NJ#\F3@qu$,&\?7C,Z",
            pydap=True,
            bootstrap=True,
        )
        # ds.to_netcdf()
        await asyncio.sleep(1)
        if point is not None:
            ds_oi = xru.nearest(ds, point["lat"], point["lon"])
            ds_oi.to_netcdf(
                query_["filename"] + "_" + point["lat"] + "_" + point["lon"] + ".nc"
            )
        elif region is not None:
            ds_oi = xru.subregion(ds, region["lat"], region["lon"])
            ds_oi.to_netcdf(
                query_["filename"]
                + "_"
                + region["lat"][0]
                + "_"
                + region["lat"][1]
                + "_"
                + region["lon"][0]
                + "_"
                + region["lon"][1]
                + ".nc"
            )

    except:
        print("Download failed: " + query_["filename"])
    return


def cordex_data(fname, credentials, path="", point=None, region=None):
    from loguru import logger

    if path != "":
        auxiliar.mkdir(path)

    queries_ = wget2queries(fname, path)

    loop = asyncio.get_event_loop()
    tasks = []
    if point is not None:
        point_ = {}
        for coord in point.itertuples():
            point_["lat"] = coord.lat
            point_["lon"] = coord.lon

            for query_ in queries_:
                logger.info("Downloading {}".format(queries_[query_]["filename"]))
                tasks.append(server_query(queries_[query_], credentials, point=point_))

    elif region is not None:
        region_ = {}
        for coord in region.itertuples():
            region["lat"] = coord.lat
            region["lon"] = coord.lon

            for query_ in queries_:
                logger.info("Downloading {}".format(queries_[query_]["filename"]))
                tasks.append(
                    server_query(queries_[query_], credentials, region=region_)
                )
    else:
        raise ValueError("Neither point or region are given. One of them is required")

    tasks, _ = loop.run_until_complete(asyncio.wait(tasks))
    return


class GoogleMapsLayers:
    ROADMAP = "v"
    TERRAIN = "p"
    ALTERED_ROADMAP = "r"
    SATELLITE = "s"
    TERRAIN_ONLY = "t"
    HYBRID = "y"


class GoogleMapDownloader:
    """
    A class which generates high resolution google maps images given
    a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12, layer=GoogleMapsLayers.ROADMAP):
        """
        GoogleMapDownloader Constructor
        Args:
            lat:    The latitude of the location required
            lng:    The longitude of the location required
            zoom:   The zoom level of the location required, ranges from 0 - 23
                    defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def getXY(self):
        """
        Generates an X,Y tile coordinate based on the latitude, longitude
        and zoom level
        Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (
            (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size
        )

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = (
            (
                (tile_size / 2)
                + 0.5
                * math.log((1 + sin_y) / (1 - sin_y))
                * -(tile_size / (2 * math.pi))
            )
            * numTiles
            // tile_size
        )

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
        Generates an image by stitching a number of google map tiles together.
        Args:
            start_x:        The top-left x-tile coordinate
            start_y:        The top-left y-tile coordinate
            tile_width:     The number of tiles wide the image should be -
                            defaults to 5
            tile_height:    The number of tiles high the image should be -
                            defaults to 5
        Returns:
            A high-resolution Google Map image.
        """

        start_x = kwargs.get("start_x", None)
        start_y = kwargs.get("start_y", None)
        tile_width = kwargs.get("tile_width", 5)
        tile_height = kwargs.get("tile_height", 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new("RGB", (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                url = (
                    f"https://mt0.google.com/vt?lyrs={self._layer}&x="
                    + str(start_x + x)
                    + "&y="
                    + str(start_y + y)
                    + "&z="
                    + str(self._zoom)
                )

                current_tile = str(x) + "-" + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img


def googleImage(lat=35, lon=0, zoom=13):
    # Create a new instance of GoogleMap Downloader
    lat, lon, zoom = 30, 0, 13
    gmd = GoogleMapDownloader(lat, lon, zoom, GoogleMapsLayers.SATELLITE)

    print("The tile coorindates are {}".format(gmd.getXY()))

    try:
        # Get the high resolution image
        img = gmd.generateImage()
    except IOError:
        print(
            "Could not generate the image - try adjusting the zoom level and checking your coordinates"
        )
    else:
        # Save the image to disk
        img.save("high_resolution_image.png")
        print("The map has successfully been created")


# Creation of urls for static bing maps
# url = f'https://bing.com/maps/default.aspx?cp=43.901683~-69.522416&lvl=12&style=r'


def openstreetImages(lon, lat, distx, disty, sitename, style: str = "satellite"):

    # style can be 'map' or 'satellite'
    # for style in ['map','satellite']:

    osm_image(lon, lat, sitename=sitename, style=style, distx=distx, disty=disty)

    return


##########################################################################


def osm_image(lon, lat, sitename="Columbo", style="satellite", distx=500, disty=500):
    """This function makes OpenStreetMap satellite or map image with circle and random points.
    Change np.random.seed() number to produce different (reproducable) random patterns of points.
    Also review 'scale' variable"""

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    projpath = "."

    if style == "map":
        ## MAP STYLE
        cimgt.OSM.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        print("no valid style")

    # stroke = [pe.Stroke(linewidth=1, foreground='w'), pe.Normal()]

    ############################################################################

    plt.close("all")
    fig = plt.figure(figsize=(10, 10))  # open matplotlib figure
    ax = plt.axes(
        projection=img.crs
    )  # project using coordinate reference system (CRS) of street map
    data_crs = ccrs.PlateCarree()
    data_crs = ccrs.UTM(zone=30)

    ax.set_title(f"{sitename} ({lat},{lon})", fontsize=15)

    # auto-calculate scale
    radius = np.max([distx, disty])
    scale = int(120 / np.log(radius))
    scale = (scale < 20) and scale or 19

    # or change scale manually
    # NOTE: scale specifications should be selected based on radius
    # but be careful not have both large scale (>16) and large radius (>1000),
    #  it is forbidden under [OSM policies](https://operations.osmfoundation.org/policies/tiles/)
    # -- 2     = coarse image, select for worldwide or continental scales
    # -- 4-6   = medium coarseness, select for countries and larger states
    # -- 6-10  = medium fineness, select for smaller states, regions, and cities
    # -- 10-12 = fine image, select for city boundaries and zip codes
    # -- 14+   = extremely fine image, select for roads, blocks, buildings

    extent = calc_extent(lon, lat, distx, disty)
    ax.set_extent(extent)  # set extents
    ax.add_image(img, int(scale))  # add OSM with zoom specification

    # add site
    # ax.plot(lon,lat, color='black', marker='x', ms=7, mew=3, transform=data_crs)
    # ax.plot(lon,lat, color='red', marker='x', ms=6, mew=2, transform=data_crs)

    # if npoints>0:
    #     # set random azimuth angles (seed for reproducablity)
    #     np.random.seed(1235)
    #     rand_azimuths_deg = np.random.random(npoints)*360

    #     # set random distances (seed for reproducablity)
    #     np.random.seed(6341)
    #     rand_distances = radius*np.sqrt((np.random.random(npoints))) # np.random.uniform(low=0, high=radius, size=npoints)

    #     rand_lon = cgeo.Geodesic().direct((lon,lat),rand_azimuths_deg,rand_distances)[:,0]
    #     rand_lat = cgeo.Geodesic().direct((lon,lat),rand_azimuths_deg,rand_distances)[:,1]

    #     ax.plot(rand_lon,rand_lat,color='black',lw=0,marker='x',ms=4.5,mew=1.0,transform=data_crs)
    #     ax.plot(rand_lon,rand_lat,color='yellow',lw=0,marker='x',ms=4,mew=0.5,transform=data_crs)

    # add cartopy geodesic circle
    # circle_points = cgeo.Geodesic().circle(lon=lon, lat=lat, radius=radius)
    # geom = shapely.geometry.Polygon(circle_points)
    # ax.add_geometries((geom,), crs=ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=2.5)

    # radius_text = cgeo.Geodesic().direct(points=(lon,lat),azimuths=30,distances=radius)[:,0:2][0]
    # stroke = [pe.Stroke(linewidth=5, foreground='w'), pe.Normal()]
    # ax.text(radius_text[0],radius_text[1],f'r={radius} m', color='red',
    #     fontsize=8, ha='left',va='bottom', path_effects=stroke, transform=data_crs)

    gl = ax.gridlines(draw_labels=True, crs=data_crs, color="k", lw=0.5)

    gl.top_labels = False
    gl.right_labels = False
    if 0:
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    plt.show()

    # fig.savefig(f'{projpath}/{sitename}_{style}_r{radius}_pts{npoints}_scale{scale}.jpg', dpi=150, bbox_inches='tight')

    return


def calc_extent(lon, lat, distx, disty):
    """This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    """

    import cartopy.geodesic as cgeo

    angle = np.rad2deg(np.arctan(disty / distx))

    dist_cnr = np.sqrt(distx**2 + disty**2)
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-(90 - angle), distances=dist_cnr
    )[:, 0:2][0]
    bot_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=90 + angle, distances=dist_cnr
    )[:, 0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent


def image_spoof(self, tile):
    """this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """

    import io
    from urllib.request import Request, urlopen

    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header("User-agent", "Anaconda 3")  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), "lower"  # reformat for cartopy
