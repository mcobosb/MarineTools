import numpy as np


def nearest(
    ds, lat, lon, lat_name=None, lon_name=None, variable_mask=None, time_mask=0
):
    """Find the index of the nearest corresponding coords."""
    if not lat_name or not lon_name:
        lat_name, lon_name = coords_name(ds)

    lats, lons = latslons_values(ds, lat_name, lon_name)

    if variable_mask:
        mask = np.isnan(ds[variable_mask].isel(time=time_mask).values)

        lats = np.ma.masked_array(lats, mask)
        lons = np.ma.masked_array(lons, mask)

    ilat, ilon = find_indexes(lats, lons, lat, lon)

    if "rlat" in ds.dims and "rlon" in ds.dims:
        poi = ds.isel(rlat=ilat, rlon=ilon)
    elif lat_name in ds.dims and lon_name in ds.dims:
        poi = ds.isel(**{lat_name: ilat, lon_name: ilon})
    elif not ilat and not ilon:
        poi = ds
    else:
        raise Exception("Dimensions coordinates not found in dataset")

    return poi


def latslons_values(ds, lat_name, lon_name):
    if len(ds[lat_name].dims) == 1 and len(ds[lon_name] == 1):
        lats, lons = create_lat_lon_matrix(ds[lat_name].values, ds[lon_name].values)
    else:
        lats, lons = ds[lat_name].values, ds[lon_name].values

    return lats, lons


def find_indexes(latvar, lonvar, lat0, lon0):
    """Find the index of the nearest corresponding coords.

    Use the length of a tunnel through the Earth between two points as distance.

    References:
        https://www.unidata.ucar.edu/blogs/developer/en/entry/accessing_netcdf_data_by_coordinates
    """
    indexes = None, None
    if latvar.size > 1 and lonvar.size > 1:
        import math

        rad_factor = np.pi / 180.0

        latvals = latvar * rad_factor
        lonvals = lonvar * rad_factor
        lat_rad = lat0 * rad_factor
        lon_rad = lon0 * rad_factor

        clat, clon = np.cos(latvals), np.cos(lonvals)
        slat, slon = np.sin(latvals), np.sin(lonvals)
        delX = np.cos(lat_rad) * np.cos(lon_rad) - clat * clon
        delY = np.cos(lat_rad) * np.sin(lon_rad) - clat * slon
        delZ = np.sin(lat_rad) - slat
        dist_sq = delX**2 + delY**2 + delZ**2
        minindex_1d = dist_sq.argmin()

        indexes = np.unravel_index(minindex_1d, latvals.shape)

    return indexes


def create_lat_lon_matrix(lat, lon):
    """Create a coords matrix from two coords vectors."""
    lat_matrix = np.tile(lat, (lon.shape[0], 1)).T
    lon_matrix = np.tile(lon, (lat.shape[0], 1))

    return lat_matrix, lon_matrix


def coords_name(ds):
    if "lat" in ds.coords and "lon" in ds.coords:
        lat_name = "lat"
        lon_name = "lon"
    elif "latitude" in ds.coords and "longitude" in ds.coords:
        lat_name = "latitude"
        lon_name = "longitude"
    else:
        raise Exception("Latitudes and longitudes not found in dataset")

    return lat_name, lon_name
