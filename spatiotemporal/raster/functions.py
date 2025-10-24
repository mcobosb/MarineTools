"""
Marinetools - Statistical Functions Module

This module provides statistical and spatial analysis functions for marine data processing.
It includes utilities for grid manipulation, contour analysis, geometry processing,
and spatial interpolation specifically designed for coastal and marine applications.

The module handles:
- Bathymetric data processing and grid operations
- Shoreline and contour extraction
- Spatial interpolation and refinement
- Geometric operations for coastal features
- File I/O operations for GIS and raster data
- Statistical analysis of seasonal and temporal data

Dependencies:
    - numpy: Array operations and mathematical functions
    - pandas: Data manipulation and analysis
    - geopandas: Geospatial data operations
    - rasterio: Raster data I/O and processing
    - matplotlib: Plotting and visualization
    - scipy: Scientific computing and interpolation
    - shapely: Geometric operations
    - loguru: Logging functionality

Author: Marinetools Development Team
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Debug mode: set DEBUG_MODE=True to enable visualizations
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")


def calculate_grid_angle_and_create_rotated_mesh(xx, yy, grid_size):
    """
    Calculate grid angle from DEM data and create a rotated mesh aligned with contours.

    Calculates the orientation angle of the DEM grid and generates a new rotated
    coordinate mesh (X, Y) inscribed within the xx, yy bounds with coordinates
    aligned to the contours.

    Parameters
    ----------
    xx : array-like
        X coordinate bounds for the new mesh.
    yy : array-like
        Y coordinate bounds for the new mesh.
    grid_size : float
        Grid spacing for the new mesh.

    Returns
    -------
    X_rotated : np.ndarray
        Rotated X coordinates of the new mesh.
    Y_rotated : np.ndarray
        Rotated Y coordinates of the new mesh.
    angle : float
        Rotation angle in radians used for the mesh alignment.

    Notes
    -----
    The function automatically detects if the DEM coordinates are 1D or 2D and
    calculates the appropriate rotation angle. A memory usage warning is issued
    if the resulting mesh would be very large (> 10 million points).
    """
    # Calculate the angle of the DEM grid
    # Use corners to calculate the main orientation
    x_dem = xx
    y_dem = yy

    # Calculate edge vectors of the grid
    # Horizontal vector (first row)
    dx1 = x_dem[0, -1] - x_dem[0, 0]
    dy1 = y_dem[0, -1] - y_dem[0, 0]
    # Vertical vector (first column)
    dx2 = x_dem[-1, 0] - x_dem[0, 0]
    dy2 = y_dem[-1, 0] - y_dem[0, 0]

    # Calculate rotation angle (use the longer vector)
    if np.sqrt(dx1**2 + dy1**2) > np.sqrt(dx2**2 + dy2**2):
        angle = np.arctan2(dy1, dx1)  # horizontal axis angle
    else:
        angle = np.arctan2(dx2, dy2) - np.pi / 2  # corrected vertical axis angle

    # Bounds of xx, yy
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)

    # Domain center
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Domain dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Calculate number of points for the new mesh
    nx = int(width / grid_size) + 1
    ny = int(height / grid_size) + 1

    # Create regular mesh in local system (without rotation)
    x_local = np.linspace(-width / 2, width / 2, nx)
    y_local = np.linspace(-height / 2, height / 2, ny)
    X_local, Y_local = np.meshgrid(x_local, y_local)

    # Apply rotation
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rotated = cos_a * X_local - sin_a * Y_local + x_center
    Y_rotated = sin_a * X_local + cos_a * Y_local + y_center

    return X_rotated, Y_rotated, angle


def band(da_dem, levels, orientation_):
    """
    Create a band mask for DEM data within specified depth/elevation levels.

    Creates a mask for areas within the DEM that fall between minimum and maximum
    levels, then expands this mask in the specified orientation to create a band
    for further analysis.

    Parameters
    ----------
    da_dem : dict
        DEM data containing 'x', 'y', and 'z' arrays.
    levels : array-like
        Depth/elevation levels to define the band boundaries.
    orientation_ : str
        Band orientation, either "WE" (West-East) or "NS" (North-South).

    Returns
    -------
    band_ : np.ndarray
        Boolean mask indicating the band area.
    coords : dict
        Dictionary containing reshaped X and Y coordinate arrays for the band.

    Notes
    -----
    In debug mode (DEBUG_MODE=True), displays a visualization of the band.
    The function extends the initial mask along the specified orientation to
    create a continuous band across the domain.
    """
    # Define level bounds and create initial mask
    z_ = np.where((da_dem["z"] < levels.max()) & (da_dem["z"] > levels.min()), 1, 0)

    # Debug mode: show band visualization
    if DEBUG_MODE:
        plt.figure()
        plt.contourf(da_dem["x"], da_dem["y"], z_, levels=2, cmap="RdBu", alpha=0.5)
        plt.axis("equal")
        plt.title("Debug: Band Visualization")
        plt.show()

    # Create extended mask based on orientation
    mask = np.copy(z_)
    len_ = 0
    if orientation_ == "WE":
        # Extend mask horizontally (West-East)
        for i in range(da_dem["y"].shape[0]):
            if np.any(mask[i, :] == 1):
                mask[i, :] = 1
                len_ += 1
    elif orientation_ == "NS":
        # Extend mask vertically (North-South)
        for i in range(da_dem["y"].shape[1]):
            if np.any(mask[:, i] == 1):
                mask[:, i] = 1
                len_ += 1

    # Convert to boolean mask and extract coordinates
    band_ = mask == 1
    xx = da_dem["x"][band_]
    yy = da_dem["y"][band_]

    # Reshape coordinates according to orientation
    if orientation_ == "NS":
        xx = np.reshape(xx, (-1, len_))
        yy = np.reshape(yy, (-1, len_))
    elif orientation_ == "WE":
        xx = np.reshape(xx, (len_, -1))
        yy = np.reshape(yy, (len_, -1))

    return band_, {"X": xx, "Y": yy}


def refinement(da_dem, band_, coords):
    """
    Perform spatial interpolation to refine elevation data on a new coordinate grid.

    Uses LinearNDInterpolator to interpolate elevation values from the DEM data
    within the band area to new coordinate positions specified in coords.

    Parameters
    ----------
    da_dem : dict
        DEM data containing 'x', 'y', and 'z' arrays.
    band_ : np.ndarray
        Boolean mask indicating the area within the band for interpolation.
    coords : dict
        Dictionary containing 'X' and 'Y' arrays with target interpolation coordinates.

    Returns
    -------
    Z : np.ndarray
        Interpolated elevation/depth values at the new coordinate positions.

    Notes
    -----
    In debug mode (DEBUG_MODE=True), displays a visualization comparing the original
    DEM points with the new interpolation grid points. The interpolation uses only
    the points within the band mask to avoid extrapolation beyond the data bounds.
    """
    X, Y = coords["X"], coords["Y"]

    # Debug mode: show band visualization
    if DEBUG_MODE:
        plt.figure()
        plt.plot(da_dem["x"].flatten(), da_dem["y"].flatten(), "ob", markersize=1)
        plt.plot(X.flatten(), Y.flatten(), "xr", markersize=1)
        plt.title("Debug: DEM points (blue) vs Interpolation grid (red)")
        plt.show()

    # Create interpolator using only band points
    interp = LinearNDInterpolator(
        list(zip(da_dem["x"][band_].flatten(), da_dem["y"][band_].flatten())),
        da_dem["z"][band_].flatten(),
    )

    # Interpolate to new coordinates
    Z = interp(X, Y)
    return Z


def save2GISapp(
    line,
    crs,
    beach_name,
    dir_results_analysis_GIS,
    dir_results_analysis_app,
):
    """
    Save geometric line data to both GIS and application-specific formats.

    Saves the line geometry to two different formats: GPKG for GIS applications
    and GeoJSON for web applications, with appropriate coordinate reference system
    transformations.

    Parameters
    ----------
    line : geopandas.GeoDataFrame
        Line geometry to be saved.
    crs : str or pyproj.CRS
        Coordinate reference system for the GIS output.
    beach_name : str
        Base name for the output files.
    dir_results_analysis_GIS : str or Path
        Directory path for GIS format output (GPKG).
    dir_results_analysis_app : str or Path
        Directory path for application format output (GeoJSON).

    Returns
    -------
    None

    Notes
    -----
    The function automatically converts the CRS to EPSG:4326 for the GeoJSON output
    to ensure compatibility with web mapping applications.
    """
    # Save to GIS format
    line = line.set_crs(crs)
    line.to_file(
        Path(dir_results_analysis_GIS) / beach_name,
        driver="GPKG",
        crs=crs,
        engine="fiona",
    )

    # Save for web application
    line = line.to_crs("EPSG:4326")
    line.to_file(
        Path(dir_results_analysis_app) / beach_name.replace("gpkg", "geojson"),
        driver="GeoJSON",
        crs="EPSG:4326",
        engine="fiona",
    )

    return


def save_matrix_to_netcdf(data, coordinates, time, info, sim_no, filename):
    import xarray as xr

    # Create dataset
    ds = xr.Dataset(
        data_vars={"prob": (("time", "dim_x", "dim_y"), data[season])},
        coords={
            "x": (("dim_x", "dim_y"), coordinates["x"]),
            "y": (("dim_x", "dim_y"), coordinates["y"]),
            "time": time,
        },
        attrs={
            "description": f"Matriz binaria anual ({season}) a partir de promedios mensuales",
            "stretch": stretch,
            "sim": str(sim_no).zfill(2),
        },
    )

    # Save to NetCDF4 with compression
    ds.to_netcdf(
        filename,
        format="NETCDF4",
        engine="netcdf4",
        encoding={"prob": {"zlib": True, "complevel": 2}},
    )
    return
