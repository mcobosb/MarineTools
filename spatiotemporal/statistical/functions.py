import json
from pathlib import Path
import os

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio import features
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, MultiPoint, Point, MultiLineString
import scipy.stats as st
from affine import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.ops import polygonize, snap, split, transform
import re
from typing import Optional
from loguru import logger
from shapely.ops import linemerge

# Modo debug: establecer DEBUG_MODE=True para activar visualizaciones
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')


def boundary_polygon(Z, level):

    """
    Asigna un valor constante a los puntos de una matriz Z que caen dentro de un polígono.

    Dada una matriz de valores Z, esta función identifica el contorno y le asigna el valor 'level'.
    El resto de los valores de Z permanecen igual.

    Parámetros
    ----------
    Z : np.ndarray
        Matriz de valores Z (2D), misma forma que X e Y.
    level : float
        Valor a asignar a los puntos dentro del polígono.

    Devuelve
    --------
    Z : np.ndarray
        Matriz Z modificada, con los valores del contorno igual a 'level'.
    """
    Z[0, :], Z[-1, :] = level  # Evitar líneas en bordes N y S
    Z[ :, 0], Z[:, -1] = level  # Evitar líneas en bordes E y O
    return Z


def sieve(data, fld_portion):
    data[data <= 0] = 0
    data[data > 0] = 1

    masked_arr = np.ma.masked_where(np.isnan(data), data)
    filled_arr = masked_arr.filled(0)

    # Convertir el array a un tipo de dato entero compatible con rasterio.sieve
    filled_arr_int = filled_arr.astype(rasterio.int16)

    # Aplicar rasterio sieve
    sieved = features.sieve(filled_arr_int, size=np.sum(filled_arr_int) / fld_portion)

    # Ahora podemos restaurar los valores enmascarados (si es necesario)
    restored_masked_arr = np.ma.masked_where(masked_arr.mask, sieved)

    mask = restored_masked_arr == 0
    restored_masked_arr[mask] = 1
    restored_masked_arr[~mask] = 0
    return restored_masked_arr


def obtain_lines(x, y, z, levels):
    """
    Devuelve una lista de arrays (N,2), uno por nivel solicitado.
    Si en un nivel no hay isolínea, devuelve None en esa posición.
    Selecciona la isolínea más larga cuando hay varias en un mismo nivel.
    """
    # Asegurar que no haya NaNs que rompan el contour
    z = np.asarray(z)
    if np.isnan(z).any():
        z = np.ma.masked_invalid(z)

    cs = plt.contour(x, y, z, levels=levels)
    try:
        segs_per_level = cs.allsegs  # robusto en múltiples versiones
    finally:
        plt.close()

    lines = []
    for segs in segs_per_level:
        if not segs:                  # no hay segmentos para este nivel
            lines.append(None)
            continue
        # Elegimos la polilínea con más vértices (suele ser la principal)
        best = max(segs, key=lambda a: a.shape[0])
        lines.append(best)         # (N,2) con columnas [x, y]
    return lines



def recortar_linea_por_playa(playas, id_playa, line_, buffer_puntos=0):
    """
    Recorta la polilínea data_aux (array Nx2) entre los puntos más cercanos
    a los extremos (inicio y fin) de la geometría de la playa indicada.

    Parámetros
    ----------
    nombre_playa : str o índice
        Identificador de la playa. Si coincide con un valor de la columna `nombre_columna`,
        se selecciona por esa columna; si coincide con el índice de `playas`, se usa el índice.
    data_aux : np.ndarray de shape (N, 2)
        Polilínea de entrada (x, y) sobre la que recortar.
    buffer_puntos : int, opcional
        Amplía el recorte incluyendo este número de puntos extra a cada lado (por defecto 0).

    Devuelve
    --------
    data_beach : np.ndarray
        Segmento recortado de data_aux.
    indices : tuple(int, int)
        Índices (i0, i1) usados para el recorte en data_aux (i0 <= i1).
    """

    # --- obtener geometría de la playa ---

    geom = playas.geometry.loc[id_playa]


    # Asegurar LineString (si es MultiLineString, tomar la parte más larga)
    if geom.geom_type == 'MultiLineString':
        geom = max(geom.geoms, key=lambda g: g.length)
    if geom.geom_type != 'LineString':
        raise TypeError(f"La geometría de la playa debe ser LineString/MultiLineString, no {geom.geom_type}.")

    # --- extremos de la playa ---
    coords_playa = np.asarray(geom.coords)
    pto_ini = coords_playa[0]      # (x0, y0)
    pto_fin = coords_playa[-1]     # (xN, yN)

    line_ = np.asarray(line_.geometry[0].coords, dtype=float)
    if line_.ndim != 2 or line_.shape[1] != 2:
        raise ValueError("line_ debe ser un array de shape (N, 2).")

    # --- índices de los puntos más cercanos en line_ a los extremos de la playa ---
    d_ini = np.linalg.norm(line_ - pto_ini, axis=1)
    d_fin = np.linalg.norm(line_ - pto_fin, axis=1)
    pos_min_ini = int(np.argmin(d_ini))
    pos_min_fin = int(np.argmin(d_fin))

    # --- construir el corte (ordenar índices y aplicar buffer) ---
    if pos_min_ini == pos_min_fin:
        i0 = max(0, pos_min_ini - 1 - buffer_puntos)
        i1 = min(len(line_) - 1, pos_min_fin + 1 + buffer_puntos)
    else:
        i0, i1 = sorted((pos_min_ini, pos_min_fin))
        i0 = max(0, i0 - buffer_puntos)
        i1 = min(len(line_) - 1, i1 + buffer_puntos)

    data_beach = line_[i0:i1+1]

    return data_beach, (i0, i1)


def nearest_point(pto, ref_point):
    # Encuentra el punto más cercano al punto de referencia
    nearest_point_ = None
    min_distance = float("inf")
    for geom in pto:
        for point in geom.geoms:
            distance = ref_point.distance(point)
            if distance < min_distance:
                nearest_point_ = point
                min_distance = distance
    return nearest_point_


def read_asci_tiff(fname, type_="row"):

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



def calculate_grid_angle_and_create_rotated_mesh(da_dem, xx, yy, grid_size):
    """
    Calcula el ángulo de la malla da_dem y genera una nueva matriz X, Y 
    inscrita en xx, yy con coordenadas alineadas a los contornos.
    """
    # Calcular el ángulo de la malla da_dem
    # Usar las esquinas para calcular la orientación principal
    x_dem = da_dem["x"]
    y_dem = da_dem["y"]
    
    # Calcular vectores de los bordes de la malla
    # Vector horizontal (primera fila)
    if x_dem.ndim == 2:
        dx1 = x_dem[0, -1] - x_dem[0, 0]
        dy1 = y_dem[0, -1] - y_dem[0, 0]
        # Vector vertical (primera columna)  
        dx2 = x_dem[-1, 0] - x_dem[0, 0]
        dy2 = y_dem[-1, 0] - y_dem[0, 0]
    else:
        # Si es 1D, crear malla regular
        dx1 = x_dem[-1] - x_dem[0]
        dy1 = 0
        dx2 = 0
        dy2 = y_dem[-1] - y_dem[0]
    
    # Calcular ángulo de rotación (usar el vector más largo)
    if np.sqrt(dx1**2 + dy1**2) > np.sqrt(dx2**2 + dy2**2):
        angle = np.arctan2(dy1, dx1)  # ángulo del eje horizontal
    else:
        angle = np.arctan2(dx2, dy2) - np.pi/2  # ángulo del eje vertical corregido
    
    # Límites de xx, yy
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)
    
    # Centro del dominio
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Dimensiones del dominio
    width = x_max - x_min
    height = y_max - y_min
    
    # Calcular número de puntos para la nueva malla
    nx = int(width / grid_size) + 1
    ny = int(height / grid_size) + 1
    
    # Crear malla regular en sistema local (sin rotación)
    x_local = np.linspace(-width/2, width/2, nx)
    y_local = np.linspace(-height/2, height/2, ny)
    X_local, Y_local = np.meshgrid(x_local, y_local)
    
    # Aplicar rotación
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rotated = cos_a * X_local - sin_a * Y_local + x_center
    Y_rotated = sin_a * X_local + cos_a * Y_local + y_center
    
    return X_rotated, Y_rotated, angle


def band(da_dem, levels, grid_size, orientation_, band_width=3):

    min_level, max_level = levels.min().min(), levels.max().max()
    z_ = np.where((da_dem["z"] < max_level) & (da_dem["z"] > min_level), 1, 0)
    
    # Modo debug: mostrar visualización de la banda
    if DEBUG_MODE:
        plt.figure()
        plt.contourf(da_dem["x"], da_dem["y"], z_, levels=2, cmap="RdBu", alpha=0.5)
        plt.axis("equal")
        plt.title("Debug: Visualización de la banda")
        plt.show()


    mask = np.copy(z_)
    len_ = 0
    if orientation_ == "WE":
        for i in range(da_dem["y"].shape[0]):
            if np.any(mask[i, :] == 1):
                mask[i, :] = 1 
                len_ += 1
    elif orientation_ == "NS":
        for i in range(da_dem["y"].shape[1]):
            if np.any(mask[:, i] == 1):
                mask[:, i] = 1
                len_ += 1

    # Convertir band_ a máscara booleana y aplicar
    band_ = mask == 1
    xx = da_dem["x"][band_]
    yy = da_dem["y"][band_]
    if orientation_ == "NS":
        xx = np.reshape(xx, (-1, len_))
        yy = np.reshape(yy, (-1, len_))
    elif orientation_ == "WE":
        xx = np.reshape(xx, (len_, -1))
        yy = np.reshape(yy, (len_, -1))

    return band_, {"X": xx, "Y": yy}



def refinement(da_dem, band_, coords):

    X, Y = coords["X"], coords["Y"]

    # Modo debug: mostrar visualización de la banda
    if DEBUG_MODE:
        plt.figure()
        plt.plot(da_dem["x"].flatten(), da_dem["y"].flatten(), "ob", markersize=1)
        plt.plot(X.flatten(), Y.flatten(), "xr", markersize=1)
        plt.show()
    
    interp = LinearNDInterpolator(
        list(zip(da_dem["x"][band_].flatten(), da_dem["y"][band_].flatten())), da_dem["z"][band_].flatten()
    )

    Z = interp(X, Y)
    return Z


def save2GISapp(
    line,
    crs,
    beach_name,
    dir_results_analysis_GIS,
    dir_results_analysis_app,
):
    # Se guarda en GIS
    line = line.set_crs(crs)
    line.to_file(
        Path(dir_results_analysis_GIS) / beach_name,
        driver="GPKG",
        crs=crs,
        engine='fiona'
    )

    # Se guarda para la app
    line = line.to_crs("EPSG:4326")
    line.to_file(
        Path(dir_results_analysis_app) / beach_name.replace("gpkg", "geojson"),
        driver="GeoJSON",
        crs="EPSG:4326",
        engine='fiona'
    )

    return


def media_movil_2d(x, y, ventana):

    x_suavizada = np.convolve(x, np.ones(ventana) / ventana, mode="valid")
    y_suavizada = np.convolve(y, np.ones(ventana) / ventana, mode="valid")

    x_ini, x_end = [], []
    y_ini, y_end = [], []
    for i in range(ventana):
        if i == 0:
            x_ini.append(x[0])
            x_end.append(x[-1])
            y_ini.append(y[0])
            y_end.append(y[-1])
        else:
            elem = i + 1
            x_ini.append(np.sum(x[:elem]) / (i + 1))
            x_end.append(np.sum(x[-elem:]) / (i + 1))
            y_ini.append(np.sum(y[:elem]) / (i + 1))
            y_end.append(np.sum(y[-elem:]) / (i + 1))

    # Ajustar longitud
    x_suavizada = x_suavizada[: len(y_suavizada)]
    x_suavizada = np.hstack([np.asarray(x_ini), x_suavizada, np.asarray(x_end)[::-1]])
    y_suavizada = np.hstack([np.asarray(y_ini), y_suavizada, np.asarray(y_end)[::-1]])
    return x_suavizada, y_suavizada

def close_geometry(geom_1, geom_2, n_geom=-1, n_coord=-1):
    if n_geom != -1:
        part_1 = geom_1.geoms[n_geom]
        part_2 = geom_2.geoms[n_geom]

        coord_1 = part_1.coords[n_coord]
        coord_2 = part_2.coords[n_coord]
        line = LineString([coord_1, coord_2])

        geometry = part_1.union(part_2).union(line)
        a = 1
    else:
        coord_1 = geom_1.coords[0]
        coord_2 = geom_2.coords[0]
        line_1 = LineString([coord_1, coord_2])

        coord_1 = geom_1.coords[-1]
        coord_2 = geom_2.coords[-1]
        line_2 = LineString([coord_1, coord_2])

        geometry = geom_1.union(geom_2).union(line_1).union(line_2)

    return polygonize(geometry)


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first


def split_geometries(geom_1, geom_2, name, beach_dir, tolerance=1e-8):
    geoms = []

    if geom_1.geometryType() == "MultiLineString":
        geom_1 = geom_1.geoms[0]

    if geom_2.geometryType() == "MultiLineString":
        geom_2 = geom_2.geoms[0]

    if geom_1.intersects(geom_2):
        intersections = geom_1.intersection(geom_2)

        if (
            intersections.geom_type == "GeometryCollection"
            or intersections.geom_type == "MultiLineString"
        ):
            # print('Es geomcoleccion', flush = True)
            no_inter = len(intersections)
            intersections = MultiPoint(
                [[intersections[i].representative_point()][0] for i in range(no_inter)]
            )

        if intersections.geom_type == "LineString":
            intersections = Point(
                intersections.coords.xy[0][0], intersections.coords.xy[1][0]
            )
            # print(intersections[0], flush = True)

        snap_1 = snap(geom_1, intersections, tolerance)
        snap_2 = snap(geom_2, intersections, tolerance)

        # print(snap_1, flush = True)
        # print(intersections, flush = True)

        split_1 = split(snap_1, intersections)
        split_2 = split(snap_2, intersections)

        n_geoms_1 = len(split_1.geoms)
        n_geoms_2 = len(split_2.geoms)

        # print('N geoms 1', n_geoms_1, flush = True)
        # print('N geoms 2', n_geoms_2, flush = True)

        n_geoms = min(n_geoms_1, n_geoms_2)

        # print('N geoms', n_geoms, flush = True)

        sign = np.zeros(n_geoms)

        set_line_secs = []

        for i in range(n_geoms):
            # print('geometry', i, flush = True)
            line_ = LineString(split_1[i])
            p_middle = line_.coords[int(len(line_.coords) / 2)]
            p_middle_ant = line_.coords[int(len(line_.coords) / 2 - 1)]
            p_middle_post = line_.coords[int(len(line_.coords) / 2)]

            vtg = [
                p_middle_post[0] - p_middle_ant[0],
                p_middle_post[1] - p_middle_ant[1],
                0,
            ]
            vvert = [0, 0, -1]
            Vnorm = np.cross(vtg, vvert)
            norm = Vnorm / np.linalg.norm(Vnorm)
            Vnorm1 = norm * 10000
            Vnorm2 = norm * 10000

            xp2 = p_middle[0] + Vnorm2[0]
            yp2 = p_middle[1] + Vnorm2[1]

            xp3 = p_middle[0] - Vnorm2[0]
            yp3 = p_middle[1] - Vnorm2[1]

            xp4 = np.linspace(xp2, xp3, 2)
            yp4 = np.linspace(yp2, yp3, 2)

            line_sec = np.vstack((xp4, yp4)).T
            line_sec = LineString(line_sec)

            # print('line_sec', line_sec, flush = True)

            set_line_secs.append(line_sec)

            inter_middle = line_sec.intersection(LineString(split_2[i]))

            # print('inter_middle', inter_middle, flush = True)

            # if inter_middle.is_empty | bool(inter_middle.geom_type == 'MultiPoint'):
            # sign[i] = 0

            if inter_middle.is_empty:
                sign[i] = 0

            elif inter_middle.geom_type == "MultiPoint":
                # print('inter_middle[0].coords.xy[1]', inter_middle[0].coords.xy[0][0], flush = True)
                if inter_middle[0].coords.xy[1][0] > p_middle[1]:
                    # print('inter_middle[0].coords.xy[1]', inter_middle[0].coords.xy[1], flush = True)
                    sign[i] = -1
                else:
                    sign[i] = +1

            elif inter_middle.geom_type == "Point":
                if beach_dir == "S":
                    if inter_middle.y > p_middle[1]:
                        sign[i] = -1
                    else:
                        sign[i] = +1
                elif beach_dir == "W":
                    if inter_middle.x < p_middle[0]:
                        sign[i] = -1
                    else:
                        sign[i] = +1
                elif beach_dir == "E":
                    if inter_middle.x < p_middle[0]:
                        sign[i] = +1
                    else:
                        sign[i] = -1

        line_secs_aux = geopandas.GeoDataFrame(
            {"geometry": set_line_secs}, geometry="geometry", crs="EPSG:25830"
        )
        line_secs_aux.to_file(f"{name}.gpkg", driver="GPKG", crs="EPSG:25830")

        # if (len(sign) != 1):
        #     sign[-1] = sign[-2] * (-1)
        #     sign[0] = sign[1] * (-1)

        geoms.append(close_geometry(split_1, split_2, 0, 0))

        if intersections.geom_type == "MultiPoint":
            for i in range(1, n_geoms - 1):
                # print(i)
                part_1 = split_1.geoms[i]
                part_2 = split_2.geoms[i]

                geom = part_1.union(part_2)

                geoms.append(polygonize(geom))

        geoms.append(close_geometry(split_1, split_2, n_geoms - 1, -1))

    else:

        # sign = 0
        set_line_secs = []

        geoms.append(close_geometry(geom_1, geom_2))

        line_ = LineString(geom_1)
        p_middle = line_.coords[int(len(line_.coords) / 2)]
        p_middle_ant = line_.coords[int(len(line_.coords) / 2 - 1)]
        p_middle_post = line_.coords[int(len(line_.coords) / 2)]

        vtg = [
            p_middle_post[0] - p_middle_ant[0],
            p_middle_post[1] - p_middle_ant[1],
            0,
        ]
        vvert = [0, 0, -1]
        Vnorm = np.cross(vtg, vvert)
        norm = Vnorm / np.linalg.norm(Vnorm)
        Vnorm1 = norm * 10000
        Vnorm2 = norm * 10000

        xp2 = p_middle[0] + Vnorm2[0]
        yp2 = p_middle[1] + Vnorm2[1]

        xp3 = p_middle[0] - Vnorm2[0]
        yp3 = p_middle[1] - Vnorm2[1]

        xp4 = np.linspace(xp2, xp3, 2)
        yp4 = np.linspace(yp2, yp3, 2)

        line_sec = np.vstack((xp4, yp4)).T
        line_sec = LineString(line_sec)

        # print('line_sec', line_sec, flush = True)

        set_line_secs.append(line_sec)

        inter_middle = line_sec.intersection(LineString(geom_2))

        # print('inter_middle', inter_middle, flush = True)

        # if inter_middle.is_empty | bool(inter_middle.geom_type == 'MultiPoint'):
        # sign[i] = 0

        if inter_middle.is_empty:
            sign = np.array([0])

        elif inter_middle.geom_type == "MultiPoint":
            # print('inter_middle[0].coords.xy[1]', inter_middle[0].coords.xy[0][0], flush = True)
            if inter_middle[0].coords.xy[1][0] > p_middle[1]:
                # print('inter_middle[0].coords.xy[1]', inter_middle[0].coords.xy[1], flush = True)
                sign = np.array([-1])
            else:
                sign = np.array([+1])

        elif inter_middle.geom_type == "Point":
            if beach_dir == "S":
                if inter_middle.y > p_middle[1]:
                    sign = np.array([-1])
                else:
                    sign = np.array([+1])
            elif beach_dir == "W":
                if inter_middle.x < p_middle[0]:
                    sign = np.array([-1])
                else:
                    sign = np.array([+1])
            elif beach_dir == "E":
                if inter_middle.x < p_middle[0]:
                    sign = np.array([+1])
                else:
                    sign = np.array([-1])

        # if max(geom_2.coords.xy[1]) > max(geom_1.coords.xy[1]):
        # sign = np.array([-1])
        # else:
        # sign = np.array([1])

        line_secs_aux = geopandas.GeoDataFrame(
            {"geometry": set_line_secs}, geometry="geometry", crs="EPSG:25830"
        )
        line_secs_aux.to_file(f"{name}.gpkg", driver="GPKG", crs="EPSG:25830")

    return geoms, sign

def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_isolines(lines, crs, out_path: Path):
    """
    Guarda una o varias líneas devueltas por get_isolines en un único GPKG.
    `lines` puede ser:
      - una secuencia de coordenadas (x,y) -> única línea
      - una lista de líneas (cada línea = secuencia de (x,y))
    """
    # if lines is None:
    #     return
    # Si es una sola línea: lista de tuplas (x,y)
    if len(lines) > 0 and isinstance(lines[0], (tuple, list)) and \
       (len(lines[0]) == 2 and not isinstance(lines[0][0], (list, tuple))):
        geoms = [LineString(lines)]
    else:
        geoms = [LineString(l) for l in lines if l is not None and len(l) >= 2]

    # if not geoms:
    #     return

    gdf = geopandas.GeoDataFrame({"geometry": geoms}, crs=crs)
    # if out_path.exists():
    #     out_path.unlink()
    gdf.to_file(out_path, driver="GPKG")
    return
    
    
def smooth_lines_gdf(gdf: geopandas.GeoDataFrame, ventana_movil: int, crs) -> geopandas.GeoDataFrame:
    """Suaviza todas las geometrías del GDF (LineString o MultiLineString)."""
    out_geoms = []

    # Si el archivo no tiene CRS, lo asignamos (no reproyecta)
    if gdf.crs is None and crs is not None:
        gdf = gdf.set_crs(crs)

    for geom in gdf.geometry:
        if geom is None:
            continue

        # Normalizamos a lista de LineStrings
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            # ignorar geometrías no lineales
            continue

        smoothed_parts = []
        for part in parts:
            xs, ys = part.xy
            sx, sy = media_movil_2d(xs, ys, int(ventana_movil))
            if len(sx) >= 2:
                smoothed_parts.append(LineString(np.vstack((sx, sy)).T))

        if not smoothed_parts:
            continue

        if len(smoothed_parts) == 1:
            out_geoms.append(smoothed_parts[0])
        else:
            out_geoms.append(MultiLineString(smoothed_parts))

    if not out_geoms:
        return geopandas.GeoDataFrame(geometry=[], crs=crs)

    return geopandas.GeoDataFrame({"geometry": out_geoms}, crs=crs)


def first_existing_path(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def parse_sim_tag_from_stem(stem: str) -> str:
    """
    Devuelve 'sim_XX' si encuentra número de simulación en el nombre (admite 'sim_01', 'sim01', 'sim-1', etc.).
    Si no hay número, devuelve 'sim'.
    """
    parts = stem.split('_')
    # Caso '..._sim_01_...'
    for i, tok in enumerate(parts):
        if tok == "sim":
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"sim_{parts[i + 1].zfill(2)}"
            return "sim"

    # Caso 'sim01', 'sim1', 'sim-02'
    m = re.search(r"\bsim[_\-]?(\d+)\b", stem)
    if m:
        return f"sim_{m.group(1).zfill(2)}"

    return "sim"

def season_months_for_year(year: int):
    """
    Devuelve los pares (year, month) para cada season en ese 'year'.
    - AN: Ene..Dic del year
    - TA: Abr..Sep del year
    - TB: Oct..Dic del year + Ene..Mar del year+1
         Para el último año (2100), solo Oct..Dic del 2100.
    """
    an = [(year, m) for m in range(1, 13)]
    ta = [(year, m) for m in range(4, 10)]
    if year < 2100:
        tb = [(year, 10), (year, 11), (year, 12), (year + 1, 1), (year + 1, 2), (year + 1, 3)]
    else:
        tb = [(year, 10), (year, 11), (year, 12)]
    return {"AN": an, "TA": ta, "TB": tb}


def load_monthly_index(csv_path: Path):
    """Lee monthly_files_per_year.csv y devuelve DataFrame con columnas:
       ['tiff','dates','year','month'] (rellena según existan)."""
    if not csv_path.exists():
        logger.error("No existe %s", csv_path)
        return None
    df = pd.read_csv(csv_path)
    # Normalizamos fechas
    if 'dates' in df.columns:
        df['dates'] = pd.to_datetime(df['dates'])
    else:
        # si la fecha está en el índice
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df.reset_index().rename(columns={'index': 'dates'})
        except Exception:
            logger.warning("monthly_files_per_year.csv sin columna/índice 'dates'. Intento inferir...")
            df['dates'] = pd.NaT
    # Año y mes
    if 'year' not in df.columns:
        df['year'] = df['dates'].dt.year
    if 'month' not in df.columns:
        df['month'] = df['dates'].dt.month
    # Ordenado para coger “primer fichero del intervalo”
    df = df.sort_values(['year', 'month', 'dates'], kind='stable').reset_index(drop=True)
    return df


def pick_first_season_file(df_monthly: pd.DataFrame, season: str, year: int):
    """Devuelve el nombre del primer TIFF del intervalo de la season para ese año.
       TA: meses 4..9 del año j
       TB: meses 10..12 del año j  (si no hay, meses 1..3 del año j+1)"""
    if df_monthly is None or df_monthly.empty:
        return None

    # TODO 02: Code selection by season
    if season == 'TA':
        # meses 4..9 del mismo año
        sel = df_monthly[(df_monthly['year'] == year) & (df_monthly['month'].between(4, 9))]
        if sel.empty:
            return None
        return sel.iloc[0]['tiff']

    elif season == 'TB':
        # preferimos oct-dic del año j
        sel = df_monthly[(df_monthly['year'] == year) & (df_monthly['month'].between(10, 12))]
        if sel.empty:
            # si no hay, ene-mar del año j+1
            sel = df_monthly[(df_monthly['year'] == year + 1) & (df_monthly['month'].between(1, 3))]
            if sel.empty:
                return None
        return sel.iloc[0]['tiff']

    else:
        return None
    

def save_matrix_to_netcdf(data, season, coordinates, time, stretch, sim_no, filename):
    import xarray as xr

    # Create dataset
    ds = xr.Dataset(
        data_vars={"prob": (("time", "dim_x", "dim_y"), data[season])},
        coords={"x": (("dim_x", "dim_y"), coordinates["x"]), "y": (("dim_x", "dim_y"), coordinates["y"]), "time": time},
        attrs={
            "description": f"Matriz binaria anual ({season}) a partir de promedios mensuales",
            "stretch": stretch,
            "sim": str(sim_no).zfill(2),
        },
    )
   

    # Save to NetCDF4 with compression
    ds.to_netcdf(filename, format="NETCDF4", engine="netcdf4", encoding={"prob": {"zlib": True, "complevel": 2}})
    return

def read_gpkg_as_gdf_or_none(path: Path, crs):
    """Lee un GPKG y lo devuelve en CRS esperado (puede contener varias líneas)."""
    try:
        gdf = geopandas.read_file(path, driver="GPKG")
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        elif str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)
        return gdf
    except Exception as e:
        logger.error("No se pudo leer %s: %s", path, e)
        return None
    
    

def read_line_as_lstring_or_none(path: Path, crs):
    """Lee una línea (LineString) desde GPKG/GeoJSON; devuelve LineString o None."""
    try:
        gdf = geopandas.read_file(path)
        gdf = gdf.set_crs(crs) if gdf.crs is None else gdf.to_crs(crs)
    except Exception as e:
        logger.exception(f"Error leyendo {path}: {e}")
        return None

    if gdf.empty or gdf.geometry.is_empty.all():
        return None

    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiLineString":
        if len(geom.geoms) == 0:
            return None
        geom = max(geom.geoms, key=lambda g: g.length)
    if geom.geom_type != "LineString":
        return None
    return geom


def try_paths_sims_por_playa(base_aux_dir, beach_id, year, sim):
    """
    Devuelve lista de posibles rutas de entrada en sims_por_playa para:
       l_orilla_{beach_id}_AN_{year}_sim_{SS}.gpkg
    Acepta sim con y sin zfill(2).
    """
    base = Path(base_aux_dir) / "sims_por_playa"
    candidates = [
        base / f"l_orilla_{beach_id}_AN_{year}_sim_{str(sim).zfill(2)}.gpkg",
        base / f"l_orilla_{beach_id}_AN_{year}_sim_{int(sim)}.gpkg",
        # tolerancias por si quedaron variantes
        base / f"l_orilla_{beach_id}_AN_{year}_sim-{str(sim).zfill(2)}.gpkg",
        base / f"l_orilla_{beach_id}_AN_{year}_sim{str(sim).zfill(2)}.gpkg",
    ]
    return candidates





def create_gdf(values_1d, label, x_mid, y_mid, year_ini=2020):
    """
    Crea un GeoDataFrame con una única feature (punto en el centro de la playa)
    y columnas etiquetadas por año con prefijo del estadístico:
    p.ej., m_2020, m_2021, ..., m_2099
    """
    cols = {f"{label}_{year_ini + i}": float(values_1d[i]) for i in range(len(values_1d))}
    df = pd.DataFrame([cols])
    gdf = geopandas.GeoDataFrame(df, geometry=[Point(x_mid, y_mid)])
    return gdf


def append_uf_stretch(path_like, uf, stretch):
    """
    Asegura que la ruta termina en /<UF>/<stretch>. Si ya lo trae, no duplica.
    """
    p = Path(str(path_like))
    uf_str = str(uf).strip()
    st_str = str(stretch).strip()
    parts_low = [x.lower() for x in p.parts]
    if len(p.parts) >= 2 and parts_low[-2] == uf_str.lower() and parts_low[-1] == st_str.lower():
        return p
    if p.name.lower() == st_str.lower():
        return p
    return p / uf_str / st_str


def sim_tag_from_str(s: str) -> str:
    """Normaliza un id de simulación a 'sim_XX' (2 dígitos)."""
    try:
        return f"sim_{int(s):02d}"
    except Exception:
        m = re.search(r"(\d+)", str(s))
        if m:
            return f"sim_{int(m.group(1)):02d}"
        return "sim"