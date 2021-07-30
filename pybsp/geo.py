import os
import pickle

import shapefile
from shapely.geometry import shape
import numpy as np
from .geometry import Point, LineSegment, Polygon

LIMITS = {
    "TEST": {
        "LON_MIN": -62.,
        "LON_MAX": -61.5,
        "LAT_MIN": 11.8,
        "LAT_MAX": 12.2
    },
    "GLOBAL": {
        "LON_MIN": -180.,
        "LON_MAX": 180.,
        "LAT_MIN": -80.,
        "LAT_MAX": 80.,
        "RES": 'c'
    },
    "FULL": {
        "LON_MIN": -84.,
        "LON_MAX": 34.5,
        "LAT_MIN": 9.5,
        "LAT_MAX": 62.,
        "RES": 'c'
    },
    "CARIBBEAN": {
        "LON_MIN": -90.,
        "LON_MAX": -60.,
        "LAT_MIN": 10.,
        "LAT_MAX": 22.,
        "RES": 'h'
    },
    "MEDITERRANEAN": {
        "LON_MIN": -6.,
        "LON_MAX": 36.5,
        "LAT_MIN": 30.,
        "LAT_MAX": 46.,
        "RES": 'l'
    },
    "GREECE": {
        "LON_MIN": 20.,
        "LON_MAX": 28.2,
        "LAT_MIN": 34.6,
        "LAT_MAX": 41.,
        "RES": 'h'
    },
    "UK": {
        "LON_MIN": -12.,
        "LON_MAX": 3.5,
        "LAT_MIN": 48.,
        "LAT_MAX": 60.,
        "RES": 'h'
    },
    "MALTA": {
        "LON_MIN": 14.04,
        "LON_MAX": 14.68,
        "LAT_MIN": 35.75,
        "LAT_MAX": 36.14,
        "RES": 'h'
    },
    "NEW_BRIGHTON": {
        "LON_MIN": -3.13,
        "LON_MAX": -2.9,
        "LAT_MIN": 53.37,
        "LAT_MAX": 53.46,
        "RES": 'h'
    },
    "MINI_MALTA": {
        "LON_MIN": 14.16,
        "LON_MAX": 14.37,
        "LAT_MIN": 36,
        "LAT_MAX": 36.1,
        "RES": 'h'
    }
}


def get_latlon_limits(target):
    lon_min = LIMITS[target]["LON_MIN"]
    lon_max = LIMITS[target]["LON_MAX"]
    lat_min = LIMITS[target]["LAT_MIN"]
    lat_max = LIMITS[target]["LAT_MAX"]

    return lon_min, lon_max, lat_min, lat_max


def merc_from_arrays(lats, lons):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x / lons
    y = 180.0 / np.pi * np.log(np.tan(np.pi / 4.0 + lats * (np.pi / 180.0) / 2.0)) * scale
    return (x, y)


def get_merc_limits(target):
    lon_min, lon_max, lat_min, lat_max = get_latlon_limits(target)
    xs, ys = merc_from_arrays(np.array([lat_min, lat_max]), np.array([lon_min, lon_max]))
    x_min, x_max = xs
    y_min, y_max = ys
    return x_min, x_max, y_min, y_max


def get_merc_target_polygon(target):
    x_min, x_max, y_min, y_max = get_merc_limits(target)
    target_polygon = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
    return target_polygon


def load_target_polygons(target):
    dirname = os.path.dirname(__file__)
    polygons_filename = 'polygons.p'
    polygons_filepath = os.path.abspath(os.path.join(dirname, '..', 'data/shapefiles', polygons_filename))
    polygons = pickle.load(open(polygons_filepath, 'rb'))
    target_polygon = get_merc_target_polygon(target)
    target_polygons = []
    for polygon in polygons:
        if target_polygon.shapely.contains(polygon):
            target_polygons.append(polygon)
    return target_polygons


def load_target_lines(target):
    dirname = os.path.dirname(__file__)
    filename = '{}.p'.format(target)
    filepath = os.path.abspath(os.path.join(dirname, '..', 'data/shapefiles/lines', filename))
    if os.path.exists(filepath):
        print('[INFO]: Loading target lines from file...', end='')
        lines = pickle.load(open(filepath, 'rb'))
        print('Done')
        return lines
    else:
        print('[INFO]: No lines backup file found. Proceeding to generating lines...')
        polygons_filename = 'merged_polygons.p'
        polygons_filepath = os.path.abspath(os.path.join(dirname, '..', 'data/shapefiles', polygons_filename))
        polygons = pickle.load(open(polygons_filepath, 'rb'))
        target_polygon = get_merc_target_polygon(target)
        lines = []
        for polygon in polygons:
            if target_polygon.shapely.contains(polygon):
                x, y = polygon.exterior.xy

                points = []
                for xi, yi in zip(x, y):
                    points.append(Point(xi, yi))

                for i in range(len(points)):
                    if i == len(points) - 1:
                        j = 0
                    else:
                        j = i + 1
                    p1 = points[i]
                    p2 = points[j]
                    p11 = (p1.x, p1.y)
                    p22 = (p2.x, p2.y)
                    if p11 != p22:
                        lines.append(LineSegment(p1, p2, -1, str(len(lines))))
        print('[INFO]: Saving lines backup file {}...'.format(filepath))
        pickle.dump(lines, open(filepath, 'wb'))
        return lines
