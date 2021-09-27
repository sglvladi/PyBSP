import pickle
import shapefile
import geopandas
from shapely.geometry import shape

from utils import read_df_from_shapefile, simplify_df, remove_small_polygons_from_df, merge_polygons

import matplotlib.pyplot as plt

def read_polygons_from_shapefile(path):
    shapes = shapefile.Reader(path)
    features = shapes.shapeRecords()

    polygons = []
    for i, feature in enumerate(features):
        first = feature.shape.__geo_interface__
        polygon = shape(first)
        polygons.append(polygon)
    return polygons


def read_big_polygons_from_shapefile_geopandas(path, tolerance):
    df = geopandas.GeoDataFrame.from_file(path)

    # Remove the south pole
    polar_idx = df.geometry.area.argmax()
    df = df.drop([polar_idx])

    # EPSG:6933 apparently has equal area projection
    df2 = df.to_crs('epsg:6933')
    # Filter polygons according to tolerance and convert to original crs
    df3 = df2[df2.geometry.apply(lambda x: x.area / (10 ** 6) > tolerance)].to_crs(df.crs.srs)

    return df3
    # for i, feature in enumerate(features):
    #     first = feature.shape.__geo_interface__
    #     polygon = shape(first)
    #     polygons.append(polygon)
    # return polygons


def simplify_polygons(polygons, tolerance=1000):
    new_polygons = []
    for polygon in polygons:
        new_polygons.append(polygon.simplify(tolerance))
    return new_polygons


def remove_small_polygons(polygons, threshold):
    new_polygons = []
    for polygon in polygons:
        area = polygon.area
        if area > threshold:
            new_polygons.append(polygon)
    return new_polygons


def smoothen_polygons(polygons, threshold):
    smooth_polygons = []
    for polygon in polygons:
        smooth_polygon = polygon.buffer(threshold, join_style=1).buffer(-threshold, join_style=1)
        smooth_polygons.append(smooth_polygon)
    return smooth_polygons


if __name__ == '__main__':
    df = read_df_from_shapefile('simplified_land_polygons.shp')

    # Remove the south pole
    polar_idx = df.geometry.area.argmax()
    df = df.drop([polar_idx])

    df = remove_small_polygons_from_df(df, 2000)

    df = simplify_df(df, 40000)

    polygons = df.geometry.to_list()
    polygons = merge_polygons(polygons)

    pickle.dump(polygons, open('oversimplified_merged_polygons.p', 'wb'))



    # polygons = smoothen_polygons(polygons, 10000)

    a = 2


# df = read_big_polygons_from_shapefile_geopandas('simplified_land_polygons.shp', 200)
#
#
#
# # big_polygons = remove_small_polygons(polygons, 100000)
# simple_big_polygons = simplify_polygons(big_polygons, 40000)
#
# a=2
#
#
#
# a = 2

