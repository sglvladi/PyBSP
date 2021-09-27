import shapefile
import pickle
import geopandas
from shapely.geometry import shape


def read_polygons_from_shapefile(path, tolerance=1000):
    shapes = shapefile.Reader(path)
    features = shapes.shapeRecords()

    polygons = []
    for i, feature in enumerate(features):
        first = feature.shape.__geo_interface__
        shp_geom = shape(first)
        polygon = shp_geom.simplify(tolerance)
        polygons.append(polygon)
    return polygons


def merge_polygons(polygons):
    df = geopandas.GeoDataFrame(geometry=polygons)

    geoms = df.geometry.unary_union
    df = geopandas.GeoDataFrame(geometry=[geoms])

    df = df.explode().reset_index(drop=True)

    # df.plot(cmap='cividis', alpha=0.7, edgecolor='black')
    # plt.show()
    return df.geometry.tolist()


def read_df_from_shapefile(path):
    return geopandas.GeoDataFrame.from_file(path)


def remove_small_polygons_from_df(df, tolerance):
    # EPSG:6933 apparently has equal area projection
    df2 = df.to_crs('epsg:6933')
    # Filter polygons according to tolerance and convert to original crs
    df3 = df2[df2.geometry.apply(lambda x: x.area / (10 ** 6) > tolerance)].to_crs(df.crs.srs)
    return df3


def simplify_df(df, tolerance=1000):
    df2 = df.simplify(tolerance)
    return df2




if __name__ == '__main__':
    # polygons = pickle.load(open('polygons.p', 'rb'))
    polygons = read_polygons_from_shapefile('simplified_land_polygons.shp')
    pickle.dump(polygons, open('simplified_polygons.p', 'wb'))
    print('Read polygons, now simplifying...')
    merged_polygons = merge_polygons(polygons)
    pickle.dump(merged_polygons, open('simplified_merged_polygons.p', 'wb'))