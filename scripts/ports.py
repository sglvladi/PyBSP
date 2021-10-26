import tqdm
import numpy as np
import pandas, geopandas
import matplotlib.pyplot as plt
import multiprocessing as mpp
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString, Polygon

from pybsp.geo import load_target_lines, get_merc_limits, load_target_polygons, merc_from_arrays
from pybsp.utils import remove_artists

def dms_to_dd(d, m, s=0, sign=1):
    dd = d + float(m)/60 + float(s)/3600
    return dd*sign


def comp_min_distance_single(args):
    index, p, polygons_gdf = args
    dist = np.zeros((len(polygons_gdf)))
    for index2, row2 in polygons_gdf.iterrows():
        polygon = row2.geometry
        dist[index2] = polygon.exterior.distance(p)
    return index, np.min(dist), np.argmin(dist)


def comp_min_distance(ports_gdf, world_gdf):
    pool = mpp.Pool(mpp.cpu_count())
    inputs = [(index, row.geometry, world_gdf) for index, row in ports_gdf.iterrows()]
    results = imap_tqdm(pool, comp_min_distance_single, inputs, desc='Distances')
    results = sorted(results, key=lambda x: x[0])
    min_dist = [res[1] for res in results]
    min_dist_pol = [res[2] for res in results]
    ports_gdf2 = ports_gdf.copy()
    ports_gdf2['MIN_DIST'] = min_dist
    ports_gdf2['MIN_DIST_POLYGON'] = min_dist_pol
    return ports_gdf2


def get_projected_ports(ports_gdf, world_gdf):
    projected_ports = []
    for index, row in tqdm.tqdm(ports_gdf.iterrows(), total=len(ports_gdf), desc='Projection'):
        p = row.geometry
        polygon_ind = row.MIN_DIST_POLYGON
        polygon = world_gdf.iloc[polygon_ind].geometry
        # Get point projection on polygon
        _, p2 = nearest_points(p, polygon.exterior)
        # Append point to projected ports
        projected_ports.append(p2)

    proj_ports_gdf = ports_gdf.copy()
    proj_ports_gdf.geometry = projected_ports
    return proj_ports_gdf


def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results


def preprocess(path_to_file):
    # Load port locations
    ports_df = pandas.read_excel(path_to_file)
    ports_df.rename(columns={'Field4': 'LAT_DEG', 'Field5': 'LAT_MIN', 'Combo353': 'LAT_HEMI',
                             'Field7': 'LON_DEG', 'Field8': 'LON_MIN', 'Combo214': 'LON_HEMI',
                             'Combo184': 'COUNTRY', 'Field2': 'NAME', 'Field1': 'INDEX_NO', 'Text357': 'REGION'},
                    inplace=True)

    lat_deg = ports_df.LAT_DEG.to_numpy()
    lat_min = ports_df.LAT_MIN.to_numpy()
    lat_hemi = ports_df.LAT_HEMI.to_numpy()
    lon_deg = ports_df.LON_DEG.to_numpy()
    lon_min = ports_df.LON_MIN.to_numpy()
    lon_hemi = ports_df.LON_HEMI.to_numpy()

    num_rows = len(lat_deg)

    # for i in range(num_rows):
    #     if lat_hemi[i] == 'S':
    #         if lat_deg[i] > 0:
    #             lat_deg[i] = -lat_deg[i]
    #     if lon_hemi[i] == 'W':
    #         if lon_deg[i] > 0:
    #             lon_deg[i] = -lon_deg[i]

    lat = np.zeros((num_rows,))
    lon = np.zeros((num_rows,))
    for i in range(num_rows):
        sign = 1
        if lat_hemi[i] == 'S':
            sign = -1
        lat[i] = dms_to_dd(lat_deg[i], lat_min[i], sign=sign)
        sign = 1
        if lon_hemi[i] == 'W':
            sign = -1
        lon[i] = dms_to_dd(lon_deg[i], lon_min[i], sign=sign)

    ports_df['LAT'] = lat.tolist()
    ports_df['LON'] = lon.tolist()
    cols = ['REGION', 'COUNTRY', 'NAME', 'LAT', 'LON', 'LAT_DEG', 'LAT_MIN',
            'LAT_HEMI', 'LON_DEG', 'LON_MIN', 'LON_HEMI', 'INDEX_NO', 'Field10', 'Field11',
            'Combo192', 'Combo206', 'Combo210', 'Combo196', 'Combo218', 'Combo198',
            'Combo216', 'Combo231', 'Combo223', 'Combo225', 'Combo234', 'Combo236',
            'Field24', 'Combo238', 'Combo240', 'Combo242', 'Combo244', 'Combo246',
            'Combo248', 'Combo254', 'Combo256', 'Combo258', 'Combo260', 'Combo264',
            'Combo262', 'Combo266', 'Combo268', 'Combo270', 'Combo272', 'Combo274',
            'Combo276', 'Combo278', 'Combo280', 'Combo282', 'Combo284', 'Combo288',
            'Combo290', 'Combo292', 'Combo294', 'Combo296', 'Combo304', 'Combo300',
            'Combo302', 'Combo306', 'Combo308', 'Combo310', 'Combo312', 'Combo317',
            'Combo321', 'Combo319', 'Combo323', 'Combo325', 'Combo327', 'Combo329',
            'Combo332', 'Combo335', 'Combo337', 'Combo339', 'Combo341', 'Combo343',
            'Combo345', 'Combo347', 'Combo349', 'Combo351']

    ports_df = ports_df[cols]

    return ports_df


def add_points_to_polygon(args):
    df, polygon, polygon_ind = args
    for index, row in df.iterrows():
        p = row.geometry
        x, y = polygon.exterior.xy
        x = list(x)
        y = list(y)
        x = x[:-1]
        y = y[:-1]

        douplets = [(i, (i+1)%len(x)) for i in range(len(x))]
        matched = -1
        exact = False
        for i, j in douplets:
            x1, y1 = x[i], y[i]
            x2, y2 = x[j], y[j]
            p1 = Point(x1, y1)
            p2 = Point(x2, y2)
            if p1.distance(p) < 1e-8 or p2.distance(p) < 1e-8:
                exact = True
                break
            line = LineString([(x1, y1), (x2, y2)])
            dist = line.distance(p)
            if dist < 1e-8:
                matched = j
                break
        if matched != -1:
            x.insert(matched, p.x)
            y.insert(matched, p.y)
            polygon = Polygon([(xi, yi) for xi, yi in zip(x, y)])
        else:
            if not exact:
                print('Ouch!')
    return polygon_ind, polygon

def get_ports_and_augmented_polygons(polygons):
    # Read processed WPI ports
    ports_df = pandas.read_csv('../data/WPIData_LV.csv')

    # Convert to geopandas
    ports_gdf = geopandas.GeoDataFrame(ports_df, geometry=geopandas.points_from_xy(ports_df.LON, ports_df.LAT))
    ports_gdf.set_crs('EPSG:4326', inplace=True)  # CRS is in Lat/lon
    ports_gdf = ports_gdf.to_crs('EPSG:3857')  # Convert to mercator

    # Get polygons df
    world_gdf = geopandas.GeoDataFrame(geometry=polygons, crs='EPSG:3857')

    # Convert to EPSG:4087 (World Equidistant Cylindrical)
    ports_gdf_eqd = ports_gdf.to_crs('EPSG:4087')
    world_gdf_eqd = world_gdf.to_crs('EPSG:4087')

    # Compute minimum distance between ports and polygons
    ports_gdf_eqd = comp_min_distance(ports_gdf_eqd, world_gdf_eqd)

    # Remove ports far from shore (10km)
    valid_ports_gdf_eqd = ports_gdf_eqd[ports_gdf_eqd['MIN_DIST'] < 10000].copy().reset_index(drop=True)

    # Project ports on land polygons
    proj_ports_eqd = get_projected_ports(valid_ports_gdf_eqd, world_gdf_eqd)

    # Convert to mercator
    proj_ports_tmp = proj_ports_eqd.to_crs('EPSG:3857')

    # Re-project to ensure points are on polygons
    proj_ports = get_projected_ports(proj_ports_tmp, world_gdf)

    groups = proj_ports.groupby(['MIN_DIST_POLYGON'])
    inputs = []
    for polygon_ind, df in groups:
        inputs.append((df, world_gdf.iloc[polygon_ind].geometry, polygon_ind))

    pool = mpp.Pool(mpp.cpu_count())
    results = imap_tqdm(pool, add_points_to_polygon, inputs)

    for result in results:
        pol_ind = result[0]
        pol = result[1]
        world_gdf.at[pol_ind, 'geometry'] = pol

    return proj_ports, world_gdf.geometry.to_list()


def main():

    TARGET = "GLOBAL"
    # heuristic = 'min'
    # backup_folder = '../data/trees/{}_{}'.format(TARGET, heuristic).lower()
    # val = input("Enter backup location: ")
    # if val:
    #     backup_folder = '../data/trees/{}'.format(val)
    # print(backup_folder)

    # Load lines
    lines = load_target_lines(TARGET, 'oversimplified_merged_polygons.p')
    polygons = load_target_polygons(TARGET, 'oversimplified_merged_polygons.p')

    # Preprocess WPI ports file
    # ports_df = preprocess('../data/WPIData.xlsx')
    # ports_df.to_csv('../data/WPIData_LV.csv', index=False)

    ports, new_polygons = get_ports_and_augmented_polygons(polygons)
    # for index, row in tqdm.tqdm(proj_ports.iterrows(), total=len(proj_ports)):
    #     polygon = world_gdf.iloc[row.MIN_DIST_POLYGON].geometry
    #     p = row.geometry
    #     x, y = polygon.exterior.xy
    #     x = list(x)
    #     y = list(y)
    #     x = x[:-1]
    #     y = y[:-1]
    #
    #     douplets = [(i, (i+1)%len(x)) for i in range(len(x))]
    #     matched = -1
    #     exact = False
    #     for i, j in douplets:
    #         x1, y1 = x[i], y[i]
    #         x2, y2 = x[j], y[j]
    #         p1 = Point(x1, y1)
    #         p2 = Point(x2, y2)
    #         if p1.distance(p) < 1e-8 or p2.distance(p) < 1e-8:
    #             exact = True
    #             break
    #         line = LineString([(x1, y1), (x2, y2)])
    #         dist = line.distance(p)
    #         if dist < 1e-8:
    #             matched = j
    #             break
    #     if matched != -1:
    #         x.insert(matched, p.x)
    #         y.insert(matched, p.y)
    #         new_polygon = Polygon([(xi, yi) for xi, yi in zip(x, y)])
    #         world_gdf.iloc[row.MIN_DIST_POLYGON].geometry = new_polygon
    #         a=2
    #     else:
    #         if not exact:
    #             a=2

    # Plot scene
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax1.plot(x, y, 'k-')
    ports.plot(ax=ax1, markersize=5, zorder=10)
    plt.show()

    # ax1.plot(port_x, port_y, 'r.')

    plt.pause(0.01)
    a=2

if __name__ == '__main__':
    main()