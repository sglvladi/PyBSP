import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
import shapely
import networkx as nx
import cProfile as profile
# In outer section of code
import tqdm
import multiprocessing as mpp

pr = profile.Profile()
pr.disable()
import datetime
from stonesoup.types.angle import Bearing
from stonesoup.functions import pol2cart

from pybsp.utils import plot_nodes, sort_fovs, remove_artists
from pybsp.bsp import BSP
from pybsp.geometry import LineSegment, Point, Polygon
from pybsp.geo import load_target_lines, get_merc_limits, load_target_polygons

import sys

sys.setrecursionlimit(10000000)
seed = 10000 # np.random.randint(10000000)
np.random.seed(seed)


def generate_ref_point():
    # return Point(1.5955e+06, 4.3e+06)
    # return Point(1.6019e+6, 4.305e+6)

    # return Point(-3.39e+5, 7.127e+6)
    return Point(-2e+5, 4.339e+6)
    # return Point(1.593e+6, 4.3e+6)


def inner_angle(p1, p2, p3, cw=True):
    """ Calculate inner angle

    Taken from: https://stackoverflow.com/questions/12083480/finding-internal-angles-of-polygon
    """
    # Incoming vector
    v1 = (p2.x - p1.x, p2.y - p1.y)

    # Outgoing vector
    v2 = (p3.x - p2.x, p3.y - p2.y)

    theta = np.pi
    if cw:
        theta += np.arctan2(np.cross(np.array(v1), np.array(v2)), np.dot(np.array(v1), np.array(v2)))
    else:
        theta -= np.arctan2(np.cross(np.array(v1), np.array(v2)), np.dot(np.array(v1), np.array(v2)))

    return theta


def compute_corners_single(polygon):
    corner_points = []
    adj = dict()
    # Sort points clockwise
    polygon = shapely.geometry.polygon.orient(polygon, -1.0)  # 1.0 for ccw
    x, y = polygon.exterior.xy
    x = x[:-1]
    y = y[:-1]
    points = []
    for xi, yi in zip(x, y):
        points.append(Point(xi, yi))

    triplets = [(i, (i + 1) % len(points), (i + 2) % len(points)) for i in range(len(points))]
    for i, inds in enumerate(triplets):
        p1 = points[inds[0]]
        p2 = points[inds[1]]
        p3 = points[inds[2]]

        theta = inner_angle(p1, p2, p3)

        if abs(theta) < np.pi:
            corner_points.append(p2)
            try:
                ind = corner_points.index(p1)
                adj[len(corner_points) - 1] = ind
            except ValueError:
                continue
            if i == len(triplets) - 1:
                try:
                    ind = corner_points.index(p3)
                    adj[ind] = len(corner_points) - 1
                except ValueError:
                    continue
    return corner_points, adj


def compute_corners(polygons):
    corner_points = []
    adj = dict()  # dict representing adjacency of corner points on same line

    inputs = [polygon for polygon in polygons]

    pool = mpp.Pool(mpp.cpu_count())
    results = imap_tqdm(pool, compute_corners_single, inputs, desc='Corners')

    for result in results:
        i = len(corner_points)
        corners = result[0]
        adj_tmp = result[1]
        corner_points += corners
        for key, value in adj_tmp.items():
            adj[key+i] = value+i

    # for polygon in tqdm.tqdm(polygons):
    #
    #     # Sort points clockwise
    #     polygon = shapely.geometry.polygon.orient(polygon, -1.0)  # 1.0 for ccw
    #     x, y = polygon.exterior.xy
    #     x = x[:-1]
    #     y = y[:-1]
    #     points = []
    #     for xi, yi in zip(x, y):
    #         points.append(Point(xi, yi))
    #
    #     triplets = [(i, (i+1)%len(points), (i+2)%len(points)) for i in range(len(points))]
    #     for i, inds in enumerate(triplets):
    #         p1 = points[inds[0]]
    #         p2 = points[inds[1]]
    #         p3 = points[inds[2]]
    #
    #         theta = inner_angle(p1, p2, p3)
    #
    #         dist = ports.geometry.distance(p2.shapely).min()
    #
    #         if dist < 1e-8 or abs(theta) < np.pi:
    #             corner_points.append(p2)
    #             try:
    #                 ind = corner_points.index(p1)
    #                 adj[len(corner_points)-1] = ind
    #             except ValueError:
    #                 continue
    #             if i == len(triplets)-1:
    #                 try:
    #                     ind = corner_points.index(p3)
    #                     adj[ind] = len(corner_points) - 1
    #                 except ValueError:
    #                     continue
    return corner_points, adj


# def compute_corners(polygons):
#     corner_points = []
#     adj = dict()  # dict representing adjacency of corner points on same line
#     for polygon in tqdm.tqdm(polygons):
#
#         # Sort points clockwise
#         polygon = shapely.geometry.polygon.orient(polygon, -1.0)  # 1.0 for ccw
#         x, y = polygon.exterior.xy
#         x = x[:-1]
#         y = y[:-1]
#         points = []
#         for xi, yi in zip(x, y):
#             points.append(Point(xi, yi))
#
#         triplets = [(i, (i+1)%len(points), (i+2)%len(points)) for i in range(len(points))]
#         for i, inds in enumerate(triplets):
#             p1 = points[inds[0]]
#             p2 = points[inds[1]]
#             p3 = points[inds[2]]
#
#             theta = inner_angle(p1, p2, p3)
#
#             if abs(theta) < np.pi:
#                 corner_points.append(p2)
#                 try:
#                     ind = corner_points.index(p1)
#                     adj[len(corner_points)-1] = ind
#                 except ValueError:
#                     continue
#                 if i == len(triplets)-1:
#                     try:
#                         ind = corner_points.index(p3)
#                         adj[ind] = len(corner_points) - 1
#                     except ValueError:
#                         continue
#     return corner_points, adj


def calc_conn_graph(bsptree, polygons):
    # Compute corners
    corner_points, adj = compute_corners(polygons)

    # Calculate adj matrix
    num_nodes = len(corner_points)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for key, value in adj.items():
        adj_matrix[key, value] = 1
        adj_matrix[value, key] = 1

    pool = mpp.Pool(mpp.cpu_count())

    inputs = tqdm.tqdm([(corner_points, ind1, bsptree) for ind1 in range(len(corner_points))], desc='Queued')
    results = imap_tqdm(pool, check_los, inputs, desc='Processed')
    for result in results:
        ind1 = result[0]
        inds = result[1]
        for ind2 in inds:
            adj_matrix[ind1, ind2] = 1
            adj_matrix[ind2, ind1] = 1

    G = nx.from_numpy_matrix(adj_matrix)

    pos = dict()
    for i in range(num_nodes):
        p = corner_points[i]
        pos[i] = {'pos': (p.x, p.y)}

    nx.set_node_attributes(G, pos)
    return G, corner_points


def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results


def check_los(args):
    corner_points, ind1, bsptree = args
    adj = []
    all_inds = [i for i in range(len(corner_points))]
    p1 = corner_points[ind1]
    for ind2 in all_inds:
        if ind2 <= ind1:
            continue

        p2 = corner_points[ind2]

        # Bring p1 closer to p2
        r1, ph1 = p1.to_polar(p2)
        r1 -= 1
        x1, y1 = pol2cart(r1, ph1)
        x1, y1 = x1 + p2.x, y1 + p2.y
        p1 = Point(x1, y1)

        # Bring p2 closer to p1
        r2, ph2 = p2.to_polar(p1)
        r2 -= 1
        x2, y2 = pol2cart(r2, ph2)
        x2, y2 = x2 + p1.x, y2 + p1.y
        p2 = Point(x2, y2)

        los = bsptree.check_los(p1, p2)

        if los:
            adj.append(ind2)

    return ind1, adj

def main():
    SHOW_PLANES = True
    TARGET = "GLOBAL"
    heuristic = 'min'
    backup_folder = '../data/trees/{}_{}'.format(TARGET, heuristic).lower()
    val = input("Enter backup location: ")
    if val:
        backup_folder = '../data/trees/{}'.format(val)
    print(backup_folder)

    # Load lines
    lines = load_target_lines(TARGET, 'oversimplified_merged_polygons.p')
    polygons = load_target_polygons(TARGET, 'oversimplified_merged_polygons.p')

    # Generate Reference point
    ref_point = generate_ref_point()

    # Plot scene
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for line in lines:
        x = (line.p1.x, line.p2.x)
        y = (line.p1.y, line.p2.y)
        ax1.plot(x, y, 'k-')
        # midPoint = line.mid_point
        # plt.text(midPoint.x + line.NormalV.x / 10, midPoint.y + line.NormalV.y / 10, line.Name)

        # midPoint = line.mid_point
        # plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)
    xlim = plt.xlim()
    ylim = plt.ylim()

    xmin, xmax, ymin, ymax = get_merc_limits(TARGET)

    bsptree = BSP.load(backup_folder, 'Stage3', 'final')

    corner_points, _ = compute_corners(polygons)

    G, corner_points = calc_conn_graph(bsptree, polygons)

    pickle.dump(G, open('../data/graphs/corner_conn_graph.pickle', 'wb'))

if __name__ == '__main__':
    main()