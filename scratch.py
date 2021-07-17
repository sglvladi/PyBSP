import matplotlib.pyplot as plt
import numpy as np
import cProfile as profile
# In outer section of code
pr = profile.Profile()
pr.disable()
import datetime
import pickle
import multiprocessing as mpp


from utils.functions import plot_nodes, sort_fovs
from utils.bsp import BSP
from utils.geometry import LineSegment, Point, Polygon

import sys

sys.setrecursionlimit(10000000)

def merc_from_arrays(lats, lons):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x / lons
    y = 180.0 / np.pi * np.log(np.tan(np.pi / 4.0 + lats * (np.pi / 180.0) / 2.0)) * scale
    return (x, y)


def generate_ref_point(polygons):
    # return Point(1.5955e+06, 4.3e+06)
    # return Point(1.6019e+6, 4.305e+6)
    return Point(1.593e+6, 4.3e+6)


def main():
    pool = mpp.Pool(mpp.cpu_count())
    SHOW_PLANES = True
    TARGET = "MALTA"
    heuristic = 'min'
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
    LON_MIN = LIMITS[TARGET]["LON_MIN"]
    LON_MAX = LIMITS[TARGET]["LON_MAX"]
    LAT_MIN = LIMITS[TARGET]["LAT_MIN"]
    LAT_MAX = LIMITS[TARGET]["LAT_MAX"]

    xs, ys = merc_from_arrays(np.array([LAT_MIN, LAT_MAX]), np.array([LON_MIN, LON_MAX]))
    X_MIN, X_MAX = xs
    Y_MIN, Y_MAX = ys
    target_polygon = Polygon([(X_MIN, Y_MIN), (X_MIN, Y_MAX), (X_MAX, Y_MAX), (X_MAX, Y_MIN)])
    polygons = pickle.load(open('shapefiles/polygons.p', 'rb'))
    print('Loaded polygons')

    print('Generating lines')
    lines = []
    for polygon in polygons:
        if target_polygon.shapely.contains(polygon):
            x, y = polygon.exterior.xy
            # plt.plot(x, y)

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

    # bsptree = BSP3()
    # bsptree.tree.data = copy(lines)
    # bsptree.generate_tree(bsptree.tree, heuristic='even')
    point1 = generate_ref_point(polygons)

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

    print('Creating tree')
    now = datetime.datetime.now()
    bsptree = BSP(lines, heuristic=heuristic, bounds=(xlim, ylim))
    t1 = datetime.datetime.now() - now
    now = datetime.datetime.now()
    bsptree2 = BSP(lines, heuristic=heuristic, bounds=(xlim, ylim), pool=pool)
    t2 = datetime.datetime.now()-now
    print("T1: {} | T2: {}".format(t1.total_seconds(), t2.total_seconds()))
    pickle.dump(bsptree, open('trees/bsp_{}_{}.p'.format(TARGET, heuristic), 'wb'))

    # bsptree = pickle.load(open('trees/bsp_malta_min.p', 'rb'))

    # Plot tree graph
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    bsptree.draw_nx(ax=ax2, show_labels=True)
    plt.pause(0.01)


    if SHOW_PLANES:
        ls = []
        bsptree.plot_planes(ax=ax1)
        # plot_planes(bsptree.tree, lines=ls, xlim=xlim, ylim=ylim)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

    ax1.plot(point1.x, point1.y, 'ko')
    plt.pause(0.01)

    print("Rendering...", end='')
    # pr.enable()
    # rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    now = datetime.datetime.now()
    rendered_lines = bsptree.render(point1, use_pvs=False)
    print(datetime.datetime.now()-now)
    now = datetime.datetime.now()
    rendered_lines = bsptree.render(point1, use_pvs=True)
    print(datetime.datetime.now()-now)
    # pr.disable()
    print("done")

    node = bsptree.get_node(441)
    pvs = node.pvs
    wall_pvs = node.wall_pvs
    art = plot_nodes(pvs)
    pol = node.polygon
    art.append(pol.plot(color='r'))
    x, y = pol.shapely.centroid.x - 15, pol.shapely.centroid.y - 15
    art.append(plt.text(x, y, node.id, color='w', fontsize='large', fontweight='bold'))
    for line in wall_pvs:
        art.append(line.plot(color='r'))
    # for n in bsptree.empty_leaves:
    #     if n in pvs or node == n:
    #         continue
    #     pol = n.polygon
    #     x, y = pol.centroid.x - 15, pol.centroid.y - 15
    #     art.append(plt.text(x, y, n.id, color='r', fontsize='large', fontweight='bold'))
    plt.pause(0.1)

    node = bsptree.get_node(18)
    pvs = node.pvs
    wall_pvs = node.wall_pvs
    plot_nodes(pvs)
    pol = node.polygon
    pol.plot(color='r')
    x, y = pol.shapely.centroid.x - 15, pol.shapely.centroid.y - 15
    plt.text(x, y, node.id, color='w', fontsize='large', fontweight='bold')
    for line in wall_pvs:
        line.plot(color='r')
    for node in bsptree.empty_leaves:
        if node in pvs:
            continue
        pol = node.polygon
        x, y = pol.shapely.centroid.x - 15, pol.shapely.centroid.y - 15
        plt.text(x, y, node.id, color='r', fontsize='large', fontweight='bold')
    plt.pause(0.1)

    colors = dict()
    for line in rendered_lines:
        color = np.random.rand(1, 3)
        colors[tuple(line.names)] = color
        line.plot(ax=ax1, color=color)
        for point in line.shapely.boundary:
            x = [point.x, point1.x]
            y = [point.y, point1.y]
            ax1.plot(x, y, 'k--', linewidth=0.2)
    plt.pause(0.1)

    for leaf in bsptree.empty_leaves:
        pol = leaf.polygon
        x, y = pol.shapely.centroid.x, pol.shapely.centroid.y
        ax1.text(x,y, leaf.id, color='r')
    plt.pause(0.1)

    connected_nodes = dict()
    for node1 in bsptree.empty_leaves:
        for node2 in bsptree.empty_leaves:
            if node1 == node2 or (node1, node2) in connected_nodes or (node2, node1) in connected_nodes:
                continue
            for portal in node1.portals:
                if portal in node2.portals:
                    if (node1, node2) in connected_nodes:
                        connected_nodes[(node1, node2)].push(portal)
                    else:
                        connected_nodes[(node1, node2)] = [portal]

    # print('Generating PVS...', end='')
    # pr.enable()
    # bsptree.gen_pvs()
    # pr.disable()
    # print('Done')
    # print("[INFO]: Dumping Profiler stats")
    # pr.dump_stats('profile_{}.pstat'.format(1))


    # connected_nodes = dict()
    # for node1 in bsptree.empty_leaves:
    #     for node2 in bsptree.empty_leaves:
    #         if node1 == node2 or (node1, node2) in connected_nodes or (node2, node1) in connected_nodes:
    #             continue
    #         for portal in node1.portals:
    #             if portal in node2.portals:
    #                 if (node1, node2) in connected_nodes:
    #                     connected_nodes[(node1, node2)].push(portal)
    #                 else:
    #                     connected_nodes[(node1, node2)] = [portal]


    # pvs = bsptree.get_node(37)
    # plot_nodes(pvs)
    # plot_ex(bsptree, pvs)

    for key, item in connected_nodes.items():
        key[0].polygon.plot()
        plt.pause(0.01)
        key[1].polygon.plot()
        plt.pause(0.01)
        item[0].plot()
        plt.pause(0.01)
        a = 2
    # edge_points = [Point(1597488.6717045617, 4301871.333391014),
    #                Point(1599417.9943272963, 4300372.638162802),
    #                Point(1582837.4682432958, 4303495.014776728),
    #                Point(1595103.9410449916, 4286341.24740092)]
    # v_points = [Point(1594367.0060159403, 4303016.2347787805),
    #             Point(1597818.0326819716, 4299503.813637052),
    #             Point(1595001.6940926984, 4287743.039718742),
    #             Point(1592100.9530655053, 4304043.338596948),
    #             Point(1587575.604257726, 4302195.461219004),
    #             Point(1593973.591803528, 4296706.851525738),
    #             Point(1594876.4151377596, 4298836.853015211),
    #             Point(1595006.5698863952, 4299209.634897823),
    #             Point(1595040.923081254, 4292446.547744543),
    #             Point(1595488.4608300899, 4289249.752548993)]
    # for line in rendered_lines:
    #     color = np.random.rand(1, 3)
    #     colors[tuple(line.names)] = color
    #     line.plot(color=color)
    #     x, y = line.linestring.xy
    #     x.append(point1.x)
    #     y.append(point1.y)
    #     ps = [(xi, yi) for xi, yi in zip(x, y)]
    #     polygon = Polygon(ps)
    #     polygon.plot(color=color)
    #     # x, y = polygon.exterior.xy
    #
    #     # line.plot(color=color)
    #     # for point in line.linestring.boundary:
    #     #     x = [point.x, point1.x]
    #     #     y = [point.y, point1.y]
    #     #     plt.plot(x, y, 'k--', linewidth=0.2)
    #     plt.pause(0.01)
    #     a=2
    # for point in edge_points:
    #     plt.plot(point.x, point.y, 'ro')
    # for point in v_points:
    #     plt.plot(point.x, point.y, 'bo')

    print("Finding leaf...", end='')
    a = bsptree.find_leaf(point1)
    print("Done")
    a.polygon.plot(color='r')
    plt.pause(0.01)

    fig = plt.figure()
    ax = plt.gca()
    fovs = []
    for line in rendered_lines:
        fovs.append(line.to_interval(point1))

    fovs = sort_fovs(fovs)
    for fov in fovs:
        color = colors[tuple(fov.name)]
        fov.plot(ax=ax, fc=color)
        # plt.pause(0.01)
        a = 2


    # print(bsptree.tree.print())
    print(bsptree.find_leaf(Point(1.58, 4.275)))
    print(bsptree.depth())

    print("[INFO]: Dumping Profiler stats")
    pr.dump_stats('profile_{}.pstat'.format(1))

    plt.show()


if __name__ == '__main__':
    main()