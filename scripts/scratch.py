import matplotlib.pyplot as plt
import numpy as np
import cProfile as profile
# In outer section of code
pr = profile.Profile()
pr.disable()
import datetime
import pickle
import multiprocessing as mpp
# from shapely.errors import ShapelyDeprecationWarning


from pybsp.utils import plot_nodes, sort_fovs, remove_artists
from pybsp.bsp import BSP, gen_pvs_single
from pybsp.geometry import LineSegment, Point, Polygon
from pybsp.geo import load_target_lines, get_merc_limits

import sys

import warnings
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

sys.setrecursionlimit(10000000)
seed = 10000 # np.random.randint(10000000)
np.random.seed(seed)


def generate_ref_point():
    # return Point(1.5955e+06, 4.3e+06)
    # return Point(1.6019e+6, 4.305e+6)

    # return Point(-3.39e+5, 7.127e+6)
    return Point(-2e+5, 4.339e+6)
    # return Point(1.593e+6, 4.3e+6)


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
    lines = load_target_lines(TARGET, 'panama/oversimplified_merged_polygons3_2.p', force=True)

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
    plt.pause(0.01)

    # line = lines[0]
    # now = datetime.datetime.now()
    # res3 = line.compare2([l for l in lines])
    # print(datetime.datetime.now() - now)
    # now = datetime.datetime.now()
    # res = [line.compare(l) for l in lines]
    # print(datetime.datetime.now() - now)
    # now = datetime.datetime.now()
    # res2 = [line.compare2(l) for l in lines]
    # print(datetime.datetime.now() - now)

    # bsptree = BSP.load(backup_folder, 'Stage3', 'final')
    # Create BSP tree
    # bsptree = BSP(heuristic=heuristic, bounds=(xlim, ylim))

    # Train the tree
    # bsptree.train(lines, parallel=True, backup_folder=backup_folder, start_stage=1, end_stage=2)

    # bsptree.gen_portals_walls(parallel=False)
    bsptree = BSP.load(backup_folder, 'Stage2', 'final')
    # # Train the tree
    bsptree.train(lines, parallel=True, backup_folder=backup_folder, start_stage=3)
    # bsptree.gen_pvs()
    # bsptree = BSP.load(backup_folder, 'Stage3', 'final')


    # pr.enable()
    # pr.disable()
    # stats_filename = 'profile_{}.pstat'.format(5)
    # print("[INFO]: Dumping Profiler stats file {}".format(stats_filename))
    # pr.dump_stats(stats_filename)

    # print('Loading BSP tree')
    # bsptree = pickle.load(open('trees/bsp_{}_{}_full.p'.format(TARGET, heuristic), 'rb'))

    # Plot tree graph
    # fig2 = plt.figure(figsize=(8, 6))
    # ax2 = fig2.add_subplot(111)
    # bsptree.draw_nx(ax=ax1, show_labels=True)
    # plt.pause(0.01)

    if SHOW_PLANES:
        ls = []
        bsptree.plot_planes(ax=ax1)
        # plot_planes(bsptree.tree, lines=ls, xlim=xlim, ylim=ylim)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

    ax1.plot(ref_point.x, ref_point.y, 'ko')
    plt.pause(0.01)

    # n1 = bsptree.empty_leaves[0]
    # for n2 in bsptree.empty_leaves[1:]:
    #     arts = [n1.polygon.plot(color='b', ax=ax1), n2.polygon.plot(color='r', ax=ax1)]
    #     now = datetime.datetime.now()
    #     res = bsptree.check_pvs(n1, n2)
    #     dt = datetime.datetime.now() - now
    #     ax1.set_title(dt.total_seconds())
    #     plt.pause(0.01)
    #     remove_artists(arts)
    #
    # points = []
    # for leaf in bsptree.empty_leaves:
    #     x, y = leaf.polygon.shapely.centroid.x, leaf.polygon.shapely.centroid.y
    #     p = Point(x, y)
    #     points.append(p)
    #
    # for p in points:
    #     p.plot('or', ax=ax1)
    #
    # plt.pause(0.01)
    #
    # p1 = points[0]
    # for point in points[1:]:
    #     l = LineSegment(p1, point)
    #     arts = [p1.plot('bo', ax=ax1), point.plot('bo', ax=ax1)]
    #     now = datetime.datetime.now()
    #     res = bsptree.check_los(p1, point)
    #     dt = datetime.datetime.now()-now
    #     color = 'g' if res else 'r'
    #     arts.append(l.plot(linestyle='--', color=color, ax=ax1))
    #     ax1.set_title(dt.total_seconds())
    #     plt.pause(0.01)
    #     remove_artists(arts)

    # id, pvs = gen_pvs_single((5981, bsptree.nodes, bsptree._portals, bsptree.node_connectivity))

    print("Rendering...", end='')
    # pr.enable()
    # rendered_lines = plot_visibility2(bsptree, ref_point, plt.gca())
    now = datetime.datetime.now()
    rendered_lines = bsptree.render(ref_point, use_pvs=True)
    print(datetime.datetime.now()-now)
    # now = datetime.datetime.now()
    # rendered_lines = bsptree.render(ref_point, use_pvs=True)
    # print(datetime.datetime.now()-now)
    # pr.disable()
    print("done")


    # node = bsptree.find_leaf(ref_point)
    # # node = bsptree.get_node(180)
    # pvs = [bsptree.nodes[n] for n in node.pvs]
    # wall_pvs = [bsptree.get_wall(w) for w in node.wall_pvs]
    # art = plot_nodes(pvs, ax=ax1)
    # pol = node.polygon
    # art.append(pol.plot(color='r', ax=ax1))
    # x, y = pol.shapely.centroid.x - 15, pol.shapely.centroid.y - 15
    # art.append(ax1.text(x, y, node.id, color='w', fontsize='large', fontweight='bold'))
    # for line in wall_pvs:
    #     art.append(line.plot(color='r', ax=ax1))
    # # for n in bsptree.empty_leaves:
    # #     if n in pvs or node == n:
    # #         continue
    # #     pol = n.polygon
    # #     x, y = pol.centroid.x - 15, pol.centroid.y - 15
    # #     art.append(plt.text(x, y, n.id, color='r', fontsize='large', fontweight='bold'))
    # plt.pause(0.1)
    # plt.show()

    for line in rendered_lines:
        x, y = line.shapely.xy
        ax1.plot(x, y, 'r')
        # for point in line.shapely.boundary:
        #     x = [point.x, ref_point.x]
        #     y = [point.y, ref_point.y]
        #     ax1.plot(x, y, 'k--', linewidth=0.2)
    plt.pause(0.1)

    # pr.enable()
    # for i in range(100):
    #     bsptree.render(ref_point, use_pvs=True)
    # pr.disable()
    # pr.dump_stats('../data/profiles/render1.pstat')

    print("Generating Waypoints...", end='')
    # pr.enable()
    # rendered_lines = plot_visibility2(bsptree, ref_point, plt.gca())
    now = datetime.datetime.now()
    waypoints = bsptree.gen_waypoints(ref_point)
    print(datetime.datetime.now() - now)
    # now = datetime.datetime.now()
    # rendered_lines = bsptree.render(ref_point, use_pvs=True)
    # print(datetime.datetime.now()-now)
    # pr.disable()
    print("done")

    for i, waypoint in enumerate(waypoints):
        waypoint.plot(ax=ax1, color='b', marker='x', markersize=10)
        ax1.text(waypoint.x+10000, waypoint.y+10000, str(i+1), fontsize=15)
        x = [ref_point.x, waypoint.x]
        y = [ref_point.y, waypoint.y]
        ax1.plot(x, y, 'k--', linewidth=0.2)

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
            x = [point.x, ref_point.x]
            y = [point.y, ref_point.y]
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
    #     x.append(ref_point.x)
    #     y.append(ref_point.y)
    #     ps = [(xi, yi) for xi, yi in zip(x, y)]
    #     polygon = Polygon(ps)
    #     polygon.plot(color=color)
    #     # x, y = polygon.exterior.xy
    #
    #     line.plot(color=color)
    #     for point in line.linestring.boundary:
    #         x = [point.x, ref_point.x]
    #         y = [point.y, ref_point.y]
    #         plt.plot(x, y, 'k--', linewidth=0.2)
    #     plt.pause(0.01)
    #     a=2
    # for point in edge_points:
    #     plt.plot(point.x, point.y, 'ro')
    # for point in v_points:
    #     plt.plot(point.x, point.y, 'bo')

    print("Finding leaf...", end='')
    a = bsptree.find_leaf(ref_point)
    print("Done")
    a.polygon.plot(color='r')
    plt.pause(0.01)

    fig = plt.figure()
    ax = plt.gca()
    fovs = []
    for line in rendered_lines:
        fovs.append(line.to_interval(ref_point))

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