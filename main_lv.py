import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from shapely.geometry import Polygon, MultiLineString, shape

from bsp import BSP
from geometry import LineSegment, Point
from utils import sort_fovs

import cProfile as profile
# In outer section of code
pr = profile.Profile()
pr.disable()

# a = np.random.randint(10000)
# print(a)
np.random.seed(6951)

def generate_polygons():
    lines = []
    polygons = []

    # Polygon 1
    offset_x = 100.
    offset_y = 100.
    width = 100.
    height = 100.
    p1 = Point(offset_x, offset_y)
    p2 = Point(offset_x, offset_y + height)
    p3 = Point(offset_x + width, offset_y)
    p4 = Point(offset_x + width, offset_y + height)
    lines.append(LineSegment(p1, p2, -1, '1'))
    lines.append(LineSegment(p1, p3, 1, '2'))
    lines.append(LineSegment(p3, p4, 1, '3'))
    lines.append(LineSegment(p2, p4, -1, '4'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p4.to_array(), p3.to_array()]))

    # Polygon 2
    p1 = Point(200., 600.)
    p2 = Point(450., 600.)
    p3 = Point(300., 700.)
    lines.append(LineSegment(p1, p2, 1, '5'))
    lines.append(LineSegment(p2, p3, 1, '6'))
    lines.append(LineSegment(p3, p1, 1, '7'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p3.to_array()]))

    # Polygon 3
    p1 = Point(500., 150.)
    p2 = Point(600., 250.)
    p3 = Point(500., 350.)
    p4 = Point(400., 250.)
    lines.append(LineSegment(p1, p2, 1, '8'))
    lines.append(LineSegment(p2, p3, 1, '9'))
    lines.append(LineSegment(p3, p4, 1, '10'))
    lines.append(LineSegment(p4, p1, 1, '11'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p3.to_array(), p4.to_array()]))

    # Polygon 4
    p1 = Point(0., 0.)
    p2 = Point(800., 0.)
    p3 = Point(800., 800.)
    p4 = Point(0., 800.)
    lines.append(LineSegment(p1, p2, -1, '12'))
    lines.append(LineSegment(p2, p3, -1, '13'))
    lines.append(LineSegment(p3, p4, -1, '14'))
    lines.append(LineSegment(p4, p1, -1, '15'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p3.to_array(), p4.to_array()]))

    return lines, polygons

def plot_waypoints(point1, bsptree):
    waypoint1 = Point(220, 220)
    waypoint2 = Point(500, 380)
    waypoint3 = Point(475, 585)
    waypoints = [waypoint1, waypoint2, waypoint3]

    dest1 = Point(250, 60)
    dest2 = Point(700, 270)
    dest3 = Point(460, 700)
    destinations = [dest1, dest2, dest3]

    points1 = [point1, waypoint1, dest1]
    points2 = [point1, waypoint2, dest2]
    points3 = [point1, waypoint3, dest3]

    pointss = [points1, points2, points3]

    for points in pointss:
        LoS = bsptree.checkLoS(points)

        for iFrom, From in enumerate(LoS):
            for iTo, To in enumerate(LoS):
                if iFrom != iTo and LoS[iFrom][iTo] == 'T':
                    x = [points[iFrom].x, points[iTo].x]
                    y = [points[iFrom].y, points[iTo].y]
                    plt.plot(x, y, 'g-')

    plt.plot(point1.x, point1.y, 'bs', markersize=8, label="Start")
    xs = []
    ys = []
    for point in waypoints:
        xs.append(point.x)
        ys.append(point.y)
    plt.plot(xs, ys, 'or', markersize=8, label="Waypoints")
    xs = []
    ys = []
    for point in destinations:
        xs.append(point.x)
        ys.append(point.y)
    plt.plot(xs, ys, 'sr', markersize=8, label="Destinations")

    plt.legend(loc="upper right", bbox_to_anchor=(0.85, 0.99))


def generate_ref_point(polygons):
    # return Point(300, 300)
    return Point(400, 218)
    # return Point(40, 400)



def main():
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    SHOW_NORMALS = True
    ANNOTATE_LINES = True

    print('Generating line segments')
    lines, polygons = generate_polygons()
    point1 = generate_ref_point(polygons)

    ## Get all 'LineString' objects
    # lines2 = [line.linestring for line in lines]
    # ## Convert to 'MultiString'
    # lines2 = MultiLineString([line.xy for line in lines2])
    # ## Convert Lines to Polygons by applying a tiny buffer
    # lines = lines2.buffer(0.0000000000001)
    # ## Get outer boundary of the lines as a polygon
    # boundary = lines2.convex_hull
    # ## Get a difference to generate a multipolygon
    # multipolygons = boundary.difference(lines2)

    print('Generating tree')
    bsptree = BSP(lines, heuristic='min', bounds=((-100, 900), (-100, 900)))
    # bsptree.tree.data = copy(lines)
    # bsptree.generate_tree(bsptree.tree, heuristic='random')

    #plt.figure(figsize=(8, 6))
    bsptree.draw_nx(plt.gca(), show_labels=True)

    plt.figure(figsize=(8, 6))
    # plot_planes(bsptree.tree, xlim=(-100, 900), ylim=(-100, 900))
    bsptree.plot_planes()
    for line in lines:
        x = [line.p1.x, line.p2.x]
        y = [line.p1.y, line.p2.y]
        plt.plot(x, y, 'k-')
        if SHOW_NORMALS:
            midPoint = line.mid_point
            plt.quiver(midPoint.x, midPoint.y, line.normalV.x, line.normalV.y, width=0.001, headwidth=0.2)
        if ANNOTATE_LINES:
            midPoint = line.mid_point
            plt.text(midPoint.x + line.normalV.x / 10, midPoint.y + line.normalV.y / 10, line.name)

    # plot_planes(bsptree.tree)
    # bsptree.draw_nx(plt.gca(), show_labels=False)
    # plot_waypoints(point1, bsptree)
    plt.axis('equal')
    plt.xlim((-100, 900))
    plt.ylim((-100, 900))

    # p = Point()
    #print(bsptree.find_leaf(Point(40, 400)).data[0].name)
    print(bsptree.depth)
    plt.pause(0.01)

    # plt.plot(point1.x, point1.y, 'ko')
    # rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    rendered_lines = bsptree.render(point1)
    # rendered_lines = bsptree.render2(point1)333333333333333333333333333313
    # merged_lines = merge_lines2(rendered_lines)

    for line in rendered_lines:
        x, y = line.linestring.xy
        plt.plot(x, y, 'r')
        for point in line.linestring.boundary:
            x = [point.x, point1.x]
            y = [point.y, point1.y]
            plt.plot(x, y, 'k--', linewidth=0.2)
    plt.pause(0.1)

    # a = bsptree.find_leaf(point1)
    # a.polygon.plot(color='r')
    # plt.pause(0.01)

    # p = a.area()

    for leaf in bsptree.empty_leaves:
        pol = leaf.polygon
        x, y = pol.centroid.x, pol.centroid.y
        plt.text(x,y, leaf.id, color='r')
    plt.pause(0.1)

    # for portal, nodes in bsptree.portal_connections.items():
    #     p = portal.plot(color='b')
    #     n1 = nodes[0].polygon.plot(color='y')
    #     n2 = nodes[1].polygon.plot(color='y')
    #     plt.pause(0.01)
    #     p.remove()
    #     n1.remove()
    #     n2.remove()
    bsptree.gen_portals()
    pr.enable()
    bsptree.gen_pvs()
    pr.disable()
    print("[INFO]: Dumping Profiler stats")
    pr.dump_stats('profile_{}.pstat'.format(1))

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

    for key, item in connected_nodes.items():
        key[0].polygon.plot()
        plt.pause(0.01)
        key[1].polygon.plot()
        plt.pause(0.01)
        item[0].plot()
        plt.pause(0.01)
        a=2
    fig = plt.figure()
    ax = plt.gca()
    fovs = []
    for line in rendered_lines:
        fovs.append(line.to_interval(point1))

    fovs = sort_fovs(fovs)
    for fov in fovs:
        color = np.random.rand(1, 3)
        fov.plot(ax=ax, fc=color)
        plt.pause(0.01)
    # rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    # for line in rendered_lines:
    #     line.plot(color='r', marker='s')

    # plt.pause(0.01)
    plt.show()


if __name__ == '__main__':
    main()
