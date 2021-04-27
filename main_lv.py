import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from shapely.geometry import Polygon

from bsp import BSP
from geometry import LineSegment, Point
from utils import plot_planes, plot_visibility2

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
    lines.append(LineSegment(p1, p2, 1, '1'))
    lines.append(LineSegment(p1, p3, -1, '2'))
    lines.append(LineSegment(p3, p4, -1, '3'))
    lines.append(LineSegment(p2, p4, 1, '4'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p4.to_array(), p3.to_array()]))

    # Polygon 2
    p1 = Point(200., 600.)
    p2 = Point(450., 600.)
    p3 = Point(300., 700.)
    lines.append(LineSegment(p1, p2, -1, '5'))
    lines.append(LineSegment(p2, p3, -1, '6'))
    lines.append(LineSegment(p3, p1, -1, '7'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p3.to_array()]))

    # Polygon 3
    p1 = Point(500., 150.)
    p2 = Point(600., 250.)
    p3 = Point(500., 350.)
    p4 = Point(400., 250.)
    lines.append(LineSegment(p1, p2, -1, '8'))
    lines.append(LineSegment(p2, p3, -1, '9'))
    lines.append(LineSegment(p3, p4, -1, '10'))
    lines.append(LineSegment(p4, p1, -1, '11'))
    polygons.append(Polygon([p1.to_array(), p2.to_array(), p3.to_array(), p4.to_array()]))

    # Polygon 4
    p1 = Point(0., 0.)
    p2 = Point(800., 0.)
    p3 = Point(800., 800.)
    p4 = Point(0., 800.)
    lines.append(LineSegment(p1, p2, 1, '12'))
    lines.append(LineSegment(p2, p3, 1, '13'))
    lines.append(LineSegment(p3, p4, 1, '14'))
    lines.append(LineSegment(p4, p1, 1, '15'))
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
    return Point(40, 400)


def main():
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    SHOW_NORMALS = False
    ANNOTATE_LINES = True

    bsptree = BSP()
    print('Generating line segments')
    lines, polygons = generate_polygons()
    point1 = generate_ref_point(polygons)


    print('Generating tree')
    bsptree.tree.data = copy(lines)
    bsptree.generate_tree(bsptree.tree, heuristic='random')


    plt.figure(figsize=(8, 6))
    bsptree.draw_nx(plt.gca(), show_labels=False)

    plt.figure(figsize=(8, 6))
    plot_planes(bsptree.tree)
    for line in lines:
        x = [line.p1.x, line.p2.x]
        y = [line.p1.y, line.p2.y]
        plt.plot(x, y, 'k-')
        if SHOW_NORMALS:
            midPoint = line.getMidPoint()
            plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)
        if ANNOTATE_LINES:
            midPoint = line.getMidPoint()
            plt.text(midPoint.x + line.NormalV.x / 10, midPoint.y + line.NormalV.y / 10, line.Name)

    plot_planes(bsptree.tree)
    plot_waypoints(point1, bsptree)
    plt.axis('equal')
    plt.xlim((0, SCREEN_WIDTH))
    plt.ylim((0, SCREEN_HEIGHT))

    # p = Point()
    print(bsptree.find_leaf(Point(40, 400)).data[0].Name)
    print(bsptree.depth())
    plt.pause(0.01)

    plt.plot(point1.x, point1.y, 'ko')
    rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    # rendered_lines = bsptree.render2(point1)333333333333333333333333333313
    # merged_lines = merge_lines2(rendered_lines)

    for line in rendered_lines:
        x, y = line.xy
        plt.plot(x, y, 'r')
        for point in line.boundary:
            x = [point.x, point1.x]
            y = [point.y, point1.y]
            plt.plot(x, y, 'k--', linewidth=0.2)
    # rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    # for line in rendered_lines:
    #     line.plot(color='r', marker='s')

    # plt.pause(0.01)
    plt.show()


if __name__ == '__main__':
    main()
