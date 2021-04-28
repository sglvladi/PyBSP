import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge
import cProfile as profile
# In outer section of code
pr = profile.Profile()
pr.disable()

from bsp import BSP, BSP2, BSP3
from geometry import LineSegment, Point
from utils import plot_planes, plot_visibility2, sort_fovs


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
    SHOW_PLANES = True
    TARGET = "MALTA"
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
    polygons = pickle.load(open('polygons.p', 'rb'))
    print('Loaded polygons')

    print('Generating lines')
    lines = []
    for polygon in polygons:
        if target_polygon.contains(polygon):
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
                    lines.append(LineSegment(p1, p2, 1, str(len(lines))))

    # bsptree = BSP3()
    # bsptree.tree.data = copy(lines)
    # bsptree.generate_tree(bsptree.tree, heuristic='even')
    point1 = generate_ref_point(polygons)

    # Plot scene
    plt.figure(figsize=(8, 6))
    for line in lines:
        x = (line.p1.x, line.p2.x)
        y = (line.p1.y, line.p2.y)
        plt.plot(x, y, 'k-')
        # midPoint = line.getMidPoint()
        # plt.text(midPoint.x + line.NormalV.x / 10, midPoint.y + line.NormalV.y / 10, line.Name)

        # midPoint = line.getMidPoint()
        # plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)
    xlim = plt.xlim()
    ylim = plt.ylim()

    print('Creating tree')
    bsptree = BSP3(lines, heuristic='even', bounds=(xlim, ylim))
    if SHOW_PLANES:
        ls = []
        plot_planes(bsptree.tree, lines=ls, xlim=xlim, ylim=ylim)
        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.plot(point1.x, point1.y, 'ko')
    plt.pause(0.01)

    print("Rendering...", end='')
    pr.enable()
    # rendered_lines = plot_visibility2(bsptree, point1, plt.gca())
    rendered_lines = bsptree.render2(point1)
    pr.disable()
    print("done")

    for line in rendered_lines:
        line.plot(color='r')
        for point in line.linestring.boundary:
            x = [point.x, point1.x]
            y = [point.y, point1.y]
            plt.plot(x, y, 'k--', linewidth=0.2)
    plt.pause(0.01)

    print("Finding leaf...", end='')
    a = bsptree.find_leaf(point1)
    print("Done")
    a.polygon.plot(color='r')
    plt.pause(0.01)

    # Plot tree graph
    plt.figure(figsize=(8, 6))
    bsptree.draw_nx(plt.gca(), show_labels=False)

    fig = plt.figure()
    ax = plt.gca()
    fovs = []
    for line in rendered_lines:
        fovs.append(line.to_interval(point1))

    fovs = sort_fovs(fovs)
    for fov in fovs:
        color = np.random.rand(1, 3)
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