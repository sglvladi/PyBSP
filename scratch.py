import pickle
import matplotlib.pyplot as plt
from geometry import LineSegment, Point
import numpy as np
from copy import copy
from shapely.geometry import Polygon

from bsp2 import BSP, BSP2

def extrapolate_line(line, xlim, ylim):
    dx = line.p2.x - line.p1.x
    dy = line.p2.y - line.p1.y
    if dx == 0:
        if dy>0:
            y = np.array([ylim[0], ylim[1]])
        else:
            y = np.array([ylim[1], ylim[0]])
        x = np.ones((len(y),)) * line.p2.x
    else:
        m = dy / dx
        if dx>0:
            x = np.array([xlim[0], xlim[1]])
        else:
            x = np.array([xlim[1], xlim[0]])
        y = m * (x - line.p1.x) + line.p1.y
    return x, y


def get_lines(tree):
    lines = []
    if tree:
        for line in tree.data:
            lines.append(line)
        lines += get_lines(tree.left)
        lines += get_lines(tree.right)
    return lines


def plot_lines(tree, line_stack=[], dir_stack=[], lines=[], xlim=(0,800), ylim=(0,800), annotate=False):

    if tree:
        line = tree.data[0]
        x, y = extrapolate_line(line, xlim, ylim)
        c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.Normal, line.Name)
        for p_l, dir in zip(reversed(line_stack), reversed(dir_stack)):
            ls = p_l.split(c_l)
            if ls:
                l1, l2 = ls

                # Unit Vector for normal
                vp = [p_l.NormalV.x, p_l.NormalV.y]
                up = vp / np.linalg.norm(vp)

                # Unit Vector for l1
                v1 = [l1.p1.x-l1.p2.x, l1.p1.y-l1.p2.y]
                u1 = v1 / np.linalg.norm(v1)
                t1 = np.arccos(np.dot(u1,up))

                # Unit Vector for l2
                v2 = [l2.p2.x-l2.p1.x, l2.p2.y-l2.p1.y]
                u2 = v2 / np.linalg.norm(v2)
                t2 = np.arccos(np.dot(u2,up))

                if dir == 'left':
                    if t1 < np.pi/2:
                        l = l1
                    else:
                        l = l2
                elif dir == 'right':
                    if t1 > np.pi/2:
                        l = l1
                    else:
                        l = l2

                x, y = ([l.p1.x, l.p2.x], [l.p1.y, l.p2.y])

                c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.Normal, line.Name)
        # midPoint = c_l.getMidPoint()
        # plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001,
        # headwidth=0.2)
        plt.plot(x, y, 'm-', linewidth=0.2)
        if annotate:
            midPoint = c_l.getMidPoint()
            plt.text(midPoint.x, midPoint.y, c_l.Name)
        # plt.pause(0.1)
        a=2
        line_stack.append(c_l)
        lines.append(c_l)
        plot_lines(tree.left, line_stack, dir_stack+['left'], lines, xlim, ylim)
        plot_lines(tree.right, line_stack, dir_stack+['right'], lines, xlim, ylim)
        line_stack.pop()


def main():
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

    def merc_from_arrays(lats, lons):
        r_major = 6378137.000
        x = r_major * np.radians(lons)
        scale = x/lons
        y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lats * (np.pi/180.0)/2.0)) * scale
        return (x, y)


    xs, ys = merc_from_arrays(np.array([LAT_MIN, LAT_MAX]), np.array([LON_MIN, LON_MAX]))
    X_MIN, X_MAX = xs
    Y_MIN, Y_MAX = ys
    target_polygon = Polygon([(X_MIN, Y_MIN), (X_MIN, Y_MAX), (X_MAX, Y_MAX), (X_MAX, Y_MIN)])
    polygons = pickle.load(open('polygons.p', 'rb'))
    print('Loaded polygons')

    plt.figure(figsize=(8, 6))

    print('Generating lines')
    lines = []
    for polygon in polygons:
        if target_polygon.contains(polygon):
            x, y = polygon.exterior.xy
            # plt.plot(x, y)

            points = []
            for xi, yi in zip(x,y):
                points.append(Point(xi, yi))

            for i in range(len(points)):
                if i == len(points)-1:
                    j = 0
                else:
                    j = i+1
                p1 = points[i]
                p2 = points[j]
                p11 = (p1.x, p1.y)
                p22 = (p2.x, p2.y)
                if p11 != p22:
                    lines.append(LineSegment(p1, p2, 1, str(len(lines))))

    for line in lines:
        x = (line.p1.x, line.p2.x)
        y = (line.p1.y, line.p2.y)
        # p1 = (line.p1.x, line.p1.y)
        # p2 = (line.p2.x, line.p2.y)
        # if p1==p2:
        #     a=2
        plt.plot(x, y, 'k-')

        # midPoint = line.getMidPoint()
        # plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)

    print('Creating tree')
    bsptree = BSP()
    bsptree.tree.data = copy(lines)
    bsptree.generate_tree(bsptree.tree, heuristic='even')
    # bsptree.draw_nx()

    # bsptree2 = BSP2(lines, heuristic='even')
    # bsptree2.draw_nx()
    print(bsptree.tree.print())
    xlim = plt.xlim()
    ylim = plt.ylim()

    # l = get_lines(bsptree.tree)
    # plot_lines(l, xlim, ylim)

    ls = []
    plot_lines(bsptree.tree, lines=ls, xlim=xlim, ylim=ylim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    bsptree.draw_nx()

    print(bsptree.find_leaf(Point(1.58, 4.275)).data[0].Name)
    #
    # print('Plotting')
    # x, y = target_polygon.exterior.xy
    # plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main()