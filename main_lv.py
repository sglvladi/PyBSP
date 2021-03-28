import sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from copy import copy

from bsp import BSP
from geometry import LineSegment, Point


def sign(x): return (x > 0) - (x < 0)

def generateRandom(n, Range, a=3, isPowerLaw=False):
    """
    Generate a random number
    :param n: int, Number of numbers to return
    :param Range: float, maximum range of a number to be generated
    :param a: float, for use in powerlaw distribution
    :param isPowerLaw: boolean, generate with powerlaw distribution if true else generate with uniform distribution
    :return: integer or list of numbers depending on argument n
    """
    if not isPowerLaw:
        if n > 1:
            return list(np.random.uniform(0, Range, n))
        else:
            return np.random.uniform(0, Range)

    else:
        if n > 1:
            return list(np.random.power(a, n) * Range)
        else:
            return np.random.power(a) * Range

def generatePolygons():
    Lines = []

    # Polygon 1
    offset_x = 100
    offset_y = 100
    width = 100
    height = 100
    p1 = Point(offset_x, offset_y)
    p2 = Point(offset_x, offset_y+height)
    p3 = Point(offset_x+width, offset_y)
    p4 = Point(offset_x+width, offset_y+height)
    Lines.append(LineSegment(p1, p2, -1, '1'))
    Lines.append(LineSegment(p1, p3, 1, '2'))
    Lines.append(LineSegment(p3, p4, 1, '3'))
    Lines.append(LineSegment(p2, p4, -1, '4'))

    # Polygon 2
    p1 = Point(200, 600)
    p2 = Point(450, 600)
    p3 = Point(300, 700)
    Lines.append(LineSegment(p1, p2, 1, '5'))
    Lines.append(LineSegment(p2, p3, 1, '6'))
    Lines.append(LineSegment(p3, p1, 1, '7'))

    # Polygon 3
    p1 = Point(500, 150)
    p2 = Point(600, 250)
    p3 = Point(500, 350)
    p4 = Point(400, 250)
    Lines.append(LineSegment(p1, p2, 1, '8'))
    Lines.append(LineSegment(p2, p3, 1, '9'))
    Lines.append(LineSegment(p3, p4, 1, '10'))
    Lines.append(LineSegment(p4, p1, 1, '11'))

    # Polygon 4
    p1 = Point(0, 0)
    p2 = Point(800, 0)
    p3 = Point(800, 800)
    p4 = Point(0, 800)
    Lines.append(LineSegment(p1, p2, -1, '12'))
    Lines.append(LineSegment(p2, p3, -1, '13'))
    Lines.append(LineSegment(p3, p4, -1, '14'))
    Lines.append(LineSegment(p4, p1, -1, '15'))

    return Lines


def generatePoints(n, width, height, isUniform=True):
    """
    Randomnly generates a list of points
    :param n: int, Number of points
    :param width: int, area width
    :param height: int, area height
    :param isUniform: boolean, whether the position of line segments should be generated with random number of uniform distribution or powerlaw distribution
    :return: list of points
    """
    Points = []
    for i in range(n):
        if isUniform:
            Points.append(Point(int(round(generateRandom(1, width))),
                                int(round(generateRandom(1, height)))))
        else:
            Points.append(Point(int(round(generateRandom(1, width, isPowerLaw=True))), int(
                round(generateRandom(1, height, isPowerLaw=True)))))

    return Points


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


# def get_lines(lines, tree, parent=None, dir=None, xlim=(0,800), ylim=(0,800)):

def plot_lines(tree, line_stack=[], dir_stack=[], xlim=(0,800), ylim=(0,800)):

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
        # plt.pause(0.1)
        a=2
        line_stack .append(c_l)
        plot_lines(tree.left, line_stack, dir_stack+['left'], xlim)
        plot_lines(tree.right, line_stack, dir_stack+['right'], ylim)
        line_stack.pop()


def plot_tree(tree, parent=None, dir=None, xlim=(0,800), ylim=(0,800)):

    if dir=='left':
        tree = tree.left
    elif dir=='right':
        tree = tree.right

    if tree:
        if parent:
            line = parent[0]
            x, y = extrapolate_line(line, xlim, ylim)
            p_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]))

        for line in tree.data:
            x, y = extrapolate_line(line, xlim, ylim)

            if parent:
                c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]))
                ls = p_l.split(c_l)
                if ls:
                    l1, l2 = ls
                    if dir=='left':
                        x, y = ([l1.p1.x, l1.p2.x], [l1.p1.y, l1.p2.y])
                    elif dir=='right':
                        x, y = ([l2.p1.x, l2.p2.x], [l2.p1.y, l2.p2.y])



            # if dx == 0:
            #     y = np.linspace(max(0, p.y), min(800, p.y), 100)
            #     x = np.ones((len(y),))*line.p2.x
            # else:
            #     m = dy / dx
            #     x = np.linspace(0, 800, 100)
            #     y = m*(x-line.p1.x)+line.p1.y
            l = LineSegment(Point(x[0], y[0]), Point(x[1], y[1]), line.Normal)
            midPoint = l.getMidPoint()
            plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)
            plt.plot(x, y,'m-', linewidth=0.1)
        plot_tree(tree, tree.data, 'left')
        plot_tree(tree, tree.data, 'right')

# def tree_to_graph(bsptree):


# def plot_tree(tree):
#     if tree:
#         for line in tree.data:
#             dx = line.p2.x-line.p1.x
#             dy = line.p2.y-line.p1.y
#             if dx == 0:
#                 y = np.linspace(0, 800, 100)
#                 x = np.ones((len(y),))*line.p2.x
#             else:
#                 m = dy / dx
#                 x = np.linspace(0, 800, 100)
#                 y = m*(x-line.p1.x)+line.p1.y
#             plt.plot(x,y,'m-', linewidth=0.1)
#         plot_tree(tree.left)
#         plot_tree(tree.right)

def main():

    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    SHOW_NORMALS = True

    bsptree = BSP()
    print('Generating line segments')
    lines = generatePolygons()
    bsptree.tree.data = copy(lines)

    points = generatePoints(4, SCREEN_WIDTH, SCREEN_HEIGHT, isUniform=True)

    plt.figure(figsize=(8, 6))
    for line in lines:
        x = [line.p1.x, line.p2.x]
        y = [line.p1.y, line.p2.y]
        plt.plot(x, y, 'k-')
        if SHOW_NORMALS:
            midPoint = line.getMidPoint()
            plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)

    # for point in points:
    #     plt.plot(point.x, point.y, 'or', markersize=4)

    print('Generating tree')
    bsptree.generateTree(bsptree.tree, UseHeuristic='min')

    # plot_tree(bsptree.tree)
    plot_lines(bsptree.tree)
    # plt.plot([0, 0, SCREEN_WIDTH, SCREEN_WIDTH, 0], [0, SCREEN_HEIGHT, SCREEN_HEIGHT, 0, 0], 'k-')

    point1 = Point(40, 400)

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

    print(bsptree.tree.printTree())
    plt.axis('equal')
    plt.xlim((0, SCREEN_WIDTH))
    plt.ylim((0, SCREEN_HEIGHT))
    plt.legend(loc="upper right", bbox_to_anchor=(0.85,0.99))
    plt.show()

if __name__ == '__main__':
    main()