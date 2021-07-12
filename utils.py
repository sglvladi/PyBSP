import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge
from stonesoup.functions import pol2cart
from stonesoup.types.angle import Bearing
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.components.connected import connected_components

from geometry import Point, LineSegment, MergedLine
from angles import mid_angle, to_range2, AngleInterval

def test2(X, alpha, delta, not_equals=False):
    a = (alpha - delta < X < alpha + delta)
    d = (np.isclose(float(alpha+delta-X), 0., atol=1e-15) or np.isclose(float(alpha-delta-X), 0., atol=1e-15))
    b = (alpha + 2*np.pi - delta <= X <= 2*np.pi)
    c = (0 <= X <= alpha - 2*np.pi + delta)
    # d = (X != alpha - delta and X!= alpha + delta)
    if not_equals:
        return a and not d  #'(a or b or c) and d
    else:
        return a or d # a or b or c

def test_between(phi, phi1, phi2, not_equals=False):
    mid, dx = to_range2(float(phi1), float(phi2))
    if not_equals:
        return test2(float(phi), mid, dx) and phi != phi1 and phi != phi2
    else:
        return test2(float(phi), mid, dx) or phi == phi1 or phi == phi2


def extrapolate_line(line, xlim, ylim):
    dx = line.p2.x - line.p1.x
    dy = line.p2.y - line.p1.y
    if dx == 0:
        if dy > 0:
            y = np.array([ylim[0], ylim[1]])
        else:
            y = np.array([ylim[1], ylim[0]])
        x = np.ones((len(y),)) * line.p2.x
    else:
        m = dy / dx
        if dx > 0:
            x = np.array([xlim[0], xlim[1]])
        else:
            x = np.array([xlim[1], xlim[0]])
        y = m * (x - line.p1.x) + line.p1.y
    return x, y

def plot_planes(tree, line_stack=[], dir_stack=[], lines=[], xlim=(0,800), ylim=(0,800), annotate=False):

    if tree:
        line = tree.data[0]
        x, y = extrapolate_line(line, xlim, ylim)
        c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.normal, line.name)
        for p_l, dir in zip(reversed(line_stack), reversed(dir_stack)):
            ls = p_l.split(c_l)
            if ls:
                l1, l2 = ls

                # Unit Vector for normal
                vp = [p_l.normalV.x, p_l.normalV.y]
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

                c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.normal, line.name)
        # midPoint = c_l.mid_point
        # plt.quiver(midPoint.x, midPoint.y, line.normalV.x, line.normalV.y, width=0.001,
        # headwidth=0.2)
        plt.plot(x, y, plt.plot(x, y, 'm-', linewidth=0.2))
        if annotate:
            midPoint = c_l.mid_point
            plt.text(midPoint.x, midPoint.y, c_l.name)
        # plt.pause(0.1)
        a=2
        line_stack.append(c_l)
        lines.append(c_l)
        plot_planes(tree.left, line_stack, dir_stack+['left'], lines, xlim, ylim)
        plot_planes(tree.right, line_stack, dir_stack+['right'], lines, xlim, ylim)
        line_stack.pop()


def merge_lists(l):
    def to_graph(l):
        G = nx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """
            treat `l` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    G = to_graph(l)
    return [t for t in connected_components(G)]



def merge_fovs2(fov):
    valid_fovs = set()
    for i, interval_i in enumerate(fov):
        for j, interval_j in reversed(list(enumerate(fov))):
            if i == j:
                continue
            if interval_i.intersects(interval_j):
                valid_fovs.add((i, j))

    valid_fovs = list(valid_fovs)
    valid_fovs = merge_lists(valid_fovs)
    # for i, lines1 in enumerate(valid_fovs):
    #     for j, lines2 in reversed(list(enumerate(valid_fovs))):
    #         if i == j:
    #             continue
    #         if bool(set(lines1) & set(lines2)):
    #             lines1 = list(set(lines1) | set(lines2))
    #             valid_fovs[i] = lines1
    #             valid_fovs.pop(j)

    merged = []
    for ids in valid_fovs:
        merged_interval = None
        names = set()
        processed = []
        while len(ids):
            for idx in ids:
                interval_i = fov[idx]
                if not merged_interval:
                    merged_interval = interval_i
                    processed.append(idx)
                else:
                    merged1 = merged_interval.union(interval_i)
                    if merged1:
                        merged_interval = merged1
                        processed.append(idx)
                        asda=2
                # if isinstance(name_i, str):
                #     names.add(name_i)
                # else:
                #     names |= name_i
            ids = [idx for idx in ids if idx not in processed]
        merged.append(merged_interval)

    if len(valid_fovs):
        merged_idx = set.union(*[set(item) for item in valid_fovs])
        for i, line in enumerate(fov):
            if i in merged_idx:
                continue
            merged.append(line)
    else:
        merged = fov

    return merged


def merge_lines(lines):
    touching_lines = set()
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i == j:
                continue
            p11, p21 = (line1.p1, line1.p2)
            p12, p22 = (line2.p1, line2.p2)
            if p11 == p12 or p11 == p22 or p21 == p12 or p21 == p22:
                touching_lines.add((i, j))

    t_lines = merge_lists(touching_lines)

    merged = []
    for ids in t_lines:
        points = [(line.p1.to_array(), line.p2.to_array()) for i, line in enumerate(lines) if i in ids]
        lines1 = [line for i, line in enumerate(lines) if i in ids]
        linestring = linemerge(points)
        merged.append(MergedLine(lines1, linestring))
    merged_idx = set.union(*[set(item) for item in touching_lines])
    for i, line in enumerate(lines):
        if i in merged_idx:
            continue
        merged.append(MergedLine([line], LineString((line.p1.to_array(), line.p2.to_array()))))

    return merged


def plot_visibility2(bsptree, ref_point, ax=None):

    l = bsptree.render2(ref_point)

    # Convert to ranges
    vis_lines = []
    fov = []
    for idx, line in enumerate(l):
        fov, vis_lines = process_line(line, fov, vis_lines, ref_point, ax)
        fov = merge_fovs2(fov)
        if len(fov) == 1 and fov[0].delta==np.pi:
            break

    merged = merge_lines(vis_lines)
    return merged


def process_line(line, fov, vis_lines, ref_point, ax=None):

    if ax is None:
        ax = plt.gca()

    a = line.name
    interval = line.to_interval(ref_point)

    if not len(fov):
        # First iteration
        vis_lines.append(line)
        fov.append(interval)
        # x, y = line.xy
        # ax.plot(x, y, 'sg-')
    else:
        visible = True
        split_lines = []
        for i, interval_i in enumerate(fov):

            if interval_i.contains_interval(interval):
                visible = False
                break
            elif interval.contains_interval(interval_i, True):
                # Split existing line, according to fov
                x, y = pol2cart(1e15, interval_i.min)
                x, y = (x + ref_point.x, y + ref_point.y)
                li = LineSegment(ref_point, Point(x, y))
                split_lines = li.split(line)
                visible = False
                break
            elif interval.intersects(interval_i):
                clip_min = interval_i.contains_angle(interval.min, not_equals=True)
                clip_max = interval_i.contains_angle(interval.max, not_equals=True)
                if not (clip_min or clip_max):
                    continue
                if clip_min:
                    phi = interval_i.max
                    x, y = pol2cart(1e15, phi)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if interval_i.contains_angle(a1, True):
                        line = l2
                        # p1, p2 = l2.to_polar(ref_point)
                    else:
                        line = l1
                        # p1, p2 = l1.to_polar(ref_point)

                    # Generate new line and interval
                    # phi0, dx1 = to_range2(p1[1], p2[1])
                    # interval = AngleInterval(phi0, dx1)
                    interval = line.to_interval(ref_point)
                if clip_max:
                    phi = interval_i.min
                    # Split existing line, according to fov
                    x, y = pol2cart(1e15, phi)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if interval_i.contains_angle(a1, True):
                        line = l2
                        # p1, p2 = l2.to_polar(ref_point)
                    else:
                        line = l1
                        # p1, p2 = l1.to_polar(ref_point)

                    # Generate new line and interval
                    # phi0, dx1 = to_range2(p1[1], p2[1])
                    # interval = AngleInterval(phi0, dx1)
                    interval = line.to_interval(ref_point)
                # else:
                #     continue
                # # Split existing line, according to fov
                # x, y = pol2cart(1e15, phi)
                # x, y = (x + ref_point.x, y + ref_point.y)
                # li = LineSegment(ref_point, Point(x, y))
                # l1, l2 = li.split(line)
                #
                # _, a1 = l1.p1.to_polar(ref_point)
                # if interval_i.contains_angle(a1, True):
                #     line = l2
                #     # p1, p2 = l2.to_polar(ref_point)
                # else:
                #     line = l1
                #     # p1, p2 = l1.to_polar(ref_point)
                #
                # # Generate new line and interval
                # # phi0, dx1 = to_range2(p1[1], p2[1])
                # # interval = AngleInterval(phi0, dx1)
                # interval = line.to_interval(ref_point)

        if visible:
            fov.append(interval)
            vis_lines.append(line)

            # Plot line
            # x, y = line.xy
            # ax.plot(x, y, 'sg-')
            # plt.pause(0.001)
            aas=2

        for line in split_lines:
            fov, vis_lines = process_line(line, fov, vis_lines, ref_point, ax)
            fov = merge_fovs2(fov)

    return fov, vis_lines


def sort_fovs(fovs):
    sorted_idx = np.argsort([fov.mid for fov in fovs])
    sorted = [fovs[idx] for idx in sorted_idx]
    return sorted


def plot_nodes(nodes, **kwargs):
    artists = []
    for node in nodes:
        artists.append(node.polygon.plot(**kwargs))
    return artists


def remove_artists(artists):
    for artist in artists:
        artist.remove()


def plot_ex(bsptree, pvs):
    node = bsptree.root
    _plot_ex(node, pvs)

def _plot_ex(node, pvs):
    if node.data and len(node.data)>1:
        a=2
    empty_leaves = [n for n in node.leaf_children if n.is_empty]
    if all([n in pvs for n in empty_leaves]):
        node.data[0].plot()
        a=2
    if node.front is not None and not node.front.is_leaf:
        _plot_ex(node.front, pvs)
    if node.back is not None and not node.back.is_leaf:
        _plot_ex(node.back, pvs)
