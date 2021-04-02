import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge
from stonesoup.functions import pol2cart
from stonesoup.types.angle import Bearing
import matplotlib.pyplot as plt

from geometry import Point, LineSegment
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
        plot_planes(tree.left, line_stack, dir_stack+['left'], lines, xlim, ylim)
        plot_planes(tree.right, line_stack, dir_stack+['right'], lines, xlim, ylim)
        line_stack.pop()


def plot_visibility(bsptree, point1):
    l = bsptree.render2(point1)

    # for line in l:
    #     plt.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'sg-')
    #     plt.pause(0.01)

    # Convert to ranges
    lines_c = []
    r_fov = []
    for line in l:
        a = line.Name
        p1 = line.p1.to_polar(point1)
        p2 = line.p2.to_polar(point1)
        phi1 = p1[1]
        phi2 = p2[1]

        if not len(r_fov):
            # First iteration
            r_fov.append([p1, p2, line.Name])
            x1, y1 = pol2cart(p1[0], p1[1])
            x1, y1 = (x1 + point1.x, y1 + point1.y)
            x2, y2 = pol2cart(p2[0], p2[1])
            x2, y2 = (x2 + point1.x, y2 + point1.y)
            plt.plot([x1, x2], [y1, y2], 'sg-')
            plt.pause(0.001)
        else:
            visible = True
            for p_1, p_2, name in r_fov:
                phi_1 = p_1[1]
                phi_2 = p_2[1]
                # Check if angles fall within rendered angles
                phi1_test = test_between(phi1, phi_1, phi_2)
                phi2_test = test_between(phi2, phi_1, phi_2)
                if phi1_test and phi2_test:
                    # If this is true for both angles, then line is not visible
                    visible = False
                    break
                elif phi1_test:
                    # if phi_1 falls between phi1 and phi2
                    if test_between(phi_1, phi1, phi2, not_equals=True):
                        phi1 = phi_1
                        p = p_1
                    # elif phi_2 falls between phi1 and phi2
                    elif test_between(phi_2, phi1, phi2, not_equals=True):
                        phi1 = phi_2
                        p = p_2
                    else:
                        continue
                    x, y = pol2cart(p[0], p[1])
                    x, y = (x + point1.x, y + point1.y)
                    li = LineSegment(point1, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(point1)
                    if test_between(a1, phi_1, phi_2):
                        p1, p2 = (l2.p1.to_polar(point1), l2.p2.to_polar(point1))
                    else:
                        p1, p2 = (l1.p1.to_polar(point1), l1.p2.to_polar(point1))
                elif phi2_test:
                    if test_between(phi_1, phi1, phi2, not_equals=True):
                        phi2 = phi_1
                        p = p_1
                    # elif phi_2 falls between phi1 and phi2
                    elif test_between(phi_2, phi1, phi2, not_equals=True):
                        phi2 = phi_2
                        p = p_2
                    else:
                        continue

                    x, y = pol2cart(p[0], p[1])
                    x, y = (x + point1.x, y + point1.y)
                    li = LineSegment(point1, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(point1)
                    if test_between(a1, phi_1, phi_2):
                        p1, p2 = (l2.p1.to_polar(point1), l2.p2.to_polar(point1))
                    else:
                        p1, p2 = (l1.p1.to_polar(point1), l1.p2.to_polar(point1))

            _, dx = to_range2(p1[1], p2[1])
            if visible and dx != 0:
                r_fov.append([p1, p2, line.Name])
                x1, y1 = pol2cart(p1[0], p1[1])
                x1, y1 = (x1 + point1.x, y1 + point1.y)
                x2, y2 = pol2cart(p2[0], p2[1])
                x2, y2 = (x2 + point1.x, y2 + point1.y)
                plt.plot([x1, x2], [y1, y2], 'sg-')
                plt.pause(0.001)
                bvc = 2


def merge_fovs(fov):
    for i, (phi1_1, phi2_1) in enumerate(fov):
        for j, (phi1_2, phi2_2) in reversed(list(enumerate(fov))):
            if i == j:
                continue
            a = np.equal([phi1_1, phi2_1], [phi1_2, phi2_2])
            b = np.equal([phi2_1, phi1_1], [phi1_2, phi2_2])
            if any(a):
                phi1 = np.array([phi1_1, phi2_1])[~a]
                phi2 = np.array([phi1_2, phi2_2])[~a]
                fov[i][1] = np.amin([phi1, phi2])
                fov[i][0] = np.amax([phi1, phi2])
                fov.pop(j)
            elif any(b):
                phi1 = np.array([phi2_1, phi1_1])[~b]
                phi2 = np.array([phi1_2, phi2_2])[~b]
                fov[i][1] = np.amin([phi1, phi2])
                fov[i][0] = np.amax([phi1, phi2])
                fov.pop(j)
        if fov[i][0]<0 and fov[i][1]>0:
            a = fov[i][0]
            b = fov[i][1]
            fov[i][0] = b
            fov[i][1] = a
            # if any(np.equal([phi1_1, phi1_2], [phi2_1, phi2_2])) or any(np.equal([phi1_2, phi1_1], [phi2_1, phi2_2])):
            #     fov[i][0] = np.amin([phi1_1, phi1_2, phi2_1, phi2_2])
            #     fov[i][1] = np.amax([phi1_1, phi1_2, phi2_1, phi2_2])
            #     fov.pop(j)
    return fov

def merge_fovs2(fov):
    valid_fovs = set()
    for i, (interval_i, name_i) in enumerate(fov):
        for j, (interval_j, name_j) in reversed(list(enumerate(fov))):
            if i == j:
                continue
            if interval_i.intersects(interval_j):
                valid_fovs.add((i, j))

    valid_fovs = list(valid_fovs)
    for i, lines1 in enumerate(valid_fovs):
        for j, lines2 in reversed(list(enumerate(valid_fovs))):
            if i == j:
                continue
            if bool(set(lines1) & set(lines2)):
                lines1 = list(set(lines1) | set(lines2))
                valid_fovs[i] = lines1
                valid_fovs.pop(j)

    merged = []
    for ids in valid_fovs:
        merged_interval = None
        names = []
        processed = []
        while len(ids):
            for idx in ids:
                interval_i = fov[idx][0]
                name_i = fov[idx][1]
                if not merged_interval:
                    merged_interval = interval_i
                    processed.append(idx)
                    asdasd=2
                else:
                    merged1 = merged_interval.union(interval_i)
                    if merged1:
                        merged_interval = merged1
                        processed.append(idx)
                        asda=2
            ids = [idx for idx in ids if idx not in processed]

        names.append(name_i)
        merged.append([merged_interval, names])

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
    num_merged = 1
    lines_copy = lines[:]
    while num_merged>0:
        num_merged = 0
        for i, line1 in enumerate(lines_copy):
            for j, line2 in reversed(list(enumerate(lines_copy))):
                if line1.compare(line2) == 'C':
                    p11, p21 = (line1.p1, line1.p2)
                    p12, p22 = (line2.p1, line2.p2)
                    if p21 == p12:
                        lines_copy[i] = LineSegment(p11, p22, line1.Normal, line1.Name)
                        lines_copy.pop(j)
                        line1 = lines_copy[i]
                        num_merged += 1
                    elif p22 == p11:
                        lines_copy[i] = LineSegment(p12, p21, line2.Normal, line2.Name)
                        lines_copy.pop(j)
                        line1 = lines_copy[i]
                        num_merged += 1
    return lines_copy

def merge_lines2(lines):
    touching_lines = set()
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i==j:
                continue
            p11, p21 = (line1.p1, line1.p2)
            p12, p22 = (line2.p1, line2.p2)
            if p11 == p12 or p11==p22 or p21==p12 or p21==p22:
               touching_lines.add((i, j))

    touching_lines = list(touching_lines)
    for i, lines1 in enumerate(touching_lines):
        for j, lines2 in reversed(list(enumerate(touching_lines))):
            if i == j:
                continue
            if bool(set(lines1) & set(lines2)):
                lines1 = list(set(lines1) | set(lines2))
                touching_lines[i] = lines1
                touching_lines.pop(j)
    merged = []
    for ids in touching_lines:
        lines1 = [(line.p1.to_array(), line.p2.to_array()) for i, line in enumerate(lines) if i in ids]
        merged.append(linemerge(lines1))
    merged_idx = set.union(*[set(item) for item in touching_lines])
    for i, line in enumerate(lines):
        if i in merged_idx:
            continue
        merged.append(LineString((line.p1.to_array(), line.p2.to_array())))

    adas = 2
    return merged


def process_line(line, fov, vis_lines, ref_point, ax=None):

    if ax is None:
        ax = plt.gca()

    a = line.Name
    c1, c2 = (line.p1, line.p2)
    p1, p2 = line.to_polar(ref_point)
    phi1 = p1[1]
    phi2 = p2[1]

    if not len(fov):
        # First iteration
        vis_lines.append(line)
        fov.append([np.amax([phi1, phi2]), np.amin([phi1, phi2])])
        ax.plot([c1.x, c2.x], [c1.y, c2.y], 'sg-')
    else:
        visible = True
        is_pain = False
        for i, (phi_1, phi_2) in enumerate(fov):

            # Check if line angles fall within rendered angles
            phi1_test = test_between(phi1, phi_1, phi_2)  # phi_1 <-> phi1 <-> phi_2
            phi2_test = test_between(phi2, phi_1, phi_2)  # phi_1 <-> phi2 <-> phi_2

            if phi1_test and phi2_test:
                # If this is true for both angles, then line is not visible
                visible = False
                break
            elif phi1_test or phi2_test:
                # Elif phi1 or phi2 is within rendered fov, but not both
                if test_between(phi_1, phi1, phi2, not_equals=True):
                    # if phi_1 falls between phi1 and phi2 (i.e. phi_2 <-> phi1 <-> phi_1 <-> phi2)
                    phi1 = phi_1 if phi1_test else phi1
                    phi2 = phi_1 if phi2_test else phi2
                elif test_between(phi_2, phi1, phi2, not_equals=True):
                    phi1 = phi_2 if phi1_test else phi1
                    phi2 = phi_2 if phi2_test else phi2
                else:
                    continue

                # Split existing line, according to fov
                phi = phi1 if phi1_test else phi2
                x, y = pol2cart(1., phi)
                x, y = (x + ref_point.x, y + ref_point.y)
                li = LineSegment(ref_point, Point(x, y))
                line = LineSegment(c1, c2, line.Normal, line.Name)
                l1, l2 = li.split(line)

                _, a1 = l1.p1.to_polar(ref_point)
                if test_between(a1, phi_1, phi_2):
                    c1, c2 = (l2.p1, l2.p2)
                    p1, p2 = l2.to_polar(ref_point)
                else:
                    c1, c2 = (l1.p1, l1.p2)
                    p1, p2 = l1.to_polar(ref_point)
            # else:
            #     # Check if rendered angles fall within line angles
            #     phi_1_test = test_between(phi_1, phi1, phi2)  # phi_1 <-> phi1 <-> phi_2
            #     phi_2_test = test_between(phi_2, phi1, phi2)  # phi_1 <-> phi2 <-> phi_2
            #
            #     if phi_1_test and phi_2_test:
            #         # Split existing line, according to fov
            #         x, y = pol2cart(1., phi1)
            #         x, y = (x + ref_point.x, y + ref_point.y)
            #         li = LineSegment(ref_point, Point(x, y))
            #         l1, l2 = li.split(line)
            #
            #         _, a1 = l1.p1.to_polar(ref_point)
            #         if test_between(a1, phi1, phi2):
            #             x, y = pol2cart(1., phi2)
            #             x, y = (x + ref_point.x, y + ref_point.y)
            #             li = LineSegment(ref_point, Point(x, y))
            #             l2, l3 = li.split(l2)
            #         else:
            #             x, y = pol2cart(1., phi2)
            #             x, y = (x + ref_point.x, y + ref_point.y)
            #             li = LineSegment(ref_point, Point(x, y))
            #             l2, l3 = li.split(l1)
            #         fov, vis_lines = process_line(l1, fov, vis_lines, ref_point, ax)
            #         fov, vis_lines = process_line(l2, fov, vis_lines, ref_point, ax)
            #         fov, vis_lines = process_line(l3, fov, vis_lines, ref_point, ax)

        _, dx = to_range2(p1[1], p2[1])
        if visible and dx != 0 and not is_pain:
            fov.append([np.amax([p1[1], p2[1]]), np.amin([p1[1], p2[1]])])
            # x1, y1 = pol2cart(p1[0], p1[1])
            # x1, y1 = (x1 + ref_point.x, y1 + ref_point.y)
            # x2, y2 = pol2cart(p2[0], p2[1])
            # x2, y2 = (x2 + ref_point.x, y2 + ref_point.y)
            line = LineSegment(c1, c2, line.Normal, line.Name)
            vis_lines.append(line)

            ax.plot([c1.x, c2.x], [c1.y, c2.y], 'sg-')
            plt.pause(0.001)
            asd = 2
    return fov, vis_lines


def plot_visibility2(bsptree, ref_point, ax=None):

    l = bsptree.render2(ref_point)

    # Convert to ranges
    vis_lines = []
    fov = []
    for idx, line in enumerate(l):
        fov, vis_lines = process_line3(line, fov, vis_lines, ref_point, ax)
        fov = merge_fovs2(fov)

    merged = merge_lines(vis_lines)
    vfd=2
    return vis_lines #merged


def process_line2(line, fov, vis_lines, ref_point, ax=None):

    if ax is None:
        ax = plt.gca()

    a = line.Name
    c1, c2 = (line.p1, line.p2)
    p1, p2 = line.to_polar(ref_point)
    phi1 = p1[1]
    phi2 = p2[1]

    if not len(fov):
        # First iteration
        vis_lines.append(line)
        mid, dx = to_range2(phi1, phi2)
        fov.append([mid, dx, line.Name])
        ax.plot([c1.x, c2.x], [c1.y, c2.y], 'sg-')
    else:
        visible = True
        is_pain = False
        for i, (phi_0, dx, name) in enumerate(fov):

            # Check if line angles fall within rendered angles
            phi1_test = test2(phi1, phi_0, dx)  # phi_1 <-> phi1 <-> phi_2
            phi2_test = test2(phi2, phi_0, dx)  # phi_1 <-> phi2 <-> phi_2

            if phi1_test and phi2_test:
                # If this is true for both angles, then line is not visible
                visible = False
                break
            elif phi1_test:
                # Elif phi1 is within rendered fov, but not phi2
                phi0, dx1 = to_range2(phi1, phi2)
                t1 = test2(phi_0-dx, phi0, dx1, not_equals=True) # phi_0-dx between phi1 and phi2
                t2 = test2(phi_0+dx, phi0, dx1, not_equals=True) # phi_0+dx between phi1 and phi2
                if t1:
                    phi1 = phi_0-dx
                elif t2:
                    phi1 = phi_0+dx
                else:
                    continue

                # Split existing line, according to fov
                x, y = pol2cart(1., phi1)
                x, y = (x + ref_point.x, y + ref_point.y)
                li = LineSegment(ref_point, Point(x, y))
                line = LineSegment(c1, c2, line.Normal, line.Name)
                l1, l2 = li.split(line)

                _, a1 = l1.p1.to_polar(ref_point)
                if test2(a1, phi_0, dx):
                    c1, c2 = (l2.p1, l2.p2)
                    p1, p2 = l2.to_polar(ref_point)
                else:
                    c1, c2 = (l1.p1, l1.p2)
                    p1, p2 = l1.to_polar(ref_point)
            elif phi2_test:
                # Elif phi1 is within rendered fov, but not phi2
                phi0, dx1 = to_range2(phi1, phi2)
                t1 = test2(phi_0-dx, phi0, dx1, not_equals=True) # phi_0-dx between phi1 and phi2
                t2 = test2(phi_0+dx, phi0, dx1, not_equals=True) # phi_0+dx between phi1 and phi2
                if t1:
                    phi2 = phi_0-dx
                elif t2:
                    phi2 = phi_0+dx
                else:
                    continue

                # Split existing line, according to fov
                x, y = pol2cart(1., phi2)
                x, y = (x + ref_point.x, y + ref_point.y)
                li = LineSegment(ref_point, Point(x, y))
                line = LineSegment(c1, c2, line.Normal, line.Name)
                l1, l2 = li.split(line)

                _, a1 = l1.p1.to_polar(ref_point)
                if test2(a1, phi_0, dx):
                    c1, c2 = (l2.p1, l2.p2)
                    p1, p2 = l2.to_polar(ref_point)
                else:
                    c1, c2 = (l1.p1, l1.p2)
                    p1, p2 = l1.to_polar(ref_point)

        _, dx = to_range2(p1[1], p2[1])
        if visible and dx != 0 and not is_pain:
            phi0, dx1 = to_range2(p1[1], p2[1])
            fov.append([phi0, dx1, line.Name])
            # x1, y1 = pol2cart(p1[0], p1[1])
            # x1, y1 = (x1 + ref_point.x, y1 + ref_point.y)
            # x2, y2 = pol2cart(p2[0], p2[1])
            # x2, y2 = (x2 + ref_point.x, y2 + ref_point.y)
            line = LineSegment(c1, c2, line.Normal, line.Name)
            vis_lines.append(line)

            ax.plot([c1.x, c2.x], [c1.y, c2.y], 'sg-')
            plt.pause(0.001)
            asd = 2
    return fov, vis_lines


def process_line3(line, fov, vis_lines, ref_point, ax=None):

    if ax is None:
        ax = plt.gca()

    a = line.Name
    p1, p2 = line.to_polar(ref_point)
    mid, dx = to_range2(p1[1], p2[1])
    interval = AngleInterval(mid, dx)
    # direction = 1 if np.isclose(interval.min-phi1, 0, atol=1e-10) else 0

    if not len(fov):
        # First iteration
        vis_lines.append(line)
        fov.append([interval, line.Name])
        x, y = line.xy
        ax.plot(x, y, 'sg-')
    else:
        visible = True
        is_pain = False
        # i = 0
        # while i < len(fov):
        #     interval_i = fov[i][0]
        #     name = fov[i][1]
        for i, (interval_i, name) in enumerate(fov):

            if interval_i.contains_interval(interval):
                visible = False
                break
            # elif interval.contains_interval(interval_i, True):
            #     # Split existing line, according to fov
            #     x, y = pol2cart(1., interval_i.min)
            #     x, y = (x + ref_point.x, y + ref_point.y)
            #     li = LineSegment(ref_point, Point(x, y))
            #     l1, l2 = li.split(line)
            #
            #     for ll in [l1, l2]:
            #         fov, vis_lines = process_line3(ll, fov, vis_lines, ref_point, ax)
            #     visible = False

            elif interval.intersects(interval_i):
                if interval_i.contains_angle(interval.min, not_equals=True):
                    # if direction:
                    #     phi1 = interval_i.max
                    # else:
                    #     phi2 = interval_i.max

                    # Split existing line, according to fov
                    x, y = pol2cart(1., interval_i.max)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if interval_i.contains_angle(a1, True):
                        line = l2
                        p1, p2 = l2.to_polar(ref_point)
                    else:
                        line = l1
                        p1, p2 = l1.to_polar(ref_point)

                    # Generate new line and interval
                    phi0, dx1 = to_range2(p1[1], p2[1])
                    interval = AngleInterval(phi0, dx1)
                elif interval_i.contains_angle(interval.max, not_equals=True):
                    # if direction:
                    #     phi2 = interval_i.min
                    # else:
                    #     phi1 = interval_i.min

                    # Split existing line, according to fov
                    x, y = pol2cart(1., interval_i.min)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if interval_i.contains_angle(a1, True):
                        line = l2
                        p1, p2 = l2.to_polar(ref_point)
                    else:
                        line = l1
                        p1, p2 = l1.to_polar(ref_point)

                    # Generate new line and interval
                    phi0, dx1 = to_range2(p1[1], p2[1])
                    interval = AngleInterval(phi0, dx1)
            i += 1

        if visible and not is_pain:
            fov.append([interval, line.Name])
            vis_lines.append(line)
            x, y = line.xy
            ax.plot(x, y, 'sg-')
            plt.pause(0.001)
            asd = 2
    return fov, vis_lines