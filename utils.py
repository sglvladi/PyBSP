import numpy as np
from geometry import Point, LineSegment
from stonesoup.functions import pol2cart
import matplotlib.pyplot as plt


def mid_angle(a1, a2):
    vec1 = (np.sin(a1), np.cos(a1))

    vec2 = (np.sin(a2), np.cos(a2))

    VecSum = tuple(sum(x) for x in zip(vec1, vec2))

    FinalVec = VecSum
    FinalVec = tuple(x / (np.sqrt(VecSum[0] ** 2 + VecSum[1] ** 2)) for x in FinalVec)

    return np.arctan2(FinalVec[0], FinalVec[1])


def to_range2(a1, a2):
    mid = mid_angle(a1, a2)
    dx = np.abs(a1 - mid)
    return mid, dx


def test2(X, alpha, delta):
    return ((alpha - delta <= X <= alpha + delta)
            or (alpha + 2*np.pi - delta <= X <= 2*np.pi)
            or (0 <= X <= alpha - 2*np.pi + delta))


def test_between(phi, phi1, phi2, not_equals=False):
    mid, dx = to_range2(float(phi1), float(phi2))
    if not_equals:
        return test2(float(phi), mid, dx) and phi != phi1 and phi!=phi2
    else:
        return test2(float(phi), mid, dx) or phi == phi1 or phi==phi2


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





def plot_visibility2(bsptree, ref_point):

    l = bsptree.render2(ref_point)

    # Convert to ranges
    vis_lines = []
    fov = []
    for line in l:
        a = line.Name
        c1, c2 = (line.p1, line.p2)
        p1, p2 = line.to_polar(ref_point)
        phi1 = p1[1]
        phi2 = p2[1]

        if not len(fov):
            # First iteration
            vis_lines.append(line)
            fov.append([np.amax([phi1, phi2]), np.amin([phi1, phi2])])
        else:
            visible = True
            for i, (phi_1, phi_2) in enumerate(fov):

                # Check if angles fall within rendered angles
                phi1_test = test_between(phi1, phi_1, phi_2)  # phi_1 <-> phi1 <-> phi_2
                phi2_test = test_between(phi2, phi_1, phi_2)  # phi_1 <-> phi2 <-> phi_2

                if phi1_test and phi2_test:
                    # If this is true for both angles, then line is not visible
                    visible = False
                    break
                elif phi1_test:
                    # Elif phi1 is within rendered fov, but phi2
                    if test_between(phi_1, phi1, phi2, not_equals=True):
                        # if phi_1 falls between phi1 and phi2 (i.e. phi_2 <-> phi1 <-> phi_1 <-> phi2)
                        phi1 = phi_1
                    elif test_between(phi_2, phi1, phi2, not_equals=True):
                        phi1 = phi_2
                    else:
                        continue

                    # Split existing line, according to fov
                    x, y = pol2cart(1., phi1)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if test_between(a1, phi_1, phi_2):
                        c1, c2 = (l2.p1, l2.p2)
                        p1, p2 = l2.to_polar(ref_point)
                    else:
                        c1, c2 = (l1.p1, l1.p2)
                        p1, p2 = l1.to_polar(ref_point)

                elif phi2_test:
                    # Elif phi2 is within rendered fov, but phi1 is not
                    # then we need to extend current fov to include phi1
                    if test_between(phi_1, phi1, phi2, not_equals=True):
                        # if phi_1 falls between phi1 and phi2 (i.e. phi_2 <-> phi2 <-> phi_1 <-> phi1)
                        phi2 = phi_1
                    elif test_between(phi_2, phi1, phi2, not_equals=True):
                        # elif phi_2 falls between phi1 and phi2 (i.e. phi1 <-> phi_2 <-> phi2 <->  phi_1)
                        phi2 = phi_2
                    else:
                        continue

                    x, y = pol2cart(1., phi2)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    li = LineSegment(ref_point, Point(x, y))
                    l1, l2 = li.split(line)

                    _, a1 = l1.p1.to_polar(ref_point)
                    if test_between(a1, phi_1, phi_2):
                        c1, c2 = (l2.p1, l2.p2)
                        p1, p2 = l2.to_polar(ref_point)
                    else:
                        c1, c2 = (l1.p1, l1.p2)
                        p1, p2 = l1.to_polar(ref_point)

            _, dx = to_range2(p1[1], p2[1])
            if visible and dx != 0:
                fov.append([np.amax([p1[1], p2[1]]), np.amin([p1[1], p2[1]])])
                # x1, y1 = pol2cart(p1[0], p1[1])
                # x1, y1 = (x1 + ref_point.x, y1 + ref_point.y)
                # x2, y2 = pol2cart(p2[0], p2[1])
                # x2, y2 = (x2 + ref_point.x, y2 + ref_point.y)
                line = LineSegment(c1, c2, line.Normal, line.Name)
                vis_lines.append(line)

                # plt.plot([x1, x2], [y1, y2], 'sg-')
                # plt.pause(0.001)
    merged = merge_lines(vis_lines)
    vfd=2