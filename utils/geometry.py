import math
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon
from matplotlib.collections import PatchCollection
from stonesoup.functions import cart2pol, pol2cart
from stonesoup.types.angle import Bearing
from shapely.geometry import LineString

from .angles import to_range2, AngleInterval

def sign(x): return int(x > 0) - int(x < 0)


DoubleTolerance = 1e-5


class Point:
    """2D cartesian coordinate representation of point"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._shapely = None

    # def __init__(self, *args):
    #     self._x = 0
    #     self._y = 0
    #     super().__init__(*args)
    #     self._x = self.coords[0][0]
    #     self._y = self.coords[0][1]

    # @property
    # def x(self):
    #     return self._x
    #
    # @property
    # def y(self):
    #     return self._y

    def __eq__(self, other):
        return np.array_equal(self.to_array(), other.to_array())

    def __repr__(self):
        return "Point(x={}, y={})".format(self.x, self.y)

    def __str__(self):
        return self.__repr__()

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)

    @property
    def shapely(self):
        if not self._shapely:
            self._shapely = shp.geometry.Point(self.x, self.y)
        return self._shapely

    def print(self):
        print(self.x, ' ', self.y)

    def get_distance(self, OtherPoint):
        return math.sqrt(math.pow((self.x - OtherPoint.x), 2) +
                         math.pow((self.y - OtherPoint.y), 2))

    def to_array(self):
        return np.array([self.x, self.y])

    def compare(self, line):
        """
        Compare point to line

        Parameters
        ----------
        line: LineSegment
            Line to compare against

        Returns
        -------
        int
            1  - Point is in direction of line normal (in front)
            -1 - Point is in opposite direction of line normal (behind)
            0  - Point is on line

        """
        dot = np.dot(line.normalV.to_array(), self.to_array() - line.mid_point.to_array())
        if dot == 0:
            return dot
        return dot / abs(dot)

    def to_polar(self, ref=None):
        if not ref:
            x_ref = 0
            y_ref = 0
        else:
            x_ref = ref.x
            y_ref = ref.y

        x, y = (self.x-x_ref, self.y-y_ref)
        rho, phi = cart2pol(x, y)
        p1 = (rho, Bearing(phi))

        return p1


class Vector:
    """A quite basic 2D vector class"""

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Vector(x={}, y={})".format(self.x, self.y)

    def to_array(self):
        return np.array([self.x, self.y])

    def dot(self, vector):
        """returns vector dot product of this vector with vector as function argument"""
        return np.dot(self.to_array(), vector.to_array())
        # return self.x * vector.x + self.y * vector.y


class LineSegment:
    """2D line segment class"""
    def __init__(self, p1, p2, normal=1, name='', linestring=None):
        """Arguments: p1 (type: Point), p2 (type: Point), normal (type: Int), name (type: String),
        arg 'normal' represents one of two possible directions of normal vector of our line segment, arg 'name
        is any arbitrary name for the purpose of identifying nodes when printing binary trees"""
        self.p1 = p1
        self.p2 = p2
        self.name = name

        Dx = p2.x - p1.x
        Dy = p2.y - p1.y
        self.normal = normal
        self.normalV = Vector(Dy, -Dx)
        if normal == -1:
            self.normalV = Vector(-Dy, Dx)

        self._shapely = linestring

    def __repr__(self):
        return "LineSegment(name={}, p1={}, p2={}, normal={}, normalV={})".format(self.name, self.p1, self.p2,
                                                                                  self.normal, self.normalV)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @property
    def points(self):
        return [self.p1, self.p2]

    @property
    def xy(self):
        return np.array([self.p1.x, self.p2.x]), np.array([self.p1.y, self.p2.y])

    @property
    def shapely(self):
        if not self._shapely:
            self._shapely = LineString((self.p1.to_array(), self.p2.to_array()))
        return self._shapely

    @property
    def mid_point(self):
        """returns middle point of our line segment"""
        return Point((self.p2.x + self.p1.x) / 2, (self.p2.y + self.p1.y) / 2)

    @property
    def length(self):
        """returns length of our line segment """
        return self.p1.get_distance(self.p2)

    def Print(self):
        """prints point coordinates and direction of normal vector"""
        self.p1.Print()
        self.p2.Print()
        print(self.normal, '\n')

    def compare(self, other):
        """Compares two line segments for space partitioning, returns a character that identify the comparison of the two lines
        if 'other' exists completely on side or other of our line segment (imagined as an infinite line segment), then it will return either 'F' or 'B' depending on the direction of 'other' to our line segment
        if our line segment (infinite) intersects the 'other' and thus causes the 'other' to split into two, it returns 'P'
        if both line segments are collinear, returns 'C'
        """

        if isinstance(other, LineSegment):
            p1 = np.array([other.p1.x - self.p1.x, other.p1.y - self.p1.y])
            dot1 = np.dot(self.normalV.to_array(), p1)
            if abs(dot1) < DoubleTolerance:
                dot1 = 0

            p2 = np.array([other.p2.x - self.p1.x, other.p2.y - self.p1.y])
            dot2 = np.dot(self.normalV.to_array(), p2)
            if abs(dot2) < DoubleTolerance:
                dot2 = 0

            if (dot1 > 0 > dot2) or (dot1 < 0 < dot2):
                return 'P'
            elif dot1 == dot2 == 0:
                return 'C'
            elif dot1 + dot2 > 0:
                return 'F'
            elif dot1 + dot2 < 0:
                return 'B'
        else:
            num_lines = len(other)
            if not num_lines:
                return []
            p1 = np.array([(line.p1.x, line.p1.y) for line in other]) - self.p1.to_array()
            dot1 = np.dot(self.normalV.to_array(), p1.T)
            dot1[abs(dot1)<DoubleTolerance] = 0

            p2 = np.array([(line.p2.x, line.p2.y) for line in other]) - self.p1.to_array()
            dot2 = np.dot(self.normalV.to_array(), p2.T)
            dot2[abs(dot2)<DoubleTolerance] = 0

            res = np.array(['' for _ in range(num_lines)])
            res[dot1 + dot2 < 0] = 'B'
            res[dot1 + dot2 > 0] = 'F'
            res[np.logical_and(dot1 == 0, dot2 == 0)] = 'C'
            res[np.logical_or(np.logical_and(dot1 > 0, dot2 < 0), np.logical_and(dot1 < 0, dot2 > 0))] = 'P'

            return list(res)

    def split(self, other):
        """
        Split other line segment, based on intersection of plane defined by line.

        Parameters
        ----------
        other: LineSegment
            Line to be split

        Returns
        -------
        LineSegments or None
            Two LineSegments if LineSegment in 'self' (as an infinite line segment) partitions 'otherLine' in space
            partitioning, otherwise returns None

        Notes
        -----
        Solution taken from: http://paulbourke.net/geometry/pointlineplane/
        """

        numer = np.dot(self.normalV.to_array(),
                       self.p1.to_array() - other.p1.to_array())
        denom = np.dot(self.normalV.to_array(),
                       other.p2.to_array() - other.p1.to_array())

        # numer = (self.normalV.x * (other.p1.x - self.p1.x)) + \
        #     (self.normalV.y * (other.p1.y - self.p1.y))
        # denom = ((-self.normalV.x) * (other.p2.x - other.p1.x)) + \
        #     ((-self.normalV.y) * (other.p2.y - other.p1.y))
        #
        # numer = self.normalV.to_array() @ (other.p1.to_array()-self.p1.to_array())
        # denom = -self.normalV.to_array() @ (other.p2.to_array()-other.p1.to_array())

        if denom != 0.0:
            t = numer / denom
        else:
            return None

        if 0 <= t <= 1.0:
            x = other.p1.x + t * (other.p2.x - other.p1.x)
            y = other.p1.y + t * (other.p2.y - other.p1.y)
            intersection = Point(x, y)
            return LineSegment(other.p1, intersection, other.normal, name=(other.name + '_a')), \
                   LineSegment(intersection, other.p2, other.normal, name=(other.name + '_b'))
        else:
            return None

    def plot(self, ax=None, plot_norm=False, **kwargs):
        x = [self.p1.x, self.p2.x]
        y = [self.p1.y, self.p2.y]
        if ax:
            if plot_norm:
                midPoint = self.mid_point
                ax.quiver(midPoint.x, midPoint.y, self.normalV.x, self.normalV.y, width=0.001, headwidth=0.2)
            return ax.plot(x, y, **kwargs)[0]
        else:
            if plot_norm:
                midPoint = self.mid_point
                plt.quiver(midPoint.x, midPoint.y, self.normalV.x, self.normalV.y, width=0.001, headwidth=0.2)
            return plt.plot(x, y, **kwargs)[0]


    def to_polar(self, ref=None):
        if not ref:
            x_ref = 0
            y_ref = 0
        else:
            x_ref = ref.x
            y_ref = ref.y

        x, y = (self.p1.x-x_ref, self.p1.y-y_ref)
        rho, phi = cart2pol(x, y)
        p1 = (rho, Bearing(phi))

        x, y = (self.p2.x-x_ref, self.p2.y-y_ref)
        rho, phi = cart2pol(x, y)
        p2 = (rho, Bearing(phi))

        return p1, p2

    def to_interval(self, ref_point=None):
        p1, p2 = self.to_polar(ref_point)
        mid, dx = to_range2(p1[1], p2[1])
        return AngleInterval(mid, dx, self.name)

    @staticmethod
    def from_linestring(linestring: LineString, normal=None, name=None):
        x, y = linestring.xy
        p1 = Point(x[0], y[0])
        p2 = Point(x[1], y[1])
        return LineSegment(p1, p2, normal, name, linestring=linestring)


class MergedLine:
    def __init__(self, lines, linestring):
        self.lines = lines
        self.shapely = linestring

    def __repr__(self):
        return "MergedLine(names={}, lines={}, linestring={})".format(self.names, self.lines, self.shapely)

    @property
    def names(self):
        return [line.name for line in self.lines]

    def to_interval(self, ref_point=None):
        points = []
        for point in self.shapely.boundary:
            points.append(Point(point.x, point.y))
        p1, p2 = (points[0].to_polar(ref_point), points[1].to_polar(ref_point))
        mid, dx = to_range2(p1[1], p2[1])
        x, y = pol2cart(1e15, mid)
        x, y = (x + ref_point.x, y + ref_point.y)
        li = LineSegment(ref_point, Point(x, y))
        if not li.shapely.intersects(self.shapely):
            a1 = mid - dx
            dx2 = np.abs(a1 - (mid + np.pi))
            return AngleInterval(mid + np.pi, dx2, self.names)
        return AngleInterval(mid, dx, self.names)

    def plot(self, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()

        x, y = self.shapely.xy
        return ax.plot(x, y, **kwargs)


class Polygon:

    def __init__(self, points, shapely=None):
        self.points = [Point(p[0], p[1]) for p in points]
        self._shapely = shapely


    def __repr__(self):
        points = [p for p in self.points]
        return "Polygon(points={})".format(points)

    def __str__(self):
        return self.__repr__()

    # @property
    # def points(self):
    #     x, y = self.exterior.xy
    #     return [Point(xi, yi) for xi, yi in zip(x, y)]

    @property
    def shapely(self):
        if not self._shapely:
            self._shapely = shp.geometry.Polygon([(p.x, p.y) for p in self.points])
        return self._shapely

    @staticmethod
    def from_shapely(pol: shp.geometry.Polygon):
        x, y = pol.exterior.xy
        points = [(xi, yi) for xi, yi in zip(x, y)]
        return Polygon(points, pol)

    def plot(self, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.3

        x, y = self.shapely.exterior.xy
        xy = np.array([x, y]).T
        polygon = pltPolygon(xy, True, **kwargs)
        p = PatchCollection([polygon], **kwargs)
        ax.add_collection(p)
        return p
