import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from stonesoup.types.angle import Bearing
from shapely.geometry import Point

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def mid_angle(a1, a2):
    vec1 = (np.sin(a1), np.cos(a1))
    vec2 = (np.sin(a2), np.cos(a2))

    vec_sum = tuple(sum(x) for x in zip(vec1, vec2))

    vec3 = vec_sum
    vec3 = tuple(x / (np.sqrt(vec_sum[0] ** 2 + vec_sum[1] ** 2)) for x in vec3)

    return Bearing(np.arctan2(vec3[0], vec3[1]))


def to_range2(a1, a2):
    mid = mid_angle(a1, a2)
    dx = np.abs(a1 - mid)
    return mid, dx


class AngleInterval:
    def __init__(self, mid, delta, name=None):
        self.mid = Bearing(mid)
        self.delta = float(delta)
        if not isinstance(name, list):
            name = [name]
        self.name = name
        if self.delta < 0:
            warnings.warn("Negative delta provided, value with be adjusted...")
        elif self.delta == 0:
            raise ValueError("Invalid delta value provided: Delta cannot be equal to 0!")
        elif self.delta > np.pi:
            raise ValueError("Invalid delta value provided: Delta cannot be highrt than pi!")

    def __repr__(self):
        return "AngleInterval(mid={}, delta={}, name={})".format(self.mid, self.delta, self.name)

    @property
    def min(self):
        return self.mid-self.delta

    @property
    def max(self):
        return self.mid + self.delta

    def contains_angle(self, angle, not_equals=False):
        alpha = np.fmod(float(self.mid)+2*np.pi, 2*np.pi)
        delta = float(self.delta)
        X = np.fmod(float(angle)+2*np.pi, 2*np.pi)

        a = (alpha - delta < X < alpha + delta)
        b = (alpha + 2*np.pi - delta < X < 2*np.pi)
        c = (0 < X < alpha - 2 * np.pi + delta)
        # Equals with account for numerical precision errors
        d = (math.isclose(float(self.mid - self.delta), float(angle),  abs_tol=1e-10)
             or math.isclose(float(self.mid + self.delta), float(angle),  abs_tol=1e-10))

        if not_equals:
            return (a or b or c) and not d
        else:
            return a or b or c or d

    def contains_interval(self, other, not_equals=False):
        a = self.contains_angle(other.min, not_equals)
        b = self.contains_angle(other.max, not_equals)
        c = other.contains_angle(self.max, not_equals)
        d = other.contains_angle(self.min, not_equals)
        if not_equals:
            return (a and b) and not (c and d)
        else:
            if (np.isclose(float(self.mid-other.mid), 0., atol=1e-10)
                    and np.isclose(self.delta-other.delta, 0., atol=1e-10)):
                return True
            return (a and b) and not (c and d)

        # return (self.contains_angle(other.min, not_equals)
        #         and self.contains_angle(other.max, not_equals))

    def intersects(self, other, not_equals=False):
        return (self.contains_angle(other.min, not_equals)
                or self.contains_angle(other.max, not_equals)
                or other.contains_angle(self.min, not_equals)
                or other.contains_angle(self.max, not_equals))

    def union(self, other):
        a = self.contains_angle(other.min)
        b = self.contains_angle(other.max)
        c = other.contains_angle(self.min)
        d = other.contains_angle(self.max)

        if not (a or b or c or d):
            # Not intersecting
            return None
        elif a and b and c and d:
            # Fully matching or mirror (accounting for numerical precision)
            if not(np.isclose(float(self.mid-other.mid), 0, atol=1e-10)):
                # Mirror
                name = list(set(self.name).union(set(other.name)))
                return AngleInterval(0, np.pi, name)
            else:
                # Matching
                return self
        elif a and b:
            # self contains other
            return self
        elif c and d:
            # other contains self
            return other
        elif a and d:
            # self.max and other.min intersect
            min = self.min
            max = other.max
            dx = other.delta + self.delta - (self.max - other.min) / 2
            mid = mid_angle(min, max)
            if dx > np.pi / 2:
                # if dx is higher than 180 degrees, offset computed mid point by 180
                mid = mid + np.pi
            elif dx == np.pi / 2:
                # mid_angle fails when dx is exactly 180, hence this fix
                mid = mid_angle(min+0.1, max)-0.05
            name = list(set(self.name).union(set(other.name)))
            return AngleInterval(mid, dx, name)
        elif b and c:
            # other.max and self.min intersect
            min = other.min
            max = self.max
            mid = mid_angle(min, max)
            dx = other.delta + self.delta - (other.max-self.min)/2
            if dx > np.pi / 2:
                # if dx is higher than 180 degrees, offset computed mid point by 180
                mid = mid + np.pi
            elif dx == np.pi / 2:
                # mid_angle fails when dx is exactly 180, hence this fix
                mid = mid_angle(min + 0.1, max) - 0.05
            name = list(set(self.name).union(set(other.name)))
            return AngleInterval(mid, dx, name)

    def plot(self, ref_point=None, radius=1, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()

        if not ref_point:
            ref_point = Point(0, 0)
        min = np.rad2deg(np.fmod(float(self.mid-self.delta)+2*np.pi, 2*np.pi))
        max = np.rad2deg(np.fmod(float(self.mid+self.delta)+2*np.pi, 2*np.pi))

        # Deal with full 360 degree interval
        if self.delta == np.pi:
            max = max-1e-10

        w = Wedge((ref_point.x, ref_point.y), radius, min, max, **kwargs)
        p = PatchCollection([w], alpha=0.3, **kwargs)
        ax.add_collection(p)
        ax.autoscale_view()
        return p











