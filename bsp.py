from shapely.geometry import MultiLineString, LineString
from copy import copy

from shapely.ops import linemerge, unary_union, polygonize

from geometry import LineSegment, Point, Polygon
from utils import extrapolate_line, test_between, to_range2, merge_lines, process_line, merge_fovs2
import random
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from stonesoup.functions import pol2cart

class BinaryTree:
    """Binary tree class"""
    def __init__(self, data=[], parent=None, bounds=((0, 800), (0, 800)), polygon=None, plane=None):
        """Constructor, declares variables for left and right sub-tree and data for the current node"""
        self.left = None
        self.right = None
        self.data = data
        self.parent = parent
        self.polygon = polygon
        self.plane = plane
        self.bounds=bounds

    def __repr__(self):
        return "BinaryTree(data={}, left={}, right={})".format(self.data, self.left, self.right)

    def __bool__(self):
        return self.data is not None

    def print(self):
        """Prints the all tree nodes 'name' attribute in a binary tree format (needs to be improved)"""
        queue = [self]
        print_str = ''

        while len(queue) > 0:
            tree = queue.pop(0)

            str_list = []
            for line in tree.data:
                str_list.append(line.name)
            print_str += str(str_list)
            if tree.left is not None:
                str_list = []
                for line in tree.left.data:
                    str_list.append(line.name)
                print_str += (' Left: ' + str(str_list))
                queue.append(tree.left)
            else:
                print_str += (' Left: ' + ' / ')

            if tree.right is not None:
                str_list = []
                for line in tree.right.data:
                    str_list.append(line.name)
                print_str += (' Right: ' + str(str_list))
                queue.append(tree.right)
            else:
                print_str += (' Right: ' + ' / ')

            print_str += '\n'
        return print_str

    def is_leaf(self, full=True):
        if full:
            return self.left is None and self.right is None
        return self.left is None or self.right is None

    def is_root(self):
        return self.parent is None

    def is_solid(self):
        if self.is_leaf() and self == self.parent.right:
            return True
        return False

    def is_empty(self):
        if self.is_leaf() and self == self.parent.left:
            return True
        return False

    def get_area(self):
        xlim = self.bounds[0]
        ylim = self.bounds[1]
        tree = self
        lines_ = []
        plt.figure()
        while tree is not None:
            line = tree.data[0]
            x, y = extrapolate_line(line, xlim, ylim)
            p_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.normal, line.name)
            # p_l.plot(color='r')
            for i, c_l in enumerate(lines_):
                ls = p_l.split(c_l)
                if ls:
                    l1, l2 = ls

                    # Unit Vector for normal
                    vp = [p_l.normalV.x, p_l.normalV.y]
                    up = vp / np.linalg.norm(vp)

                    # Unit Vector for l1
                    v1 = [l1.p1.x - l1.p2.x, l1.p1.y - l1.p2.y]
                    u1 = v1 / np.linalg.norm(v1)
                    t1 = np.arccos(np.dot(u1, up))

                    # Unit Vector for l2
                    v2 = [l2.p2.x - l2.p1.x, l2.p2.y - l2.p1.y]
                    u2 = v2 / np.linalg.norm(v2)
                    t2 = np.arccos(np.dot(u2, up))

                    if c_l.name in self._child_names(tree.left):
                        if t1 < np.pi / 2:
                            l = l1
                        else:
                            l = l2
                    elif c_l.name in self._child_names(tree.right):
                        if t1 > np.pi / 2:
                            l = l1
                        else:
                            l = l2
                    x, y = ([l.p1.x, l.p2.x], [l.p1.y, l.p2.y])
                    lines_[i] = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), c_l.normal, c_l.name)
                    # lines_[i].plot(color='b')
            lines_.append(p_l)
            # plt.clf()
            tree = tree.parent

        for line in lines_:
            line.plot()
        plt.pause(0.1)
        buffer = 0.0000000000001
        lines = MultiLineString([line.linestring.xy for line in lines_]).buffer(buffer)
        boundary = Polygon(((xlim[0], ylim[0]), (xlim[0], ylim[1]),
                            (xlim[1], ylim[1]), (xlim[1], ylim[0])))  # lines.convex_hull
        multipolygons = boundary.difference(lines)
        polygons = list(multipolygons)
        for p in polygons:
            x,y = p.exterior.xy
            plt.plot(x,y)
        plt.pause(0.01)
        # l1 = lines_[0].linestring.buffer(0.0000000000001)
        # l2 = lines_[1].linestring.buffer(0.0000000000001)
        ps = [p for p in polygons if np.all([l.linestring.buffer(0.0000000000001).intersects(p) for l in lines_])]
        polygon = ps[-1]
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'k-')
        plt.pause(0.01)

        return polygon

    def _child_names(self, tree, names=[]):
        if tree:
            names = names + [l.name for l in tree.data] + self._child_names(tree.left) + self._child_names(tree.right)
        return names


class BSP:
    """Binary Space Partition class, optimally generates BSP tree from a list of line segments by using a heuristic"""
    def __init__(self, lines=None, heuristic='random', bounds=((0, 800), (0, 800))):
        """Constructor, initializes binary tree"""
        self.tree = BinaryTree()
        self.heuristic = heuristic
        self.bounds = bounds
        if lines:
            self.generate_tree(lines)

    @property
    def nodes(self):
        nodes = self._nodes(self.tree)
        return nodes

    def _nodes(self, tree):
        nodes = [tree]
        if tree.left is not None:
            nodes += self._nodes(tree.left)
        if tree.right is not None:
            nodes += self._nodes(tree.right)
        return nodes

    def heuristic_min_partition(self, lines):
        """
        Returns the index of the line segment in which causes the least amount of partitions with
        other line segments in the list.
        """
        min_idx = 0
        min_partition = np.inf

        for idx1, line1 in enumerate(lines):
            partition_count = 0
            for idx2, line2 in enumerate(lines):
                # print("{}|{} out of {}".format(idx1, idx2, len(lines)))
                if idx1 != idx2:
                    res = line1.compare(line2)
                    if res == 'P':
                        partition_count += 1

            if partition_count < min_partition:
                min_partition = partition_count
                min_idx = idx1

        return min_idx

    def heuristic_even_partition(self, lines):
        """
        Returns the index of the line segment which produces the most balanced tree
        """
        best_idx = 0
        min_divide = np.inf
        min_nodes = np.inf

        for idx1, line1 in enumerate(lines):
            left_count = 0
            right_count = 0
            for idx2, line2 in enumerate(lines):
                # print("{}|{} out of {}".format(idx1, idx2, len(lines)))
                if idx1 != idx2:
                    res = line1.compare(line2)
                    if res == 'P':
                        left_count += 1
                        right_count += 1
                    elif res == 'F':
                        left_count += 1
                    elif res == 'B':
                        right_count += 1

            if abs(left_count - right_count) < min_divide:
                min_nodes = left_count + right_count
                min_divide = abs(left_count - right_count)
                best_idx = idx1
            elif abs(left_count - right_count) == min_divide:
                if left_count + right_count < min_nodes:
                    min_nodes = left_count + right_count
                    best_idx = idx1

        return best_idx

    def generate_tree(self, lines, heuristic=None):
        if heuristic:
            self.heuristic = heuristic
        polygon = Polygon(((self.bounds[0][0], self.bounds[1][0]), (self.bounds[0][0], self.bounds[1][1]),
                           (self.bounds[0][1], self.bounds[1][1]), (self.bounds[0][1], self.bounds[1][0])))
        self.tree = BinaryTree(copy(lines), polygon=polygon)
        self._generate_tree(self.tree, self.heuristic)
        # self._generate_areas(self.tree)
        a = 2

    def _generate_tree(self, tree, heuristic='even'):
        """
        Generates the binary space partition tree recursively using the specified heuristic at each sub-tree
        :param tree: BinaryTree, value should be self.tree on the first call, this argument exists so we can traverse the tree recursively
        :param UseHeuristic: string, either 'even' for balanced tree or 'min' for least number of nodes
        :return: nothing
        """
        best_idx = 0
        # print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(tree.data)
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(tree.data)
        elif heuristic == 'random':
            best_idx = np.random.randint(len(tree.data))

        print('....')
        data = []
        data_left = []
        data_right = []
        line = tree.data.pop(best_idx)
        x, y = extrapolate_line(line, self.bounds[0], self.bounds[1])
        l = LineString(((x[0], y[0]), (x[1], y[1])))
        pols = cut_polygon_by_line(tree.polygon, l)
        try:
            pol_left = next(
                Polygon.from_shapely(pol) for pol in pols if Point(pol.centroid.x, pol.centroid.y).compare(line) == 1)
        except StopIteration:
            pol_left = None

        try:
            pol_right = next(
                Polygon.from_shapely(pol) for pol in pols if Point(pol.centroid.x, pol.centroid.y).compare(line) == -1)
        except StopIteration:
            pol_right = None

        data.append(line)

        if pol_left and pol_right:
            tree.plane = LineSegment.from_linestring(pol_left.intersection(pol_right))
        elif pol_left:
            tree.plane = LineSegment.from_linestring(pol_left.intersection(l))
        else:
            tree.plane = LineSegment.from_linestring(pol_right.intersection(l))

        for i, line2 in enumerate(tree.data):
            # print('{}/{}'.format(i, len(tree.data)))
            result = line.compare(line2)
            if result == 'P':
                split_lines = line.split(line2)

                for split_line in split_lines:
                    res = line.compare(split_line)
                    if res == 'F':
                        data_left.append(split_line)
                    elif res == 'B':
                        data_right.append(split_line)
                    else:
                        print('Error!!', res)
                        # split_lines = line.split(line2)
                        # line.compare(split_line)
            elif result == 'C':
                data.append(line2)
            elif result == 'F':
                data_left.append(line2)
            elif result == 'B':
                data_right.append(line2)

        tree.data = data
        if not len(data_left):
            data_left = None
        tree.left = BinaryTree(data_left, parent=tree, polygon=pol_left)
        if data_left:
            self._generate_tree(tree.left, heuristic)

        if not len(data_right):
            data_right = None
        tree.right = BinaryTree(data_right, parent=tree, polygon=pol_right)
        if data_right:
            self._generate_tree(tree.right, heuristic)

    def num_nodes(self, tree):
        """returns the number of nodes in the entire tree by traversing the tree"""
        g = self.to_graph()
        return g.number_of_nodes()

    def depth(self):
        return self._depth(self.tree)

    def _depth(self, tree):
        depth = 1
        depth_left = 0
        if tree.left:
            depth_left = self._depth(tree.left)
        depth_right = 0
        if tree.right:
            depth_right = self._depth(tree.right)
        return depth + np.amax([depth_left, depth_right])

    def to_graph(self):
        g = nx.Graph()
        return self._traverse_tree_nx(self.tree, g)

    def _traverse_tree_nx(self, tree, g, parent=None):
        child = g.number_of_nodes() + 1
        data = tree.data
        polygon = tree.polygon
        leaf = tree.is_leaf()
        solid = tree.is_solid()
        empty = tree.is_empty()
        if polygon is not None:
            g.add_node(child, data=data, leaf=leaf, polygon=polygon, solid=solid, empty=empty)
            if parent:
                g.add_edge(parent, child)
            if tree.left is not None:
                g = self._traverse_tree_nx(tree.left, g, child)
            if tree.right is not None:
                g = self._traverse_tree_nx(tree.right, g, child)
        return g

    def draw_nx(self, ax=None, show_labels=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.to_graph()
        pos = graphviz_layout(g, prog="dot")
        # pos = {n: g.nodes[n]['data'][0].getMidPoint().to_array() if g.nodes[n]['data'] else np.array([g.nodes[n]['polygon'].centroid.x, g.nodes[n]['polygon'].centroid.y])
        #        for n in g.nodes}

        nx.draw(g, pos, ax=ax)
        leaf_nodes = [n for n in g.nodes if (g.nodes[n]['leaf'] and not g.nodes[n]['solid'])]
        solid_nodes = [n for n in g.nodes if (g.nodes[n]['leaf'] and g.nodes[n]['solid'])]
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=leaf_nodes, node_color='green', node_shape='s')
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=solid_nodes, node_color='red', node_shape='s')

        if show_labels:
            labels = {n: [l.name for l in g.nodes[n]['data']] if g.nodes[n]['data'] else '' for n in g.nodes}
            pos_labels = {}
            for node, coords in pos.items():
                pos_labels[node] = (coords[0] + 10, coords[1])
            nx.draw_networkx_labels(g, pos_labels, labels=labels)

    def draw_on_plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.to_graph()
        # pos = graphviz_layout(g, prog="dot")
        pos = {n: (g.nodes[n]['data'][0].getMidPoint().to_array()) for n in g.nodes}
        labels = {n: [l.name for l in g.nodes[n]['data']] for n in g.nodes}
        nx.draw(g, pos, ax=ax)
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, labels=labels)

    def plot_planes(self, ax=None, color='m', linewidth=0.2, **kwargs):
        nodes = self.nodes
        for node in nodes:
            if node.plane:
                node.plane.plot(ax=ax, color=color, linewidth=linewidth, **kwargs)

    def find_leaf(self, p):
        tree = self.tree
        while True:
            line = tree.data[0]
            if p.compare(line) > 0:
                if not tree.left:
                    return tree.left
                tree = tree.left
            elif p.compare(line) < 0:
                if not tree.right:
                    return tree.right
                tree = tree.right

    def render(self, ref_point, ax=None):
        # Find the leaf node
        leaf = self.find_leaf(ref_point)

        r_lines = [] # self._render_child2(leaf, ref_point)
        parent = leaf.parent

        p_lines = []
        if not leaf.is_root():
            p_lines = self._render_parent(parent, leaf, ref_point)

        l = r_lines + p_lines

        # Convert to ranges
        vis_lines = []
        fov = []
        for idx, line in enumerate(l):
            fov, vis_lines = process_line(line, fov, vis_lines, ref_point, ax)
            fov = merge_fovs2(fov)
            if len(fov) == 1 and fov[0].delta == np.pi:
                break

        merged = merge_lines(vis_lines)
        return merged

    def _render_parent(self, parent, child, p):
        a = parent.data[0].name
        r_lines = parent.data

        if parent.left == child:
            if parent.right:
                if p.compare(parent.data[0]) > 0:
                    r_lines = r_lines + self._render_child(parent.right, p)
                else:
                    r_lines = self._render_child(parent.right, p) + r_lines
        else:
            if parent.left:
                if p.compare(parent.data[0]) > 0:
                    r_lines = self._render_child(parent.left, p) + r_lines
                else:
                    r_lines = r_lines + self._render_child(parent.left, p)

        p_lines = []
        if not parent.is_root():
            p_lines = self._render_parent(parent.parent, parent, p)

        r_lines = r_lines + p_lines
        return r_lines

    def _render_child(self, child, p):
        a = child.data[0].name
        r_lines = child.data
        if child.is_leaf():
            return r_lines
        else:
            r_lines_left = []
            r_lines_right = []

            if child.left:
                r_lines_left = self._render_child(child.left, p)
            if child.right:
                r_lines_right = self._render_child(child.right, p)

            if p.compare(child.data[0]) > 0:
                # if point is in front of dividing plane
                r_lines = r_lines_left + r_lines + r_lines_right
            else:
                r_lines = r_lines_right + r_lines + r_lines_left
        return r_lines

    def construct_portals(self):
        pass


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)