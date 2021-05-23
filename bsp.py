from shapely.geometry import MultiLineString, LineString
from copy import copy

from shapely.ops import linemerge, unary_union, polygonize

from angles import angle_between
from geometry import LineSegment, Point, Polygon
from utils import extrapolate_line, merge_lines, process_line, merge_fovs2
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


class BSPNode:
    """BSP Node class"""
    def __init__(self, data=[], parent=None, bounds=((0, 800), (0, 800)), polygon=None, plane=None, id=None):
        """Constructor, declares variables for left and right sub-tree and data for the current node"""
        self.left = None
        self.right = None
        self.data = data
        self.parent = parent
        self.polygon = polygon
        self.plane = plane
        self.bounds = bounds
        self.portals = []
        self.id = id

    def __repr__(self):
        return "BSPNode(id={}, data={}, left={}, right={})".format(self.id, self.data, self.left, self.right)

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_solid(self):
        if self.is_leaf and self == self.parent.right:
            return True
        return False

    @property
    def is_empty(self):
        if self.is_leaf and self == self.parent.left:
            return True
        return False


class BSP:
    """Binary Space Partition class, optimally generates BSP tree from a list of line segments by using a heuristic"""
    def __init__(self, lines=None, heuristic='random', bounds=((0, 800), (0, 800))):
        """Constructor, initializes binary tree"""
        self.tree = BSPNode()
        self.heuristic = heuristic
        self.bounds = bounds
        self._portals = []
        self._last_node_id = 0
        if lines:
            self.generate_tree(lines)

    @property
    def nodes(self) -> [BSPNode]:
        """
        Return the nodes in the BSP tree, ordered by a depth-first search around the back of each node.

        Returns
        -------
        list of BSPNode objects

        """
        nodes = self._nodes(self.tree)
        return nodes

    def _nodes(self, tree):
        nodes = [tree]
        if tree.right is not None:
            nodes += self._nodes(tree.right)
        if tree.left is not None:
            nodes += self._nodes(tree.left)
        return nodes

    @property
    def solid_leaves(self) -> [BSPNode]:
        """
        Return all leaf nodes that represent a solid area (behind a wall)

        Returns
        -------
        list of BSPNode objects

        """
        return [n for n in self.nodes if n.is_solid]

    @property
    def empty_leaves(self) -> [BSPNode]:
        """
        Return all leaf nodes that represent an empty area (in front of walls)

        Returns
        -------
        list of BSPNode objects

        """
        return [n for n in self.nodes if n.is_empty]

    def heuristic_min_partition(self, lines) -> int:
        """
        Returns the index of the line segment which causes the least amount of partitions with
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
        self.tree = BSPNode(copy(lines), polygon=polygon, id=self._last_node_id)
        self._last_node_id += 1
        self._generate_tree(self.tree, self.heuristic)
        # self._generate_areas(self.tree)
        a = 2

    def _generate_tree(self, tree, heuristic='even'):
        """
        Generates the binary space partition tree recursively using the specified heuristic at each sub-tree
        :param tree: BSPNode, value should be self.tree on the first call, this argument exists so we can traverse the tree recursively
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
            tree.plane = LineSegment.from_linestring(pol_left.intersection(pol_right), line.normal, line.name)
        elif pol_left:
            tree.plane = LineSegment.from_linestring(pol_left.intersection(l), line.normal, line.name)
        else:
            tree.plane = LineSegment.from_linestring(pol_right.intersection(l), line.normal, line.name)

        a = angle_between(tree.plane.normalV.to_array(), line.normalV.to_array())
        if abs(a) > np.pi/100:
            tree.plane = LineSegment(tree.plane.p1, tree.plane.p2, -tree.plane.normal, tree.plane.name)


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
        tree.left = BSPNode(data_left, parent=tree, polygon=pol_left, id=self._last_node_id)
        self._last_node_id += 1
        if data_left:
            self._generate_tree(tree.left, heuristic)

        if not len(data_right):
            data_right = None
        tree.right = BSPNode(data_right, parent=tree, polygon=pol_right, id=self._last_node_id)
        self._last_node_id += 1
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
        leaf = tree.is_leaf
        solid = tree.is_solid
        empty = tree.is_empty
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
        if not leaf.is_root:
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
        if not parent.is_root:
            p_lines = self._render_parent(parent.parent, parent, p)

        r_lines = r_lines + p_lines
        return r_lines

    def _render_child(self, child, p):
        a = child.data[0].name
        r_lines = child.data
        if child.is_leaf:
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

    def gen_portals2(self):
        self._portals = []
        self._removed_portals = set()
        self._replacement_portals = dict()
        for node in self.nodes:
            # update node portals in case replacement or removal has occurred
            node.portals = self._update_portals(node.portals)
            #
            if node.is_solid:
                self._removed_portals |= {p.name for p in node.portals}
                self._portals = [p for p in self._portals if p.name not in self._removed_portals]
                node.portals = []
                continue
            elif node.is_empty:
                continue
            portal = node.plane
            node.portals.append(portal)
            self._portals.append(portal)
            # Push down
            for p in node.portals:
                res = node.plane.compare(p)
                if res == 'F':
                    node.left.portals.append(p)
                elif res == 'B':
                    node.right.portals.append(p)
                elif res == 'P':
                    lines = node.plane.split(p)
                    if node.plane.compare(lines[0]) == 'F':
                        node.left.portals.append(lines[0])
                        node.right.portals.append(lines[1])
                    else:
                        node.right.portals.append(lines[0])
                        node.left.portals.append(lines[1])
                    self._replacement_portals[p.name] = [lines[0].name, lines[1].name]
                    self._portals.remove(p)
                    self._portals += lines
                elif res == 'C':
                    node.left.portals.append(p)
                    node.right.portals.append(p)

        for node in self.empty_leaves:
            node.portals = self._update_portals(node.portals)

    def _update_portals(self, portals):
        portal_names = [p.name for p in portals]
        processed_portals = set()
        while any([po in self._replacement_portals for po in set(portal_names)-processed_portals]):
            for portal_name in portal_names:
                if portal_name in self._replacement_portals.keys()-processed_portals:
                    portal_names += set(self._replacement_portals[portal_name])
                    processed_portals.add(portal_name)
        return [p for p in self._portals if p.name in set(portal_names)-processed_portals]

def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)