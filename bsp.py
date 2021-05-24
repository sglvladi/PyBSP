import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from copy import copy
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import linemerge, unary_union, polygonize, split
from stonesoup.functions import pol2cart

from angles import angle_between
from geometry import LineSegment, Point, Polygon
from utils import extrapolate_line, merge_lines, process_line, merge_fovs2


class BSPNode:
    """BSP Node class"""
    def __init__(self, data=[], parent=None, polygon=None, plane=None, id=None):
        """Constructor, declares variables for left and right sub-tree and data for the current node"""
        self.parent = parent
        self.front = None
        self.back = None
        self.data = data
        self.polygon = polygon
        self.plane = plane
        self.portals = []
        self.pvs = set()
        self.id = id

    def __repr__(self):
        return "BSPNode(id={}, data={}, left={}, right={})".format(self.id, self.data, self.front, self.back)

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @property
    def is_leaf(self):
        return self.front is None and self.back is None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_solid(self):
        if self.is_leaf and self == self.parent.back:
            return True
        return False

    @property
    def is_empty(self):
        if self.is_leaf and self == self.parent.front:
            return True
        return False


class BSP:
    """Binary Space Partition class, optimally generates BSP tree from a list of line segments by using a heuristic"""
    def __init__(self, lines=None, heuristic='random', bounds=((0, 800), (0, 800))):
        """Constructor, initializes binary tree"""
        self.root = BSPNode()
        self.heuristic = heuristic
        self.bounds = bounds
        self._portals = []
        self._last_node_id = 0
        self.node_connectivity = None
        self.portal_connections = None
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
        nodes = self._nodes(self.root)
        return nodes

    def _nodes(self, node):
        nodes = [node]
        if node.back is not None:
            nodes += self._nodes(node.back)
        if node.front is not None:
            nodes += self._nodes(node.front)
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

    @property
    def num_nodes(self) -> int:
        """
        Return number of nodes in BSP tree

        Returns
        -------
        int
            The number of nodes in the tree
        """
        return len(self.nodes)

    @property
    def depth(self) -> int:
        """
        Return the depth of the BSP tree

        Returns
        -------
        int
            Depth of the BSP tree
        """
        return self._depth(self.root)

    def _depth(self, node):
        depth = 1
        depth_left = 0
        if node.front is not None:
            depth_left = self._depth(node.front)
        depth_right = 0
        if node.back is not None:
            depth_right = self._depth(node.back)
        return depth + np.amax([depth_left, depth_right])

    @property
    def nx_graph(self) -> nx.Graph:
        """
        Returns tree as a NetworkX graph object

        Returns
        -------
        nx.Graph
            NetworkX graph representation of the BSP tree
        """
        g = nx.Graph()
        return self._traverse_tree_nx(self.root, g)

    def get_node(self, id):
        return next(filter(lambda node: node.id == id, self.nodes), None)

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
        """
        Generates the binary space partition tree recursively using the provided lines and specified heuristic

        Parameters
        ----------
        lines: list of LineSegment objects
            Wall segments to be used for building the tree.

        heuristic: str
            Selected method for building the tree. Possible values are: (i) 'even' for balanced tree; (ii) 'min' for
            least number of nodes; (iii) 'rand' for random generation

        """
        if heuristic:
            self.heuristic = heuristic
        polygon = Polygon(((self.bounds[0][0], self.bounds[1][0]), (self.bounds[0][0], self.bounds[1][1]),
                           (self.bounds[0][1], self.bounds[1][1]), (self.bounds[0][1], self.bounds[1][0])))
        self._last_node_id = 0
        self.root = BSPNode(copy(lines), polygon=polygon, id=self._last_node_id)
        self._last_node_id += 1
        self._generate_tree(self.root, self.heuristic)
        #self.gen_portals()

    def _generate_tree(self, node: BSPNode, heuristic='even'):
        best_idx = 0
        # print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(node.data)
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(node.data)
        elif heuristic == 'random':
            best_idx = np.random.randint(len(node.data))

        print('....')
        data = []
        data_left = []
        data_right = []
        line = node.data.pop(best_idx)
        data.append(line)

        # Compute the polygon of the area represented by the node
        x, y = extrapolate_line(line, self.bounds[0], self.bounds[1])
        l = LineString(((x[0], y[0]), (x[1], y[1])))
        pols = cut_polygon_by_line(node.polygon, l)
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

        # Compute the node's splitting plane
        if pol_left and pol_right:
            node.plane = LineSegment.from_linestring(pol_left.intersection(pol_right), line.normal, line.name)
        elif pol_left:
            node.plane = LineSegment.from_linestring(pol_left.intersection(l), line.normal, line.name)
        else:
            node.plane = LineSegment.from_linestring(pol_right.intersection(l), line.normal, line.name)
        # Ensure normal is preserved
        a = angle_between(node.plane.normalV.to_array(), line.normalV.to_array())
        if abs(a) > np.pi/100:
            node.plane = LineSegment(node.plane.p1, node.plane.p2, -node.plane.normal, node.plane.name)

        #
        for i, line2 in enumerate(node.data):
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
            elif result == 'C':
                data.append(line2)
            elif result == 'F':
                data_left.append(line2)
            elif result == 'B':
                data_right.append(line2)

        # Generate front node
        node.data = data
        if not len(data_left):
            data_left = None
        node.front = BSPNode(data_left, parent=node, polygon=pol_left, id=self._last_node_id)
        self._last_node_id += 1
        if data_left:
            self._generate_tree(node.front, heuristic)

        # Generate back node
        if not len(data_right):
            data_right = None
        node.back = BSPNode(data_right, parent=node, polygon=pol_right, id=self._last_node_id)
        self._last_node_id += 1
        if data_right:
            self._generate_tree(node.back, heuristic)

    def _traverse_tree_nx(self, node, g, parent=None):
        child = g.number_of_nodes() + 1
        id = node.id
        data = node.data
        polygon = node.polygon
        leaf = node.is_leaf
        solid = node.is_solid
        empty = node.is_empty
        if polygon is not None:
            g.add_node(child, data=data, leaf=leaf, polygon=polygon, solid=solid, empty=empty, id=id)
            if parent:
                g.add_edge(parent, child)
            if node.front is not None:
                g = self._traverse_tree_nx(node.front, g, child)
            if node.back is not None:
                g = self._traverse_tree_nx(node.back, g, child)
        return g

    def draw_nx(self, ax=None, show_labels=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.nx_graph
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
            labels = {n: g.nodes[n]['id'] for n in g.nodes}
            pos_labels = {}
            for node, coords in pos.items():
                pos_labels[node] = (coords[0] + 10, coords[1])
            nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels)

    def draw_on_plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.nx_graph
        # pos = graphviz_layout(g, prog="dot")
        pos = {n: (g.nodes[n]['data'][0].getMidPoint().to_array()) for n in g.nodes}
        labels = {n: [l.name for l in g.nodes[n]['data']] for n in g.nodes}
        nx.draw(g, pos, ax=ax)
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, labels=labels)

    def plot_planes(self, ax=None, color='m', linewidth=0.2, **kwargs):
        """
        Plot all splitting planes using matplotlib

        Parameters
        ----------
        ax: plt.axis
            Axis on which to draw planes
        color: string
            Line color
        linewidth: float
            Line width
        kwargs:
            Keyword arguments to be passed to pyplot.plot function

        """
        nodes = self.nodes
        for node in nodes:
            if node.plane:
                node.plane.plot(ax=ax, color=color, linewidth=linewidth, **kwargs)
                midpoint = node.plane.mid_point
                if ax:
                    ax.text(midpoint.x, midpoint.y, node.plane.name, color='y')
                else:
                    plt.text(midpoint.x, midpoint.y, node.plane.name, color='y')

    def find_leaf(self, p: Point) -> BSPNode:
        """
        Return the leaf node in which a given point lies.

        Parameters
        ----------
        p: Point
            Point to be searched for

        Returns
        -------
        BSPNode
            Leaf node whose polygon contains the query point.

        """
        node = self.root
        while True:
            line = node.data[0]
            if p.compare(line) > 0:
                if not node.front:
                    return node.front
                node = node.front
            elif p.compare(line) < 0:
                if not node.back:
                    return node.back
                node = node.back

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

        if parent.front == child:
            if parent.back:
                if p.compare(parent.data[0]) > 0:
                    r_lines = r_lines + self._render_child(parent.back, p)
                else:
                    r_lines = self._render_child(parent.back, p) + r_lines
        else:
            if parent.front:
                if p.compare(parent.data[0]) > 0:
                    r_lines = self._render_child(parent.front, p) + r_lines
                else:
                    r_lines = r_lines + self._render_child(parent.front, p)

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

            if child.front:
                r_lines_left = self._render_child(child.front, p)
            if child.back:
                r_lines_right = self._render_child(child.back, p)

            if p.compare(child.data[0]) > 0:
                # if point is in front of dividing plane
                r_lines = r_lines_left + r_lines + r_lines_right
            else:
                r_lines = r_lines_right + r_lines + r_lines_left
        return r_lines

    def gen_portals(self):
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
                    node.front.portals.append(p)
                elif res == 'B':
                    node.back.portals.append(p)
                elif res == 'P':
                    lines = node.plane.split(p)
                    if node.plane.compare(lines[0]) == 'F':
                        node.front.portals.append(lines[0])
                        node.back.portals.append(lines[1])
                    else:
                        node.back.portals.append(lines[0])
                        node.front.portals.append(lines[1])
                    self._replacement_portals[p.name] = [lines[0].name, lines[1].name]
                    self._portals.remove(p)
                    self._portals += lines
                elif res == 'C':
                    node.front.portals.append(p)
                    node.back.portals.append(p)

        for node in self.empty_leaves:
            node.portals = self._update_portals(node.portals)

        self.portal_connections= dict()
        for portal in self._portals:
            nodes = [n for n in self.empty_leaves if portal in n.portals]
            self.portal_connections[portal] = nodes

        self.node_connectivity = dict()
        for node in self.empty_leaves:
            self.node_connectivity[node] = dict()
            for portal in node.portals:
                self.node_connectivity[node][portal] = next(n for n in self.portal_connections[portal] if n != node)

    def _update_portals(self, portals):
        portal_names = [p.name for p in portals]
        processed_portals = set()
        while any([po in self._replacement_portals for po in set(portal_names)-processed_portals]):
            for portal_name in portal_names:
                if portal_name in self._replacement_portals.keys()-processed_portals:
                    portal_names += set(self._replacement_portals[portal_name])
                    processed_portals.add(portal_name)
        return [p for p in self._portals if p.name in set(portal_names)-processed_portals]

    def gen_pvs(self):
        for source_node in self.empty_leaves:
            a=2
            for source_portal in source_node.portals:
                target_node = self.node_connectivity[source_node][source_portal]
                source_node.pvs.add(target_node)
                target_portals = target_node.portals
                self._gen_pvs2(source_node, target_node, source_portal, target_portals, [source_node, target_node], [source_portal])

    def _gen_pvs(self, source_node, current_node, source_portal, target_portals: list,
                 visited_nodes: list, visited_portals: list, last_penumbra=None):
        target_portals = copy(target_portals)
        visited_nodes = copy(visited_nodes)
        visited_portals = copy(visited_portals)

        # Check all portals except the one we are looking through
        valid_target_portals = set(target_portals)-{visited_portals[-1]}
        a=2
        for target_portal in valid_target_portals:
            if source_portal.compare(target_portal) == 'C' or source_portal.linestring.touches(target_portal.linestring):
                a=2
                continue
            dest_node = self.node_connectivity[current_node][target_portal]

            # Avoid circle back to visited nodes
            # if dest_node in visited_nodes:
            #     continue

            source_node.pvs.add(dest_node)
            ref_point, interval = self.compute_anti_penumbra(source_portal, target_portal)

            penumbra = self.compute_anti_penumbra2(source_portal, target_portal)
            if last_penumbra:
                penumbra = last_penumbra.intersection(penumbra)

            if not ref_point:
                continue

            # Plot portals
            # source_portal.plot()
            # plt.pause(0.01)
            # target_portal.plot()
            # plt.pause(0.01)
            #
            # interval.plot(ref_point=ref_point, radius=1000)
            # plt.pause(0.01)

            dest_portals = []
            # Check all portals, except the once we are looking through
            valid_destination_portals = set(dest_node.portals) - {target_portal}
            a=2
            for dest_portal in valid_destination_portals:
                #dest_leaf2 =
                # If the destination portal is collinear to the target or source portals, then we can not see the node
                # it leads to (at least not through the target node)
                if dest_portal.compare(target_portal) == 'C' or dest_portal.compare(source_portal) == 'C':
                    continue

                # dest_penumbra = penumbra.intersection(self.compute_anti_penumbra2(target_portal, dest_portal))
                a=2
                try:
                    dest_interval = dest_portal.to_interval(ref_point)
                except ValueError:
                    continue
                if dest_interval.contains_interval(interval, True):
                    phi1 = interval.min
                    x, y = pol2cart(1e15, phi1)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    l1 = LineSegment(ref_point, Point(x, y))
                    phi2 = interval.max
                    x, y = pol2cart(1e15, phi2)
                    x, y = (x + ref_point.x, y + ref_point.y)
                    l2 = LineSegment(ref_point, Point(x, y))
                    p1 = l1.linestring.intersection(dest_portal.linestring)
                    p1 = Point(p1.x, p1.y)
                    p2 = l2.linestring.intersection(dest_portal.linestring)
                    if isinstance(p2, LineString):
                        a=2
                    p2 = Point(p2.x, p2.y)

                    line = LineSegment(p1, p2, name=dest_portal.name)
                    dest_portals.append(line)
                elif interval.contains_interval(dest_interval):
                    dest_portals.append(dest_portal)
                elif interval.intersects(dest_interval):
                    clip_min = dest_interval.contains_angle(interval.min, not_equals=True)
                    clip_max = dest_interval.contains_angle(interval.max, not_equals=True)
                    if not (clip_min or clip_max):
                        continue
                    if clip_min:
                        phi = interval.min
                        x, y = pol2cart(1e15, phi)
                        x, y = (x + ref_point.x, y + ref_point.y)
                        li = LineSegment(ref_point, Point(x, y))
                        try:
                            l1, l2 = li.split(dest_portal)
                        except:
                            a=2
                        _, a1 = l1.p1.to_polar(ref_point)
                        if interval.contains_angle(a1, True):
                            line = l1
                            # p1, p2 = l2.to_polar(ref_point)
                        else:
                            line = l2
                            # p1, p2 = l1.to_polar(ref_point)

                        # Generate new line and interval
                        # phi0, dx1 = to_range2(p1[1], p2[1])
                        # interval = AngleInterval(phi0, dx1)
                        #dest_interval = line.to_interval(ref_point)
                        line.name = dest_portal.name
                    if clip_max:
                        phi = interval.max
                        # Split existing line, according to fov
                        x, y = pol2cart(1e15, phi)
                        x, y = (x + ref_point.x, y + ref_point.y)
                        li = LineSegment(ref_point, Point(x, y))
                        l1, l2 = li.split(dest_portal)

                        _, a1 = l1.p1.to_polar(ref_point)
                        if interval.contains_angle(a1, True):
                            line = l1
                            # p1, p2 = l2.to_polar(ref_point)
                        else:
                            line = l2
                            # p1, p2 = l1.to_polar(ref_point)

                        # Generate new line and interval
                        # phi0, dx1 = to_range2(p1[1], p2[1])
                        # interval = AngleInterval(phi0, dx1)
                        #dest_interval = line.to_interval(ref_point)
                        line.name = dest_portal.name
                    dest_portals.append(line)
            if len(dest_portals):
                self._gen_pvs(source_node, dest_node, source_portal, dest_portals,
                              visited_nodes+[dest_node], visited_portals+[target_portal])

    def _gen_pvs2(self, source_node, current_node, source_portal, target_portals: list,
                  visited_nodes: list, visited_portals: list, last_penumbra=None):
        target_portals = copy(target_portals)
        visited_nodes = copy(visited_nodes)
        visited_portals = copy(visited_portals)

        # Check all portals except the one we are looking through
        valid_target_portals = set(target_portals) - {visited_portals[-1]}
        a = 2
        for target_portal in valid_target_portals:
            if source_portal.compare(target_portal) == 'C':
                a = 2
                continue
            dest_node = self.node_connectivity[current_node][target_portal]

            # Avoid circle back to visited nodes
            # if dest_node in visited_nodes:
            #     continue

            source_node.pvs.add(dest_node)

            penumbra = self.compute_anti_penumbra2(visited_portals[-1], target_portal)
            if not penumbra.is_valid:
                continue
            if last_penumbra:
                intersection = last_penumbra.intersection(penumbra)
                if isinstance(intersection, GeometryCollection):
                    intersection = next(p for p in intersection if isinstance(p, ShapelyPolygon))
                # elif isinstance(intersection, MultiPolygon):
                #     a=2
                elif isinstance(intersection, ShapelyPolygon) and intersection.is_empty:
                    continue
                elif not isinstance(intersection, ShapelyPolygon):
                    continue
                penumbra = Polygon.from_shapely(intersection)
            if not penumbra.is_valid:
                continue

            # Plot portals
            # source_portal.plot()
            # plt.pause(0.01)
            # target_portal.plot()
            # plt.pause(0.01)
            #
            # interval.plot(ref_point=ref_point, radius=1000)
            # plt.pause(0.01)

            dest_portals = []
            # Check all portals, except the once we are looking through
            valid_destination_portals = set(dest_node.portals) - {target_portal}
            a=2
            for dest_portal in valid_destination_portals:
                # If the destination portal is collinear to the target or source portals, then we can not see the node
                # it leads to (at least not through the target node)
                if dest_portal.compare(target_portal) == 'C' or dest_portal.compare(source_portal) == 'C':
                    a=2
                    continue

                if not penumbra.intersects(dest_portal.linestring):
                    continue

                ls = dest_portal.linestring.intersection(penumbra)
                if not ls or not isinstance(ls, LineString) or ls.length < 0.1:
                    continue

                # try:
                #     lines = split(dest_portal.linestring, penumbra)
                # except:
                #     # Destination portal coincides with the side of the penumbra
                #     continue
                #
                # contains = [penumbra.contains(l) for l in lines]
                # truths = [i for i in contains if i]
                # if len(truths)==0 or len(truths)>1:
                #     a=2
                #
                # ls = next(l for l in lines if penumbra.contains(l))
                #
                # # intersections = [penumbra.intersection(l) for l in lines]
                # # ls = next(filter(lambda l: isinstance(l, LineString) and not l.is_empty and l.length>1e-10, intersections), None)
                #
                # if not ls:
                #     a=2
                #     continue
                if ls.length< 1e-1:
                    continue
                dest_portal = LineSegment.from_linestring(ls, name=dest_portal.name)
                a = 2
                dest_portals.append(dest_portal)

            if len(dest_portals):
                self._gen_pvs2(source_node, dest_node, source_portal, dest_portals,
                              visited_nodes + [dest_node], visited_portals + [target_portal], penumbra)


    def compute_anti_penumbra2(self, source, target):
        line1 = LineSegment(source.p1, target.p1)
        line2 = LineSegment(source.p2, target.p2)
        if not line1.linestring.intersects(line2.linestring):
            if not line1.linestring.buffer(1e-7).intersects(line2.linestring):
                line1 = LineSegment(source.p1, target.p2)
                line2 = LineSegment(source.p2, target.p1)

        _, phi1 = line1.p2.to_polar(line1.p1)
        _, phi2 = line2.p2.to_polar(line2.p1)

        p1 = line1.p2
        p2 = line2.p2
        x3, y3 = pol2cart(1e7, phi2)
        x3, y3 = x3 + line2.p2.x, y3 + line2.p2.y
        p3 = Point(x3, y3)
        x4, y4 = pol2cart(1e7, phi1)
        x4, y4 = x4 + line1.p2.x, y4 + line1.p2.y
        p4 = Point(x4, y4)
        points = [p1, p2, p3, p4]
        return Polygon([(p.x, p.y) for p in points])


    def compute_anti_penumbra(self, source: LineSegment, target:LineSegment):
        line1 = LineSegment(source.p1, target.p1)
        line2 = LineSegment(source.p2, target.p2)
        if not line1.linestring.intersects(line2.linestring):
            line1 = LineSegment(source.p1, target.p2)
            line2 = LineSegment(source.p2, target.p1)


        p = line1.linestring.intersection(line2.linestring)
        if not p:
            return None, None
        p = Point(p.x, p.y)

        try:
            interval = target.to_interval(p)
        except:
            return None, None

        return p, interval


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)