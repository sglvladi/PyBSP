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

from .angles import angle_between
from .geometry import LineSegment, Point, Polygon
from .functions import extrapolate_line, merge_lines, process_line, merge_fovs2


# from multiprocessing import Pool, cpu_count
import tqdm
import random
# from joblib import Parallel, delayed

from collections.abc import Mapping

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
        self.walls = []
        self.pvs = set()
        self.wall_pvs = set()
        self.id = id

    def __repr__(self):
        return "BSPNode(id={}, data={}, left={}, right={})".format(self.id, self.data, self.front, self.back)

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class BSP:
    """Binary Space Partition class, optimally generates BSP tree from a list of line segments by using a heuristic"""
    def __init__(self, lines=None, heuristic='random', bounds=((0, 800), (0, 800)), pool=None):
        """Constructor, initializes binary tree"""
        self.heuristic = heuristic
        self.bounds = bounds
        self.node_connectivity: Mapping[int, Mapping[str, int]] = dict()
        self.portal_connections = None
        self.node_pvs = dict()
        self.wall_pvs = dict()

        self._walls = []
        self._portals = []
        self._nodes = []
        self._nodes_sorted = False

        if lines:
            self.generate_tree(lines, pool=pool)

    @property
    def root(self) -> [BSPNode]:
        return self.nodes[0]

    @property
    def nodes(self) -> [BSPNode]:
        """
        Return the nodes in the BSP tree, ordered by a depth-first search around the back of each node.

        Returns
        -------
        list of BSPNode objects

        """
        nodes = self._nodes
        return nodes

    def get_node(self, id) -> [BSPNode]:
        if self._nodes_sorted:
            return self.nodes[id]
        return next(filter(lambda node: node.id == id, self.nodes), None)

    def get_portal(self, name):
        return next(filter(lambda portal: portal.name == name, self._portals), None)

    def get_wall(self, name):
        return next(filter(lambda wall: wall.name == name, self._walls), None)

    def is_leaf(self, node):
        return node.front is None and node.back is None

    def is_root(self, node):
        return node.parent is None

    def is_solid(self, node):
        if self.is_leaf(node) and node.id == self.get_node(node.parent).back:
            return True
        return False

    def is_empty(self, node):
        if self.is_leaf(node) and node.id == self.get_node(node.parent).front:
            return True
        return False

    def sort_nodes_back(self):
        return self._sort_nodes(self.root)

    def _sort_nodes(self, node):
        nodes = [node]
        if node.back is not None:
            nodes += self._sort_nodes(self.nodes[node.back])
        if node.front is not None:
            nodes += self._sort_nodes(self.nodes[node.front])
        return nodes

    @property
    def solid_leaves(self) -> [BSPNode]:
        """
        Return all leaf nodes that represent a solid area (behind a wall)

        Returns
        -------
        list of BSPNode objects

        """
        return [n for n in self.nodes if self.is_solid(n)]

    @property
    def empty_leaves(self) -> [BSPNode]:
        """
        Return all leaf nodes that represent an empty area (in front of walls)

        Returns
        -------
        list of BSPNode objects

        """
        return [n for n in self.nodes if self.is_empty(n)]

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
            depth_left = self._depth(self.nodes[node.front])
        depth_right = 0
        if node.back is not None:
            depth_right = self._depth(self.nodes[node.back])
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

    def heuristic_min_partition(self, lines, pool=None) -> int:
        """
        Returns the index of the line segment which causes the least amount of partitions with
        other line segments in the list.
        """
        min_idx = 0
        if pool:
            inputs = [(idx, line, lines) for idx, line in enumerate(lines)]
            # mins = list(tqdm.tqdm(self.pool.istarmap(min_partitions, inputs), total=len(inputs)))
            # mins = []
            # for _ in tqdm.tqdm(self.pool.imap_unordered(min_partitions, inputs), total=len(inputs)):
            #     pass
            #     mins.append(result)
            # mins = self.pool.starmap(min_partitions, [(idx, line, lines) for idx, line in enumerate(lines)])
            # mins = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(min_partitions)(idx, line, lines) for idx, line in enumerate(lines))

            # mins = self.pool.imap_unordered(min_partitions, inputs)
            # mins = pool.imap(min_partitions, inputs)
            mins = pool.starmap(min_partitions, tqdm.tqdm(inputs, total=len(inputs)))
            mins_0 = [m[0] for m in mins]
            mins_1 = [m[1] for m in mins]
            min_idx_tmp = np.argmin(mins_1)
            min_idx = mins_0[min_idx_tmp]
        else:
            min_partition = np.inf
            for idx1, line1 in enumerate(tqdm.tqdm(lines)):
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

                # print('Done {}'.format(idx1))

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

    def generate_tree(self, lines, heuristic=None, pool=None):
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
        self._last_node_id = len(self.nodes)
        root = BSPNode(copy(lines), polygon=polygon, id=self._last_node_id)
        self.nodes.append(root)
        self._last_node_id += 1
        print('\nGenerating tree...')
        self._generate_tree(root, self.heuristic, pool)
        self._nodes.sort(key=lambda x: x.id)
        self._nodes_sorted = True

    def _generate_tree(self, node: BSPNode, heuristic='even', pool=None):
        best_idx = 0
        # print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(node.data, pool)
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(node.data, pool)
        elif heuristic == 'random':
            best_idx = np.random.randint(len(node.data))

        # print('....')
        data = []
        data_front = []
        data_back = []
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
            node.plane = LineSegment.from_linestring(pol_left.shapely.intersection(pol_right.shapely), line.normal, line.name)
        elif pol_left:
            node.plane = LineSegment.from_linestring(pol_left.shapely.intersection(l), line.normal, line.name)
        else:
            node.plane = LineSegment.from_linestring(pol_right.shapely.intersection(l), line.normal, line.name)
        # Ensure normal is preserved
        a = angle_between(node.plane.normalV.to_array(), line.normalV.to_array())
        if abs(a) > np.pi/100:
            node.plane = LineSegment(node.plane.p1, node.plane.p2, -node.plane.normal, node.plane.name)

        # Process lines
        for i, line2 in enumerate(node.data):
            result = line.compare(line2)
            if result == 'P':
                split_lines = line.split(line2)
                for split_line in split_lines:
                    res = line.compare(split_line)
                    if res == 'F':
                        data_front.append(split_line)
                    elif res == 'B':
                        data_back.append(split_line)
                    else:
                        print('Error!!', res)
            elif result == 'C':
                data.append(line2)
            elif result == 'F':
                data_front.append(line2)
            elif result == 'B':
                data_back.append(line2)

        # Generate front node
        node.data = data
        if not len(data_front):
            data_front = None
        node_front = BSPNode(data_front, parent=node.id, polygon=pol_left, id=len(self._nodes))
        self._nodes.append(node_front)
        node.front = node_front.id
        if data_front:
            self._generate_tree(node_front, heuristic, pool=pool)

        # Generate back node
        if not len(data_back):
            data_back = None
        node_back = BSPNode(data_back, parent=node.id, polygon=pol_right, id=len(self._nodes))
        self._nodes.append(node_back)
        node.back = node_back.id
        if data_back:
            self._generate_tree(node_back, heuristic, pool=pool)

    def gen_portals(self, pool=None, chunk_size=50):
        self._portals = []
        self._removed_portals = set()
        self._replacement_portals = dict()
        empty_leaves = self.empty_leaves

        sorted_nodes = self.sort_nodes_back()

        for node in tqdm.tqdm(sorted_nodes, total=len(self.nodes)):
            # update node portals in case replacement or removal has occurred
            node.portals = self._update_portals(node.portals)
            #
            if self.is_solid(node):
                self._removed_portals |= {p for p in node.portals}
                self._portals = [p for p in self._portals if p.name not in self._removed_portals]
                node.portals = []
                continue
            elif self.is_empty(node):
                continue
            portal = node.plane
            self._portals.append(portal)
            node.portals.append(portal.name)
            # Push down
            for p_name in node.portals:
                p = self.get_portal(p_name)
                res = node.plane.compare(p)
                if res == 'F':
                    self.nodes[node.front].portals.append(p.name)
                elif res == 'B':
                    self.nodes[node.back].portals.append(p.name)
                elif res == 'P':
                    lines = node.plane.split(p)
                    if node.plane.compare(lines[0]) == 'F':
                        self.nodes[node.front].portals.append(lines[0].name)
                        self.nodes[node.back].portals.append(lines[1].name)
                    else:
                        self.nodes[node.back].portals.append(lines[0].name)
                        self.nodes[node.front].portals.append(lines[1].name)
                    self._replacement_portals[p.name] = [lines[0].name, lines[1].name]
                    self._portals.remove(p)
                    self._portals += lines
                elif res == 'C':
                    self.nodes[node.front].portals.append(p.name)
                    self.nodes[node.back].portals.append(p.name)

        # node_portals = dict()
        # for node in tqdm.tqdm(empty_leaves, total=len(empty_leaves)):
        #     node_portals[node] = self._update_portals(node.portals)

        if pool:
            # Step 1 - Update portals
            node_ids = [node.id for node in empty_leaves]
            chunks = get_chunks(node_ids, chunk_size)
            inputs = [(self, chunk) for chunk in chunks]
            updated_portals_chunks = pool.starmap(update_portals, tqdm.tqdm(inputs, total=len(inputs)))
            updated_portals = [item for sublist in updated_portals_chunks for item in sublist]
            updated_portals = {item[0]: item[1] for item in updated_portals}
            for node in empty_leaves:
                node.portals = updated_portals[node.id]

            # Step 2 - Generate connectivity look-up tables
            portal_names = [portal.name for portal in self._portals]
            chunks = get_chunks(portal_names, chunk_size)
            inputs = [(chunk, empty_leaves) for chunk in chunks]
            portal_connections_chunks = pool.starmap(process_portal_connections, tqdm.tqdm(inputs, total=len(inputs)))
            portal_connections_list = [item for sublist in portal_connections_chunks for item in sublist]
            self.portal_connections = {item[0]: item[1] for item in portal_connections_list}
            self.node_connectivity = {node.id: dict() for node in empty_leaves}
            for item in portal_connections_list:
                portal = item[0]
                nodes = item[1]
                self.node_connectivity[nodes[0]][portal] = nodes[1]
                self.node_connectivity[nodes[1]][portal] = nodes[0]
        else:
            # Step 1 - Update portals
            for node in tqdm.tqdm(empty_leaves, total=len(empty_leaves)):
                node.portals = self._update_portals(node.portals)

            # Step 2 - Generate connectivity look-up tables
            self.portal_connections = dict()
            self.node_connectivity = {node.id: dict() for node in empty_leaves}
            for portal in tqdm.tqdm(self._portals, total=len(self._portals)):
                nodes = [n.id for n in empty_leaves if portal.name in n.portals]
                self.portal_connections[portal.name] = nodes
                self.node_connectivity[nodes[0]][portal.name] = nodes[1]
                self.node_connectivity[nodes[1]][portal.name] = nodes[0]

        # if pool:
        #     portal_names = [portal.name for portal in self._portals]
        #     chunks = get_chunks(portal_names, chunk_size)
        #     inputs = [(chunk, empty_leaves) for chunk in chunks]
        #     portal_connections_chunks = pool.starmap(process_portal_connections, tqdm.tqdm(inputs, total=len(inputs)))
        #     portal_connections_list = [item for sublist in portal_connections_chunks for item in sublist]
        #     self.portal_connections = {item[0]: item[1] for item in portal_connections_list}
        #     self.node_connectivity = {node.id: dict() for node in empty_leaves}
        #     for item in portal_connections_list:
        #         portal = item[0]
        #         nodes = item[1]
        #         self.node_connectivity[nodes[0]][portal] = nodes[1]
        #         self.node_connectivity[nodes[1]][portal] = nodes[0]
        # else:
        #     self.portal_connections = dict()
        #     self.node_connectivity = {node.id: dict() for node in empty_leaves}
        #     for portal in tqdm.tqdm(self._portals, total=len(self._portals)):
        #         nodes = [n.id for n in empty_leaves if portal.name in n.portals]
        #         self.portal_connections[portal.name] = nodes
        #         self.node_connectivity[nodes[0]][portal.name] = nodes[1]
        #         self.node_connectivity[nodes[1]][portal.name] = nodes[0]

        # if pool:
        #     inputs = [(node, {portal: self.portal_connections[portal] for portal in node.portals}) for node in empty_leaves]
        #     node_connectivity = pool.starmap(process_node_connectivity, tqdm.tqdm(inputs, total=len(inputs)))
        #     self.node_connectivity = {item[0]: item[1] for item in node_connectivity}
        # else:
        #     self.node_connectivity = dict()
        #     for node in tqdm.tqdm(empty_leaves, total=len(empty_leaves)):
        #         self.node_connectivity[node.id] = {portal: next(n for n in self.portal_connections[portal]
        #                                                         if n != node.id)
        #                                            for portal in node.portals}
        #         # self.node_connectivity[node.id] = dict()
        #         # for portal in node.portals:
        #         #     self.node_connectivity[node][portal] = next(n for n in self.portal_connections[portal] if n != node)

    def _update_portals(self, portals):
        portal_names = [p for p in portals]
        processed_portals = set()
        while any([po in self._replacement_portals for po in set(portal_names)-processed_portals]):
            for portal_name in portal_names:
                if portal_name in self._replacement_portals.keys()-processed_portals:
                    portal_names += set(self._replacement_portals[portal_name])
                    processed_portals.add(portal_name)
        return [p.name for p in self._portals if p.name in set(portal_names)-processed_portals]

    def gen_walls(self, pool=None, chunk_size=50):
        self._replacement_portals = dict()
        self._walls = []
        sorted_nodes = self.sort_nodes_back()
        for node in tqdm.tqdm(sorted_nodes, total=len(self.nodes)):
            # update node portals in case replacement or removal has occurred
            node.walls = self._update_walls(node.walls)
            if self.is_solid(node) or self.is_empty(node):
                continue
            portal = node.plane
            self._walls.append(portal)
            node.walls.append(portal.name)
            # Push down
            for p_name in node.walls:
                p = self.get_wall(p_name)
                res = node.plane.compare(p)
                if res == 'F':
                    self.nodes[node.front].walls.append(p.name)
                elif res == 'B':
                    self.nodes[node.back].walls.append(p.name)
                elif res == 'P':
                    lines = node.plane.split(p)
                    if node.plane.compare(lines[0]) == 'F':
                        self.nodes[node.front].walls.append(lines[0].name)
                        self.nodes[node.back].walls.append(lines[1].name)
                    else:
                        self.nodes[node.back].walls.append(lines[0].name)
                        self.nodes[node.front].walls.append(lines[1].name)
                    self._replacement_portals[p.name] = [lines[0].name, lines[1].name]
                    self._walls.remove(p)
                    self._walls += lines
                elif res == 'C':
                    self.nodes[node.front].walls.append(p.name)
                    self.nodes[node.back].walls.append(p.name)

        empty_leaves = self.empty_leaves


        # updated_walls_1 = dict()
        # for node in tqdm.tqdm(self.empty_leaves, total=len(self.empty_leaves)):
        #     updated_walls_1[node.id] = self._update_walls(node.walls)

        if pool:
            # Step 1 - Update walls
            node_ids = [node.id for node in empty_leaves]
            chunks = get_chunks(node_ids, chunk_size)
            inputs = [(self, chunk) for chunk in chunks]
            updated_walls_chunks = pool.starmap(update_walls, tqdm.tqdm(inputs, total=len(inputs)))
            updated_walls = [item for sublist in updated_walls_chunks for item in sublist]
            updated_walls = {item[0]: item[1] for item in updated_walls}
            for node in empty_leaves:
                node.walls = updated_walls[node.id]

            # Step 2 - Remove portals
            portal_names = [p.name for p in self._portals]
            self._walls = [w for w in self._walls if w.name not in portal_names]
            inputs = [(self, chunk) for chunk in chunks]
            corrected_walls_chunks = pool.starmap(correct_walls, tqdm.tqdm(inputs, total=len(inputs)))
            corrected_walls = [item for sublist in corrected_walls_chunks for item in sublist]
            corrected_walls = {item[0]: item[1] for item in corrected_walls}
            for node in empty_leaves:
                node.walls = corrected_walls[node.id]
        else:
            # Step 1 - Update walls
            for node in tqdm.tqdm(self.empty_leaves, total=len(self.empty_leaves)):
                node.walls = self._update_walls(node.walls)

            # Step 2 - Remove portals
            portal_names = [p.name for p in self._portals]
            self._walls = [w for w in self._walls if w.name not in portal_names]
            wall_names = [w.name for w in self._walls]
            for node in tqdm.tqdm(self.empty_leaves, total=len(self.empty_leaves)):
                # node.walls = [p for p in node.walls if p not in portal_names]
                node.walls = [w for w in node.walls if w in wall_names]

        # corrected_walls_1 = dict()
        # wall_names = [w.name for w in self._walls]
        # for node in tqdm.tqdm(self.empty_leaves, total=len(self.empty_leaves)):
        #     # node.walls = [p for p in node.walls if p not in portal_names]
        #     corrected_walls_1[node.id] = [w for w in node.walls if w in wall_names]

        # if pool:
        #     inputs = [(self, chunk) for chunk in chunks]
        #     corrected_walls_chunks = pool.starmap(correct_walls, tqdm.tqdm(inputs, total=len(inputs)))
        #     corrected_walls = [item for sublist in corrected_walls_chunks for item in sublist]
        #     corrected_walls = {item[0]: item[1] for item in corrected_walls}
        #     for node in empty_leaves:
        #         node.walls = corrected_walls[node.id]
        # else:
        #     wall_names = [w.name for w in self._walls]
        #     for node in tqdm.tqdm(self.empty_leaves, total=len(self.empty_leaves)):
        #         # node.walls = [p for p in node.walls if p not in portal_names]
        #         node.walls = [w for w in node.walls if w in wall_names]

    def _update_walls(self, portals):
        portal_names = [p for p in portals]
        processed_portals = set()
        while any([po in self._replacement_portals for po in set(portal_names)-processed_portals]):
            for portal_name in portal_names:
                if portal_name in self._replacement_portals.keys()-processed_portals:
                    portal_names += set(self._replacement_portals[portal_name])
                    processed_portals.add(portal_name)
        return [p.name for p in self._walls if p.name in set(portal_names)-processed_portals]

    def sim_pvs(self, path):
        penumbras = []
        artists = []
        for i in range(1, len(path) - 1):
            previous_node = path[i - 1]
            current_node = path[i]
            next_node = path[i + 1]
            source_portal = next(
                line for line, node in self.node_connectivity[previous_node].items() if node == current_node)
            target_portal = next(
                line for line, node in self.node_connectivity[current_node].items() if node == next_node)
            if len(penumbras):
                ls = target_portal.shapely.intersection(penumbras[-1].shapely)
                target_portal = LineSegment.from_linestring(ls, name=target_portal.name)
            penumbra = compute_anti_penumbra2(source_portal, target_portal)
            if len(penumbras):
                last_penumbra = penumbras[-1]
                intersection = last_penumbra.shapely.intersection(penumbra.shapely)
                penumbra = Polygon.from_shapely(intersection)
            penumbras.append(penumbra)
            artists.append(penumbra.plot())
            plt.pause(1)
        return penumbras, artists

    def gen_pvs(self, pool=None):
        self.node_pvs = dict()
        if pool:
            inputs = [(self, source_node.id) for source_node in self.empty_leaves]

            results = pool.starmap(BSP._gen_pvs, tqdm.tqdm(inputs, total=len(inputs)))
            for res in tqdm.tqdm(results):
                source_node = self.nodes[res[0]]
                source_node.pvs = res[1]
                source_node.wall_pvs = res[2]
        else:
            for source_node in tqdm.tqdm(self.empty_leaves):
                source_node.pvs = set()
                self.node_pvs[source_node.id] = dict()
                source_node.wall_pvs = set(source_node.walls)
                a=2
                for source_portal_name in source_node.portals:
                    source_portal = self.get_portal(source_portal_name)
                    target_node = self.nodes[self.node_connectivity[source_node.id][source_portal_name]]
                    source_node.pvs.add(target_node.id)
                    source_node.wall_pvs |= set(target_node.walls)
                    target_portals = [self.get_portal(p_name) for p_name in target_node.portals]
                    self._gen_pvs_recursive(source_node, target_node, source_portal, target_portals,
                                            [source_node.id, target_node.id], [source_portal])
                # source_node.pvs = list(source_node.pvs)
                # source_node.wall_pvs = list(source_node.wall_pvs)

    def _gen_pvs(self, source_node_id):
        source_node = self.nodes[source_node_id]
        source_node.pvs = set()
        self.node_pvs[source_node] = dict()
        source_node.wall_pvs = set(source_node.walls)
        a = 2
        for source_portal_name in source_node.portals:
            source_portal = self.get_portal(source_portal_name)
            target_node = self.nodes[self.node_connectivity[source_node.id][source_portal_name]]
            source_node.pvs.add(target_node.id)
            source_node.wall_pvs |= set(target_node.walls)
            target_portals = [self.get_portal(p_name) for p_name in target_node.portals]
            self._gen_pvs_recursive(source_node, target_node, source_portal, target_portals,
                                    [source_node.id, target_node.id], [source_portal])
        # source_node.pvs = list(source_node.pvs)
        # source_node.pvs = [n.id for n in source_node.pvs]
        return source_node_id, source_node.pvs, source_node.wall_pvs

    def _gen_pvs_recursive(self, source_node, current_node, source_portal, target_portals: list,
                           visited_node_ids: list, visited_portals: list, last_penumbra=None):
        target_portals = copy(target_portals)
        visited_node_ids = copy(visited_node_ids)
        visited_portals = copy(visited_portals)

        # Check all portals except the one we are looking through
        valid_target_portals = set(target_portals) - {visited_portals[-1]}
        a = 2
        for target_portal in valid_target_portals:
            if source_portal.compare(target_portal) == 'C':
                a = 2
                continue
            dest_node = self.nodes[self.node_connectivity[current_node.id][target_portal.name]]

            # Avoid circle back to visited nodes
            # if len(visited_nodes>200):
            #     a=2
            # if dest_node == source_node or target_portal in visited_portals:
            #     continue

            source_node.pvs.add(dest_node.id)
            source_node.wall_pvs |= set(dest_node.walls)
            # if dest_node.id in self.node_pvs[source_node]:
            #     self.node_pvs[source_node][dest_node].append(visited_nodes+[dest_node])
            # else:
            #     self.node_pvs[source_node][dest_node] = [visited_nodes+[dest_node]]

            # visited_portals = [self.get_portal(p_name) for p_name in visited_portal_names]
            penumbra = compute_anti_penumbra2(visited_portals[-1], target_portal)
            if not penumbra.shapely.is_valid:
                continue
            if last_penumbra:
                intersection = last_penumbra.shapely.intersection(penumbra.shapely)
                if isinstance(intersection, GeometryCollection):
                    intersection = next(p for p in intersection if isinstance(p, ShapelyPolygon))
                elif isinstance(intersection, MultiPolygon):
                    # TODO: Need to handle this!!!!
                    continue
                     # a=2
                elif isinstance(intersection, ShapelyPolygon) and intersection.is_empty:
                    continue
                elif not isinstance(intersection, ShapelyPolygon):
                    continue
                penumbra = Polygon.from_shapely(intersection)
            if not penumbra.shapely.is_valid:
                continue

            dest_portals = []
            # Check all portals, except the one we are looking through
            p_names = set(dest_node.portals) - {target_portal.name}
            valid_destination_portals = set([self.get_portal(p_name) for p_name in p_names]) - {target_portal}
            a=2
            for dest_portal in valid_destination_portals:
                # If the destination portal is collinear to the target or source portals, then we can not see the node
                # it leads to (at least not through the target node)
                if dest_portal.compare(target_portal) == 'C' or dest_portal.compare(source_portal) == 'C':
                    a=2
                    continue

                if not penumbra.shapely.intersects(dest_portal.shapely):
                    continue

                ls = dest_portal.shapely.intersection(penumbra.shapely)
                if not ls or not isinstance(ls, LineString) or ls.length < 0.1:
                    continue

                if ls.length< 1e-1:
                    continue
                dest_portal = LineSegment.from_linestring(ls, name=dest_portal.name)
                a = 2
                dest_portals.append(dest_portal)

            if len(dest_portals):
                self._gen_pvs_recursive(source_node, dest_node, source_portal, dest_portals,
                                        visited_node_ids + [dest_node.id], visited_portals + [target_portal], penumbra)

    def _traverse_tree_nx(self, node, g, parent=None):
        child = g.number_of_nodes() + 1
        id = node.id
        data = node.data
        polygon = node.polygon
        leaf = self.is_leaf(node)
        solid = self.is_solid(node)
        empty = self.is_empty(node)
        if polygon is not None:
            g.add_node(child, data=data, leaf=leaf, polygon=polygon, solid=solid, empty=empty, id=id)
            if parent:
                g.add_edge(parent, child)
            if node.front is not None:
                g = self._traverse_tree_nx(self.get_node(node.front), g, child)
            if node.back is not None:
                g = self._traverse_tree_nx(self.get_node(node.back), g, child)
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
            labels = {n: g.nodes[n]['id'] for n in g.nodes} #  if g.nodes[n]['leaf']}
            pos_labels = {}
            for node, coords in pos.items():
                # if g.nodes[node]['leaf']:
                pos_labels[node] = (coords[0], coords[1])
            nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels, font_color='white')

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

    def plot_planes(self, ax=None, color='m', linewidth=0.2, annotate=False, **kwargs):
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
        if not ax:
            ax = plt.gca()
        nodes = self.nodes
        for node in nodes:
            if node.plane:
                node.plane.plot(ax=ax, color=color, linewidth=linewidth, **kwargs)
                midpoint = node.plane.mid_point
                if annotate:
                    ax.text(midpoint.x, midpoint.y, node.plane.name, color='y')

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
            plane = node.plane
            node_front = self.nodes[node.front]
            node_back = self.nodes[node.back]
            if p.compare(plane) > 0:
                if self.is_leaf(node_front):
                    return node_front
                node = node_front
            elif p.compare(plane) < 0:
                if self.is_leaf(node_back):
                    return node_back
                node = node_back
            else:
                if self.is_empty(node_front):
                    return node_front
                if self.is_empty(node_back):
                    return node_back
                return node_front

    def render(self, ref_point, ax=None, use_pvs=False):
        # Find the leaf node
        leaf = self.find_leaf(ref_point)

        r_lines = [] # self._render_child2(leaf, ref_point)
        parent = self.nodes[leaf.parent]

        p_lines = []
        if not self.is_root(leaf):
            p_lines = self._render_parent(parent, leaf, ref_point)

        l = r_lines + p_lines

        # Convert to ranges
        vis_lines = []
        fov = []
        for idx, line in enumerate(l):
            if not use_pvs:
                fov, vis_lines = process_line(line, fov, vis_lines, ref_point, ax)
                fov = merge_fovs2(fov)
                if len(fov) == 1 and fov[0].delta == np.pi:
                    break
            else:
                w_pvs_names = set([line.name.split('_')[0] for line in leaf.wall_pvs])
                if line.name.split('_')[0] in w_pvs_names:
                    fov, vis_lines = process_line(line, fov, vis_lines, ref_point, ax)
                    fov = merge_fovs2(fov)
                    if len(fov) == 1 and fov[0].delta == np.pi:
                        break
                else:
                    a=2
                    continue

        merged = merge_lines(vis_lines)
        return merged

    def _render_parent(self, parent, child, p):
        a = parent.data[0].name
        r_lines = parent.data

        if self.nodes[parent.front] == child:
            if parent.back is not None and self.nodes[parent.back]:
                if p.compare(parent.data[0]) > 0:
                    r_lines = r_lines + self._render_child(self.nodes[parent.back], p)
                else:
                    r_lines = self._render_child(self.nodes[parent.back], p) + r_lines
        else:
            if parent.front is not None and self.nodes[parent.front]:
                if p.compare(parent.data[0]) > 0:
                    r_lines = self._render_child(self.nodes[parent.front], p) + r_lines
                else:
                    r_lines = r_lines + self._render_child(self.nodes[parent.front], p)

        p_lines = []
        if not self.is_root(parent):
            p_lines = self._render_parent(self.nodes[parent.parent], parent, p)

        r_lines = r_lines + p_lines
        return r_lines

    def _render_child(self, child, p):
        a = child.data[0].name
        r_lines = child.data
        if self.is_leaf(child):
            return r_lines
        else:
            r_lines_left = []
            r_lines_right = []

            if child.front is not None and self.nodes[child.front]:
                r_lines_left = self._render_child(self.nodes[child.front], p)
            if child.back is not None and self.nodes[child.back]:
                r_lines_right = self._render_child(self.nodes[child.back], p)

            if p.compare(child.data[0]) > 0:
                # if point is in front of dividing plane
                r_lines = r_lines_left + r_lines + r_lines_right
            else:
                r_lines = r_lines_right + r_lines + r_lines_left
        return r_lines

    def plot_longest_path(self, n1, n2):
        path_lens = [len(p) for p in self.node_pvs[n1][n2]]
        path_idx = np.argmax(path_lens)
        path = self.node_pvs[n1][n2][path_idx]
        _, artists = self.sim_pvs(path)
        return artists


# //////////////////////////////////////////////////////////////////////

def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.shapely.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def compute_anti_penumbra2(source, target):
    line1 = LineSegment(source.p1, target.p1)
    line2 = LineSegment(source.p2, target.p2)
    if not line1.shapely.intersects(line2.shapely):
        if not line1.shapely.buffer(1e-7).intersects(line2.shapely):
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


def min_partitions(idx, line, lines):
# def min_partitions(args):
#     idx, line, lines = args
    partition_count = 0
    # if len(lines) > 1000:
    #     samples = random.sample(lines, 1000)
    # else:
    samples = lines
    for idx2, line2 in enumerate(samples):
        # print("{}|{} out of {}".format(idx1, idx2, len(lines)))
        if idx != idx2:
            res = line.compare(line2)
            if res == 'P':
                partition_count += 1
    # print('Done {}'.format(idx))
    return idx, partition_count


def update_portals(obj, node_ids):
    updated_portals = []
    for node_id in node_ids:
        node = obj.nodes[node_id]
        updated_portals.append((node_id, obj._update_portals(node.portals)))
    return updated_portals


def process_portal_connections(portal_names, empty_leaves):
    portal_conns = []
    for portal_name in portal_names:
        portal_conns.append((portal_name, [n.id for n in empty_leaves if portal_name in n.portals]))
    return portal_conns


def update_walls(obj, node_ids):
    updated_walls = []
    for node_id in node_ids:
        node = obj.nodes[node_id]
        updated_walls.append((node_id, obj._update_walls(node.walls)))
    return updated_walls


def correct_walls(obj, node_ids):
    wall_names = [w.name for w in obj._walls]
    corrected_walls = []
    for node_id in node_ids:
        node = obj.nodes[node_id]
        corrected_walls.append((node_id, [w for w in node.walls if w in wall_names]))
    return corrected_walls


# def process_node_connectivity(node_id, node_portals, portal_connections):
#     node_connectivity = dict()
#     for portal in node.portals:
#         node_connectivity[portal] = next(n for n in portal_connections[portal] if n != node)
#     return node, node_connectivity

def get_chunks(lst, n):
    """Split lst into n-sized chunks"""
    chunks = [lst[i:i + n] for i in range(0, len(lst), n)]
    return chunks
