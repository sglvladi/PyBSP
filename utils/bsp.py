import datetime
import os.path
import pickle
import signal
import random
from copy import copy
import multiprocessing as mpp
from collections.abc import Mapping

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import linemerge, unary_union, polygonize
from stonesoup.functions import pol2cart

from .angles import angle_between
from .geometry import LineSegment, Point, Polygon
from .functions import extrapolate_line, merge_lines, process_line, merge_fovs2

panic = None

class BSPNode:
    """BSP Node class"""
    def __init__(self, data=None, parent=None, polygon=None, plane=None, id=None):
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
    def __init__(self, heuristic='random', bounds=((0, 800), (0, 800)), backup_folder=None):
        """Constructor, initializes binary tree"""
        self.heuristic = heuristic
        self.bounds = bounds
        self.backup_folder = backup_folder

        self.node_connectivity: Mapping[int, Mapping[str, int]] = dict()
        self.portal_connections = None
        self.node_pvs = dict()
        self.wall_pvs = dict()

        self._walls = []
        self._portals = dict()
        self._nodes = []
        self._nodes_sorted = False

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
        return self._portals[name]

    def get_wall(self, name):
        return self._walls[name]

    @staticmethod
    def is_leaf(node):
        return node.front is None and node.back is None

    @staticmethod
    def is_root(node):
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
            mins = imap_tqdm(pool, min_partitions, inputs, chunksize=None)
            mins_idx = [m[0] for m in mins]
            mins_val = [m[1] for m in mins]
            min_idx_tmp = np.argmin(mins_val)
            min_idx = mins_idx[min_idx_tmp]
        else:
            min_partition = np.inf
            for idx1, line1 in enumerate(tqdm.tqdm(lines)):
                if len(lines) > 1000:
                    samples = random.sample(lines, 1000)
                else:
                    samples = lines
                results = np.array(line1.compare(samples))
                partition_count = np.sum(results == 'P')

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

    def train(self, lines, parallel=True, backup_folder=None, start_stage=1, end_stage=3):
        if backup_folder:
            self.backup_folder = backup_folder

        if start_stage<1 or start_stage>3:
            raise ValueError('start_stage should take values between 1-3. Given: start_stage={}'.format(start_stage))
        if end_stage<1 or end_stage>3:
            raise ValueError('end_stage should take values between 1-3. Given: end_stage={}'.format(end_stage))
        if end_stage<start_stage:
            raise ValueError('end_stage cannot be smaller than start_stage! Given: start_stage={} and end_stage={}'
                             .format(start_stage, end_stage))

        # Stage 1 - Tree generation
        if start_stage == 1:
            self.generate_tree(lines, parallel=parallel)

        # Stage 2 - Portal/Wall generation
        if end_stage >= 2:
            if start_stage <= 2:
                self.gen_portals_walls(parallel=parallel)

        # Stage 3 - PVS generation
        if end_stage >= 3:
            if start_stage <= 3:
                self.gen_pvs(parallel=parallel)

    def generate_tree(self, lines, heuristic=None, parallel=None):
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

        print('\nGenerating tree...')

        pool = None
        if parallel:
            print('Initializing process pool...')
            pool = mpp.Pool(mpp.cpu_count())

        if heuristic:
            self.heuristic = heuristic
        polygon = Polygon(((self.bounds[0][0], self.bounds[1][0]), (self.bounds[0][0], self.bounds[1][1]),
                           (self.bounds[0][1], self.bounds[1][1]), (self.bounds[0][1], self.bounds[1][0])))
        self._last_node_id = len(self.nodes)
        root = BSPNode(copy(lines), polygon=polygon, id=self._last_node_id)
        self.nodes.append(root)
        self._last_node_id += 1

        # Run recursive tree generation
        self._generate_tree(root, self.heuristic, pool)
        self._nodes.sort(key=lambda x: x.id)
        self._nodes_sorted = True
        if self.backup_folder:
            self.serialize(os.path.join(self.backup_folder, 'Stage1', 'final.pickle'))

        if pool:
            pool.close()
            pool.join()

    def _generate_tree(self, node: BSPNode, heuristic='even', pool=None):
        best_idx = 0
        # print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(node.data, pool)
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(node.data)
        elif heuristic == 'rand':
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

    def gen_portals_walls(self, parallel=None, chunk_size=None, backup_folder=None):

        print('\nGenerating portals and walls...')

        pool = None
        if parallel:
            print('Initializing process pool...')
            pool = mpp.Pool(mpp.cpu_count())

        if not backup_folder and self.backup_folder:
            backup_folder = os.path.join(self.backup_folder, 'Stage2')

        self._portals = dict()
        self._known_walls = set()
        self._replacement_portals = dict()
        empty_leaves = self.empty_leaves

        # Step 1 - Main generation process
        sorted_nodes = self.sort_nodes_back()  # Nodes need to be sorted in depth-first search, priority to back nodes
        for node in tqdm.tqdm(sorted_nodes, desc='Step 1'):
            # update node portals in case replacement has occurred
            node.portals = update_portals_walls(node.portals, self._replacement_portals)
            #
            if self.is_solid(node):
                self._known_walls |= {p for p in node.portals}
                node.portals = []
                continue
            elif self.is_empty(node):
                continue
            portal = node.plane
            self._portals[portal.name] = portal
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

                    parts = p.name.split('_')
                    it = 0
                    if parts[0] in self._replacement_portals:
                        current_dict = self._replacement_portals[parts[0]]
                        it += 1
                        while True:
                            if parts[it] not in current_dict or not current_dict[parts[it]]:
                                parts_0 = lines[0].name.split('_')
                                parts_1 = lines[1].name.split('_')
                                nested_set(self._replacement_portals, parts, {parts_0[-1]: dict(), parts_1[-1]: dict()})
                                break
                            else:
                                current_dict = current_dict[parts[it]]
                                it += 1
                    else:
                        parts_0 = lines[0].name.split('_')
                        parts_1 = lines[1].name.split('_')
                        if len(parts)>1:
                            nested_set(self._replacement_portals, parts, {parts_0[-1]: dict(), parts_1[-1]: dict()})
                        else:
                            self._replacement_portals[p.name] = {parts_0[-1]: dict(), parts_1[-1]: dict()}

                    self._portals.pop(p.name)
                    self._portals.update({l.name: l for l in lines})
                elif res == 'C':
                    self.nodes[node.front].portals.append(p.name)
                    self.nodes[node.back].portals.append(p.name)

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'checkpoint1.pickle'))

        # Step 2 - Update portals
        if pool:
            inputs = [[node.id, node.portals] for node in empty_leaves]
            results_chunks = imap_tqdm_chunk(pool, update_portals_walls_chunk, inputs, (self._replacement_portals,),
                                             chunksize=chunk_size, desc='Step 2')
            results = {item[0]: item[1] for chunk in results_chunks for item in chunk}  # Unpack chunks
            for node in empty_leaves:
                node.portals = results[node.id]
        else:
            for node in tqdm.tqdm(empty_leaves, total=len(empty_leaves)):
                node.portals = update_portals_walls(node.portals, self._replacement_portals)

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'checkpoint2.pickle'))

        # Step 3 - Extract walls and replace
        wall_names = [w for w in self._known_walls]
        wall_names = update_portals_walls(wall_names, self._replacement_portals)
        self._walls = {w_name: self._portals[w_name] for w_name in tqdm.tqdm(wall_names, desc='Step 3')}

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'checkpoint3.pickle'))

        # Step 4 - Filter out walls from portals
        portal_names = self._portals.keys() - self._walls.keys()
        self._portals = {p_name: self._portals[p_name] for p_name in portal_names}
        if pool:
            inputs = [[node.id, node.portals] for node in empty_leaves]
            results_chunks = imap_tqdm_chunk(pool, filter_node_portal_walls_chunk, inputs, (wall_names,),
                                             chunksize=chunk_size, desc='Step 4')
            results = {item[0]: (item[1], item[2]) for chunk in results_chunks for item in chunk}  # Unpack chunks
            for node in empty_leaves:
                node.portals = results[node.id][0]
                node.walls = results[node.id][1]
        else:
            for node in tqdm.tqdm(empty_leaves, desc='Step 4'):
                node.walls = list(set(node.portals).intersection(set(wall_names)))
                node.portals = list(set(node.portals) - set(node.walls))

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'checkpoint4.pickle'))

        # Step 5 - Refactor portals and walls
        portal_name_mapping = dict()
        new_portals = dict()
        wall_name_mapping = dict()
        new_walls = dict()
        for i, (portal_name, portal) in enumerate(tqdm.tqdm(self._portals.items(), desc='Step 5.1')):
            new_name = str(i)
            portal_name_mapping[portal_name] = new_name
            portal.name = new_name
            new_portals[new_name] = portal
        for i, (wall_name, wall) in enumerate(tqdm.tqdm(self._walls.items(), desc='Step 5.2')):
            new_name = str(i)
            wall_name_mapping[wall_name] = new_name
            wall.name = new_name
            new_walls[new_name] = wall
        for node in tqdm.tqdm(empty_leaves, desc='Step 5.3'):
            node.portals = [portal_name_mapping[portal_name] for portal_name in node.portals]
            node.walls = [wall_name_mapping[wall_name] for wall_name in node.walls]
        self._portals = new_portals
        self._walls = new_walls

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'checkpoint5.pickle'))

        # Step 6 - Generate connectivity look-up tables
        if pool:
            portal_names = [portal_name for portal_name in self._portals]
            results_chunks = imap_tqdm_chunk(pool, process_portal_connections, portal_names, (empty_leaves,),
                                             chunksize=chunk_size, desc='Step 6')
            portal_connections_list = [item for chunk in results_chunks for item in chunk] # Unpack chunks
            self.portal_connections = {item[0]: item[1] for item in portal_connections_list}
            self.node_connectivity = {node.id: dict() for node in empty_leaves}
            for item in portal_connections_list:
                portal = item[0]
                node1, node2 = item[1]
                self.node_connectivity[node1][portal] = node2
                self.node_connectivity[node2][portal] = node1
        else:
            self.portal_connections = dict()
            self.node_connectivity = {node.id: dict() for node in empty_leaves}
            for portal_name in tqdm.tqdm(self._portals, total=len(self._portals)):
                nodes = [n.id for n in empty_leaves if portal_name in n.portals]
                self.portal_connections[portal_name] = nodes
                self.node_connectivity[nodes[0]][portal_name] = nodes[1]
                self.node_connectivity[nodes[1]][portal_name] = nodes[0]

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'final.pickle'))

        if pool:
            pool.close()
            pool.join()

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

    def gen_pvs(self, parallel=None, backup_folder=None, chunk_size=5000):
        print('\nGenerating PVS...')

        pool = None
        if parallel:
            print('Initializing process pool...')
            manager = mpp.Manager()
            global panic
            panic = manager.Value('i', False)
            pool = mpp.Pool(mpp.cpu_count(), initializer=init_worker, initargs=(panic,))

        if not backup_folder and self.backup_folder:
            backup_folder = os.path.join(self.backup_folder, 'Stage3')

        self.node_pvs = dict()
        if pool:
            self._processed_pvs = np.zeros((self.num_nodes,), dtype=bool)
            bar = tqdm.tqdm(total=len(self.empty_leaves), position=0, desc='Overall')
            chunks = get_chunks([n.id for n in self.empty_leaves], chunk_size)
            last_processed_node_id = 0
            for i, chunk in enumerate(chunks):
                inputs = [(id, self.nodes, self._portals, self.node_connectivity) for id in chunk]
                try:
                    results = imap_tqdm(pool, gen_pvs_single, inputs, position=1,
                                        desc='Chunk {}/{}'.format(i+1, len(chunks)))
                    self._processed_pvs[chunk] = True

                    for res in results:
                        source_node = self.nodes[res[0]]
                        source_node.pvs = np.flatnonzero(res[1]).tolist()
                        source_node.wall_pvs = set()
                        for node_id in source_node.pvs:
                            node = self.nodes[node_id]
                            source_node.wall_pvs |= set(node.walls)

                    bar.update(len(chunk))
                    last_processed_node_id = chunk[-1]
                    if backup_folder:
                        self.serialize(os.path.join(backup_folder, 'checkpoint1_{}.pickle'.format(last_processed_node_id)))
                except KeyboardInterrupt:
                    print('Interrupt handled smoothly...')
                    pool.close()
                    pool.join()
                    self.serialize(os.path.join(backup_folder, 'checkpoint1_{}.pickle'.format(last_processed_node_id)))
                    return

        else:
            for source_node in tqdm.tqdm(self.empty_leaves):
                _, source_node_pvs = gen_pvs_single((source_node.id, self.nodes, self._portals, self.node_connectivity))
                source_node.pvs = np.flatnonzero(source_node_pvs).tolist()
                source_node.wall_pvs = set()
                for node_id in source_node.pvs:
                    node = self.nodes[node_id]
                    source_node.wall_pvs |= set(node.walls)

        if backup_folder:
            self.serialize(os.path.join(backup_folder, 'final.pickle'))

        if pool:
            pool.close()
            pool.join()

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
        # pos = {n: g.nodes[n]['data'][0].getMidPoint().to_array()
        #        if g.nodes[n]['data']
        #        else np.array([g.nodes[n]['polygon'].centroid.x, g.nodes[n]['polygon'].centroid.y])
        #        for n in g.nodes}

        nx.draw(g, pos, ax=ax)
        leaf_nodes = [n for n in g.nodes if (g.nodes[n]['leaf'] and not g.nodes[n]['solid'])]
        solid_nodes = [n for n in g.nodes if (g.nodes[n]['leaf'] and g.nodes[n]['solid'])]
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=leaf_nodes, node_color='green', node_shape='s')
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=solid_nodes, node_color='red', node_shape='s')

        if show_labels:
            labels = {n: [l.name for l in g.nodes[n]['data']] if g.nodes[n]['data'] else '' for n in g.nodes}
            labels = {n: g.nodes[n]['id'] for n in g.nodes}  # if g.nodes[n]['leaf']}
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

    def serialize(self, path):
        print('Saving file: {}... '.format(path), end='')
        dir_path, filename = os.path.split(path)
        create_dirs(dir_path)
        pickle.dump(self, open(path, 'wb'))
        print('Done')

    @staticmethod
    def load(path, stage=None, checkpoint=None):
        if os.path.isdir(path):
            if stage is None and checkpoint is None:
                filepath = os.path.join(path, 'final.pickle')
            else:
                if isinstance(stage, int):
                    stage = 'Stage{}'.format(stage)
                if checkpoint == 'final':
                    filename = 'final.pickle'
                else:
                    filename = 'checkpoint{}.pickle'.format(checkpoint)
                filepath = os.path.join(path, stage, filename)
            return pickle.load(open(filepath, 'rb'))
        else:
            return pickle.load(open(path, 'rb'))


# //////////////////////////////////////////////////////////////////////
def imap_tqdm_chunk(pool, f, inputs, batch_inputs, chunksize=None, **tqdm_kwargs):
    if chunksize:
        chunks = get_chunks(inputs, chunksize)
    else:
        chunks = get_n_chunks(inputs, mpp.cpu_count())
    batch_inputs = tuple(batch_inputs)
    inputs = [(chunk, *batch_inputs) for chunk in chunks]
    p_chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
    if extra:
        p_chunksize += 1
    return list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=p_chunksize), total=len(inputs), **tqdm_kwargs))


def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results

# def gen_pvs_single(source_node_id, nodes, portals, node_connectivity):
def gen_pvs_single(args):
    source_node_id, nodes, portals, node_connectivity = args
    global panic
    if panic and panic.value:
        # print('Skipping {}...'.format(source_node_id))
        return None, None
    # else:
    #     print('Running {}'.format(source_node_id))
    source_node = nodes[source_node_id]
    source_node.pvs = np.zeros((len(nodes),), dtype=bool)
    source_node.pvs[source_node_id] = True
    for source_portal_name in source_node.portals:
        source_portal = portals[source_portal_name]
        target_node = nodes[node_connectivity[source_node.id][source_portal_name]]
        source_node.pvs[target_node.id] = True
        target_portals = [portals[p_name] for p_name in target_node.portals]
        if source_portal.shapely.length < 1.0:
            continue
        gen_pvs_recursive(source_node, target_node, source_portal, target_portals, source_portal, nodes, portals,
                          node_connectivity)
    return source_node_id, source_node.pvs


def gen_pvs_chunk(source_node_ids, nodes, portals, node_connectivity):
    global panic
    if panic and panic.value:
        print('Skipping {}...'.format(source_node_ids))
        return None
    else:
        print('Running {}'.format(source_node_ids))
    for source_node_id in source_node_ids:
        source_node = nodes[source_node_id]
        source_node.pvs = np.zeros((len(nodes),), dtype=bool)
        source_node.pvs[source_node_id] = True
        for source_portal_name in source_node.portals:
            source_portal = portals[source_portal_name]
            target_node = nodes[node_connectivity[source_node.id][source_portal_name]]
            source_node.pvs[target_node.id] = True
            target_portals = [portals[p_name] for p_name in target_node.portals]
            if source_portal.shapely.length < 1.0:
                continue
            gen_pvs_recursive(source_node, target_node, source_portal, target_portals, source_portal, nodes, portals,
                              node_connectivity)
        return source_node_id, source_node.pvs


def gen_pvs_recursive(source_node, current_node, source_portal, target_portals, last_portal,
                      nodes, portals, node_connectivity, last_penumbra=None):
    global panic
    if panic and panic.value:
        # print('Skipping {}...'.format(source_node_ids))
        return
    # Check all portals except the one we are looking through
    for target_portal in set(target_portals) - {last_portal}:
        if target_portal.shapely.length < 1.0 or source_portal.compare(target_portal) == 'C':
            continue
        dest_node = nodes[node_connectivity[current_node.id][target_portal.name]]

        # Add destination node to source node's pvs
        source_node.pvs[dest_node.id] = True

        penumbra = compute_anti_penumbra2(last_portal, target_portal)
        if not penumbra.shapely.is_valid:
            continue
        if last_penumbra:
            intersection = last_penumbra.shapely.intersection(penumbra.shapely)
            if isinstance(intersection, GeometryCollection):
                if intersection.is_empty:
                    continue
                intersection = next(p for p in intersection if isinstance(p, ShapelyPolygon))
            elif isinstance(intersection, MultiPolygon):
                # TODO: Need to handle this!!!!
                continue
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
        valid_destination_portals = set([portals[p_name] for p_name in p_names]) - {target_portal}
        for dest_portal in valid_destination_portals:
            # If the destination portal is collinear to the target or source portals, then we can not see the node
            # it leads to (at least not through the target node)
            if dest_portal.compare(source_portal) == 'C' or dest_portal.compare(target_portal) == 'C':
                continue

            if not penumbra.shapely.intersects(dest_portal.shapely):
                continue

            ls = dest_portal.shapely.intersection(penumbra.shapely)
            if not ls or not isinstance(ls, LineString) or ls.length < 1.:
                continue

            dest_portal = LineSegment.from_linestring(ls, name=dest_portal.name)

            dest_portals.append(dest_portal)

        if len(dest_portals):
            gen_pvs_recursive(source_node, dest_node, source_portal, dest_portals, target_portal, nodes,
                              portals, node_connectivity, penumbra)


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.shapely.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def compute_anti_penumbra2(source, target):
    line1 = LineSegment(source.p1, target.p1)
    line2 = LineSegment(source.p2, target.p2)
    # if not line1.shapely.intersects(line2.shapely):
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


# def min_partitions(idx, line, lines):
def min_partitions(args):
    idx, line, lines = args
    if len(lines) > 1000:
        samples = random.sample(lines, 1000)
    else:
        samples = lines
    results = np.array(line.compare(samples))
    partition_count = np.sum(results == 'P')
    # print('Done {}'.format(idx))
    return idx, partition_count


def update_portals_walls(portals, replacement_portals):
    portal_names = [p for p in portals]
    new_portals = []
    for p in portal_names:
        parts = p.split('_')
        new_portal = p
        current_dict = nested_get(replacement_portals, parts)
        if not current_dict:
            new_portals.append(new_portal)
            continue
        new_parts = iter_dict(current_dict)
        for part in new_parts:
            new_portals.append(new_portal + '_' + part)
    return new_portals


# def update_portals_walls_chunk(chunks, replacement_portals):
def update_portals_walls_chunk(args):
    chunks, replacement_portals = args
    updated_portals = []
    for chunk in chunks:
        node_id, node_portals = chunk
        updated_portals.append((node_id, update_portals_walls(node_portals, replacement_portals)))
    return updated_portals


# def filter_node_portal_walls_chunk(chunks, wall_names):
def filter_node_portal_walls_chunk(args):
    chunks, wall_names = args
    filtered_portals_walls = []
    for chunk in chunks:
        node_id, node_portals = chunk
        node_walls = list(set(node_portals).intersection(set(wall_names)))
        node_portals = list(set(node_portals) - set(node_walls))
        filtered_portals_walls.append((node_id, node_portals, node_walls))
    return filtered_portals_walls


# def process_portal_connections(portal_names, empty_leaves):
def process_portal_connections(args):
    portal_names, empty_leaves = args
    portal_conns = []
    for portal_name in portal_names:
        portal_conns.append((portal_name, [n.id for n in empty_leaves if portal_name in n.portals]))
    return portal_conns


def get_chunks(lst, n):
    """Split lst into n-sized chunks"""
    chunks = [lst[i:i + n] for i in range(0, len(lst), n)]
    return chunks


def get_n_chunks(lst, num_chunks):
    """Split lst into n chunks"""
    num_elements = len(lst)
    n = int(np.ceil(num_elements/num_chunks))
    chunks = [lst[i:i + n] for i in range(0, len(lst), n)]
    return chunks


def nested_get(dic, keys):
    for key in keys:
        if key not in dic:
            return None
        dic = dic[key]
    return dic


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def iter_dict(current_dict):
    vals = []
    for key in current_dict:
        if current_dict[key]:
            vals_rec = iter_dict(current_dict[key])
            for val in vals_rec:
                vals.append(key + '_' + val)
        else:
            vals.append(key)
    return vals


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def create_dirs(path):
    parts = splitall(path)
    for i in range(len(parts)):
        path_tmp = ''.join([p+'/' for p in parts[:i+1]])
        if os.path.exists(path_tmp):
            continue
        os.mkdir(path_tmp)


def handler(signal_receiver, frame):
    global panic
    panic.value = True
    print('Keyboard Interrupt Received!!!')


def init_worker(args):
    global panic
    panic = args
    signal.signal(signal.SIGINT, handler)