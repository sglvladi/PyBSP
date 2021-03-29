from geometry import LineSegment, Point
from utils import extrapolate_line, test_between, to_range2, merge_lines
import random
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from stonesoup.functions import pol2cart

class BinaryTree:
    """Binary tree class"""
    def __init__(self, data=[], parent=None):
        """Constructor, declares variables for left and right sub-tree and data for the current node"""
        self.left = None
        self.right = None
        self.data = data
        self.parent = parent

    def print(self):
        """Prints the all tree nodes 'Name' attribute in a binary tree format (needs to be improved)"""
        queue = [self]
        print_str = ''

        while len(queue) > 0:
            tree = queue.pop(0)

            str_list = []
            for line in tree.data:
                str_list.append(line.Name)
            print_str += str(str_list)
            if tree.left is not None:
                str_list = []
                for line in tree.left.data:
                    str_list.append(line.Name)
                print_str += (' Left: ' + str(str_list))
                queue.append(tree.left)
            else:
                print_str += (' Left: ' + ' / ')

            if tree.right is not None:
                str_list = []
                for line in tree.right.data:
                    str_list.append(line.Name)
                print_str += (' Right: ' + str(str_list))
                queue.append(tree.right)
            else:
                print_str += (' Right: ' + ' / ')

            print_str += '\n'
        return print_str

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_root(self):
        return self.parent is None

class BSP2:
    def __init__(self, lines, heuristic='even'):
        """Constructor, initializes binary tree"""
        self.tree = nx.DiGraph()
        self.tree.add_node(0, data=lines)

        self.generate_tree(0, heuristic)

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
                print("{}|{} out of {}".format(idx1, idx2, len(lines)))
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
                print("{}|{} out of {}".format(idx1, idx2, len(lines)))
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

    def generate_tree(self, node, heuristic='even'):
        """
        Generates the binary space partition tree recursively using the specified heuristic at each sub-tree
        :param tree: BinaryTree, value should be self.tree on the first call, this argument exists so we can traverse the tree recursively
        :param UseHeuristic: string, either 'even' for balanced tree or 'min' for least number of nodes
        :return: nothing
        """
        best_idx = 0
        print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(self.tree.nodes[node]['data'])
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(self.tree.nodes[node]['data'])
        elif heuristic == 'random':
            best_idx = random.randrange(len(self.tree.nodes[node]['data']) - 1)

        print('....')
        data = []
        data_left = []
        data_right = []
        line = self.tree.nodes[node]['data'].pop(best_idx)
        data.append(line)

        for i, line2 in enumerate(self.tree.nodes[node]['data']):
            print('{}/{}'.format(i, len(self.tree.nodes[node]['data'])))
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

        self.tree.nodes[node]['data'] = data
        if len(data_left) > 0:
            left_node = self.tree.number_of_nodes()
            self.tree.add_node(left_node, data=data_left)
            self.tree.add_edge(node, left_node, label='F')
            if len(data_left) > 1:
                self.generate_tree(left_node, heuristic)

        if len(data_right) > 0:
            right_node = self.tree.number_of_nodes()
            self.tree.add_node(right_node, data=data_right)
            self.tree.add_edge(node, right_node, label='B')
            if len(data_right) > 1:
                self.generate_tree(right_node, heuristic)

    def draw_nx(self, ax=None, pos=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.tree
        if not pos:
            pos = graphviz_layout(g, prog="dot")
        labels = {n: [l.Name for l in g.nodes[n]['data']] for n in g.nodes}
        nx.draw(g, pos, ax=ax)
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, labels=labels)
        edge_labels = {e: g.edges[e]['label'] for e in g.edges}
        nx.draw_networkx_edge_labels(g, pos, edge_labels)


class BSP:
    """Binary Space Partition class, optimally generates BSP tree from a list of line segments by using a heuristic"""
    def __init__(self):
        """Constructor, initializes binary tree"""
        self.tree = BinaryTree()

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
                print("{}|{} out of {}".format(idx1, idx2, len(lines)))
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
                print("{}|{} out of {}".format(idx1, idx2, len(lines)))
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

    def generate_tree(self, tree, heuristic='even'):
        """
        Generates the binary space partition tree recursively using the specified heuristic at each sub-tree
        :param tree: BinaryTree, value should be self.tree on the first call, this argument exists so we can traverse the tree recursively
        :param UseHeuristic: string, either 'even' for balanced tree or 'min' for least number of nodes
        :return: nothing
        """
        best_idx = 0
        print('heuristic')
        if heuristic == 'min':
            best_idx = self.heuristic_min_partition(tree.data)
        elif heuristic == 'even':
            best_idx = self.heuristic_even_partition(tree.data)
        elif heuristic == 'random':
            best_idx = random.randrange(len(tree.data)-1)

        print('....')
        data = []
        data_left = []
        data_right = []
        line = tree.data.pop(best_idx)
        data.append(line)

        for i, line2 in enumerate(tree.data):
            print('{}/{}'.format(i, len(tree.data)))
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
        if len(data_left) > 0:
            tree.left = BinaryTree(parent=tree)
            tree.left.data = data_left
            if len(data_left) > 1:
                self.generate_tree(tree.left, heuristic)

        if len(data_right) > 0:
            tree.right = BinaryTree(parent=tree)
            tree.right.data = data_right
            if len(data_right) > 1:
                self.generate_tree(tree.right, heuristic)

    def count_nodes(self, tree):
        """returns the number of nodes in the entire tree by traversing the tree"""
        count = len(tree.data)
        if tree.left is not None:
            count += self.count_nodes(tree.left)
        if tree.right is not None:
            count += self.count_nodes(tree.right)
        return count

    def checkLoS(self, points):
        """Determine line of sight between all points in the list by constructing a line segment for the two points
        in question and comparing it for intersection with line segments in the BSP tree
        :param points: a list of Point objects
        :return: a list of lists, n by n, an entry at [i][j] tells wether point i and point j have line of sight with each other
        """
        LoS = []
        for point in points:
            LoS.append(['X'] * len(points))

        for FromIndex, FromPoint in enumerate(points):
            for ToIndex, ToPoint in enumerate(points):
                # if LoS is not determined
                if (FromIndex != ToIndex) and (LoS[FromIndex][ToIndex] == 'X'):
                    # Assume there is LoS
                    LoS[FromIndex][ToIndex] = 'T'
                    LoS[ToIndex][FromIndex] = 'T'

                    SightSegment = LineSegment(
                        points[FromIndex], points[ToIndex])

                    # Point to root node
                    stack = [self.tree]
                    IsIntersection = False
                    # NumOfIntersections = 0
                    NumOfTraversals = 0
                    while len(stack) != 0 and IsIntersection == False:
                        TreePointer = stack.pop()
                        NumOfTraversals += 1

                        compareLoS = TreePointer.data[0].compare(SightSegment)
                        if compareLoS == 'P':
                            if SightSegment.split(TreePointer.data[0]) is not None:
                                LoS[FromIndex][ToIndex] = 'F'
                                LoS[ToIndex][FromIndex] = 'F'
                            else:
                                if TreePointer.left is not None:
                                    stack.append(TreePointer.left)
                                if TreePointer.right is not None:
                                    stack.append(TreePointer.right)

                        elif compareLoS == 'F':
                            if TreePointer.left is not None:
                                stack.append(TreePointer.left)

                        elif compareLoS == 'B':
                            if TreePointer.right is not None:
                                stack.append(TreePointer.right)

                    distance = points[FromIndex].get_distance(points[ToIndex])
                    if IsIntersection:
                        print(('Distance: %0.1f' % distance) +
                              ', # of traversals(F): ' + str(NumOfTraversals))
                    else:
                        print(('Distance: %0.1f' % distance) +
                              ', # of traversals(T): ' + str(NumOfTraversals))

        return LoS

    def to_graph(self):
        g = nx.Graph()
        return self._traverse_tree_nx(self.tree, g)

    def _traverse_tree_nx(self, tree, g, parent=None):
        child = g.number_of_nodes() + 1
        data = [l.Name for l in tree.data]
        g.add_node(child, data=data)
        if parent:
            g.add_edge(parent, child)
        if tree.left:
            g = self._traverse_tree_nx(tree.left, g, child)
        if tree.right:
            g = self._traverse_tree_nx(tree.right, g, child)
        return g

    def draw_nx(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        g = self.to_graph()
        pos = graphviz_layout(g, prog="dot")
        labels = {n: g.nodes[n]['data'] for n in g.nodes}
        nx.draw(g, pos, ax=ax)
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, labels=labels)

    def find_leaf(self, p):
        tree = self.tree
        while True:
            line = tree.data[0]
            if p.compare(line) > 0:
                if not tree.left:
                    return tree
                tree = tree.left
            elif p.compare(line) < 0:
                if not tree.right:
                    return tree
                tree = tree.right

    def render(self, p):
        t = self.find_leaf(p)

        r_lines = self._render_child(t, p)
        parent = t.parent

        p_lines = self._render_parent(parent, t, p)

        return r_lines + p_lines

    def _render_parent(self, parent, child, p):
        a = parent.data[0].Name
        r_lines = parent.data

        if parent.left == child:
            if parent.right:
                r_lines = r_lines + self._render_child(parent.right, p)
        else:
            if parent.left:
                r_lines = r_lines + self._render_child(parent.left, p)

        p_lines = []
        if not parent.is_root():
            p_lines = self._render_parent(parent.parent, parent, p)

        r_lines = r_lines + p_lines
        return r_lines

    def _render_child(self, child, p):
        a = child.data[0].Name
        r_lines = []
        if child.is_leaf():
            r_lines = child.data
        else:
            r_lines_left = []
            r_lines_right = []

            r_lines = child.data

            if child.left and child.right:
                min_left = np.inf
                min_right = np.inf
                for line in child.left.data:
                    dist = p.get_distance(line.getMidPoint())
                    if dist < min_left:
                        min_left = dist
                for line in child.right.data:
                    dist = p.get_distance(line.getMidPoint())
                    if dist < min_right:
                        min_right = dist
                r_lines_left = self._render_child(child.left, p)
                r_lines_right = self._render_child(child.right, p)

                if min_left<=min_right:
                    r_lines = r_lines + r_lines_left + r_lines_right
                else:
                    r_lines = r_lines + r_lines_right + r_lines_left
            else:
                if child.left:
                    r_lines_left = self._render_child(child.left, p)
                if child.right:
                    r_lines_right = self._render_child(child.right, p)

                r_lines =  r_lines + r_lines_left + r_lines_right
        return r_lines

    def render2(self, ref_point):
        # Find the leaf node
        leaf = self.find_leaf(ref_point)

        r_lines = self._render_child2(leaf, ref_point)
        parent = leaf.parent

        p_lines = self._render_parent2(parent, leaf, ref_point)

        l = r_lines + p_lines

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
                    line = LineSegment(c1, c2, line.Normal, line.Name)
                    vis_lines.append(line)

        merged = merge_lines(vis_lines)
        return merged

    def _render_parent2(self, parent, child, p):
        a = parent.data[0].Name
        r_lines = parent.data

        if parent.left == child:
            if parent.right:
                if p.compare(parent.data[0]) > 0:
                    r_lines = r_lines + self._render_child2(parent.right, p)
                else:
                    r_lines = self._render_child2(parent.right, p) + r_lines
        else:
            if parent.left:
                if p.compare(parent.data[0]) > 0:
                    r_lines = self._render_child2(parent.left, p) + r_lines
                else:
                    r_lines = r_lines + self._render_child2(parent.left, p)

        p_lines = []
        if not parent.is_root():
            p_lines = self._render_parent2(parent.parent, parent, p)

        r_lines = r_lines + p_lines
        return r_lines

    def _render_child2(self, child, p):
        a = child.data[0].Name
        r_lines = child.data
        if child.is_leaf():
            return r_lines
        else:
            r_lines_left = []
            r_lines_right = []

            r_lines = child.data

            if child.left:
                r_lines_left = self._render_child2(child.left, p)
            if child.right:
                r_lines_right = self._render_child2(child.right, p)

            if p.compare(child.data[0]) > 0:
                # if point is in front of dividing plane
                r_lines = r_lines_left + r_lines + r_lines_right
            else:
                r_lines = r_lines_right + r_lines + r_lines_left
        return r_lines

    # def get_area(self, tree, xlim=(0, 800), ylim=(0, 800)):
    #
    #     lines = []
    #     line = tree.data[0]
    #     x, y = extrapolate_line(line, xlim, ylim)
    #     c_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.Normal, line.Name)
    #     lines.append(c_l)
    #     root = tree.parent
    #     while not root.is_root():
    #         line = root.data[0]
    #         x, y = extrapolate_line(line, xlim, ylim)
    #         p_l = LineSegment(Point(x[0], y[0]), Point(x[-1], y[-1]), line.Normal, line.Name)
    #
    #         for c_l in
    #
    #         root = root.parent
