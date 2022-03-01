import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
import shapely
import networkx as nx
import geopandas
import cProfile as profile
# In outer section of code
import tqdm
import multiprocessing as mpp

pr = profile.Profile()
pr.disable()
import datetime
from stonesoup.types.angle import Bearing
from stonesoup.functions import pol2cart
from stonesoup.custom.graph import CustomDiGraph, graph_to_dict


from pybsp.utils import plot_nodes, sort_fovs, remove_artists
from pybsp.bsp import BSP, get_n_chunks
from pybsp.geometry import LineSegment, Point, Polygon
from pybsp.geo import load_target_lines, get_merc_limits, load_target_polygons

import sys

sys.setrecursionlimit(10000000)
seed = 10000 # np.random.randint(10000000)
np.random.seed(seed)


from ports import get_ports_and_augmented_polygons


def generate_ref_point():
    # return Point(1.5955e+06, 4.3e+06)
    # return Point(1.6019e+6, 4.305e+6)

    # return Point(-3.39e+5, 7.127e+6)
    return Point(-2e+5, 4.339e+6)
    # return Point(1.593e+6, 4.3e+6)


def inner_angle(p1, p2, p3, cw=True):
    """ Calculate inner angle

    Taken from: https://stackoverflow.com/questions/12083480/finding-internal-angles-of-polygon
    """
    # Incoming vector
    v1 = (p2.x - p1.x, p2.y - p1.y)

    # Outgoing vector
    v2 = (p3.x - p2.x, p3.y - p2.y)

    theta = np.pi
    if cw:
        theta += np.arctan2(np.cross(np.array(v1), np.array(v2)), np.dot(np.array(v1), np.array(v2)))
    else:
        theta -= np.arctan2(np.cross(np.array(v1), np.array(v2)), np.dot(np.array(v1), np.array(v2)))

    return theta


def compute_corners_single(args):
    polygon, ports = args
    corner_points = []
    adj = dict()
    # Sort points clockwise
    polygon = shapely.geometry.polygon.orient(polygon, -1.0)  # 1.0 for ccw
    x, y = polygon.exterior.xy
    x = x[:-1]
    y = y[:-1]
    points = []
    for xi, yi in zip(x, y):
        points.append(Point(xi, yi))

    triplets = [(i, (i + 1) % len(points), (i + 2) % len(points)) for i in range(len(points))]
    for i, inds in enumerate(triplets):
        p1 = points[inds[0]]
        p2 = points[inds[1]]
        p3 = points[inds[2]]

        theta = inner_angle(p1, p2, p3)

        dist = ports.geometry.distance(p2.shapely).min()

        if dist < 1e-8 or abs(theta) < np.pi:
            corner_points.append(p2)
            try:
                ind = corner_points.index(p1)
                adj[len(corner_points) - 1] = ind
            except ValueError:
                continue
            if i == len(triplets) - 1:
                try:
                    ind = corner_points.index(p3)
                    adj[ind] = len(corner_points) - 1
                except ValueError:
                    continue
    return corner_points, adj


def compute_corners(polygons, ports):
    corner_points = []
    adj = dict()  # dict representing adjacency of corner points on same line

    inputs = [(polygon, ports) for polygon in polygons]

    pool = mpp.Pool(mpp.cpu_count())
    results = imap_tqdm(pool, compute_corners_single, inputs, desc='Corners')

    for result in results:
        i = len(corner_points)
        corners = result[0]
        adj_tmp = result[1]
        corner_points += corners
        for key, value in adj_tmp.items():
            adj[key+i] = value+i

    # for polygon in tqdm.tqdm(polygons):
    #
    #     # Sort points clockwise
    #     polygon = shapely.geometry.polygon.orient(polygon, -1.0)  # 1.0 for ccw
    #     x, y = polygon.exterior.xy
    #     x = x[:-1]
    #     y = y[:-1]
    #     points = []
    #     for xi, yi in zip(x, y):
    #         points.append(Point(xi, yi))
    #
    #     triplets = [(i, (i+1)%len(points), (i+2)%len(points)) for i in range(len(points))]
    #     for i, inds in enumerate(triplets):
    #         p1 = points[inds[0]]
    #         p2 = points[inds[1]]
    #         p3 = points[inds[2]]
    #
    #         theta = inner_angle(p1, p2, p3)
    #
    #         dist = ports.geometry.distance(p2.shapely).min()
    #
    #         if dist < 1e-8 or abs(theta) < np.pi:
    #             corner_points.append(p2)
    #             try:
    #                 ind = corner_points.index(p1)
    #                 adj[len(corner_points)-1] = ind
    #             except ValueError:
    #                 continue
    #             if i == len(triplets)-1:
    #                 try:
    #                     ind = corner_points.index(p3)
    #                     adj[ind] = len(corner_points) - 1
    #                 except ValueError:
    #                     continue
    return corner_points, adj


def comp_adj_matrix(corner_points, adj, bsptree, inds=None):
    # Calculate adj matrix
    num_nodes = len(corner_points)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for key, value in adj.items():
        adj_matrix[key, value] = 1
        adj_matrix[value, key] = 1

    pool = mpp.Pool(mpp.cpu_count() - 4)
    if inds is None:
        inds = (ind1 for ind1 in range(len(corner_points)))

    inputs = tqdm.tqdm([(corner_points, ind1, bsptree) for ind1 in inds],
                       desc='Queued')
    results = imap_tqdm(pool, check_los, inputs, desc='Processed')
    for result in results:
        ind1 = result[0]
        inds = result[1]
        for ind2 in inds:
            adj_matrix[ind1, ind2] = 1
            adj_matrix[ind2, ind1] = 1

    return adj_matrix


def comp_conn_graph(adj_matrix, corner_points):
    # G = nx.from_numpy_matrix(adj_matrix)
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

    pos = dict()
    for i in range(num_nodes):
        p = corner_points[i]
        pos[i] = {'pos': (p.x, p.y)}

    nx.set_node_attributes(G, pos)
    pos = nx.get_node_attributes(G, 'pos')
    weight = dict()
    for edge in G.edges:
        n1 = edge[0]
        n2 = edge[1]
        p1 = Point(*pos[n1])
        p2 = Point(*pos[n2])
        weight[edge] = p1.get_distance(p2)

    nx.set_edge_attributes(G, name='weight', values=weight)
    return G


def calc_conn_graph(bsptree, polygons, ports):
    # Compute corners
    corner_points, adj = compute_corners(polygons, ports)

    adj_matrix = comp_adj_matrix(corner_points, adj)

    G = comp_conn_graph(adj_matrix, corner_points)

    return G, corner_points



def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results


def check_los(args):
    corner_points, ind1, bsptree = args
    adj = []
    all_inds = [i for i in range(len(corner_points))]
    p1 = corner_points[ind1]
    for ind2 in all_inds:
        if ind2 <= ind1:
            continue

        p2 = corner_points[ind2]

        # Bring p1 closer to p2
        r1, ph1 = p1.to_polar(p2)
        r1 -= 2
        x1, y1 = pol2cart(r1, ph1)
        x1, y1 = x1 + p2.x, y1 + p2.y
        p11 = Point(x1, y1)

        # Bring p2 closer to p1
        r2, ph2 = p2.to_polar(p1)
        r2 -= 2
        x2, y2 = pol2cart(r2, ph2)
        x2, y2 = x2 + p1.x, y2 + p1.y
        p22 = Point(x2, y2)

        los = bsptree.check_los(p11, p22)

        if los:
            adj.append(ind2)

    return ind1, adj


def get_nearest_node_single(args):
    i, p, pos = args
    for node_id, position in pos.items():
        dx = np.isclose(p.x, position[0])
        dy = np.isclose(p.y, position[1])
        if dx and dy:
            break
    else:
        node_id = -1
    return i, node_id


# def graph_to_dict(G):
#
#     weights = nx.get_edge_attributes(G, 'weight')
#     S = dict()
#     S['Edges'] = dict()
#     S['Edges']['EndNodes'] = []
#     S['Edges']['Weight'] = []
#     for edge in G.edges:
#         S['Edges']['EndNodes'].append([edge[0]+1, edge[1]+1])
#         S['Edges']['Weight'].append(weights[edge])
#
#     pos = nx.get_node_attributes(G, 'pos')
#     S['Nodes'] = dict()
#     S['Nodes']['Longitude'] = []
#     S['Nodes']['Latitude'] = []
#     for node in G.nodes:
#         S['Nodes']['Longitude'].append(pos[node][0])
#         S['Nodes']['Latitude'].append(pos[node][1])
#
#     return S


def main():
    SHOW_PLANES = True
    TARGET = "GLOBAL"
    heuristic = 'min'
    backup_folder = '../data/trees/{}_{}'.format(TARGET, heuristic).lower()
    val = input("Enter backup location: ")
    if val:
        backup_folder = '../data/trees/{}'.format(val)
    print(backup_folder)

    # Load lines
    # lines = load_target_lines(TARGET, 'oversimplified_merged_polygons.p')
    polygons = load_target_polygons(TARGET, 'panama/oversimplified_merged_polygons.p')

    # Compute new ports and polygons
    # ports, new_polygons = get_ports_and_augmented_polygons(polygons)
    # pickle.dump([ports, new_polygons], open('../data/ports_polygons.pickle', 'wb'))
    # ports, new_polygons = pickle.load(open('../data/ports_polygons.pickle', 'rb'))

    # Gen duplicated (offset) ports
    # geom_east = ports.translate(-40_075_016.685578)
    # ports_e = ports.copy(True)
    # ports_e.geometry = geom_east
    # geom_west = ports.translate(40_075_016.685578)
    # ports_w = ports.copy(True)
    # ports_w.geometry = geom_west
    # ports_all = ports.append([ports_e, ports_w])
    #
    # # Gen duplicated (offset) polygons
    # df = geopandas.GeoDataFrame(geometry=new_polygons, crs=ports.crs)
    # geom_east = df.translate(-40_075_016.685578)
    # df_e = df.copy(True)
    # df_e.geometry = geom_east
    # geom_west = df.translate(40_075_016.685578)
    # df_w = df.copy(True)
    # df_w.geometry = geom_west
    # df_all = df.append([df_e, df_w])
    # geoms = df_all.geometry.unary_union
    # df_all2 = geopandas.GeoDataFrame(geometry=[geoms], crs=df_all.crs)
    # df_all3 = df_all2.explode().reset_index(drop=True)
    # new_polygons_all = df_all3.geometry.to_list()
    # pickle.dump([ports_all, new_polygons_all], open('../data/ports_polygons3.pickle', 'wb'))
    ports, new_polygons = pickle.load(open('../data/ports_polygons3.pickle', 'rb'))

    # Construct Graph
    bsptree = BSP.load(backup_folder, 'Stage2', 'final')

    # Compute corner points
    # corner_points, adj = compute_corners(new_polygons, ports)
    # pickle.dump([corner_points, adj], open(f'../data/corners/corners_adj3.pickle', 'wb'))
    corner_points, adj = pickle.load(open(f'../data/corners/corners_adj3.pickle', 'rb'))

    # Compute adjacency matrix
    load = 0
    num_nodes = len(corner_points)
    all_inds = [i for i in range(num_nodes)]
    chunks = get_n_chunks(all_inds, 10)
    if load is None:
        adj_matrix = np.zeros((num_nodes, num_nodes))
    else:
        adj_matrix = pickle.load(open(f'../data/corners/adj_matrix_{load}.pickle', 'rb'))

    for i, chunk in enumerate(chunks):
        print(f'Processing chunk {i+1} of {len(chunks)}')
        if i <= load:
             continue
        adj_matrix_chunk = comp_adj_matrix(corner_points, adj, bsptree, chunk)
        adj_matrix = np.logical_or(adj_matrix, adj_matrix_chunk)
        pickle.dump(adj_matrix, open(f'../data/corners/adj_matrix_{i}.pickle', 'wb'))
        print()


    G, corner_points = calc_conn_graph(bsptree, new_polygons, ports)
    pickle.dump(G, open('../data/graphs/corner_conn_digraph_v5.pickle', 'wb'))
    # G = pickle.load(open('../data/graphs/corner_conn_digraph_v5.pickle', 'rb'))
    # G = pickle.load(open('../data/graphs/custom_digraph_v4.1.pickle', 'rb'))

    # Compute weights for each edge
    # pos = nx.get_node_attributes(G, 'pos')
    # weight = dict()
    # for edge in G.edges:
    #     n1 = edge[0]
    #     n2 = edge[1]
    #     p1 = Point(*pos[n1])
    #     p2 = Point(*pos[n2])
    #     weight[edge] = p1.get_distance(p2)
    # nx.set_edge_attributes(G, name='weight', values=weight)
    # pickle.dump(G, open('../data/graphs/corner_conn_digraph_v4.1.pickle', 'wb'))
    # Convert to custom digraph
    S = graph_to_dict(G)
    G2 = CustomDiGraph.from_dict(S)
    pickle.dump(G2, open('../data/graphs/custom_digraph_v5.pickle', 'wb'))

    # Find corresponding node id for each port
    port_points = ports['geometry'].to_list()
    pos = nx.get_node_attributes(G, 'pos')
    node_ids = -1*np.ones((len(port_points),), dtype=int)
    pool = mpp.Pool(mpp.cpu_count())
    inputs = [(i, p, pos) for i, p in enumerate(port_points)]
    results = imap_tqdm(pool, get_nearest_node_single, inputs, desc='Nearest node')
    for i, node_id in results:
        node_ids[i] = node_id
    ports['Node'] = node_ids
    pickle.dump([ports, new_polygons], open('../data/ports_polygons3.pickle', 'wb'))

    # Plot scene
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    # for polygon in new_polygons:
    #     x, y = polygon.exterior.xy
    #     ax1.plot(x, y, 'k-')
        # midPoint = line.mid_point
        # plt.text(midPoint.x + line.NormalV.x / 10, midPoint.y + line.NormalV.y / 10, line.Name)

        # midPoint = line.mid_point
        # plt.quiver(midPoint.x, midPoint.y, line.NormalV.x, line.NormalV.y, width=0.001, headwidth=0.2)
    xlim = plt.xlim()
    ylim = plt.ylim()

    xmin, xmax, ymin, ymax = get_merc_limits(TARGET)


    # import networkx as nx
    pos = nx.get_node_attributes(G, 'pos')
    nodes = nx.draw_networkx_nodes(G, pos, node_size=5, node_color='r', ax=ax1)
    nx.draw_networkx_edges(G.to_undirected(), pos, width=0.2, edge_color='g')

    annot = ax1.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)


    def update_annot(ind):
        node = ind["ind"][0]
        xy = pos[node]
        annot.xy = xy
        node_attr = {'node': node}
        node_attr.update(G.nodes[node])
        text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax1:
            cont, ind = nodes.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig1.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig1.canvas.draw_idle()

    fig1.canvas.mpl_connect('button_press_event', hover)
    plt.tight_layout()

    edges = [e for e in G.edges]
    a = ports.loc[ports['NAME'] == 'TOKYO KO']
    node_path = nx.shortest_path(G, 6327, 2431, 'weight')
    path_edges = list(zip(node_path, node_path[1:]))
    edge_path = [edges.index(edge) for edge in path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=list(path_edges), arrows=False, edge_color='b', ax=ax1)

    plt.show()
    a=2


if __name__ == '__main__':
    main()