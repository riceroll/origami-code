import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys


if __name__ == '__main__':
    nodes = np.array([
        [0, 0],
        [1, 1],
        [1, -1],
        [2, 0]
    ])
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [2, 3],
        [1, 3]
    ], dtype=np.int)
    
    faces = np.array([
        [0, 2, 1],
        [2, 3, 1]
    ], dtype=np.int)
    
    # ===== info =====
    is_facet = np.zeros([len(edges), 1])
    is_boundary_edges = np.zeros([len(edges), 1])
    is_boundary_nodes = np.ones([len(nodes), 1])
    k_edges = np.ones([len(edges), 1])
    target_angles = np.zeros([len(edges), 1])
    i_handle = 0
    
    is_boundary_edges[0] = 1
    is_boundary_edges[2] = 1
    is_boundary_edges[3] = 1
    is_boundary_edges[4] = 1
    
    target_angles += 1
    target_angles *= np.pi
    
    nodes_out = np.zeros([len(nodes), 3])
    nodes_out[:, :2] += nodes

    name = 'fold'
    np.savez('data/'+name,
             nodes=nodes_out,
             edges=edges,
             faces=faces,
             is_facet=is_facet.reshape(-1),
             is_boundary_edges=is_boundary_edges.reshape(-1),
             is_boundary_nodes=is_boundary_nodes.reshape(-1),
             k_edges=k_edges.reshape(-1),
             target_angles=target_angles.reshape(-1),
             i_handle=i_handle)
    
    # ================
    
    # ===== filter =====
    # x_lim = 2
    # y_lim = 2
    # nodes_unfiltered = (nodes[:, 0] < x_lim) * (nodes[:, 0] > - x_lim) * (nodes[:, 1] < y_lim) * (nodes[:, 1] > -y_lim)
    
    # nodes = nodes[nodes_unfiltered]
    # ids = np.arange(len(nodes_unfiltered))[nodes_unfiltered].tolist()
    #
    # edges_new = []
    # for e in edges:
    #     if (not (e[0] in ids)) or (not (e[1] in ids)):
    #         continue
    #     else:
    #         edges_new.append([ids.index(e[0]), ids.index(e[1])])
    # edges = edges_new
    #
    # faces_new = []
    # for f in faces:
    #     if not (f[0] in ids and f[1] in ids and f[2] in ids):
    #         continue
    #     else:
    #         faces_new.append([ids.index(f[0]), ids.index(f[1]), ids.index(f[2])])
    # faces = faces_new
    # ==================
    