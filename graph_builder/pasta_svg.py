import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.optimize

name = 'pasta_modified'

n_quads = 15
# look from +z, length of the strip is along x axis, width of quad is along x-axis
width_quad = 1
length_quad = 2
theta = 30 / 180 * np.pi    # angle vertical edges are tilted anti-clock-wise by

width_quad = 1
length_quad = 5
theta = 0 / 180 * np.pi    # angle vertical edges are tilted anti-clock-wise by
DISTORTED = True

nodes = []
edges = []
faces = []

x0 = 0
y0 = 0
nodes.append(np.array([x0, y0]))    # first left_bottom
nodes.append(np.array([x0 - np.sin(theta) * length_quad, np.cos(theta) * length_quad]))     # first left_top

for i_quad in range(n_quads):
    n_0 = nodes[-2]     # left_bottom
    n_1 = nodes[-1]     # left_top
    n_2 = (n_0 + n_1) / 2 + np.array([width_quad/2, 0])     # middle
    n_3 = n_0 + np.array([width_quad, 0])   # right_bottom
    n_4 = n_1 + np.array([width_quad, 0])   # right_top
    nodes.append(n_2)
    nodes.append(n_3)
    nodes.append(n_4)
    
    ids = np.arange(len(nodes)-5, len(nodes))   # ids of n_0 ~ n_5
    
    if i_quad == 0:
        edges.append([ids[0], ids[1], 2])    # 0: facet, 2: boundary, 1: positive crease, 2: negative crease
    else:
        edges.append([ids[0], ids[1], 1])
    
    if i_quad == n_quads - 1:
        edges.append([ids[3], ids[4], 2])
    else:
        edges.append([ids[3], ids[4], 1])

    edges.append([ids[0], ids[2], 0])
    edges.append([ids[1], ids[2], 0])
    edges.append([ids[3], ids[2], 0])
    edges.append([ids[4], ids[2], 0])
    edges.append([ids[0], ids[3], 2])
    edges.append([ids[1], ids[4], 2])
    
    faces.append([ids[1], ids[0], ids[2]])
    faces.append([ids[0], ids[3], ids[2]])
    faces.append([ids[3], ids[4], ids[2]])
    faces.append([ids[4], ids[1], ids[2]])
    
k_edges = []
is_facet = []
is_boundary_edges = []
is_boundary_nodes = []
target_angles = []
i_handle = len(nodes) // 2

for e in edges:
    k_edges.append(1)
    if e[-1] == 0:
        is_facet.append(1)
    else:
        is_facet.append(0)
    if e[-1] == 2:
        is_boundary_edges.append(1)
    else:
        is_boundary_edges.append(0)
    if e[-1] == 1:
        target_angles.append(np.pi)
    else:
        target_angles.append(0)

nodes = np.array(nodes)
nodes_out = np.zeros([len(nodes), 3])
nodes_out[:, :2] += nodes
nodes = nodes_out
edges = np.array(edges)[:, :2]
faces = np.array(faces)

k_edges = np.array(k_edges)
is_facet = np.array(is_facet)
is_boundary_edges = np.array(is_boundary_edges)
is_boundary_nodes = np.zeros(len(nodes))
target_angles = np.array(target_angles)


if __name__ == '__main__':

    if DISTORTED:
        alpha = 85 / 180 * np.pi
        l = width_quad * n_quads
        r = l / alpha
        for i, node in enumerate(nodes):
            a = node[0] / r
            x = np.cos(a) * (r + node[1])
            y = np.sin(a) * (r + node[1])
            
            nodes[i][0] = x
            nodes[i][1] = y

    np.savez('data/'+name,
             nodes=nodes,
             edges=edges,
             faces=faces,
             k_edges=k_edges,
             is_facet=is_facet,
             is_boundary_edges=is_boundary_edges,
             is_boundary_nodes=is_boundary_nodes,
             target_angles=target_angles,
             i_handle=i_handle)
