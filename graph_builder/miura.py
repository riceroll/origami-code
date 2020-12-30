import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.optimize

n_col = 10
n_row = 10
l = 1
w = 1
a = 30 / 180 * np.pi
target_angle = np.pi
name = 'miura0'
distort = False
kawazaki = False

def rc_2_i(i_row, i_col, n_row, n_col):
    return i_row * (n_col + 1) + i_col

def i_2_rc(i, n_row, n_col):
    i_row = i // (n_col + 1)
    i_col = i % (n_col + 1)
    return i_row, i_col

def generate_Miura(n_col=n_col, n_row=n_row, l=l, w=w, a=a):
    disp = l * np.sin(a)
    h = l * np.cos(a)
    nodes = np.zeros([(n_row + 1) * (n_col + 1), 2])
    faces = []
    edges = []
    k_edges = []
    
    i_handle = rc_2_i( int((n_row) / 2), int((n_col) / 2), n_row, n_col)
    
    is_boundary_nodes = []
    is_boundary_edges = []
    is_facet = []
    target_angles = []
    
    inner_nodes = []
    up_nodes = []
    down_nodes = []
    left_nodes = []
    right_nodes = []
    
    # assign locations
    for i_row in range(n_row + 1):
        for i_col in range(n_col + 1):
            x = i_col * w
            y = i_row * -h
            if i_row % 2 == 0:
                x += disp
                
            i_v = rc_2_i(i_row, i_col, n_row, n_col)
            
            dist = np.sqrt((x - 8)**2 + (y+4)**2)
            dir = np.array([8 - x, -4 - y])
            dir = dir / np.sqrt(np.sum(dir ** 2))
            
            # distortion
            if distort:
                if dist > 4:
                    delta = 0
                elif dist > 2:
                    delta = -(dist - 1) + 2
                else:
                    delta = dist
                displacement = dir * delta * 0.5
                x += displacement[0]
                y += displacement[1]
            
            nodes[i_v] = [x, y]
            
            if i_col != 0 and i_col != n_col and i_row != 0 and i_row != n_row:
                inner_nodes.append(i_v)
                
                get_i = lambda ro, co: int(ro * (n_col + 1) + co)      # ?
                assert(i_v == get_i(i_row, i_col))
                
                up_nodes.append(get_i(i_row-1, i_col))
                down_nodes.append(get_i(i_row+1, i_col))
                left_nodes.append(get_i(i_row, i_col-1))
                right_nodes.append(get_i(i_row, i_col+1))
            
            # # fix nodes
            if i_row == 0 or i_row == n_row or i_col == 0 or i_col == n_col:
                is_boundary_nodes.append(1)
            else:
                is_boundary_nodes.append(0)

            # if i_col == 0 or i_col == n_col:
            #     is_boundary_nodes.append(1)
            # else:
            #     is_boundary_nodes.append(0)
    
    def energy(nodes):
        nodes = nodes.reshape(-1, 2)
        x = nodes[inner_nodes]
        x_u = nodes[up_nodes]
        x_d = nodes[down_nodes]
        x_l = nodes[left_nodes]
        x_r = nodes[right_nodes]
        
        norm = lambda x: np.sqrt(np.sum(x**2, axis=1))
        
        theta_1 = np.arccos( np.sum((x_u - x)*(x_l - x), axis=1) / (norm(x_u - x) * norm(x_l - x)))
        theta_2 = np.arccos( np.sum((x_d - x)*(x_r - x), axis=1) / (norm(x_d - x) * norm(x_r - x)))
        
        return np.mean((theta_1 + theta_2 - np.pi)**2)

    nodes_prev = nodes.copy()
    if kawazaki:
        print(energy(nodes))
        res = scipy.optimize.minimize(energy, nodes, method='CG', tol=1e-4)
        nodes = res.x
        nodes = nodes.reshape(-1, 2)
        print(energy(nodes))

    # add edges
    for i_row in range(n_row):
        for i_col in range(n_col):
            i_0 = rc_2_i(i_row, i_col, n_row, n_col)    # left-top node
            i_1 = rc_2_i(i_row, i_col + 1, n_row, n_col)    # right-top node
            i_2 = rc_2_i(i_row + 1, i_col, n_row, n_col)    # left_bottom node
            i_3 = rc_2_i(i_row + 1, i_col + 1, n_row, n_col)    # right_bottom node
            
            r_handle, c_handle = i_2_rc(i_handle, n_row, n_col)
            
            # stiffness
            k = 1
            # if i_col < c_handle - 1:
            #     k = 0.5
            # if i_col >= c_handle - 1:
            #     k = 20
            
            if i_row == 0:
                edges.append([i_0, i_1, 2])     # 2: boundary
                k_edges.append(k)
            if i_col == 0:
                edges.append([i_0, i_2, 2])
                k_edges.append(k)
            
            if i_row % 2 == 0:
                edges.append([i_0, i_3, 0])     # 0: facet
                k_edges.append(k)
                faces.append([i_0, i_2, i_3])
                k_edges.append(k)
                faces.append([i_0, i_3, i_1])
                k_edges.append(k)
            else:
                edges.append([i_1, i_2, 0])
                k_edges.append(k)
                faces.append([i_0, i_2, i_1])
                k_edges.append(k)
                faces.append([i_1, i_2, i_3])
                k_edges.append(k)
            
            if i_col == n_col - 1:
                edges.append([i_1, i_3, 2])
            else:
                if i_col % 2 == 0:
                    edges.append([i_1, i_3, 1])     # 1: positive crease
                else:
                    edges.append([i_1, i_3, -1])    # -1: negative crease
            
            if i_row == n_row - 1:
                edges.append([i_2, i_3, 2])
            else:
                if (i_col % 2 == 0 and i_row % 2 == 0) or (i_row % 2 != 0 and i_col % 2 != 0):
                    edges.append([i_2, i_3, 1])
                else:
                    edges.append([i_2, i_3, -1])

    # stiffness

    theta = lambda x, y, x_0, y_0: np.arctan2((y - y_0), (x - x_0))
    x_0, y_0 = nodes[i_handle]
    for i_e, edge in enumerate(edges):
        p1 = nodes[edge[0]]
        p2 = nodes[edge[1]]
        x, y = (p1 + p2) / 2
        the = theta(x, y, x_0, y_0)
        period = 45 / 180 * np.pi
        # k = (the % period) / period * 10 + 1
        
        # k = abs((the % period) / period)
        k = the / np.pi
        #
        # k = 1 if k < 0.5 else 20
        
        if 0 < k < 0.5 or k < - 0.5:
            k = 1
        else:
            k = 20
        
        # k_edges[i_e] = k

    # edge properties
    for e in edges:
        if e[-1] == 0:
            is_facet.append(1)
            is_boundary_edges.append(0)
            target_angles.append(0)
        if e[-1] == 2:
            is_facet.append(0)
            is_boundary_edges.append(1)
            target_angles.append(0)
        if e[-1] == -1:
            is_facet.append(0)
            is_boundary_edges.append(0)
            target_angles.append(-target_angle)
        if e[-1] == 1:
            is_facet.append(0)
            is_boundary_edges.append(0)
            target_angles.append(target_angle)
    
    edges = np.array(edges, dtype=np.int)
    edges = edges[:, :2]
    
    zs = np.zeros([len(nodes), 1])
    nodes = np.hstack([nodes, zs])
    
    k_edges = np.array(k_edges, dtype=np.float).reshape(-1, 1)
    
    return nodes, edges, faces, is_facet, is_boundary_edges, target_angles, is_boundary_nodes, nodes_prev, k_edges


if __name__ == '__main__':
    nodes, edges, faces, is_facet, is_boundary_edges, target_angles, is_boundary_nodes, nodes_prev, k_edges = generate_Miura()
    
    i_handle = rc_2_i( int((n_row) / 2), int((n_col) / 2), n_row, n_col)
    
    def plot_2d(nodes):
        lines = []
        colors = []
        for i_e, e in enumerate(edges):
            v0 = nodes[e[0]][:2]
            v1 = nodes[e[1]][:2]
            vs = np.vstack([v0, v1])
            lines.append(vs)
            if is_facet[i_e]:
                c = (0.8, 0.8, 0.8)
            if target_angles[i_e] < 0:
                c = (0, 0, 1)
            if target_angles[i_e] > 0:
                c = (1, 0, 0)
            if is_boundary_edges[i_e]:
                c = (0, 0, 0)
            colors.append(c)
        fig, ax = plt.subplots()
        lc = LineCollection(lines, linewidths=1, colors=colors)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal')
        for i, node in enumerate(nodes):
            ax.text(node[0], node[1], str(i))
        ax.plot([nodes[i_handle][0]], [nodes[i_handle][1]], 'o')
        
        plt.show()
        
    plot_2d(nodes_prev)
    plot_2d(nodes)
    
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
