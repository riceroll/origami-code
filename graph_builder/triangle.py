import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys

fig, ax = plt.subplots()

nodes = []
edges = []
faces = []

angle2vec = lambda degree: np.array([np.cos(degree / 180 * np.pi), np.sin(degree / 180 * np.pi)]).round(5)


class Node:
    index = 0
    members = []
    
    def __init__(self, coord=[0, 0]):
        self.id = Node.index
        Node.index += 1
        Node.members.append(self)

        self.coord = np.array(coord)
        
        self.edges = []
        
        ax.text(self.coord[0], self.coord[1], str(self.id))
        
    @staticmethod
    def has(coord):
        coords = np.array([n.coord for n in Node.members]).reshape(-1, 2)
        try:
            found = (((coords - coord) ** 2).sum(axis=1) < 1e-4)
        except:
            import pdb
            pdb.set_trace()
        if found.any():
            return found.nonzero()[0][0]
        else:
            return False

class Edge:
    index = 0
    members = []
    
    def __init__(self, pair=[None, None], target_angle=None):
        self.id = Edge.index
        Edge.index += 1
        Edge.members.append(self)
        
        self.pair = pair
        pair[0].edges.append(self)
        pair[1].edges.append(self)
        
        self.target_angle = target_angle
        
        centroid = (pair[0].coord + pair[1].coord) / 2
        # ax.text(centroid[0], centroid[1], str(round(target_angle,2)))
        
    @staticmethod
    def has(pair):
        pairs = [e.pair for e in Edge.members]
        if pair in pairs:
            return pairs.index(pair)
        elif pair[::-1] in pairs:
            return pairs.index(pair[::-1])
        else:
            return False

class Face:
    index = 0
    members = []
    
    def __init__(self, nodes=[None]*3):
        self.id = Face.index
        Face.index += 1
        Face.members.append(self)
        
        self.nodes = nodes     # anti-clock wise
        
        

class Unit:
    max_num = 30
    max_radius = 10
    unit_length = 1
    angle = 30 / 180 * np.pi
    
    num = 0
    members = []
    to_generate = []
    
    def __init__(self, coord=(0, 0), inverted=False):
        self.id = Unit.num
        Unit.num += 1
        Unit.members.append(self)
        Unit.to_generate.append(self)
        
        self.coord = np.array(coord)    # center coord
        
        self.inverted = inverted       # inverted triangle has bottom corner
        self.neighbors = [None] * 3     # 0: up/down triangle, 1-2: go anti-clockwise
        self.nodes = [None] * 6  # 0: up/down corner node, 1-2: anti-clock, 3: down/up edge node, 4-5: anti-clock
        self.edges = [None] * 9   # 0: outer triangle, anti-clock next to #0 node, 1-5: go anti-clock
                            # 6: inner triangle, anti-clock next to #3 node, 7-8: go anti-clock
        self.faces = [None] * 4   # 0: face containing node 0, 1-2: go anti-clockwise, 3: center face
        
    @staticmethod
    def has(coord):
        coords = np.array([u.coord for u in Unit.members]).reshape(-1, 2)
        found = (((coords - coord) ** 2).sum(axis=1) < 1e-4)
        if found.any():
            return found.nonzero()[0][0]
        else:
            return False
        
    def generate_pattern(self):

        dist = Unit.unit_length / np.sqrt(3)
        coords = [None] * 6
        coords[0] = self.coord + dist * angle2vec(90) if not self.inverted else \
                    self.coord - dist * angle2vec(90)
        coords[1] = self.coord + dist * angle2vec(-90-60) if not self.inverted else \
                    self.coord - dist * angle2vec(-90-60)
        coords[2] = self.coord + dist * angle2vec(-90+60) if not self.inverted else \
                    self.coord - dist * angle2vec(-90+60)
        mid_dist = np.tan(Unit.angle) * dist / 2
        coords[3] = (coords[1] + coords[2]) / 2 + np.array([mid_dist, 0])
        coords[4] = (coords[0] + coords[2]) / 2 + mid_dist * angle2vec(90+30)
        coords[5] = (coords[0] + coords[1]) / 2 + mid_dist * angle2vec(-90-30)
        for i, coord in enumerate(coords):
            has = Node.has(coord)
            if has is False:
                self.nodes[i] = Node(coord)
            else:
                self.nodes[i] = Node.members[has]
                
        pairs = [None] * 9
        pairs[0] = [0, 5, np.pi]
        pairs[1] = [5, 1, -60 / 180 * np.pi]
        pairs[2] = [1, 3, np.pi]
        pairs[3] = [3, 2, -60 / 180 * np.pi]
        pairs[4] = [2, 4, np.pi]
        pairs[5] = [4, 0, -60 / 180 * np.pi]
        pairs[6] = [4, 5, -90 / 180 * np.pi]
        pairs[7] = [5, 3, -90 / 180 * np.pi]
        pairs[8] = [3, 4, -90 / 180 * np.pi]
        if self.inverted:
            pairs[0][2], pairs[1][2] = pairs[1][2], pairs[0][2]
            pairs[2][2], pairs[3][2] = pairs[3][2], pairs[2][2]
            pairs[4][2], pairs[5][2] = pairs[5][2], pairs[4][2]
        
        for i, pair in enumerate(pairs):
            has = Edge.has([self.nodes[pair[0]], self.nodes[pair[1]]])
            if has is False:
                self.edges[i] = Edge([self.nodes[pairs[i][0]], self.nodes[pairs[i][1]]], pairs[i][2])
            else:
                self.edges[i] = Edge.members[has]
                
        faces = [None] * 4
        faces[0] = [0, 5, 4]
        faces[1] = [5, 1, 3]
        faces[2] = [2, 4, 3]
        faces[3] = [4, 5, 3]
        for i, face in enumerate(faces):
            self.faces[i] = Face([self.nodes[faces[i][0]], self.nodes[faces[i][1]], self.nodes[faces[i][2]]])
        
        for i in range(len(self.neighbors)):
            # if Unit.num >= Unit.max_num:
            #     continue

            if self.neighbors[i]:
                continue
            
            if i == 0:  # up / down neighbor
                displacement = np.array([0, -1]) * dist
                if self.inverted:
                    displacement *= -1
                coord_neighbor = self.coord + displacement
                
            if i == 1:
                displacement = np.array([np.cos(30 / 180 * np.pi), np.sin(30 / 180 * np.pi)]) * dist
                if self.inverted:
                    displacement *= -1
                coord_neighbor = self.coord + displacement
                
            if i == 2:
                displacement = np.array([-np.cos(30 / 180 * np.pi), np.sin(30 / 180 * np.pi)]) * dist
                if self.inverted:
                    displacement *= -1
                coord_neighbor = self.coord + displacement
                
            if (coord_neighbor ** 2).sum() > Unit.max_radius ** 2:
                continue
                
            has = Unit.has(coord_neighbor)
            if has is False:
                self.neighbors[i] = Unit(coord=coord_neighbor, inverted=not self.inverted)
            else:
                self.neighbors[i] = Unit.members[has]
                
if __name__ == '__main__':
    Unit.max_num = 50
    Unit.max_radius = 3
    Unit.max_radius = float(sys.argv[1])

    Unit.unit_length = 1
    Unit.angle = 30 / 180 * np.pi
    Unit.angle = float(sys.argv[2]) / 180 * np.pi

    unit = Unit()
    while Unit.to_generate:
        u = Unit.to_generate[0]
        Unit.to_generate = Unit.to_generate[1:]
        u.generate_pattern()
        
    nodes = np.array([n.coord for n in Node.members])
    edges = np.array([[e.pair[0].id, e.pair[1].id] for e in Edge.members], dtype=np.int)
    faces = np.array([[f.nodes[0].id, f.nodes[1].id, f.nodes[2].id] for f in Face.members])
    
    # ===== info =====
    is_facet = np.zeros([len(edges), 1])
    is_boundary_edges = np.zeros([len(edges), 1])
    is_boundary_nodes = np.zeros([len(nodes), 1])
    k_edges = np.ones([len(edges), 1])
    target_angles = np.zeros([len(edges), 1])
    i_handle = 0
    
    for i, n in enumerate(Node.members):
        if len(n.edges) < 6:
            is_boundary_nodes[i] = 1

    for i, e in enumerate(Edge.members):
        num_adj_faces = 0
        for f in Face.members:
            if e.pair[0] in f.nodes and e.pair[1] in f.nodes:
                num_adj_faces += 1
                if num_adj_faces == 2:
                    break
        if num_adj_faces == 1:
            is_boundary_edges[i] = 1
            
        target_angles[i] = e.target_angle
    
    nodes_out = np.zeros([len(nodes), 3])
    nodes_out[:, :2] += nodes

    name = 'triangle'
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
    
    segs = [[nodes[e[0]], nodes[e[1]]] for e in edges]
    colors = [plt.get_cmap('rainbow')((e.target_angle + np.pi)/np.pi / 2 ) for e in Edge.members]
    
    line_segments = LineCollection(segs, colors=colors)
    ax.add_collection(line_segments)
    m = max( abs(max(nodes[:, 0])), abs(max(nodes[:, 1])), abs(min(nodes[:, 0])), abs(min(nodes[:, 1])))
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect('equal')
    plt.show()
