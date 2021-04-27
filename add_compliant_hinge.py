import sys
import pdb
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Parameter:
    def __init__(self):
        self.hinge_width = 0.2
        self.EA_facet = 100
        self.EA_hinge = 10
        self.k_hinge = 1
        self.k_facet = 10

class Focus:
    focus = None
    type = None
    
    def __init__(self):
        pass
    
    @staticmethod
    def at(obj):
        Focus.focus = obj
        Focus.type = type(obj)

def reindex(cls):
    old_all = cls.all[:]
    cls.all = []
    for item in old_all:
        item.id = len(cls.all)
        cls.all.append(item)

class Node:
    all = []
    
    def __init__(self, p, is_boundary):
        self.halfedge = None
        self.id = len(Node.all)
        Node.all.append(self)
        
        self.p = p      # np.array [2]
        self.is_boundary = is_boundary

class Edge:
    all = []
    
    def __init__(self, k, is_facet, is_boundary, is_stretchable, target_angle, is_foldable, is_offset=False):
        self.halfedge = None
        self.id = len(Edge.all)
        Edge.all.append(self)

        self.k = k
        self.is_facet = is_facet
        self.is_boundary = is_boundary
        self.is_stretchable = is_stretchable
        self.is_foldable = is_foldable
        if self.is_foldable is None:
            self.is_foldable = False
            if not is_facet and not is_boundary:
                self.is_foldable = True
        self.target_angle = target_angle
        self.is_offset = is_offset  # if the edge is created with offsetting
    
    def remove(self):
        Edge.all.remove(self)
    
    def collapse(self):
        if self.is_boundary:
            return False
        h = self.halfedge
        h_twin = h.twin
        n_0 = h.node
        n_1 = h_twin.node
        h_0 = h.next
        h_1 = h.prev
        h_twin_0 = h_twin.next
        h_twin_1 = h_twin.prev
        f = h.face
        f_twin = h_twin.face
        
        n_0.halfedge = h_twin_0
        n_1.halfedge = h_0
        self.remove()
        h.remove()
        h_twin.remove()
        h_twin_1.next = h_0
        h_0.prev = h_twin_1
        h_1.next = h_twin_0
        h_twin_0.prev = h_1
        f.remove()
        f_twin.remove()
        f_new = Face()
        f_new.halfedge = h_0
        h_0.face = h_1.face = h_twin_0.face = h_twin_1.face = f_new
        f_new.update()

        Focus.at(f_new)
        
    def create_compliant_hinge(self):
        h_bottom = self.halfedge.after_offset.twin
        h_top = self.halfedge.twin.after_offset.twin
        
        n_lb = h_bottom.node
        n_rb = h_bottom.twin.node
        n_lt = h_top.twin.node
        n_rt = h_top.node
        
        # p_center = (n_lb.p + n_rb.p + n_lt.p + n_rt.p) / 4
        # n_center = Node(p_center, is_boundary=False)
        # e_left = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_left = Halfedge()
        # h_left_twin = Halfedge()
        # e_right = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_right = Halfedge()
        # h_right_twin = Halfedge()
        # e_lt = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_lt_out = Halfedge()
        # h_lt_in = Halfedge()
        # e_rt = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_rt_out = Halfedge()
        # h_rt_in = Halfedge()
        # e_lb = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_lb_out = Halfedge()
        # h_lb_in = Halfedge()
        # e_rb = Edge(h_top.edge.k, is_facet=False, is_boundary=True, target_angle=0)
        # h_rb_out = Halfedge()
        # h_rb_in = Halfedge()
        # f_top = Face()
        # f_bottom = Face()
        # f_left = Face()
        # f_right = Face()

        p_center = (n_lb.p + n_rb.p + n_lt.p + n_rt.p) / 4
        n_center = Node(p_center, is_boundary=False)
        p_left = (n_lt.p + n_lb.p) / 2
        n_left = Node(p_left, is_boundary=True)
        p_right = (n_rt.p + n_rb.p) / 2
        n_right = Node(p_right, is_boundary=True)
        f_top = Face()
        f_lt = Face()
        f_lb = Face()
        f_bottom = Face()
        f_rb = Face()
        f_rt = Face()
        
        e_tl = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_tl_out = Halfedge()
        h_tl_in = Halfedge()
        e_left = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_left_out = Halfedge()
        h_left_in = Halfedge()
        e_bl = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_bl_out = Halfedge()
        h_bl_in = Halfedge()
        e_br = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_br_out = Halfedge()
        h_br_in = Halfedge()
        e_right = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_right_out = Halfedge()
        h_right_in = Halfedge()
        e_tr = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=True, target_angle=0)
        h_tr_out = Halfedge()
        h_tr_in = Halfedge()
        
        e_lt = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=False, target_angle=0)
        h_lt_up = Halfedge()
        h_lt_down = Halfedge()
        e_lb = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=False, target_angle=0)
        h_lb_up = Halfedge()
        h_lb_down = Halfedge()
        e_rb = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=False, target_angle=0)
        h_rb_up = Halfedge()
        h_rb_down = Halfedge()
        e_rt = Edge(h_top.edge.k, is_facet=False, is_boundary=True, is_stretchable=True, is_foldable=False, target_angle=0)
        h_rt_up = Halfedge()
        h_rt_down = Halfedge()
        
        # # halfedge assignment       <editor-fold>
        # h_top.face, f_top.halfedge = f_top, h_top
        # h_top.next = h_lt_in
        # h_top.prev = h_rt_out
        # h_bottom.face, f_bottom.halfedge = f_bottom, h_bottom
        # h_bottom.next = h_rb_in
        # h_bottom.prev = h_lb_out
        #
        # h_left.node = n_lt
        # h_left.edge, e_left.halfedge = e_left, h_left
        # h_left.face, f_left.halfedge = f_left, h_left
        # h_left.twin, h_left_twin.twin = h_left_twin, h_left
        # h_left.next = h_lb_in
        # h_left.prev = h_lt_out
        #
        # h_right.node = n_rb
        # h_right.edge, e_right.halfedge = e_right, h_right
        # h_right.face, f_right.halfedge = f_right, h_right
        # h_right.twin, h_right_twin.twin = h_right_twin, h_right
        # h_right.next = h_rt_in
        # h_right.prev = h_rb_out
        #
        # h_left_twin.node = n_lb
        # h_left_twin.edge = e_left
        # h_left_twin.face = None
        # h_left_twin.next = None  # TODO
        # h_left_twin.prev = None  # TODO
        #
        # h_right_twin.node = n_rt
        # h_right_twin.edge = e_right
        # h_right_twin.face = None
        # h_right_twin.next = None  # TODO
        # h_right_twin.prev = None  # TODO
        #
        # h_lt_out.node, n_center.halfedge = n_center, h_lt_out
        # h_lt_out.edge, e_lt.halfedge = e_lt, h_lt_out
        # h_lt_out.face = f_left
        # h_lt_out.twin, h_lt_in.twin = h_lt_in, h_lt_out
        # h_lt_out.next = h_left
        # h_lt_out.prev = h_lb_in
        #
        # h_lb_out.node = n_center
        # h_lb_out.edge, e_lb.halfedge = e_lb, h_lb_out
        # h_lb_out.face = f_bottom
        # h_lb_out.twin, h_lb_in.twin = h_lb_in, h_lb_out
        # h_lb_out.next = h_bottom
        # h_lb_out.prev = h_rb_in
        #
        # h_rb_out.node = n_center
        # h_rb_out.edge, e_rb.halfedge = e_rb, h_rb_out
        # h_rb_out.face = f_right
        # h_rb_out.twin, h_rb_in.twin = h_rb_in, h_rb_out
        # h_rb_out.next = h_right
        # h_rb_out.prev = h_rt_in
        #
        # h_rt_out.node = n_center
        # h_rt_out.edge, e_rt.halfedge = e_rt, h_rt_out
        # h_rt_out.face = f_top
        # h_rt_out.twin, h_rt_in.twin = h_rt_in, h_rt_out
        # h_rt_out.next = h_top
        # h_rt_out.prev = h_lt_in
        #
        # h_lt_in.node = n_lt
        # h_lt_in.edge = e_lt
        # h_lt_in.face = f_top
        # h_lt_in.next = h_rt_out
        # h_lt_in.prev = h_top
        #
        # h_lb_in.node = n_lb
        # h_lb_in.edge = e_lb
        # h_lb_in.face = f_left
        # h_lb_in.next = h_lt_out
        # h_lb_in.prev = h_left
        #
        # h_rb_in.node = n_rb
        # h_rb_in.edge = e_rb
        # h_rb_in.face = f_bottom
        # h_rb_in.next = h_lb_out
        # h_rb_in.prev = h_bottom
        #
        # h_rt_in.node = n_rt
        # h_rt_in.edge = e_rt
        # h_rt_in.face = f_right
        # h_rt_in.next = h_rb_out
        # h_rt_in.prev = h_right
        #
        # f_top.update()
        # f_left.update()
        # f_bottom.update()
        # f_right.update()
        # # </editor-fold>
        
        # halfedge assignment <editor-fold>
        h_top.face, f_top.halfedge = f_top, h_top
        h_top.next = h_tl_in
        h_top.prev = h_tr_out
        h_bottom.face, f_bottom.halfedge = f_bottom, h_bottom
        h_bottom.next = h_br_in
        h_bottom.prev = h_bl_out
        
        h_lt_down.node = n_lt
        h_lt_down.face, f_lt.halfedge = f_lt, h_lt_down
        h_lt_down.edge, e_lt.halfedge = e_lt, h_lt_down
        h_lt_down.twin = h_lt_up
        h_lt_down.next = h_left_in
        h_lt_down.prev = h_tl_out
        h_lt_up.node, n_left.halfedge = n_left, h_lt_up
        h_lt_up.face = None
        h_lt_up.edge = e_lt
        h_lt_up.twin = h_lt_down
        h_lt_up.next = None     # TODO
        h_lt_up.prev = h_lb_up
        
        h_lb_down.node = n_left
        h_lb_down.face, f_lb.halfedge = f_lb, h_lb_down
        h_lb_down.edge, e_lb.halfedge = e_lb, h_lb_down
        h_lb_down.twin = h_lb_up
        h_lb_down.next = h_bl_in
        h_lb_down.prev = h_left_out
        h_lb_up.node = n_lb
        h_lb_up.face = None
        h_lb_up.edge = e_lb
        h_lb_up.twin = h_lb_down
        h_lb_up.next = h_lt_up
        h_lb_up.prev = None     # TODO
        
        h_rb_up.node = n_rb
        h_rb_up.face, f_rb.halfedge = f_rb, h_rb_up
        h_rb_up.edge, e_rb.halfedge = e_rb, h_rb_up
        h_rb_up.twin = h_rb_down
        h_rb_up.next = h_right_in
        h_rb_up.prev = h_br_out
        h_rb_down.node, n_right.halfedge = n_right, h_rb_down
        h_rb_down.face = None
        h_rb_down.edge = e_rb
        h_rb_down.twin = h_rb_up
        h_rb_down.next = None   # TODO
        h_rb_down.prev = h_rt_down
        
        h_rt_up.node = n_right
        h_rt_up.face, f_rt.halfedge = f_rt, h_rt_up
        h_rt_up.edge, e_rt.halfedge = e_rt, h_rt_up
        h_rt_up.twin = h_rt_down
        h_rt_up.next = h_tr_in
        h_rt_up.prev = h_right_out
        h_rt_down.node = n_rt
        h_rt_down.face = None
        h_rt_down.edge = e_rt
        h_rt_down.twin = h_rt_up
        h_rt_down.next = h_rb_down
        h_rt_down.prev = None       # TODO
        
        h_tl_out.node, n_center.halfedge = n_center, h_tl_out
        h_tl_out.face = f_lt
        h_tl_out.edge, e_tl.halfedge = e_tl, h_tl_out
        h_tl_out.twin = h_tl_in
        h_tl_out.next = h_lt_down
        h_tl_out.prev = h_left_in
        h_tl_in.node = n_lt
        h_tl_in.face = f_top
        h_tl_in.edge = e_tl
        h_tl_in.twin = h_tl_out
        h_tl_in.next = h_tr_out
        h_tl_in.prev = h_top
        
        h_left_out.node = n_center
        h_left_out.face = f_lb
        h_left_out.edge, e_left.halfedge = e_left, h_left_out
        h_left_out.twin = h_left_in
        h_left_out.next = h_lb_down
        h_left_out.prev = h_bl_in
        h_left_in.node = n_left
        h_left_in.face = f_lt
        h_left_in.edge = e_left
        h_left_in.twin = h_left_out
        h_left_in.next = h_tl_out
        h_left_in.prev = h_lt_up
        
        h_bl_out.node = n_center
        h_bl_out.face = f_bottom
        h_bl_out.edge, e_bl.halfedge = e_bl, h_bl_out
        h_bl_out.twin = h_bl_in
        h_bl_out.next = h_bottom
        h_bl_out.prev = h_br_in
        h_bl_in.node = n_lb
        h_bl_in.face = f_lb
        h_bl_in.edge = e_bl
        h_bl_in.twin = h_bl_out
        h_bl_in.next = h_left_out
        h_bl_in.prev = h_lb_down
        
        h_br_out.node = n_center
        h_br_out.face = f_rb
        h_br_out.edge, e_br.halfedge = e_br, h_br_out
        h_br_out.twin = h_br_in
        h_br_out.next = h_rb_up
        h_br_out.prev = h_right_in
        h_br_in.node = n_rb
        h_br_in.face = f_bottom
        h_br_in.edge, e_br.halfedge = e_br, h_br_in
        h_br_in.twin = h_br_out
        h_br_in.next = h_bl_out
        h_br_in.prev = h_bottom
        
        h_right_out.node = n_center
        h_right_out.face = f_rt
        h_right_out.edge, e_right.halfedge = e_right, h_right_out
        h_right_out.twin = h_right_in
        h_right_out.next = h_rt_up
        h_right_out.prev = h_tr_in
        h_right_in.node = n_right
        h_right_in.face = f_rb
        h_right_in.edge = e_right
        h_right_in.twin = h_right_out
        h_right_in.next = h_br_out
        h_right_in.prev = h_rb_up
        
        h_tr_out.node = n_center
        h_tr_out.face = f_top
        h_tr_out.edge, e_tr.halfedge = e_tr, h_tr_out
        h_tr_out.twin = h_tr_in
        h_tr_out.next = h_top
        h_tr_out.prev = h_tl_in
        h_tr_in.node = n_rt
        h_tr_in.face = f_rt
        h_tr_in.edge = e_tr
        h_tr_in.twin = h_tr_out
        h_tr_in.next = h_right_out
        h_tr_in.prev = h_rt_up
        
        f_top.update()
        f_lt.update()
        f_lb.update()
        f_bottom.update()
        f_rb.update()
        f_rt.update()
        # </editor-fold>
        
class Face:
    all = []
    
    def __init__(self):
        self.id = len(Face.all)
        Face.all.append(self)

        self.halfedge = None
        
        self.halfedges = [None, None, None]
        self.nodes = []
        
        self.rigid = False
        
    def remove(self):
        Face.all.remove(self)
        
    def update(self):
        # update self.halfedges and self.nodes
        h_0 = self.halfedge
        self.halfedges = [h_0]
        self.nodes = [h_0.node]
        
        h = h_0.next
        while h != h_0:
            self.halfedges.append(h)
            self.nodes.append(h.node)
            h = h.next
            
    def centroid(self):
        ps = [n.p for n in self.nodes]
        return np.array(ps).mean(axis=0)
    
    def offset(self):
        vectors = []     # vectors of halfedges
        points_offset = []  # points of halfedges after offsetting
        for h in self.halfedges:
            if h.edge.is_boundary:
                offset_length = 0
            else:
                offset_length = param.hinge_width / 2
            point, vec = h.offset(offset_length)
            vectors.append(vec)
            points_offset.append(point)
        
        nodes_new = []
        for i in range(len(points_offset)):
            i_prev = (i-1) % len(points_offset)
            
            # intersect two lines
            # l1: P + t_1 * v
            # l2: Q + t_2 * wa
            # u = P - Q
            # t_1 = (w x u) / (v x w)
            P = points_offset[i_prev]
            Q = points_offset[i]
            v = vectors[i_prev]
            w = vectors[i]
            u = P - Q
            t_1 = np.linalg.norm(np.cross(w, u)) / np.linalg.norm(np.cross(v, w))
            p_new = P + t_1 * v
            node_new = Node(p_new, self.nodes[i].is_boundary)
            nodes_new.append(node_new)
        
        face_new = Face()
        edges_new = []
        halfedges_in_new = []
        halfedges_out_new = []
        for i in range(len(self.halfedges)):
            h_in = Halfedge()
            h_out = Halfedge()
            halfedges_in_new.append(h_in)
            halfedges_out_new.append(h_out)
            self.halfedges[i].after_offset = h_in
            h_in.before_offset = self.halfedges[i]
        
        for i in range(len(self.halfedges)):
            e_old = self.halfedges[i].edge    # edgebefore offsetting
            edge = Edge(k=e_old.k, is_facet=e_old.is_facet, is_boundary=e_old.is_boundary, is_stretchable=e_old.is_stretchable, is_foldable=e_old.is_foldable,
                        target_angle=e_old.target_angle, is_offset=True)
            
            h = halfedges_in_new[i]
            h_twin = halfedges_out_new[i]
            i_prev = (i - 1) % len(self.halfedges)
            i_next = (i + 1) % len(self.halfedges)
            
            h.twin, h_twin.twin = h_twin, h
            h.node, nodes_new[i].halfedge = nodes_new[i], h
            h.edge, edge.halfedge = edge, h
            h.face, face_new.halfedge = face_new, h
            h.next = halfedges_in_new[i_next]
            h.prev = halfedges_in_new[i_prev]
            h_twin.node = nodes_new[i_next]
            h_twin.edge = edge
            h_twin.face = face_new
            h_twin.next = halfedges_out_new[i_prev]
            h_twin.prev = halfedges_out_new[i_next]
        face_new.update()
        
    def triangulate_with_facet_edge(self):
        # triangulate the face with facet edge
        self.update()
        
        if len(self.halfedges) == 4:
            def connect_two_nodes(i_0, i_1):
                n_0 = self.halfedges[i_0].node
                n_1 = self.halfedges[i_1].node
                
                e = Edge(0, is_facet=True, is_boundary=False, is_stretchable=False, is_foldable=False, target_angle=0)
                h_0 = Halfedge()
                h_1 = Halfedge()
                f_1 = Face()
                
                h_0.edge = e
                e.halfedge = h_0
                h_0.node = n_1
                h_0.next = self.halfedges[i_0]
                h_0.prev = self.halfedges[i_0].next
                h_0.prev.next = h_0
                h_0.twin = h_1
                h_0.face = self
                self.halfedge = h_0
                
                h_1.edge = e
                h_1.node = n_0
                h_1.face = f_1
                f_1.halfedge = h_1
                h_1.next = self.halfedges[i_1]
                h_1.prev = self.halfedges[i_1].next
                h_1.prev.next = h_1
                h_1.twin = h_0
                
                self.update()
                f_1.update()
                f_1.rigid = self.rigid = True
                
            l_0 = np.linalg.norm(self.nodes[0].p - self.nodes[2].p)
            l_1 = np.linalg.norm(self.nodes[1].p - self.nodes[3].p)
            
            if l_0 < l_1:
                connect_two_nodes(0, 2)
            else:
                connect_two_nodes(1, 3)
            
            return True
        
        return False
        
class Halfedge:
    all = []
    
    def __init__(self):
        self.node = None
        self.edge = None
        self.face = None
        self.twin = None
        self.next = None
        self.prev = None
        self.id = len(Halfedge.all)
        Halfedge.all.append(self)
        
        self.before_offset = None
        self.after_offset = None
    
    def remove(self):
        Halfedge.all.remove(self)

    @staticmethod
    def rot_2d(theta):
        # rotate a vector anti-clockwise by theta
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    def offset(self, distance):
        # offset the current halfedge towards the left
        # return: node position: numpy array [2]
        #         direction: numpy array [2]
        vec = self.twin.node.p - self.node.p
        vec /= np.sqrt((vec**2).sum())
        vec_left = self.rot_2d(np.pi / 2) @ vec[:2].reshape(-1, 1)
        vec_left = np.concatenate([vec_left.reshape(-1), np.array([0])])
        displacement = vec_left * distance
        p_new = self.node.p.copy()
        p_new += displacement
        return p_new, vec
        
        
def build_halfedge_mesh(pattern_dir):
    data = np.load(pattern_dir)
    
    nodes_in = data['nodes']
    edges_in = data['edges']
    faces_in = data['faces']
    k_edges = data['k_edges']
    is_facet_edges = data['is_facet']
    is_boundary_edges = data['is_boundary_edges']
    is_boundary_nodes = data['is_boundary_nodes']
    target_angles = data['target_angles']
    i_handle = data['i_handle']
    
    for i_node in range(len(nodes_in)):
        p = nodes_in[i_node]
        is_boundary = is_boundary_nodes[i_node]
        n = Node(p, is_boundary)
    
    for i_face in range(len(faces_in)):
        f = Face()
        for i_node in faces_in[i_face]:
            f.nodes.append(Node.all[i_node])
        
    for i_edge in range(len(edges_in)):
        k = k_edges[i_edge]
        is_facet = is_facet_edges[i_edge]
        is_boundary = is_boundary_edges[i_edge]
        target_angle = target_angles[i_edge]
        e = Edge(k, is_facet, is_boundary, target_angle=target_angle, is_stretchable=False, is_foldable=None)
        
        n_0 = Node.all[edges_in[i_edge][0]]
        n_1 = Node.all[edges_in[i_edge][1]]
        
        neighbor_faces = []
        for f in Face.all:
            if n_0 in f.nodes and n_1 in f.nodes:
                neighbor_faces.append(f)
                
        f_0 = neighbor_faces[0]
        i_n_0 = f_0.nodes.index(n_0)
        i_n_1 = f_0.nodes.index(n_1)
        if i_n_1 == (i_n_0 + 1) % 3:    # n_0 -> n_1 is along anti-clockwise direction of face 0 (not face 1)
            pass
        else:
            n_0, n_1 = n_1, n_0
        
        h_0 = Halfedge()
        h_1 = Halfedge()
        
        h_0.edge = h_1.edge = e
        e.halfedge = h_0
        
        h_0.twin, h_1.twin = h_1, h_0
        
        h_0.node = n_0
        h_1.node = n_1
        n_0.halfedge = h_0
        n_1.halfedge = h_1
        
        h_0.face = f_0
        f_0.halfedge = h_0
        f_0.halfedges[f_0.nodes.index(n_0)] = h_0
        if len(neighbor_faces) > 1:
            assert(len(neighbor_faces) == 2)
            f_1 = neighbor_faces[1]
            h_1.face = f_1
            f_1.halfedge = h_1
            f_1.halfedges[f_1.nodes.index(n_1)] = h_1
        
    for f in Face.all:
        for i in range(3):
            i_next = (i+1) % 3
            f.halfedges[i].next = f.halfedges[i_next]
            f.halfedges[i_next].prev = f.halfedges[i]
    
    halfedges_boundary = []
    for h in Halfedge.all:
        if h.face is None:
            halfedges_boundary.append(h)
    
    for h in halfedges_boundary:
        if h.prev is not None and h.next is not None:
            continue
        for h_1 in halfedges_boundary:
            if h_1.node == h.twin.node:
                h.next = h_1
                h_1.prev = h
            if h_1.twin.node == h.node:
                h_1.next = h
                h.prev = h_1

def save_for_fabrication():
    v = []
    for n in Node.all:
        p = n.p.tolist()
        v.append(p)
        
    f = []
    f_soft = []
    for face in Face.all:
        
        ids = []
        for n in face.nodes:
            ids.append(n.id)
        f_soft.append(ids)
        
        if face.rigid:
            f.append(ids)
    
    data = {'rigid':
                {'v': v, 'f': f},
            'soft':
                {'v': v, 'f': f_soft},
            }
    
    string = json.dumps(data)
    with open('./data/fabrication.json', 'w') as ofile:
        ofile.write(string)
    
def show():
    lines = []
    colors = []
    for e in Edge.all:
        p_0 = e.halfedge.node.p[:2]
        try:
            p_1 = e.halfedge.twin.node.p[:2]
        except:
            pdb.set_trace()
        seg = np.array([p_0, p_1])
        lines.append(seg)
        
        # if e.is_facet:
        #     colors.append('red')
        # elif e.is_foldable:
        #     colors.append
    
    
    lc = LineCollection(lines, linewidths=1)
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    
    if Focus.type is Halfedge:
        x_0, y_0 = Focus.focus.node.p[:2]
        x_1, y_1 = Focus.focus.twin.node.p[:2]
        dx = x_1 - x_0
        dy = y_1 - y_0
        ax.arrow(x_0, y_0, dx, dy, length_includes_head=True, head_length=0.05, shape='right', width=0.05, color='red')
    
    if Focus.type is Edge:
        x_0, y_0 = Focus.focus.halfedge.node.p[:2]
        x_1, y_1 = Focus.focus.halfedge.twin.node.p[:2]
        ax.plot( [x_0, x_1], [y_0, y_1], '-', color='red', linewidth=2)
    
    if Focus.type is Node:
        ax.plot(Focus.focus.p[0], Focus.focus.p[1], 'o', color='red')
    
    if Focus.type is Face:
        p_mean = Focus.focus.centroid()
        ax.plot(p_mean[0], p_mean[1], 'o', color='red')

    def press(event):
        if Focus.type is Halfedge:
            if event.key == 'n':
                Focus.at(Focus.focus.next)
            if event.key == 'p':
                Focus.at(Focus.focus.prev)
            if event.key == 't':
                Focus.at(Focus.focus.twin)
            if event.key == 'e':
                Focus.at(Focus.focus.edge)
                print('is_foldable: ', Focus.focus.is_foldable)
                print('is_facet: ', Focus.focus.is_facet)
                print('is_stretchable: ', Focus.focus.is_stretchable)
            if event.key == 'v':
                Focus.at(Focus.focus.node)
            if event.key == 'f':
                Focus.at(Focus.focus.face)
            if event.key == 'd':    # duo
                if Focus.focus.before_offset is not None:
                    Focus.at(Focus.focus.before_offset)
                else:
                    Focus.at(Focus.focus.after_offset)
                
        else:
            if Focus.type is Edge and event.key == 'c':
                Focus.focus.collapse()
            
            if Focus.type is Edge and event.key == 'f':     # fold
                Focus.focus.create_compliant_hinge()
            
            if Focus.type is Face and event.key == 't':     # triangulate
                Focus.focus.triangulate_with_facet_edge()
                
            if event.key == 'h':
                Focus.at(Focus.focus.halfedge)
        
        plt.close()
        
        if event.key == 'q':
            return False
        show()
        
    fig.canvas.mpl_connect('key_press_event', press)
    
    plt.draw()
    plt.waitforbuttonpress()
    
if __name__ == '__main__':
    
    pattern_name = 'miura' if len(sys.argv) == 1 else str(sys.argv[1])
    pattern_dir = 'data/pattern/{}.npz'.format(pattern_name)
    
    param = Parameter()
    
    build_halfedge_mesh(pattern_dir)
    
    for e in Edge.all:
        if e.is_facet:
            e.collapse()
    
    # record old mesh ids
    num_nodes_old = len(Node.all)
    num_edges_old = len(Edge.all)
    num_faces_old = len(Face.all)
    num_halfedges_old = len(Halfedge.all)
    
    for f in Face.all[:]:
        f.offset()
        
    for i_e, e in enumerate(Edge.all[:]):
        if e.is_foldable and not e.is_offset:
            e.create_compliant_hinge()
            
    # remove old mesh
    Node.all = Node.all[num_nodes_old:]
    Edge.all = Edge.all[num_edges_old:]
    Face.all = Face.all[num_faces_old:]
    Halfedge.all = Halfedge.all[num_halfedges_old:]
    reindex(Node)
    reindex(Face)
    reindex(Edge)
    reindex(Halfedge)
    
    # triangulate quad faces
    for f in Face.all[:]:
        f.triangulate_with_facet_edge()
    
    Focus.at(Halfedge.all[0])
    
    show()
    save_for_fabrication()

