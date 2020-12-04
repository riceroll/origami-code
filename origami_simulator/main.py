import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
from draw_lines import LineMesh

vector3d = lambda v: o3.utility.Vector3dVector(v)
vector3i = lambda v: o3.utility.Vector3iVector(v)
vector2i = lambda v: o3.utility.Vector2iVector(v)

vis = o3.visualization.VisualizerWithKeyCallback()
vis.create_window()
render_opt = vis.get_render_option()

render_opt.mesh_show_back_face = True
render_opt.mesh_show_wireframe = True
render_opt.point_size = 8
render_opt.line_width = 10
render_opt.light_on = True
show_lines = False

# const
EA = 200
k_fold = 10
k_facet = 200
k_face = 2000
h = 0.01
folding_percent = 0

t0_trajs = 0
T_trajs = 0
forcing = False

# 00
name = 'miura'
name = 'resch'
name = '3miura_arm'
with open('../data/{}.json'.format(name)) as f:
    c = f.read()
    data = json.loads(c)
    
points_in = np.array(data['points'])
faces_in = data['faces']
creases_in = data['creases']
trajs_in = data['trajs']


class Node:
    selected = 0
    
    def __init__(self, idx, pos, fixed=False):
        self.P = torch.Tensor(pos)
        
        self.F = torch.zeros(3)
        self.V = torch.zeros(3)

        self.idx = idx
        self.fixed = fixed
ns = []
for n in points_in:
    ns.append(Node(len(ns), n[:3], n[-1]))
nodes = ns


class Face:
    def __init__(self, ns):
        self.ns = [nodes[n] for n in ns]    # anti-clockwise from the top view
        self.a0 = [self.angle(i) for i in range(3)]     # initial angles
        self.ls0 = self.lens()

    def lens(self):
        l0 = (self.ns[1].P - self.ns[0].P).norm()
        l1 = (self.ns[2].P - self.ns[1].P).norm()
        l2 = (self.ns[0].P - self.ns[2].P).norm()
        return [l0, l1, l2]
    
    def energy(self):
        e = 0
        ls = self.lens()
        for i, l in enumerate(ls):
            e += (l - self.ls0[i]) ** 2
        return e
    
    def angle(self, i):     # interior angle around the ith node
        i_prev = (i - 1) % 3
        i_next = (i + 1) % 3
        vec_1 = self.ns[i_next].P - self.ns[i].P
        vec_2 = self.ns[i_prev].P - self.ns[i].P
        cos = vec_1.dot(vec_2) / vec_1.norm() / vec_2.norm()
        return torch.acos(cos)

    def normal(self):
        n = (self.ns[1].P - self.ns[0].P).cross(self.ns[2].P - self.ns[1].P)
        return n / n.norm()

    def area(self):
        n = (self.ns[1].P - self.ns[0].P).cross(self.ns[2].P - self.ns[1].P)
        return n.norm() / 2

    def cot(self, i):   # interior angle cot around node i
        i_prev = (i-1) % 3
        i_next = (i+1) % 3
        vec_1 = self.ns[i_next].P - self.ns[i].P
        vec_2 = self.ns[i_prev].P - self.ns[i].P
        sin = vec_1.cross(vec_2).norm() / vec_1.norm() / vec_2.norm()
        cos = vec_1.dot(vec_2) / vec_1.norm() / vec_2.norm()
        return cos / sin

    def idx(self, n):   # index in self.ns of a Node
        return self.ns.index(n)
faces = [Face(f) for f in faces_in]


class Edge:
    def __init__(self, n1, n2, foldable=False, target_fold_angle=0, stiffness=0.1):
        self.n1 = nodes[n1]
        self.n2 = nodes[n2]
        self.a0 = self.angle()
        self.target_dangle = torch.Tensor([target_fold_angle])    # target_fold_angle
        self.target_length = self.length()
        self.foldable = foldable
        self.stiffness = stiffness

    def vec(self):  # pointing from n1 to n2
        return self.n2.P - self.n1.P
    
    def energy(self):
        return (self.angle() - self.target_dangle * folding_percent) ** 2

    def length(self):
        return self.vec().norm()

    def faces(self):
        fs_adj = []
        for f in faces:
            if self.n1 in f.ns and self.n2 in f.ns:
                fs_adj.append(f)
        assert len(fs_adj) == 2

        i = fs_adj[0].ns.index(self.n1)
        i_next = (i + 1) % 3
        if fs_adj[0].ns[i_next] != self.n2:
            fs_adj = fs_adj[::-1]
        return fs_adj[0], fs_adj[1]     # fs[0] is the face on the left looking from n1 to n2

    def f1(self):   # the face on the left looking from n1 to n2
        return self.faces()[0]

    def f2(self):
        return self.faces()[1]

    def n3(self):   # the node on the left looking n1 to n2, on f1
        for n in self.f1().ns:
            if n not in [self.n1, self.n2]:
                return n

    def n4(self):
        for n in self.f2().ns:
            if n not in [self.n1, self.n2]:
                return n

    def angle(self):    # fold_angle
        f1, f2 = self.faces()
        n1 = f1.normal()
        n2 = f2.normal()
        sign = -torch.sign(n1.cross(n2).dot(self.vec()))
        dot = n1.dot(n2)
        dot = dot if dot < 1 else dot * 0 + 1
        theta = sign * torch.acos(dot)
        return theta
edges = [Edge(e[0], e[1], e[2], e[3], e[4]) for e in creases_in]

# trajs
trajs = dict()

for key in trajs_in.keys():
    ps = np.array(trajs_in[key])
    xs = np.linspace(0, 1, len(ps))
    trajs[int(key)] = CubicSpline(xs, ps)

# initt
ps = vector3d(np.array([np.array(n.P*2) for n in nodes]))
fs = vector3i(np.array(faces_in))
es = []
for e in edges:
    if e.foldable:
        es.append([e.n1.idx, e.n2.idx])
es = vector2i(np.array(es))
colors = [[1, 0, 0] for i in range(len(es))]

mesh = o3.geometry.TriangleMesh(ps, fs)
vis.add_geometry(mesh)
pc = o3.geometry.PointCloud(ps)
# vis.add_geometry(pc)

lm = LineMesh(ps, es, colors, radius=0.2)
lseg = lm.cylinder_segments
for l in lseg:
    vis.add_geometry(l, reset_bounding_box=False)

n_steps = 0
t0 = time.time()
t_prev = t0

def energy(angle=True):
    en = 0
    for f in faces:
        en += f.energy()
    if angle:
        for e in edges:
            en += e.energy()[0]
    return en


def update():
    for n in nodes:
        n.F = torch.zeros(3)
        
    for e in edges:
        # axial
        k_axial = EA / e.length()
        F_axial = -k_axial * (e.length() - e.target_length)
        e.n1.F += F_axial * (-e.vec())
        e.n2.F += F_axial * e.vec()
        
        # crease
        f1, f2 = e.faces()
        N1, N2 = f1.normal(), f2.normal()
        h1 = f1.area() * 2 / e.length()
        h2 = f2.area() * 2 / e.length()
        cot11 = f1.cot(f1.idx(e.n1))
        cot12 = f2.cot(f2.idx(e.n1))
        cot21 = f1.cot(f1.idx(e.n2))
        cot22 = f2.cot(f2.idx(e.n2))

        dth_dp3 = N1 / h1   # partial theta ove partial p3
        dth_dp4 = N2 / h2
        dth_dp1 = -cot21 / (cot21 + cot11) / h1 * N1 \
                + -cot22 / (cot22 + cot12) / h2 * N2
        dth_dp2 = -cot11 / (cot21 + cot11) / h1 * N1 \
                + -cot12 / (cot22 + cot12) / h2 * N2
        dtheta = e.angle() - (e.a0 + e.target_dangle * folding_percent)

        k_crease = k_fold * e.stiffness if e.foldable else k_facet
        k_crease = k_crease * e.length()
        
        e.n1.F += -k_crease * dtheta * dth_dp1
        e.n2.F += -k_crease * dtheta * dth_dp2
        e.n3().F += -k_crease * dtheta * dth_dp3
        e.n4().F += -k_crease * dtheta * dth_dp4
        
    for f in faces:
        for i_n, n in enumerate(f.ns):
            i_next = (i_n + 1) % 3
            i_prev = (i_n - 1) % 3
            n_next = f.ns[i_next]
            n_prev = f.ns[i_prev]
            dadp_prev = f.normal().cross(n_prev.P - n.P) / (n_prev.P - n.P).norm() ** 2   # partial angle partial p_prev
            dadp_next = -f.normal().cross(n_next.P - n.P) / (n_next.P - n.P).norm() ** 2
            dadp_n = -(dadp_prev + dadp_next)
            n.F += -k_face * (f.angle(i_n) - f.a0[i_n]) * dadp_n
            n_next.F += -k_face * (f.angle(i_n) - f.a0[i_n]) * dadp_next
            n_prev.F += -k_face * (f.angle(i_n) - f.a0[i_n]) * dadp_prev


energies = []
fs = []


def step():
    global n_steps
    n_steps += 1
    # print('fps: ', n_steps / (time.time() - t0))
    
    update()
    energies.append(energy())
    
    for i, n in enumerate(nodes):
        if i in trajs.keys():
            t = time.time() - t0
            try:
                if forcing:
                    vec = torch.Tensor(trajs[i](1)) - torch.Tensor(trajs[i](0))
                    force_mag = 10000
                    F = vec / vec.norm() * force_mag
                    n.F += F
            except:
                pass
            
        if n.fixed:
            continue
        
        n.V += n.F * h
        n.V *= 0.9
        while n.V.norm() > 5:
            n.V *= 0.9
        n.P += n.V * h


def draw():
    global fs, es, mesh, lines, pc, lseg
    ps = vector3d(np.array([np.array(n.P) for n in nodes]))

    mesh.vertices = ps
    mesh.compute_vertex_normals()

    pc.points = ps
    colors = []
    for n in nodes:
        color = [0, 0, 0]
        if n.fixed:
            color = [1, 0, 0]
        # if n.idx == Node.selected:
            # color = [0, 1, 0]
        colors.append(color)
    colors = np.array(colors)
    pc.colors = vector3d(colors)

    vis.update_geometry(mesh)
    # vis.update_geometry(pc)
    
    for l in lseg:
        vis.remove_geometry(l, False)
    colors = [[1, 0, 0] for i in range(len(es))]
    lm = LineMesh(ps, es, colors, radius=0.1)
    lseg = lm.cylinder_segments
    for l in lseg:
        if show_lines:
            vis.add_geometry(l, reset_bounding_box=False)

    for i, f in enumerate(forces):
        key = list(trajs.keys())[i]
        head = nodes[key].P
        vec = torch.Tensor(trajs[key](1)) - torch.Tensor(trajs[key](0))
        end = head + vec
        ps = np.array([np.array(head), np.array(end)])
        if not forcing:
            ps *= 0
        f.points = vector3d(ps)
        vis.update_geometry(f)


def drawForces():
    global t0_trajs, T_trajs, forces
    try:
        forces[0]
    except:
        forces = []
        
    t = time.time() - t0
    tt = (t - t0_trajs) / (T_trajs - t0_trajs + 1e-5)
    if 0 <= tt:
        print(2)
        for key in trajs.keys():
            head = nodes[key].P
            vec = torch.Tensor(trajs[key](1)) -torch.Tensor(trajs[key](0))
            end = head + vec
            ps = np.array([np.array(head), np.array(end)])
            ps = vector3d(ps)
            es = vector2i(np.array([[0, 1]], dtype=np.int))
            cs = vector3d(np.array([[0, 0.9, 0]]))
            f = o3.geometry.LineSet(ps, es)
            f.colors = cs
            forces.append(f)
            vis.add_geometry(f)
    

def drawGround():
    z = -5
    # gap = 10
    gap = 1
    n = 100
    darkness = 0.5
    vs = []
    es = []
    for i, x in enumerate(np.arange(1 - n, n)):
        y = n - 1
        vs.append([x * gap, -y * gap, z])
        vs.append([x * gap, y * gap, z])
        es.append([i * 2, i * 2 + 1])
    cs = np.ones(3) * darkness
    lines = o3.geometry.LineSet(points=vector3d(vs), lines=vector2i(es))
    lines.paint_uniform_color(cs)
    vis.add_geometry(lines)
    
    vs = []
    es = []
    for i, y in enumerate(np.arange(1 - n, n)):
        x = n - 1
        vs.append([-x * gap, y * gap, z])
        vs.append([x * gap, y * gap, z])
        es.append([i * 2, i * 2 + 1])
    cs = np.ones(3) * darkness
    lines = o3.geometry.LineSet(points=vector3d(vs), lines=vector2i(es))
    lines.paint_uniform_color(cs)
    vis.add_geometry(lines)
drawGround()
drawForces()


def timerCallback(vis):
    step()
    draw()

vis.register_animation_callback(timerCallback)

def stop(a,b,c):
    vis.destroy_window()

def fold(a):
    global folding_percent
    folding_percent += 0.01
    print(round(folding_percent, 3))
    if folding_percent > 1:
        folding_percent = 1
    
def unfold(a):
    global folding_percent
    folding_percent -= 0.01
    print(round(folding_percent, 3))
    if folding_percent < 0:
        folding_percent = 0

def force(a):
    global T_trajs, t0_trajs, forcing
    forcing = not forcing
    

def show_line(a):
    global show_lines
    show_lines = not show_lines

vis.register_key_callback(85, fold)
vis.register_key_callback(74, unfold)
vis.register_key_callback(70, force)
vis.register_key_callback(83, show_line)


vis.run()
vis.destroy_window()


ts = [i for i in range(len(energies))]
plt.plot(ts, energies)
plt.plot(ts, fs)