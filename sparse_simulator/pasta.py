import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
import pdb


vector3d = lambda v: o3.utility.Vector3dVector(v)
vector3i = lambda v: o3.utility.Vector3iVector(v)
vector2i = lambda v: o3.utility.Vector2iVector(v)
LineSet = lambda v, e: o3.geometry.LineSet(points=vector3d(v), lines=vector2i(e))
PointCloud = lambda v: o3.geometry.PointCloud(points=vector3d(v))

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
EA = 100
k_fold = 1
k_facet = 20
k_face = 20000
h = 0.005
p_fold = 0.5      # folding percentage

# var
n_steps = 0
t0 = time.time()
t_prev = t0
last_click = None

steps_0 = None

# state
folding = True
fixed = False
forcing = False
trajectory = False

# recordings
energies = []

# ================ load ==================
name = 'pasta.npz'
name = 'data/' + name
with open(name) as f:
    if name[-4] == json:
        c = f.read()
        data = json.loads(c)
    else:
        data = np.load(name)

nodes_in = torch.tensor(data['nodes'], dtype=torch.float)  # nv x 3
edges_in = torch.tensor(data['edges'], dtype=torch.long)  # ne x 2
faces_in = torch.tensor(data['faces'], dtype=torch.long)  # nf x 3

is_boundary_nodes = torch.tensor(data['is_boundary_nodes'], dtype=torch.int).reshape(-1, 1)  # nv x 1
is_facet = torch.tensor(data['is_facet'], dtype=torch.int)  # ne x 1
is_boundary_edges = torch.tensor(data['is_boundary_edges'], dtype=torch.int)  # ne x 1
i_handle = data['i_handle']     # ()
rho_target = torch.tensor(data['target_angles'], dtype=torch.float)  # ne x 1

# ================ init var ==================
x = nodes_in.clone()
edges = edges_in.clone()
faces = faces_in.clone()
v = torch.zeros([len(x), 3])

# bar
M_bar = torch.zeros([len(edges), len(x)])
i_edges = torch.arange(len(edges))
M_bar[i_edges, edges[:, 1]] = 1     # x_2
M_bar[i_edges, edges[:, 0]] = -1    # x_1
d_0 = M_bar @ x
L_0 = d_0.norm(dim=1, keepdim=True)
# L_0 = torch.ones_like(L_0)* L_0.mean()

# hinge
hinges = edges[is_boundary_edges.logical_not()]
rho_target = rho_target[is_boundary_edges.logical_not()]

hinge_face_pairs = (
        (hinges.unsqueeze(1).unsqueeze(3) == faces.unsqueeze(0).unsqueeze(2))
        .sum(-1).sum(-1) == 2
).nonzero()   # (2 x n_hinges) x 2   [[hinge_id, neighbor_face_id]]
repeating_01 = torch.tensor([[0, 1]]).T.repeat(len(hinges), 1)
triplet_faces = torch.index_select(
    torch.cat([repeating_01, hinge_face_pairs], axis=1), 1, torch.tensor([1, 0, 2]))
face_pairs = torch.zeros([len(hinges), 2], dtype=torch.long)
face_pairs[triplet_faces[:, 0], triplet_faces[:, 1]] = triplet_faces[:, 2]
faces_k = faces[face_pairs][:, 0, :]    # n_hinges x 3
faces_l = faces[face_pairs][:, 1, :]    # n_hinges x 3

nx_i = torch.zeros([len(hinges)], dtype=torch.long)   # index of x_i
nx_j = torch.zeros([len(hinges)], dtype=torch.long)
nx_k = torch.zeros([len(hinges)], dtype=torch.long)
nx_l = torch.zeros([len(hinges)], dtype=torch.long)
for i_hinge in range(len(hinges)):
    face_k = faces_k[i_hinge]
    face_l = faces_l[i_hinge]
    n_ij = []
    for i_node in range(3):
        if face_k[i_node] not in face_l:
            n_k = face_k[i_node]
        else:
            n_ij.append(face_k[i_node])
        if face_l[i_node] not in face_k:
            n_l = face_l[i_node]
    n_i, n_j = n_ij
    if (x[n_j] - x[n_i]).cross(x[n_k] - x[n_j])[-1] < 0:
        n_i, n_j = n_j, n_i
    nx_i[i_hinge] = n_i
    nx_j[i_hinge] = n_j
    nx_k[i_hinge] = n_k
    nx_l[i_hinge] = n_l

k_hinge = (is_facet * k_facet + (~is_facet) * k_fold).reshape(-1, 1)
k_hinge = k_hinge[is_boundary_edges.logical_not()]

# initial noise
# x += torch.normal(torch.zeros_like(x), torch.ones_like(x) * 1)


# ================ init vis ================
vs = vector3d(x.detach().numpy())
es = vector2i(edges.detach().numpy())
ls = LineSet(vs, es)
pc = PointCloud(vs)
vis.add_geometry(ls)
vis.add_geometry(pc)
# ==========================================

def step():
    global f_hinge_k, f
    f = torch.zeros([len(x), 3])
    
    # bar energy
    d = M_bar @ x
    d_norm = (d * d).sum(axis=1, keepdim=True).sqrt()
    f_bar_i = EA / L_0 * (d_norm - L_0) / d_norm * d
    f_bar = -M_bar.T @ f_bar_i
    f.add_(f_bar)
    
    # hinge energy
    x_i = x[nx_i]
    x_j = x[nx_j]
    x_k = x[nx_k]
    x_l = x[nx_l]
    d_ij = x_j - x_i
    d_ij_norm = d_ij.norm(dim=1, keepdim=True)  # norm of d_ij
    d_ik = x_k - x_i
    d_ik_norm = d_ik.norm(dim=1, keepdim=True)
    d_il = x_l - x_i
    d_il_norm = d_il.norm(dim=1, keepdim=True)
    d_jk = x_k - x_j
    d_jk_norm = d_jk.norm(dim=1, keepdim=True)
    d_jl = x_l - x_j
    d_jl_norm = d_jl.norm(dim=1, keepdim=True)
    cos_jik = (d_ij_norm ** 2 + d_jk_norm ** 2 - d_ik_norm ** 2) / (2 * d_ij_norm * d_jk_norm)
    cos_ijk = (d_ij_norm ** 2 + d_ik_norm ** 2 - d_jk_norm ** 2) / (2 * d_ij_norm * d_ik_norm)
    cos_jil = (d_ij_norm ** 2 + d_jl_norm ** 2 - d_il_norm ** 2) / (2 * d_ij_norm * d_jl_norm)
    cos_ijl = (d_ij_norm ** 2 + d_il_norm ** 2 - d_jl_norm ** 2) / (2 * d_ij_norm * d_il_norm)
    cot = lambda t: t / torch.sqrt(1 - t ** 2)
    cot_jik = cot(cos_jik)
    cot_ijk = cot(cos_ijk)
    cot_jil = cot(cos_jil)
    cot_ijl = cot(cos_ijl)
    n_k = d_ij.cross(d_jk) / d_ij.cross(d_jk).norm(dim=1, keepdim=True)
    h_k = d_ij.cross(d_jk).norm(dim=1, keepdim=True) / d_ij_norm
    n_l = -d_ij.cross(d_il) / (-d_ij.cross(d_il)).norm(dim=1, keepdim=True)
    h_l = (-d_ij.cross(d_il)).norm(dim=1, keepdim=True) / d_ij_norm

    sign = -torch.sign((n_k.cross(n_l) * d_ij).sum(dim=1))
    dot = (n_k * n_l).sum(dim=1)
    dot = dot.clamp(max=1)
    rho = sign * torch.acos(dot)    # n_hinges
    
    par_rho_xk = n_k / h_k         # partial rho partial x_k, n_hinges x 3
    par_rho_xl = n_l / h_l
    par_rho_xi = -cot_jik / (cot_jik + cot_ijk) * n_k / h_k - cot_jil / (cot_jil + cot_ijl) * n_l / h_l
    par_rho_xj = -cot_ijk / (cot_jik + cot_ijk) * n_k / h_k - cot_ijl / (cot_jil + cot_ijl) * n_l / h_l
    
    f_hinge_k = -k_hinge * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xk
    f_hinge_l = -k_hinge * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xl
    f_hinge_i = -k_hinge * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xi
    f_hinge_j = -k_hinge * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xj
    
    f[nx_i] += f_hinge_i
    f[nx_j] += f_hinge_j
    f[nx_k] += f_hinge_k
    f[nx_l] += f_hinge_l
    
    # update
    a = f
    v.add_(a * h)
    v.mul_(0.9)
    x.add_(v)
    x.add_(-v * is_boundary_nodes)

def draw():
    vs = vector3d(x.detach().numpy())
    ls.points = vs
    ls.lines = es
    pc.points = vs
    vis.update_geometry(ls)
    vis.update_geometry(pc)
    
def timerCallback(vis):
    step()
    draw()
    
vis.register_animation_callback(timerCallback)

vis.run()
vis.destroy_window()


