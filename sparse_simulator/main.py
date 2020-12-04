import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
import pdb
from sparse_simulator import forcer
from sparse_simulator import force_viewer


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

# ======== const ========
# name = 'miura0.npz'
# name = 'triangle.npz'
# name = 'fold.npz'
name = 'pasta2.npz'

# scripting
paused = False
AUTO_FORCING = False
p_fold = 0.1      # folding percentage
n_sub_steps = 100

SHOW_FORCE = False
t0_folding = 1500 / 3 * 5
duration_folding = 3000

# model
EA = 100
k_fold = 1
k_facet = 10
k_face = 20000
v_damping = 0.1
h = 0.007

# visual
force_length_multiplier = 20
force_color = np.array([[0.8, 0.1, 0.1]]).T

# ========================

# var
n_steps = 0
t0 = time.time()
t_prev = t0
last_click = None

steps_0 = None

# state
folding = True
pinned = False
forcing = False
trajectory = False

# recordings
force = forcer.Force()
energies = []
forces = []
traj = []

# ================ load ==================
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

k_edges = torch.tensor(data['k_edges']) if 'k_edges' in data else torch.zeros([len(edges_in), 1])  # ne x 1
is_boundary_nodes = torch.tensor(data['is_boundary_nodes'], dtype=torch.long)  # nv x 1
is_facet = torch.tensor(data['is_facet'], dtype=torch.int)  # ne x 1
is_boundary_edges = torch.tensor(data['is_boundary_edges'], dtype=torch.int)  # ne x 1
i_handle = data['i_handle']     # ()
rho_target = torch.tensor(data['target_angles'], dtype=torch.float)  # ne x 1


# ================ init var ==================
x = nodes_in.clone()
edges = edges_in.clone()
faces = faces_in.clone()
v = torch.zeros([len(x), 3])
f = torch.zeros([len(x), 3])

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

k_crease = k_fold * is_facet[is_boundary_edges.logical_not()].logical_not() + \
           k_facet * is_facet[is_boundary_edges.logical_not()]
k_crease = k_crease.reshape(-1, 1)

k_crease_2 = k_crease * k_edges[is_boundary_edges.logical_not()].reshape(-1, 1)

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
    
    # not sure
    if (x[n_j] - x[n_i]).cross(x[n_k] - x[n_j])[-1] < 0:
        n_i, n_j = n_j, n_i
    nx_i[i_hinge] = n_i
    nx_j[i_hinge] = n_j
    nx_k[i_hinge] = n_k
    nx_l[i_hinge] = n_l

# noise
# x += torch.normal(torch.zeros_like(x), torch.ones_like(x) * 0.05)

# ================ init vis ================
vs = vector3d(x.detach().numpy())
es = vector2i(edges.detach().numpy())
ls = LineSet(vs, es)
pc = PointCloud(vs)
vis.add_geometry(ls)
vis.add_geometry(pc)

# vis force
vs_force = vector3d(torch.cat([x, x + f*force_length_multiplier ], 0).detach().numpy())
es_force = vector2i(
    torch.stack([torch.arange(len(x)), torch.arange(len(x), len(x)*2)]).T.detach().numpy()
)
ls_force = LineSet(vs_force, es_force)
ls_force.paint_uniform_color(force_color)
if SHOW_FORCE:
    vis.add_geometry(ls_force)
# ==========================================

def step():
    global f, n_steps, steps_0, forcing
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
    n_k = d_ij.cross(d_jk, 1) / d_ij.cross(d_jk, 1).norm(dim=1, keepdim=True)
    h_k = d_ij.cross(d_jk, 1).norm(dim=1, keepdim=True) / d_ij_norm
    n_l = -d_ij.cross(d_il, 1) / (-d_ij.cross(d_il, 1)).norm(dim=1, keepdim=True)
    h_l = (-d_ij.cross(d_il, 1)).norm(dim=1, keepdim=True) / d_ij_norm

    sign = -torch.sign((n_k.cross(n_l, 1) * d_ij).sum(dim=1))
    dot = (n_k * n_l).sum(dim=1)
    dot = dot.clamp(max=1)
    rho = sign * torch.acos(dot)    # n_hinges
    
    par_rho_xk = n_k / h_k         # partial rho partial x_k, n_hinges x 3
    par_rho_xl = n_l / h_l
    par_rho_xi = -cot_jik / (cot_jik + cot_ijk) * n_k / h_k - cot_jil / (cot_jil + cot_ijl) * n_l / h_l
    par_rho_xj = -cot_ijk / (cot_jik + cot_ijk) * n_k / h_k - cot_ijl / (cot_jil + cot_ijl) * n_l / h_l
    
    if forcing:
        f_hinge_k = -k_crease_2 * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xk
        f_hinge_l = -k_crease_2 * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xl
        f_hinge_i = -k_crease_2 * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xi
        f_hinge_j = -k_crease_2 * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xj
    else:
        f_hinge_k = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xk
        f_hinge_l = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xl
        f_hinge_i = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xi
        f_hinge_j = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xj
        
    f.index_add_(0, nx_i, f_hinge_i)
    f.index_add_(0, nx_j, f_hinge_j)
    f.index_add_(0, nx_k, f_hinge_k)
    f.index_add_(0, nx_l, f_hinge_l)
    
        
    # update
    n_steps += 1
    
    a = f
    
    v.add_(a * h)
    v.mul_(1 - v_damping)

    x.add_(v)
    
    if pinned:
        x.add_(is_boundary_nodes.reshape(-1, 1) * v * -1)
    
    if forcing:
        x[i_handle].add_(v[i_handle].mul_(-1))
        x_handle, y_handle = force.move(n_steps)
        
        x[i_handle].add_(torch.tensor([x_handle, y_handle, 0]) - x[i_handle])
        forces.append(f[i_handle])
        traj.append([x_handle, y_handle])
    

def draw():
    vs = vector3d(x.detach().numpy())
    ls.points = vs
    ls.lines = es
    pc.points = vs
    vis.update_geometry(ls)
    vis.update_geometry(pc)
    
    vs_force = vector3d(torch.cat([x, x + f * force_length_multiplier ], 0).detach().numpy())
    ls_force.points = vs_force
    ls_force.paint_uniform_color(force_color)
    if SHOW_FORCE:
        vis.update_geometry(ls_force)
    
def timerCallback(vis):
    global pinned, h, forcing, x
    
    if AUTO_FORCING:
        t_stop = t0_folding + duration_folding
        if n_steps == t0_folding:
            pinned = True
            h = 0.005
            forcing = True
            x_0 = x[i_handle][0]
            y_0 = x[i_handle][1]
            force.start(x_0, y_0, n_steps, h)
            print('start forcing')
        if n_steps == t_stop:
            forcing = False
            print('stop forcing')
        if n_steps < t0_folding:
            for i in range(20 - 1):
                step()
    
    for i in range(n_sub_steps):
        step()
    draw()
    
vis.register_animation_callback(timerCallback)


def fold(a):
    global p_fold, last_click
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            p_fold += 0.05
            p_fold = 1 if p_fold > 1 else p_fold
            print('p_fold: ', p_fold)


def unfold(a):
    global p_fold, last_click
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            p_fold -= 0.05
            p_fold = 0 if p_fold < 0 else p_fold
            print('p_fold: ', p_fold)

def toggle_forcing(a):
    global forcing, last_click, steps_0, n_steps, x_0, y_0, x
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            forcing = not forcing

            x_0 = x[i_handle][0]
            y_0 = x[i_handle][1]
            force.start(x_0, y_0, n_steps, h)
            print('forcing: ', forcing)


def toggle_pin(a):
    global forcing, last_click, steps_0, n_steps, x_0, y_0, x, pinned
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            pinned = not pinned
            print('pinned: ', pinned)

def increase_h(a):
    global h, last_click
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            h += 0.001
            print('h: ', h)

def decrease_h(a):
    global h, last_click
    t = time.time()
    if not last_click:
        last_click = t
        return
    else:
        if t - last_click < 0.2:
            return
        else:
            last_click = t
            h -= 0.001
            print('h: ', h)

def pause(a):
    global paused
    paused = not paused
    print('pause: ', paused)

def speed_up(a):
    global n_sub_steps
    n_sub_steps = int(n_sub_steps*10)
    if n_sub_steps > 1000:
        n_sub_steps = 1000
    print('n_sub_steps: ', n_sub_steps)
    
def speed_down(a):
    global n_sub_steps
    n_sub_steps = int(n_sub_steps / 10)
    if n_sub_steps < 1:
        n_sub_steps = 1
    print('n_sub_steps: ', n_sub_steps)

vis.register_key_callback(85, fold)
vis.register_key_callback(74, unfold)
vis.register_key_callback(70, toggle_forcing)
vis.register_key_callback(80, toggle_pin)
vis.register_key_callback(73, increase_h)
vis.register_key_callback(75, decrease_h)
vis.register_key_callback(83, pause)
vis.register_key_callback(79, speed_up)
vis.register_key_callback(76, speed_down)

vis.run()
vis.destroy_window()



traj = np.array(traj[1:])
forces = np.array([f.numpy() for f in forces[1:]])

np.savez('tmp', traj=traj, forces=forces)

forces_mag = np.sum(forces ** 2, axis=1)
forces_mag = np.log(forces_mag)
f_m = np.max(forces_mag)

colors = np.array([f / f_m for f in forces_mag])

map = plt.get_cmap('rainbow')

colors = map(colors)

x = traj[:, 0]
y = traj[:, 1]
z = forces_mag

interval = force.steps_interval()

xx = x[interval-1::interval]
yy = y[interval-1::interval]
downsampled_x = torch.tensor(traj[interval-1::interval], dtype=torch.float)
zz = torch.tensor(z[interval-1::interval], dtype=torch.float).reshape(-1, 1)


# model = force_viewer.Model()
# model.fit(downsampled_x, zz)

fig = plt.figure(dpi=160)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz)
plt.show()

a = x[interval-1::interval]
b = y[interval-1::interval]
u = (forces[interval-1::interval])[:, 0]
v = (forces[interval-1::interval])[:, 1]
plt.quiver(a,b,u,v, color=colors)
plt.axis('equal')
plt.show()


