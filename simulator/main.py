import sys
import time
import json
import yaml
import pdb

import numpy as np
import torch
import open3d as o3
import matplotlib.pyplot as plt

from simulator import spiral_force, circle_force, line_force, force_viewer


# ===========================================================================
# configuration
# ============================================================= <editor-fold>

pattern = sys.argv[1] if len(sys.argv) > 1 else 'triangle'
configuration = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1] if len(sys.argv) > 1 else 'default'

class Setting:
    def __init__(self):
        self.STIFFNESS_CHANGE = True
        self.paused = False
        self.n_sub_steps = 10
        self.t0_folding = 500
        self.T_folding = 8000
        self.p_fold = 0.6  # folding percentage
        self.pinned = False

        # force
        self.AUTO_FORCING = True
        self.FORCE_GAP_SCALE = 30
        
        # visual
        self.SHOW_FORCE = False
        self.SHOW_ENERGY = False
        self.FORCE_LENGTH_SCALE = 20
        self.FORCE_COLOR = np.array([[0.8, 0.1, 0.1]]).T
        self.color_map = plt.get_cmap('coolwarm')
        
        self.input_name = 'data/pattern/{}.npz'.format('triangle')
    
class Parameter:
    # simulation parameters
    def __init__(self):
        self.EA = 100
        self.k_fold = 1
        self.k_facet = 10
        self.k_face = 20000
        self.damping = 0.1
        self.h = 0.001

class Config:
    pattern_folder = 'data/pattern/'
    force_folder = 'data/force/'
    
    def __init__(self, param, setting, pattern='triangle', config=None):
        self.param = param
        self.setting = setting
        self.param_dict = None
        self.setting_dict = None
        
        self.pattern = pattern
        self.config = pattern if config is None else config
        self.pattern_dir = '{}{}.npz'.format(self.pattern_folder, self.pattern)
        self.force_dir = 'f_{}_{}.npz'.format(self.pattern, self.config)
        
    def load_config(self, profile_name=None):
        if profile_name is None:
            profile_name = self.pattern
        with open('data/config.yaml') as in_file:
            data = yaml.full_load(in_file.read())
            data = data[profile_name]
            if 'Param' in data:
                self.param_dict = data['Param']
            if 'Setting' in data:
                self.setting_dict = data['Setting']
        for key in self.param_dict.keys():
            exec('self.param.{} = self.param_dict["{}"]'.format(key, key))
        for key in self.setting_dict.keys():
            exec('self.setting.{} = self.setting_dict["{}"]'.format(key, key))

    def export(self):
        for key in self.param_dict.keys():
            exec('self.param_dict["{}"] = self.param.{}'.format(key, key))
        for key in self.setting_dict.keys():
            exec('self.setting_dict["{}"] = self.setting.{}'.format(key, key))
        # TODO: write into the file
        
        return False

setting = Setting()
param = Parameter()
config = Config(param, setting, pattern, "default")
config.load_config('default')
config.load_config(configuration)
# setting.__init__()
# param.__init__()

# </editor-fold>


# ===========================================================================
# init variables
# ============================================================= <editor-fold>
# time & step
num_steps = 0       # number of steps so far
t0 = time.time()    # initial time
t_prev = t0         # time of the previous click
step_prev = None    # i_step of the previous click

# simulation
p_fold = setting.p_fold
pinned = setting.pinned
forcing = False
forced = False

# force
# force = spiral_force.SpiralForce()
force = circle_force.CircleForce()
# force = line_force.LineForce()
force.gap *= setting.FORCE_GAP_SCALE

# recordings
traj = []       # trajectory of the handle
forces = []     # forces on the trajectory
f_traj = []     # visualize the force on the fly
frames = []

# loading
with open(config.pattern_dir) as f:
    if config.pattern_dir[-4] == json:
        c = f.read()
        data = json.loads(c)
    else:
        data = np.load(config.pattern_dir)

nodes_in = torch.tensor(data['nodes'], dtype=torch.float)  # nv x 3
edges_in = torch.tensor(data['edges'], dtype=torch.long)  # ne x 2
faces_in = torch.tensor(data['faces'], dtype=torch.long)  # nf x 3
k_edges = torch.tensor(data['k_edges']) if 'k_edges' in data else torch.zeros([len(edges_in), 1],
                                                                              dtype=torch.float32)  # ne x 1
rho_target = torch.tensor(data['target_angles'], dtype=torch.float)  # ne x 1
is_facet = torch.tensor(data['is_facet'], dtype=torch.int)  # ne x 1
is_boundary_nodes = torch.tensor(data['is_boundary_nodes'], dtype=torch.long)  # nv x 1
is_boundary_edges = torch.tensor(data['is_boundary_edges'], dtype=torch.int)  # ne x 1
i_handle = data['i_handle']     # ()

# </editor-fold>


# ===========================================================================
# assembling
# ============================================================= <editor-fold>

x = nodes_in.clone()
v = torch.zeros([len(x), 3])
f = torch.zeros([len(x), 3])

edges = edges_in.clone()
faces = faces_in.clone()
hinges_energy = torch.zeros([len(edges), 1])

# bar energy
M_bar = torch.zeros([len(edges), len(x)])
i_edges = torch.arange(len(edges))
M_bar[i_edges, edges[:, 1]] = 1     # x_2
M_bar[i_edges, edges[:, 0]] = -1    # x_1
d_0 = M_bar @ x
L_0 = d_0.norm(dim=1, keepdim=True)

# hinge energy
hinges = edges[is_boundary_edges.logical_not()]
rho_target = rho_target[is_boundary_edges.logical_not()]

k_crease = param.k_fold * is_facet[is_boundary_edges.logical_not()].logical_not() + \
           param.k_facet * is_facet[is_boundary_edges.logical_not()]
k_crease = k_crease.reshape(-1, 1)
if setting.STIFFNESS_CHANGE:
    k_crease = k_crease * k_edges[is_boundary_edges.logical_not()].reshape(-1, 1).to(torch.float32)

hinge_face_pairs = (
        (hinges.unsqueeze(1).unsqueeze(3) == faces.unsqueeze(0).unsqueeze(2))
        .sum(-1).sum(-1) == 2
).nonzero()   # (2 x n_hinges) x 2   [[hinge_id, neighbor_face_id]]
repeating_01 = torch.tensor([[0, 1]]).T.repeat(len(hinges), 1)
triplet_faces = torch.index_select(
    torch.cat([repeating_01, hinge_face_pairs], dim=1), 1, torch.tensor([1, 0, 2]))
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
    n_k, n_l = None, None
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

# </editor-fold>

def step():
    global f, num_steps, forcing, hinges_energy, p_fold
    f = torch.zeros([len(x), 3])
    
    # bar energy
    d = M_bar @ x
    d_norm = (d * d).sum(axis=1, keepdim=True).sqrt()
    f_bar_i = param.EA / L_0 * (d_norm - L_0) / d_norm * d
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
    
    f_hinge_k = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xk
    f_hinge_l = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xl
    f_hinge_i = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xi
    f_hinge_j = -k_crease * (rho - rho_target * p_fold).reshape(-1, 1) * par_rho_xj
    
    f.index_add_(0, nx_i, f_hinge_i)
    f.index_add_(0, nx_j, f_hinge_j)
    f.index_add_(0, nx_k, f_hinge_k)
    f.index_add_(0, nx_l, f_hinge_l)
    
    # info: hinge energy
    hinges_energy = ((rho - rho_target * p_fold) / np.pi) ** 2 * 30
    
    # update
    num_steps += 1
    
    a = f
    
    v.add_(a * param.h)
    v.mul_(1 - param.damping)

    x.add_(v)
    
    if pinned:
        x.add_(is_boundary_nodes.reshape(-1, 1) * v * -1)
        
    if forcing:
        x[i_handle].add_(v[i_handle].mul_(-1))
        ret = force.move(num_steps)
        if not ret:
            forcing = False
            print('stop forcing')
            
        else:
            x_handle, y_handle = ret
            reset_vec = x[i_handle] * torch.tensor([1, 1, 0])
            x[i_handle].add_(torch.tensor([x_handle, y_handle, 0]) - reset_vec)
            forces.append(f[i_handle])
            traj.append([x_handle, y_handle])

# ===========================================================================
# visualize
# ============================================================= <editor-fold>

# init visualizer   <editor-fold>
vis = o3.visualization.VisualizerWithKeyCallback()
vis.create_window()

render_opt = vis.get_render_option()
render_opt.mesh_show_back_face = True
render_opt.mesh_show_wireframe = True
render_opt.point_size = 8
render_opt.line_width = 10
render_opt.light_on = True

vector3d = lambda v: o3.utility.Vector3dVector(v)
vector3i = lambda v: o3.utility.Vector3iVector(v)
vector2i = lambda v: o3.utility.Vector2iVector(v)
LineSet = lambda v, e: o3.geometry.LineSet(points=vector3d(v), lines=vector2i(e))
PointCloud = lambda v: o3.geometry.PointCloud(points=vector3d(v))

# </editor-fold>

# init draw         <editor-fold>

# visualize nodes
vs = vector3d(x.detach().numpy())
pc = PointCloud(vs)
vis.add_geometry(pc)

# visualize edges
es = vector2i(edges.detach().numpy())
ls = LineSet(vs, es)
ls.colors = vector3d(setting.color_map(hinges_energy.reshape(-1))[:, :3])
if setting.SHOW_ENERGY:
    cc = np.zeros([len(ls.colors), 3])
    ls.colors = ls_colors = vector3d(cc.copy())
vis.add_geometry(ls)

# visualize forces
vs_force = vector3d(torch.cat([x, x + f * setting.FORCE_LENGTH_SCALE], 0).detach().numpy())
es_force = vector2i(torch.stack([torch.arange(len(x)), torch.arange(len(x), len(x)*2)]).T.detach().numpy())
ls_force = LineSet(vs_force, es_force)
ls_force.paint_uniform_color(setting.FORCE_COLOR)
if setting.SHOW_FORCE:
    vis.add_geometry(ls_force)

# </editor-fold>

def draw():
    vs = vector3d(x.detach().numpy())
    pc.points = vs
    vis.update_geometry(pc)
    
    ls.points = vs
    ls.lines = es
    c = np.zeros([len(es), 3])
    c[is_boundary_edges.logical_not()] = setting.color_map(hinges_energy * k_crease.reshape(-1))[:, :3]
    c[is_boundary_edges.logical_not()] = setting.color_map(
        ((k_crease ** (1 / 10) - k_crease.min()) / k_crease.max() * 256).reshape(-1))[:, :3]
    
    c *= np.logical_not((np.zeros(is_facet.shape[0]) + is_facet.numpy())).reshape(-1, 1)
    c += (np.zeros(is_facet.shape[0]) + is_facet.numpy()).reshape(-1, 1)
    
    ls.colors = vector3d(c)
    vis.update_geometry(ls)
    
    vs_force = vector3d(torch.cat([x, x + f * setting.FORCE_LENGTH_SCALE], 0).detach().numpy())
    ls_force.points = vs_force
    ls_force.paint_uniform_color(setting.FORCE_COLOR)
    if setting.SHOW_FORCE:
        vis.update_geometry(ls_force)

# </editor-fold>


# ===========================================================================
# callbacks
# ============================================================= <editor-fold>

def timerCallback(vis):
    global forcing, forced, f_traj, pinned
    
    if setting.AUTO_FORCING:
        t_stop = setting.t0_folding + setting.T_folding
        if num_steps >= setting.t0_folding and num_steps < t_stop and not forcing and not forced:
            forced = True
            pinned = True
            # h = 0.005
            forcing = True
            x_0 = x[i_handle][0]
            y_0 = x[i_handle][1]
            force.start(x_0, y_0, num_steps, param.h)
            f_traj = force.traj
            vs = vector3d(np.hstack([np.array(f_traj), np.zeros([len(f_traj), 1])]))
            
            es = vector2i(np.hstack([np.arange(len(vs)-1).reshape(-1, 1), (np.arange(len(vs)-1) + 1).reshape(-1, 1)]))
            ls = LineSet(vs, es)
            vis.add_geometry(ls)
            vis.reset_view_point(True)
            print('forcing')
            
        if num_steps >= t_stop and forcing:
            forcing = False
            print('stop forcing')
    
    vs = x.numpy().tolist()
    fs = faces.numpy().tolist()
    es = edges.numpy().tolist()
    frame = {
        'v': vs,
        'e': es,
        'f': fs
    }
    frames.append(frame)
    
    for i in range(setting.n_sub_steps):
        step()
    draw()
    
vis.register_animation_callback(timerCallback)

def fold(a):
    global step_last_click, p_fold
    t = time.time()
    try:
        step_last_click
    except:
        step_last_click = t
        return

    if t - step_last_click < 0.2:
        return
    else:
        step_last_click = t
        p_fold += 0.05
        p_fold = 0.99 if p_fold >= 1 else p_fold
        print('p_fold: ', p_fold)

def unfold(a):
    global step_last_click, p_fold
    t = time.time()
    try:
        step_last_click
    except:
        step_last_click = t
        return

    if t - step_last_click < 0.2:
        return
    else:
        step_last_click = t
        p_fold -= 0.05
        p_fold = 0 if p_fold < 0 else p_fold
        print('p_fold: ', p_fold)

def toggle_forcing(a):
    global forcing, step_last_click, num_steps, x_0, y_0, x
    t = time.time()
    if not step_last_click:
        step_last_click = t
        return
    else:
        if t - step_last_click < 0.2:
            return
        else:
            step_last_click = t
            forcing = not forcing

            x_0 = x[i_handle][0]
            y_0 = x[i_handle][1]
            force.start(x_0, y_0, num_steps, param.h)
            print('forcing: ', forcing)

def toggle_pin(a):
    global forcing, step_last_click, num_steps, x_0, y_0, x, pinned
    t = time.time()
    if not step_last_click:
        step_last_click = t
        return
    else:
        if t - step_last_click < 0.2:
            return
        else:
            step_last_click = t
            pinned = not pinned
            print('pinned: ', pinned)

def increase_h(a):
    global step_last_click
    t = time.time()
    try:
        step_last_click
    except:
        step_last_click = t
        return

    if t - step_last_click < 0.2:
        return
    else:
        step_last_click = t
        param.h += 0.001
        print('h: ', param.h)
            
def decrease_h(a):
    global step_last_click
    t = time.time()
    try:
        step_last_click
    except:
        step_last_click = t
        return
    
    if t - step_last_click < 0.2:
        return
    else:
        step_last_click = t
        param.h -= 0.001
        print('h: ', param.h)

def pause(a):
    setting.paused = not setting.paused
    print('pause: ', setting.paused)

def speed_up(a):
    setting.n_sub_steps = int(setting.n_sub_steps*10)
    if setting.n_sub_steps > 1000:
        setting.n_sub_steps = 1000
    print('n_sub_steps: ', setting.n_sub_steps)
    
def speed_down(a):
    setting.n_sub_steps = int(setting.n_sub_steps / 10)
    if setting.n_sub_steps < 1:
        setting.n_sub_steps = 1
    print('n_sub_steps: ', setting.n_sub_steps)

vis.register_key_callback(85, fold)
vis.register_key_callback(74, unfold)
vis.register_key_callback(70, toggle_forcing)
vis.register_key_callback(80, toggle_pin)
vis.register_key_callback(73, increase_h)
vis.register_key_callback(75, decrease_h)
vis.register_key_callback(83, pause)
vis.register_key_callback(79, speed_up)
vis.register_key_callback(76, speed_down)

# </editor-fold>


vis.run()
vis.destroy_window()


# ===========================================================================
# visualize force
# ============================================================= <editor-fold>

traj = np.array(traj[1:])
forces = np.array([f.numpy() for f in forces[1:]])

np.savez(config.pattern_dir+'_force', traj=traj, forces=forces)

def visualize():
    forces_mag = np.sum(forces ** 2, axis=1)
    forces_mag = np.log(forces_mag)
    f_m = np.max(forces_mag)
    
    colors = np.array([f / f_m for f in forces_mag])
    colors = setting.color_map(colors)
    
    x = traj[:, 0]
    y = traj[:, 1]
    z = forces_mag
    
    interval = force.steps_interval()
    
    xx = x[interval - 1::interval]
    yy = y[interval - 1::interval]
    zz = torch.tensor(z[interval - 1::interval], dtype=torch.float).reshape(-1, 1)
    
    fig = plt.figure(dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_zlim3d(3, 10)
    plt.show()
    
    a = x[interval-1::interval]
    b = y[interval-1::interval]
    u = (forces[interval-1::interval])[:, 0]
    v = (forces[interval-1::interval])[:, 1]
    plt.quiver(a, b, u, v, color=colors)
    plt.axis('equal')
    plt.show()
    
visualize()

def save_video(name='car'):
    for i, frame in enumerate(frames):
        with open('./video/{}/frame_{}.json'.format(name, str(i)), 'w') as ofile:
            data = json.dumps(frame)
            ofile.write(data)

# </editor-fold>