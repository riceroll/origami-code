import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
import pdb


class SpiralForce:
    def __init__(self):
        self.x_0 = None     # the initial x value of the handle at the beginning of force applying
        self.y_0 = None     # the initial y value of the handle at the beginning of force applying
        self.n_steps_0 = None       # the number of steps passed at the beginning of force applying
        self.h = None       # time step
        self.interval = None
        self.gap = 0.02     # gap size
        self.on = False

    def start(self, x_0, y_0, n_steps_0, h, interval=0.02):
        self.x_0 = np.copy(x_0)
        self.y_0 = np.copy(y_0)
        self.n_steps_0 = n_steps_0
        self.h = h
        self.interval = interval
        self.on = True
        
    def steps_interval(self):
        return int(self.interval // self.h)

    def move(self, n_steps):
        # n_steps: current time steps
        n_steps = (n_steps - self.n_steps_0) // (self.interval // self.h) + 1
        
        vs = 0.01  # velocity along the trajectory
        w = np.pi * 2 / 10  # polar angular velocity
        T = np.pi * 2 / w
        v = self.gap / T   # polar radius velocity
        
        s = n_steps * vs
        t = np.sqrt(2 * s / v / w)
        x = v * t * np.cos(w * t)
        y = v * t * np.sin(w * t)
        
        x = x + self.x_0
        y = y + self.y_0
        
        return x, y


class CircleForce:
    def __init__(self):
        self.x_0 = None  # the initial x value of the handle at the beginning of force applying
        self.y_0 = None  # the initial y value of the handle at the beginning of force applying
        self.n_steps_0 = None  # the number of steps passed at the beginning of force applying
        self.h = None  # time step
        self.interval = None
        self.gap = 0.1  # radius gap
        self.r = 0.1    # initial radius
        self.n_circles = 5 # number of circles
        self.on = False
        self.traj = []
    
    def start(self, x_0, y_0, n_steps_0, h, interval=0.02):
        self.x_0 = np.copy(x_0)
        self.y_0 = np.copy(y_0)
        x = np.copy(x_0)
        y = np.copy(y_0)
        self.n_steps_0 = n_steps_0
        self.h = h
        self.interval = interval
        self.on = True
        
        vs = 0.001  # velocity along the trajectory
        self.traj = []
        r = self.r
        gap = 0.2
        
        for i in range(self.n_circles):
            n_move = int( (r + self.x_0 - x ) // vs)
            dx = (r + self.x_0 - x) / n_move
            
            for j in range(n_move):
                x += dx
                self.traj.append([float(x), float(y)])
                # import pdb
                # pdb.set_trace()
            
            n_move = int(np.pi * r * 2 // vs)
            theta = 0
            for j in range(n_move):
                theta += 2 * np.pi / n_move
                x = np.cos(theta) * r + self.x_0
                y = np.sin(theta) * r + self.y_0
                self.traj.append([x, y])
                
            r += gap
            
            
    def steps_interval(self):
        return int(self.interval // self.h)
    
    def move(self, n_steps):
        if len(self.traj) == 0:
            return False
        x, y = self.traj[0][0], self.traj[0][1]
        self.traj = self.traj[1:]
        return x, y
