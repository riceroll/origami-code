import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
import pdb


class Force:
    def __init__(self):
        self.x_0 = None     # the initial x value of the handle at the beginning of force applying
        self.y_0 = None     # the initial y value of the handle at the beginning of force applying
        self.n_steps_0 = None       # the number of steps passed at the beginning of force applying
        self.h = None       # time step
        self.interval = None
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
        l = 0.02     # gap size
        T = np.pi * 2 / w
        v = l / T   # polar radius velocity
        
        s = n_steps * vs
        t = np.sqrt(2 * s / v / w)
        x = v * t * np.cos(w * t)
        y = v * t * np.sin(w * t)
        
        x = x + self.x_0
        y = y + self.y_0
        
        return x, y