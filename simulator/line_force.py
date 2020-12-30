import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import json
import time
import pdb




class LineForce:
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
        x = float(np.copy(x_0))
        y = float(np.copy(y_0))
        self.n_steps_0 = n_steps_0
        self.h = h
        self.interval = interval
        self.on = True
        
        vs = 0.001  # velocity along the trajectory
        self.traj = []
        r = self.r
        gap = 0.2
        
        n = 800
        step = 0.002
        step = 0.002 * 0.8
        
        for i in range(n):
            x += step
            y += step
            self.traj.append([x, y])

        for i in range(n * 2):
            x -= step
            y -= step
            self.traj.append([x, y])
            
            
    def steps_interval(self):
        return int(self.interval // self.h)
    
    def move(self, n_steps):
        if len(self.traj) == 0:
            return False
        x, y = self.traj[0][0], self.traj[0][1]
        self.traj = self.traj[1:]
        return x, y
