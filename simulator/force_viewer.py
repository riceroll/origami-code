import numpy as np
from scipy.interpolate import CubicSpline
import open3d as o3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import pdb
from simulator import spiral_force


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 1)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def fit(self, x, z):
        for i in range(40):
            output = self.forward(x)
            loss = self.loss(z, output)
            print(loss.detach().numpy())
            loss.backward()
            self.optimizer.step()
            self.zero_grad()
        
        self.plot(x, z)
        
    def plot(self, x, z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        l = int(np.max([x[:, 0].abs().max().numpy(), x[:, 1].abs().max().numpy()])) + 2
        
        # Make data
        x = np.linspace(-l, l, 100)
        y = np.linspace(-l, l, 100)
        x, y = np.meshgrid(x, y)
        xy = torch.tensor(np.stack([x,y], axis=2), dtype=torch.float)
        
        z = self.forward(xy).reshape(len(x), len(y)).detach().numpy()
        
        # Plot the surface
        # ax.plot_surface(x, y, z, color='b')
        
        ax.scatter(x, y, z)
        ax.set_xlabe('X axis')
        ax.set_ylabe('Y axis')
        ax.set_zlabe('Z axis')
        
        plt.show()
        
        

