import numpy as np
import matplotlib.pyplot as plt
import json

name = 'car'
name = 'motorcycle'

with open('./data/outline/{}.json'.format(name)) as ifile:
    content = ifile.read()
    points = json.loads(content)
    points = np.array(points)[:, :2]
    segments = []
    for i in range(len(points)):
      i_next = (i+1) % len(points)
      p_0 = points[i]
      p_1 = points[i_next]
      seg = np.array([p_0, p_1])
      segments.append(seg)

# Mathematical function we need to plot
def z_func(xs, ys):
  zs = np.zeros_like(xs)
  for i in range(len(xs)):
    for j in range(len(xs[0])):
      p = np.array([xs[i,j], ys[i,j]])
      center = np.array([5, -4.5])

      n_left = 0
      n_right = 0
      for seg in segments:
        if (seg[0,1] - p[1]) * (seg[1,1] - p[1]) <= 0:
          t = (p[1] - seg[0,1]) / (seg[1,1] - seg[0,1]) 
          intersect = t * seg[1] + (1-t) * seg[0]
          if intersect[0] < p[0]:
            n_left += 1
          else:
            n_right += 1

      if n_left % 2 == 1 and n_right % 2 == 1:
        inside = True
      else:
        inside = False

      distance = np.linalg.norm(p - center)
      q = distance ** 2 + (not inside) * 2.5

      zs[i, j] = q 
    zs += np.random.normal(0, 0.05, [len(xs), len(xs[0])])
  return zs





# Setting up input values
x = np.arange(0, 10.0, 0.05)
y = np.arange(-9, 0, 0.05)
X, Y = np.meshgrid(x, y)
 
# Calculating the output and storing it in the array Z
Z = z_func(X, Y)

Z /= 10
 
im = plt.imshow(Z, cmap=plt.cm.coolwarm, extent=(0, 10, 0, -9), interpolation='bilinear')
 
plt.colorbar(im);
 
plt.title(name)
plt.axis('off')
 
plt.show()