import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.array([
  [2, 4, 0],
  [3, 4, 6],
  [3, 0, 10],
  [2, 2, 0],
  [4, 1, 1]
])

simplices = Delaunay(points).simplices

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(points[:, 0], points[:, 1], points[:,2], triangles=simplices, cmap=plt.cm.Spectral)

ax.set_zlim(-1, 11)

plt.show()