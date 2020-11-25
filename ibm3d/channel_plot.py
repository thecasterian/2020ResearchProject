import numpy as np
from stl.mesh import Mesh
from tvtk.api import tvtk
from mayavi import mlab
import matplotlib.pyplot as plt

Nx = 192
Ny = 160
Nz = 129

xf = np.linspace(-2*np.pi, 2*np.pi, Nx+1)
yf = np.linspace(-np.pi, np.pi, Ny+1)
zf = -np.cos(np.linspace(0, np.pi, Nz+1))

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2
zc = (zf[1:] + zf[:-1]) / 2

X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')

prefix = 'nurion/'
postfix = '-10800'

u1 = np.fromfile(f'{prefix}channel_u1{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[1:-1, 1:-1, 1:-1]
u2 = np.fromfile(f'{prefix}channel_u2{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[1:-1, 1:-1, 1:-1]
u3 = np.fromfile(f'{prefix}channel_u3{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[1:-1, 1:-1, 1:-1]
p = np.fromfile(f'{prefix}channel_p{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[1:-1, 1:-1, 1:-1]

V = np.sqrt(u1**2 + u2**2 + u3**2)
# V = p

pts = np.empty(Z.shape + (3,), dtype=float)
pts[..., 0] = X
pts[..., 1] = Y
pts[..., 2] = Z

pts = pts.transpose(2, 1, 0, 3).copy()
pts.shape = pts.size // 3, 3
V = V.T.copy()

sg = tvtk.StructuredGrid(dimensions=X.shape, points=pts)
sg.point_data.scalars = V.ravel()
sg.point_data.scalars.name = 'velocity magnitude'

d = mlab.pipeline.add_dataset(sg)

gz = mlab.pipeline.grid_plane(d)
gz.grid_plane.axis = 'y'

cut_plane = mlab.pipeline.scalar_cut_plane(d, plane_orientation='y_axes')
cut_plane.implicit_plane.origin = (0, 0, 0)
# cut_plane.implicit_plane.widget.enabled = False

mlab.show()

C = plt.contourf(xc, zc, V[:, Ny//2, :], 50)
plt.colorbar(C)
plt.axis('equal')
plt.show()

# plt.plot(zc, V[:, Ny//2, 0])
# plt.show()
