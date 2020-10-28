import numpy as np
from stl.mesh import Mesh
from tvtk.api import tvtk
from mayavi import mlab

# read input
f = open("ibm3d.in", "r")
mesh_file_name = f.readline().split()[1]
f.readline()
Nx = int(f.readline().split()[1])
xf = np.array(list(map(float, f.readline().split())))
Ny = int(f.readline().split()[1])
yf = np.array(list(map(float, f.readline().split())))
Nz = int(f.readline().split()[1])
zf = np.array(list(map(float, f.readline().split())))
f.close()

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2
zc = (zf[1:] + zf[:-1]) / 2

X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')

# read stl
m = Mesh.from_file(mesh_file_name)
verts_dict = {}
verts = []
tris = []
for v0, v1, v2 in zip(m.v0, m.v1, m.v2):
    v0 = tuple(v0)
    v1 = tuple(v1)
    v2 = tuple(v2)

    if v0 not in verts_dict:
        verts_dict[v0] = len(verts_dict)
        verts.append(v0)
    if v1 not in verts_dict:
        verts_dict[v1] = len(verts_dict)
        verts.append(v1)
    if v2 not in verts_dict:
        verts_dict[v2] = len(verts_dict)
        verts.append(v2)

    tris.append([verts_dict[v0], verts_dict[v1], verts_dict[v2]])

verts = np.array(verts)
tris = np.array(tris)

# read variables
u1 = np.fromfile('u1a.out').reshape((Nx+2, Ny+2, Nz+2))
u2 = np.fromfile('u2a.out').reshape((Nx+2, Ny+2, Nz+2))
u3 = np.fromfile('u3a.out').reshape((Nx+2, Ny+2, Nz+2))
p = np.fromfile('pa.out').reshape((Nx+2, Ny+2, Nz+2))
# lvset = np.fromfile('lvset.out').reshape((Nx+2, Ny+2, Nz+2))
V = np.sqrt(u1**2 + u2**2 + u3**2)[1:-1, 1:-1, 1:-1]

# plot
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=(0.7, 0.7, 0.7))

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
gz.grid_plane.axis = 'z'

cut_plane_y = mlab.pipeline.scalar_cut_plane(d, plane_orientation='y_axes')
cut_plane_y.implicit_plane.origin = (0, 0, 0)
cut_plane_y.implicit_plane.widget.enabled = False

mlab.show()
