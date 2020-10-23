import numpy as np
from stl.mesh import Mesh
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

# read lvset
lvset = np.loadtxt("lvset.txt").reshape(Nx, Ny, Nz)

# plot
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=(0.7, 0.7, 0.7), opacity=0.5)

vc = mlab.volume_slice(X, Y, Z, lvset, plane_orientation='x_axes')
vc.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
vc.update_pipeline()

mlab.show()
