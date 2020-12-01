import numpy as np
from stl.mesh import Mesh
from tvtk.api import tvtk
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

Nx = 384
Ny = 204
Nz = 120

xf = np.linspace(-1, 31, Nx+1)
yf = np.linspace(-1, 16, Ny+1)
zf = np.linspace(-1, 9, Nz+1)

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2
zc = (zf[1:] + zf[:-1]) / 2

xc = xc[30:-30]
yc = yc[15:-15]
zc = zc[10:]

X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')

# read stl
m = Mesh.from_file("../stl/building3.stl")
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

# verts *= 25.4

# structured data
pts = np.empty(Z.shape + (3,), dtype=float)
pts[..., 0] = X
pts[..., 1] = Y
pts[..., 2] = Z

pts = pts.transpose(2, 1, 0, 3).copy()
pts.shape = pts.size // 3, 3


for i in range(5, 350, 5):
    # read data
    prefix = 'data/'
    postfix = f'-{i:05d}'

    u1 = np.fromfile(f'{prefix}building3_u1{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[31:-31, 16:-16, 11:-1]
    u2 = np.fromfile(f'{prefix}building3_u2{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[31:-31, 16:-16, 11:-1]
    u3 = np.fromfile(f'{prefix}building3_u3{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[31:-31, 16:-16, 11:-1]
    # p = np.fromfile(f'{prefix}building3_p{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))[31:-31, 16:-16, 11:-1]

    V = np.sqrt(u1**2 + u2**2 + u3**2)
    # V = p
    # V = u1
    # V = np.fromfile('data/lvset.out').reshape((Nx+2, Ny+2, Nz+2))[31:-16, 16:-16, 11:-1]
    # V = np.fromfile('data/flag.out', dtype=np.int32).reshape((Nx+2, Ny+2, Nz+2))[1:-1, 1:-1, 1:-1]

    U = V.copy()
    U[np.isnan(V)] = 0
    UU = gaussian_filter(U, sigma=0.75, truncate=1)
    W = 0*V.copy()+1
    W[np.isnan(V)]=0
    WW=gaussian_filter(W, sigma=0.75, truncate=1)
    V = UU/WW

    # plot
    mlab.figure(bgcolor=(0, 0, 0), size=(1600, 900))

    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=(0.7, 0.7, 0.7), opacity=1)

    sg = tvtk.StructuredGrid(dimensions=X.shape, points=pts)
    sg.point_data.scalars = V.T.copy().ravel()
    sg.point_data.scalars.name = 'velocity magnitude'

    d = mlab.pipeline.add_dataset(sg)

    # gz = mlab.pipeline.grid_plane(d)
    # gz.grid_plane.axis = 'y'

    cut_plane = mlab.pipeline.scalar_cut_plane(d, plane_orientation='z_axes', opacity=.6, vmax=2.25)
    cut_plane.implicit_plane.origin = (0, 0, 1.5)
    cut_plane.implicit_plane.widget.enabled = False

    cut_plane.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    cut_plane.update_pipeline()

    # mlab.colorbar(object=cut_plane)

    mlab.view(-90, 45, distance=40)

    # mlab.savefig(f'vel-horiz-{i:05d}.png')

    mlab.close()
