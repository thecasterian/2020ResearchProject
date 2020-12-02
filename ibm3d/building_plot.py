import numpy as np
from stl.mesh import Mesh
from tvtk.api import tvtk
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

Nx = 184
Ny = 126
Nz = 103

xf = np.array([0, 1.48600694085909, 2.65489267350826, 3.71751606682569, 4.6835373334779, 5.56173848497991, 6.36010316816356, 7.08588924378506, 7.74569476707732, 8.3455179700703, 8.890811790973, 9.38653344633909, 9.8371894966719, 10.2468768151563, 10.6193198319602, 10.9579043926911, 11.2657085388101, 11.5455304898274, 11.7999140816613, 12.0311718924193, 12.2414062658358, 12.432528423487, 12.6062758395337, 12.7642280359397, 12.9078209417634, 13.0383599470576, 13.1570317700524, 13.2649152455022, 13.3629911322747, 13.4521510293406, 13.5332054812188, 13.6068913465625, 13.673878496875, 13.73477590625, 13.7901371875, 13.840465625, 13.88621875, 13.9278125, 13.965625, 14, 14.03125, 14.0625, 14.09375, 14.125, 14.15625, 14.1875, 14.21875, 14.25, 14.28125, 14.3125, 14.34375, 14.375, 14.40625, 14.4375, 14.46875, 14.5, 14.53125, 14.5625, 14.59375, 14.625, 14.65625, 14.6875, 14.71875, 14.75, 14.78125, 14.8125, 14.84375, 14.875, 14.90625, 14.9375, 14.96875, 15, 15.03125, 15.0625, 15.09375, 15.125, 15.15625, 15.1875, 15.21875, 15.25, 15.28125, 15.3125, 15.34375, 15.375, 15.40625, 15.4375, 15.46875, 15.5, 15.53125, 15.5625, 15.59375, 15.625, 15.65625, 15.6875, 15.71875, 15.75, 15.78125, 15.8125, 15.84375, 15.875, 15.90625, 15.9375, 15.96875, 16, 16.03203125, 16.06486328125, 16.0985161132813, 16.1330102661133, 16.1683667727661, 16.2046071920853, 16.2417536218874, 16.2798287124346, 16.3188556802454, 16.3588583222516, 16.3998610303079, 16.4418888060656, 16.4849672762172, 16.5291227081226, 16.5743820258257, 16.6207728264713, 16.6683233971331, 16.7170627320615, 16.767020550363, 16.8182273141221, 16.8707142469751, 16.9245133531495, 16.9796574369782, 17.0361801229027, 17.0941158759753, 17.1535000228746, 17.2143687734465, 17.2767592427827, 17.3407094738522, 17.4062584606985, 17.473446172216, 17.5423135765214, 17.6129026659344, 17.6852564825828, 17.7594191446474, 17.8354358732636, 17.9133530200951, 17.9932180955975, 18.077076424875, 18.1651276706164, 18.2575814786448, 18.3546579770747, 18.4565883004261, 18.563615139945, 18.6759933214398, 18.7939904120095, 18.9178873571076, 19.0479791494606, 19.1845755314312, 19.3280017325004, 19.478599243623, 19.6367266303018, 19.8027603863145, 19.9770958301279, 20.1601480461319, 20.3523528729361, 20.5541679410805, 20.7660737626322, 20.9885748752614, 21.2222010435221, 21.4675085201958, 21.7250813707032, 21.995532863736, 22.2795069314204, 22.5776797024891, 22.8907611121111, 23.2194965922143, 23.5646688463226, 23.9270997131364, 24.3076521232908, 24.707232153953, 25.1267911861483, 25.5673281699533, 26.0298920029486, 26.5155840275937, 27.025560653471, 27.5610361106421, 28.1232853406719, 28.7136470322031, 29.3335268083108, 30])
yf = np.array([0, 0.845517970070288, 1.39081179097299, 1.88653344633908, 2.33718949667189, 2.74687681515627, 3.11931983196024, 3.45790439269113, 3.76570853881012, 4.04553048982738, 4.29991408166126, 4.53117189241932, 4.74140626583575, 4.93252842348704, 5.10627583953368, 5.26422803593971, 5.40782094176337, 5.53835994705761, 5.65703177005237, 5.76491524550216, 5.86299113227469, 5.95215102934063, 6.03320548121875, 6.1068913465625, 6.173878496875, 6.23477590625, 6.2901371875, 6.340465625, 6.38621875, 6.4278125, 6.465625, 6.5, 6.53125, 6.5625, 6.59375, 6.625, 6.65625, 6.6875, 6.71875, 6.75, 6.78125, 6.8125, 6.84375, 6.875, 6.90625, 6.9375, 6.96875, 7, 7.03125, 7.0625, 7.09375, 7.125, 7.15625, 7.1875, 7.21875, 7.25, 7.28125, 7.3125, 7.34375, 7.375, 7.40625, 7.4375, 7.46875, 7.5, 7.53125, 7.5625, 7.59375, 7.625, 7.65625, 7.6875, 7.71875, 7.75, 7.78125, 7.8125, 7.84375, 7.875, 7.90625, 7.9375, 7.96875, 8, 8.03125, 8.0625, 8.09375, 8.125, 8.15625, 8.1875, 8.21875, 8.25, 8.28125, 8.3125, 8.34375, 8.375, 8.40625, 8.4375, 8.46875, 8.5, 8.534375, 8.5721875, 8.61378125, 8.659534375, 8.7098628125, 8.76522409375, 8.826121503125, 8.8931086534375, 8.96679451878125, 9.04784897065937, 9.13700886772531, 9.23508475449784, 9.34296822994763, 9.46164005294239, 9.59217905823663, 9.73577196406029, 9.89372416046632, 10.067471576513, 10.2585937341643, 10.4688281075807, 10.7000859183387, 10.9544695101726, 11.2342914611899, 11.5420956073089, 11.8806801680398, 12.2531231848437, 12.6628105033281, 13.1134665536609, 13.609188209027, 14.1544820299297, 15])
zf = np.array([0, 0.0328125, 0.067265625, 0.10344140625, 0.1414259765625, 0.181309775390625, 0.223187764160156, 0.267159652368164, 0.313330134986572, 0.361809141735901, 0.412712098822696, 0.466160203763831, 0.522280713952022, 0.581207249649624, 0.643080112132105, 0.705459578052105, 0.767839043972105, 0.830218509892105, 0.892597975812105, 0.954977441732105, 1.01735690765211, 1.07973637357211, 1.14211583949211, 1.20449530541211, 1.26687477133211, 1.32925423725211, 1.39163370317211, 1.45401316909211, 1.51639263501211, 1.57877210093211, 1.64115156685211, 1.70353103277211, 1.76591049869211, 1.82828996461211, 1.89066943053211, 1.95304889645211, 2.01542836237211, 2.07780782829211, 2.14018729421211, 2.20256676013211, 2.26494622605211, 2.32732569197211, 2.38970515789211, 2.45208462381211, 2.51446408973211, 2.57684355565211, 2.63922302157211, 2.70160248749211, 2.7639819534121, 2.8263614193321, 2.8887408852521, 2.9511203511721, 3.0134998170921, 3.0758792830121, 3.1382587489321, 3.2006382148521, 3.2630176807721, 3.3253971466921, 3.3877766126121, 3.4501560785321, 3.5125355444521, 3.5749150103721, 3.6372944762921, 3.6996739422121, 3.7620534081321, 3.8244328740521, 3.8868123399721, 3.9491918058921, 4.0115712718121, 4.0770697110281, 4.1458430722049, 4.21805510144054, 4.29387773213796, 4.37349149437026, 4.45708594471417, 4.54486011757527, 4.63702299907943, 4.73379402465879, 4.83540360151713, 4.94209365721838, 5.05411821570469, 5.17174400211532, 5.29525107784648, 5.4249335073642, 5.56110005835781, 5.70407493690109, 5.85419855937154, 6.01182836296551, 6.17733965673918, 6.35112651520154, 6.53360271658701, 6.72520272804176, 6.92638274006924, 7.1376217526981, 7.3594227159584, 7.59231372738171, 7.83684928937619, 8.0936116294704, 8.36321208656931, 8.64629256652317, 8.94352707047473, 9.25562329962386, 9.58332434023045, 10])

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2
zc = (zf[1:] + zf[:-1]) / 2

xc = np.hstack(([2*xf[0]-xc[0]], xc, [2*xf[-1]-xc[-1]]))
yc = np.hstack(([2*yf[0]-yc[0]], yc, [2*yf[-1]-yc[-1]]))
zc = np.hstack(([2*zf[0]-zc[0]], zc, [2*zf[-1]-zc[-1]]))

X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')

# read stl
m = Mesh.from_file("../stl/building.stl")
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

# read data
prefix = 'data/'
postfix = '-10000'

u1 = np.fromfile(f'{prefix}building3_u1{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))
u2 = np.fromfile(f'{prefix}building3_u2{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))
u3 = np.fromfile(f'{prefix}building3_u3{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))
# p = np.fromfile(f'{prefix}building3_p{postfix}.out').reshape((Nx+2, Ny+2, Nz+2))

V = np.sqrt(u1**2 + u2**2 + u3**2)
# V = p
# V = u1
# V = np.fromfile('data/lvset.out').reshape((Nx+2, Ny+2, Nz+2))
# V = np.fromfile('data/flag.out', dtype=np.int32).reshape((Nx+2, Ny+2, Nz+2))

# U = V.copy()
# U[np.isnan(V)] = 0
# UU = gaussian_filter(U, sigma=0.75, truncate=1)
# W = 0*V.copy()+1
# W[np.isnan(V)]=0
# WW=gaussian_filter(W, sigma=0.75, truncate=1)
# V = UU/WW

# plot
mlab.figure(bgcolor=(0, 0, 0), size=(1600, 900))

mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=(0.7, 0.7, 0.7), opacity=1)

sg = tvtk.StructuredGrid(dimensions=X.shape, points=pts)
sg.point_data.scalars = V.T.copy().ravel()
sg.point_data.scalars.name = 'velocity magnitude'

d = mlab.pipeline.add_dataset(sg)

# gz = mlab.pipeline.grid_plane(d)
# gz.grid_plane.axis = 'y'

cut_plane = mlab.pipeline.scalar_cut_plane(d, plane_orientation='y_axes', opacity=1, vmax=1.5)
cut_plane.implicit_plane.origin = (0, 7.5, 2)
# cut_plane.implicit_plane.widget.enabled = False

cut_plane.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
cut_plane.update_pipeline()

mlab.colorbar(object=cut_plane)

mlab.view(-90, 45, distance=40)

mlab.show()
