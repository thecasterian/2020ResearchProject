import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton

data = nc.Dataset("/home/jeonukim/data/sphere/sphere-Re50.nc", "r")

x = data["x"][:]
y = data["y"][:]
z = data["z"][:]

Nx = len(x)
Ny = len(y)
Nz = len(z)

u = data["u"][0, :, :, :].T
w = data["w"][0, :, :, :].T

uc = np.mean(u[:, Ny//2-1:Ny//2+1, Nz//2-1:Nz//2+1], axis=(1, 2))
fu = interp1d(x, uc, kind='linear')
x0 = newton(fu, 1)

print(x0-0.5)

plt.plot(x, uc, [x0], [0], 'ro')
plt.grid(True)

plt.show()
