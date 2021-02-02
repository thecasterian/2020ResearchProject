import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

data = nc.Dataset("/home/jeonukim/data/channel/channel-17300.nc", "r")

Re = 3300

x = data["x"][2:-2]
y = data["y"][2:-2]
z = data["z"][2:-2]

Nx = len(x)
Ny = len(y)
Nz = len(z)

u = data["u"][0, 2:-2, 2:-2, 2:-2].T
v = data["v"][0, 2:-2, 2:-2, 2:-2].T
w = data["w"][0, 2:-2, 2:-2, 2:-2].T

z0 = z[0] + 1
z1 = z[1] + 1

ugrad = (z1**2*u[:, :, 0] - z0**2*u[:, :, 1]) / (z0*z1*(z1-z0))
wall_fric_vel = np.mean(np.sqrt(Re * ugrad))

uplus = np.mean(u, axis=(0, 1)) / wall_fric_vel
yplus = Re * (z+1)*wall_fric_vel

plt.loglog(yplus[:5], uplus[:5])
# plt.plot(z, np.mean(u, axis=(0, 1)))
plt.show()
