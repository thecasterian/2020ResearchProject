import numpy as np
import matplotlib.pyplot as plt

# Read output files
fp_in = open("mesh.in", "r")
Nx = int(fp_in.readline().split()[1])
xf = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
Ny = int(fp_in.readline().split()[1])
yf = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
fp_in.close()

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2

X, Y = np.meshgrid(xc, yc)

lvset = np.loadtxt("lvset.out").T

plt.figure()
plt.contour(X, Y, lvset, 100, cmap='hot')
plt.axis('equal')
plt.show()
