import numpy as np
import matplotlib.pyplot as plt

# Read output files
fp_in = open("cylinder.in", "r")
Nx = int(fp_in.readline().split()[1])
xf = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
Ny = int(fp_in.readline().split()[1])
yf = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
fp_in.close()

xc = (xf[1:] + xf[:-1]) / 2
yc = (yf[1:] + yf[:-1]) / 2

X, Y = np.meshgrid(xc, yc)

u1 = np.loadtxt("u1.txt")[1:-1, 1:-1].T
u2 = np.loadtxt("u2.txt")[1:-1, 1:-1].T
p = np.loadtxt("p.txt")[1:-1, 1:-1].T

vel = np.sqrt(u1**2 + u2**2)

# Add mask
mask = np.zeros(u1.shape, dtype=bool)
mask[np.sqrt(X**2 + Y**2) < 0.5] = True
u1 = np.ma.array(u1, mask=mask)
u2 = np.ma.array(u2, mask=mask)

circle1 = plt.Circle((0, 0), 0.5, color='gray')
circle2 = plt.Circle((0, 0), 0.5, color='gray')

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Velocity')
C1 = ax1.contourf(X, Y, vel, 100, cmap='hot')
# ax1.streamplot(X, Y, u1, u2, color='k', linewidth=1)
ax1.set_ylim(yf[0], yf[-1])
ax1.set_aspect('equal', 'box')
ax1.add_artist(circle1)
fig.colorbar(C1, ax=ax1)

ax2.set_title('Pressure')
C2 = ax2.contourf(X, Y, p, 100, cmap='hot')
ax2.set_ylim(yf[0], yf[-1])
ax2.set_aspect('equal', 'box')
ax2.add_artist(circle2)
fig.colorbar(C2, ax=ax2)

plt.show()
