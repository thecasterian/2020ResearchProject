import numpy as np
import matplotlib.pyplot as plt

fp_in = open("cylinder.in", "r")
Nx = int(fp_in.readline().split()[1])
x = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
Ny = int(fp_in.readline().split()[1])
y = np.fromiter(map(float, fp_in.readline().split()), dtype=np.float64)
fp_in.close()

x = (x[1:] + x[:-1]) / 2
y = (y[1:] + y[:-1]) / 2

Y, X = np.meshgrid(y, x)

u1 = np.loadtxt("u1.txt")[1:-1, 1:-1]
u2 = np.loadtxt("u2.txt")[1:-1, 1:-1]
p = np.loadtxt("p.txt")[1:-1, 1:-1]

norm = np.sqrt(u1**2 + u2**2)
norm[np.sqrt(X**2 + Y**2) < 0.5] = np.nan
u1 /= norm
u2 /= norm

circle1 = plt.Circle((0, 0), 0.5, color='gray')
circle2 = plt.Circle((0, 0), 0.5, color='gray')

plt.figure(figsize=(8, 8))
plt.axis('equal')
plt.quiver(X, Y, u1, u2, scale=60)
plt.gcf().gca().add_artist(circle1)

plt.figure(figsize=(8, 8))
plt.axis('equal')
plt.contourf(X, Y, p)
plt.gcf().gca().add_artist(circle2)

plt.show()
