import numpy as np
import matplotlib.pyplot as plt

fp_in = open("cylinder.in", "r")
Nx = fp_in.readline().split()[1]
x = list(map(float, fp_in.readline().split()))
Ny = fp_in.readline().split()[1]
y = list(map(float, fp_in.readline().split()))
fp_in.close()

Y, X = np.meshgrid(y, x)

psi = np.loadtxt("cylinder_result.txt")

plt.figure(figsize=(8, 8))
plt.contour(X, Y, psi,
            10,
            colors='k', linestyles='solid')
plt.show()
