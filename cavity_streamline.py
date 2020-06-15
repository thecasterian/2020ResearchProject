import numpy as np
import matplotlib.pyplot as plt

fp_in = open("cavity.in", "r")
Nx = fp_in.readline().split()[1]
x = list(map(float, fp_in.readline().split()))
Ny = fp_in.readline().split()[1]
y = list(map(float, fp_in.readline().split()))
fp_in.close()

Y, X = np.meshgrid(y, x)

psi = np.loadtxt("./result/cavity_result.txt")

plt.figure(figsize=(8, 8))
plt.contour(X, Y, psi,
            [-0.09, -0.07, -0.05, -0.03, -0.01, -0.001, 0, 0.0001, 0.00033],
            colors='k', linestyles='solid')
plt.show()
