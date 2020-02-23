import numpy as np
import matplotlib.pyplot as plt

psi = np.loadtxt("./cavity_result.txt")
n = psi.shape[0] - 1

x = np.linspace(0, 1, n+1)
y = np.linspace(0, 1, n+1)
Y, X = np.meshgrid(x, y)

plt.figure(figsize=(8, 8))
plt.contour(X, Y, psi,
            [-0.09, -0.07, -0.05, -0.03, -0.01, -0.001, 0, 0.0001, 0.00033],
            colors='k', linestyles='solid')
plt.show()
