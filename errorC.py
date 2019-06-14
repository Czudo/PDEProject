from analytical import analytical
from meshfree import meshfree
import numpy as np
import matplotlib.pyplot as plt

N = 100
S_max = 100
T = 0.5
M = 100
E = 10
r = 0.05
sigma = 0.2
dS = S_max / N
dt = T / M
c = np.arange(0.01, 1.5, 0.02)

e = np.zeros(len(c))
theta = 0.5

aV = analytical(N, S_max, T, M, E, r, sigma)
for i in range(len(c)):
    mV = meshfree(N, S_max, T, M, E, r, sigma, theta, c[i])
    e[i] = np.max(np.abs(mV[:, 0]-aV[:, 0]))

fig = plt.figure()
plt.plot(c, e, '-o', label='error')
plt.xlabel(r'shape parameter, $c$')
plt.ylabel(r'max error, $e(c)$')
plt.show()
