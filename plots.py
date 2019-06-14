from analytical import analytical
from meshfree import meshfree
import numpy as np
import matplotlib.pyplot as plt

N = 20
S_max = 100
T = 0.5
M = 20
E = 1
r = 0.5
sigma = 0.2

theta = 0.25
h = (S_max-S_max/N)/(N-1)
c = 2*h

aV = analytical(N, S_max, T, M, E, r, sigma)
mV = meshfree(N, S_max, T, M, E, r, sigma, theta, c)

dS = S_max / N
dt = T / M

time = np.arange(dt, T + dt, dt)
S = np.arange(dS, S_max + dS, dS)

# 3D plot
xs, ys = np.meshgrid(time, S)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, aV, alpha=0.7, color='blue', label='analytical')
ax.plot_surface(xs, ys, mV, color='red', label='meshfree method')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$S$')
ax.set_zlabel(r'$V(S,t)$')
ax.view_init(azim=10)

plt.show()

e = np.abs(aV-mV)

# plot of error in t=0
fig = plt.figure()
plt.plot(S, e[:, 0], 'o')
plt.xlabel(r'$S$')
plt.ylabel(r'$e(S,t=0)$')
plt.show()