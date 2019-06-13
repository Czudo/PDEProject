from analytical import analytical
from meshfree import meshfree
import numpy as np
import matplotlib.pyplot as plt

N = 121  # punkty kolokacji
S_max = 30
T = 0.5  # horyzont czasowy
M = 100  # liczba kroków czasowych
E = 10
r = 0.05
sigma = 0.2

aV = analytical(N, S_max, T, M, E, r, sigma)

theta = 0.5
c = 0.32  # shape parameter (????)
mV = meshfree(N, S_max, T, M, E, r, sigma, theta, c)

dS = S_max / N
dt = T / M

time = np.arange(dt, T + dt, dt)
S = np.arange(dS, S_max + dS, dS)

xs, ys = np.meshgrid(time, S)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, aV, alpha=0.7, color='blue', label='analytical')
ax.scatter(xs, ys, mV, color='red', label='meshfree method')
plt.show()