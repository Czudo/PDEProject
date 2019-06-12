import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#V - cena opcji

#  initial conditions
N = 100  # punkty kolokacji
S_max = 30
T = 0.5  # horyzont czasowy
M = 100  # liczba kroków czasowych
E = 10
r = 0.05
sigma = 0.2
theta = 0.02
c = 0.01  # shape parameter (????)
V = np.zeros([N, M])

phi = np.zeros([N, N])
R = np.zeros([N, N])

dS = S_max / N
dt = T / M

time = np.arange(dt, T+dt, dt)
S = np.arange(dS, S_max+dt, dS)
i = np.where(time == T)[0][0]

V[:, i] = np.array([k - E if k - E > 0 else 0 for k in S])  # Call - (max(S-E, 0)), Put - (max(E-S,0))

for i in range(N):
    for j in range(N):
        phi[i,j] = np.exp(-np.abs(S[i]-S[j])**2/c**2)
        R[i,j] = 1/2*sigma**2*S[i]**2*((4*(S[i]-S[j])**2-2*c**2)/c**4)*phi[i,j] + r*S[i]*(-2*(S[i]-S[j])/c**2)*phi[i,j] -r*phi[i,j]

M1 = np.matmul(phi, np.linalg.inv(phi-(1-theta)*dt*R))
M2 = np.matmul(phi+theta*dt*R, np.linalg.inv(phi))
M = np.linalg.inv(np.matmul(M1, M2))
for i in range(1, N):
    V[:, N-i-1] = np.matmul(V[:, N-i],M)

xs, ys = np.meshgrid(time, S)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, V, alpha=0.7, color='blue')
plt.show()
