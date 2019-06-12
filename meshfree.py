import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#V - cena opcji

#  initial conditions
N = 121  # punkty kolokacji
S_max = 30
T = 0.5  # horyzont czasowy
M = 100  # liczba kroków czasowych
E = 10
r = 0.05
sigma = 0.2
theta = 0.5
c = 0.01  # shape parameter (????)


phi = np.zeros([N, M])
dphi = np.zeros([N, M])
ddphi = np.zeros([N, M])

F1 = np.zeros([N, M])
H1 = np.zeros([N, M])
G1 = np.zeros([N, M])

dS = S_max / N
dt = T / M

time = np.arange(dt, T + dt, dt)
S = np.arange(0, S_max, dS)

V = np.zeros([N, M])
lamb = np.zeros([N, M])

i = np.where(time == T)[0][0]

V[:, i] = np.array([k - E if k - E > 0 else 0 for k in S])  # Call - (max(S-E, 0)), Put - (max(E-S,0))
print([k - E if k -E > 0 else 0 for k in S])
#######################################################################
#?????????????????????????????????????????????????????????????????????
for j in range(M):
    phi[:, j] = np.sqrt(c**2 + np.abs(S - S[j])**2)
    dphi[:, j] = (c**2 + np.abs(S - S[j])**2)**(-1/2) * (S - S[j])
    ddphi[:, j] = (c**2 + np.abs(S - S[j])**2)**(-3/2) * (c**2+abs(S-S[j])**2 + (S-S[j]))
#?????????????????????????????????????????????????????????????????????

lamb[:, i] = 1/phi[:, i]*V[:, i]

for j in range(M):
    F1[:, j] = 1/2*sigma**2*S**2*ddphi[:, j] + r*S*dphi[:, j] - r*phi[:, j]
    H1[:, j] = 1 + (1-theta)*dt*F1[:, j]
    G1[:, j] = 1 - theta*dt*F1[:, j]


for t in time:
    if t!=T:
        i = np.where(time == t)[0][0]
        print(i)
        lamb[:, i] = G1[:, i+1] / lamb[:, i+1]*H1[:, i+1]


V[:, 0] = np.sum(lamb[:, 0]*phi[:, 0])
print(V[:, 0] )

xs, ys = np.meshgrid(time, S)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, V, alpha=0.7, color='blue')
plt.show()

plt.figure()
plt.plot(S,V[:,0])
plt.show()