import numpy as np

#V - cena opcji

#  initial conditions
N = 3  # punkty kolokacji
S_max = 3*10**10
T = 360  # horyzont czasowy
M = 5  # liczba kroków czasowych
E = 10
r = 0.1
sigma = 0.5
theta = 0.8
c = 3  # shape parameter (????)


phi = np.zeros(N)
dphi = np.zeros(N)
ddphi = np.zeros(N)

F1 = np.zeros([N+1, M+1])
H1 = np.zeros([N+1, M+1])
G1 = np.zeros([N+1, M+1])

dS = S_max / N
dt = T / M
time = np.arange(0, T+dt, dt)

S = np.arange(0, S_max+dS, dS)
V = np.zeros([N+1, M+1])
lamb = np.zeros([N+1, M+1])


t = T
i = np.where(time == t)[0][0]

V[:, i] = np.array([i-E if i > 0 else 0 for i in S])  # Call - (max(S-E, 0)), Put - (max(E-S,0))
#######################################################################
for j in range(0, N):
    phi[j] = np.sqrt(c**2+np.abs(S-S[j])**2)
    dphi[j] = 1/phi[j]*(S-S[j])
    ddphi[j] = 1/phi[j]**3*(c**2+abs(S-S[j])**2+(S-S[j]))

lamb[:, i] = 1/phi*V[:, i]

for j in range(0, N):
    F1[j] = 1/2 * sigma**2 * S**2*ddphi+ r*S*dphi - r*phi
    H1[j] = 1+ (1-theta)*dt* F1
    G1[j] = 1-theta*dt*F1

while t > 0:
    i = np.where(time == t)[0][0]
    lamb[:, i-1] = G1 / lamb[:, i]*H1
    t = t-dt

V[:, 0] = np.sum(lamb[:, 0]*phi[j])
