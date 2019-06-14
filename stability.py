from meshfree import meshfreeStability
import numpy as np
import matplotlib.pyplot as plt

N = 100 # punkty kolokacji
S_max = 100
T = 0.5  # horyzont czasowy
M = 100 # liczba kroków czasowych
E = 1
r = 0.05
sigma = 0.2
I = np.identity(N)
theta = np.arange(0.1, 0.2, 0.1)
dt = T / M
stab = np.zeros([len(theta), N])
c = 0.5  # shape parameter

for i in range(len(theta)):
    M = meshfreeStability(N, S_max, T, M, E, r, sigma, theta, c)
    lamb =  np.linalg.eig(M)
    print(lamb)
    stab[i,:] = (1+theta[i]*dt*lamb)/(1-(1-theta[i])*dt*lamb)

fig = plt.figure()
plt.plot(theta, stab)
plt.show()