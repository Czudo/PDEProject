import numpy as np
from scipy.stats import norm


def analytical(N, S_max, T, M, E, r, sigma):
    dS = S_max / N
    dt = T / M

    time = np.arange(dt, T + dt, dt)
    S = np.arange(dS, S_max+dS, dS)

    V = np.zeros([N, M])

    for i in range(N):
        for j in range(M):
            d1 = (np.log(S[i]/E)+(r+1/2*sigma**2)*(T-time[j]))/(sigma*np.sqrt(T-time[j]))
            d2 = d1-sigma*np.sqrt(T)
            V[i, j] = S[i]*norm.cdf(d1)-E*np.exp(r*(T-time[j]))*norm.cdf(d2)

    return V


if __name__ == '__main__':

    N = 121
    S_max = 30
    T = 0.5
    M = 100
    E = 10
    r = 0.05
    sigma = 0.2

    analytical(N, S_max, T, M, E, r, sigma)
