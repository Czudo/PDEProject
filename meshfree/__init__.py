import numpy as np


def meshfree(N, S_max, T, M, E, r, sigma, theta, c):
    V = np.zeros([N, M])

    phi = np.zeros([N, N])
    R = np.zeros([N, N])

    dS = S_max / N
    dt = T / M

    time = np.arange(dt, T+dt, dt)
    S = np.arange(dS, S_max+dS, dS)
    temp=S-E
    V[:, M-1] = [k if k > 0 else 0 for k in temp]  # Call - (max(S-E, 0)), Put - (max(E-S,0))
    for i in range(N):
        for j in range(N):
            phi[i, j] = np.exp(-np.abs(S[i]-S[j])**2/c**2)
            R[i, j] = 1/2*sigma**2*S[i]**2*((4*(S[i]-S[j])**2-2*c**2)/c**4)*phi[i,j] + r*S[i]*(-2*(S[i]-S[j])/c**2)*phi[i,j] +r*phi[i,j]

    M1 = np.matmul(phi, np.linalg.inv(phi-(1-theta)*dt*R))
    M2 = np.matmul(phi+theta*dt*R, np.linalg.inv(phi))
    Mac = np.matmul(M2, M1)

    for i in range(1, M):
        V[:, M-i-1] = np.matmul(Mac, V[:,M-i])

    return V

def meshfreeStability(N, S_max, T, M, E, r, sigma, theta, c):
    V = np.zeros([N, M])

    phi = np.zeros([N, N])
    R = np.zeros([N, N])

    dS = S_max / N
    dt = T / M

    time = np.arange(dt, T+dt, dt)
    S = np.arange(dS, S_max+dS, dS)
    temp=S-E
    V[:, M-1] = [k if k > 0 else 0 for k in temp]  # Call - (max(S-E, 0)), Put - (max(E-S,0))
    for i in range(N):
        for j in range(N):
            phi[i, j] = np.exp(-np.abs(S[i]-S[j])**2/c**2)
            R[i, j] = 1/2*sigma**2*S[i]**2*((4*(S[i]-S[j])**2-2*c**2)/c**4)*phi[i,j] + r*S[i]*(-2*(S[i]-S[j])/c**2)*phi[i,j] +r*phi[i,j]

    return np.matmul(R, np.linalg.inv(phi))


if __name__=='__main__':
    #  initial conditions
    N = 101  # punkty kolokacji
    S_max = 30
    T = 0.5  # horyzont czasowy
    M = 100  # liczba kroków czasowych
    E = 10
    r = 0.02
    sigma = 0.2
    #h = (S_max-S_max/M)/(N-1)
    #print(h)
    theta = 0.5
    c = 0.32  # shape parameter (????)
    meshfree(N, S_max, T, M, E, r, sigma, theta, c)
