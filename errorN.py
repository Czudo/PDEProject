from analytical import analytical
from meshfree import meshfree
import numpy as np
import matplotlib.pyplot as plt

N = np.arange(50, 150, 10)
S_max = 100
T = 0.5
M = np.arange(50, 150, 10)
E = 10
r = 0.05
sigma = 0.2

e = np.zeros(len(N))
theta = 0.5

c = 0.5

for i in range(len(N)):
    aV = analytical(N[i], S_max, T, M[i], E, r, sigma)
    mV = meshfree(N[i], S_max, T, M[i], E, r, sigma, theta, c)
    dS = S_max / N[i]
    dt = T / M[i]
    e[i] = np.max(np.abs(mV[:,0]-aV[:,0]))

log_e = np.log10(e)
log_NM = np.log10(N*M)

m_log_e = np.mean(log_e)
m_log_NM = np.mean(log_NM)
a = np.sum((log_e-m_log_e)*(log_NM-m_log_NM))/np.sum((log_NM-m_log_NM)**2)
b = m_log_e - a*m_log_NM

fig = plt.figure()
plt.loglog(N*M, e, '-o', label='error')
plt.loglog(N*M, (N*M)**a*10**b, label='regression, a='+str(a)[0:5]+', b='+str(b)[0:4])
plt.legend()

plt.xlabel(r'number of points, $N\cdot M$')
plt.ylabel(r'max error, $e(N\cdot M)$')
plt.yticks([np.min(e), np.max(e)])
plt.show()
