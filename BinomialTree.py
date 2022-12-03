import numpy as np

# generates multiplicative binomial tree 
def BinomialTree(S0, sig, T, N):
    nparts = int(T)*N + int(N * (T - np.floor(T)))
    dt = T/nparts
    u = np.exp(sig*np.sqrt(dt))
    d = 1/u

    prices_array = [np.array([S0])]
    for i in range(1, nparts+1):
        prices_array.append(S0 * (d ** np.arange(i, -1, -1) * (u ** np.arange(0, i+1, 1))))
    
    return prices_array