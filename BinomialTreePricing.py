from scipy.stats import norm
import numpy as np
from BinomialTree import BinomialTree

# computes option price using binom tree
def btree_pricer(S0, K, sig, r, T, N, op_type):
    nparts = int(T)*N + int(N * (T - np.floor(T)))
    prices_array = BinomialTree(S0, sig, T, N)
    dt = T/nparts
    u = np.exp(sig*np.sqrt(dt))
    d = 1/u
    disc = np.exp(-r*dt)
    q = (np.exp(r*dt) - d)/(u-d)

    # European Option
    if op_type == "ec" or op_type == "ep":
        S = prices_array[-1]       # good news, for European you only need the last day's prices
        V = 0

        # value of option at terminal time
        if op_type == "ec":
            V = np.maximum(S - K, np.zeros(nparts+1))
        elif op_type == "ep":
            V = np.maximum(K - S, np.zeros(nparts+1))

        # take 2 at a time and get weighted-discounted value
        for i in np.arange(nparts,0,-1):
            V = disc * ( q* V[1:i+1] + (1-q) * V[0:i])
        # final result is option price
        return V[0]

    # American Option
    elif op_type == "ac" or op_type == "ap":
        # terminal value of stock
        S = prices_array[nparts]

        V = 0
        # value of payoff at terminal time
        if op_type == "ac":
            V = np.maximum(0, S - K)
        elif op_type == "ap":
            V = np.maximum(0, K - S)

        # backtrack through options tree
        for i in np.arange(nparts-1, -1, -1):
            # recalculate prices at current level
            S = prices_array[i]

            V[:i+1] = disc * ( q*V[1:i+2] + (1-q)*V[0:i+1] )                
            V = V[:-1]

            if op_type == 'ap':
                V = np.maximum(V, K - S)
            elif op_type == "ac":
                V = np.maximum(V, S - K)
        return V[0]