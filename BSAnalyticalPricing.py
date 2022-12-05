import numpy as np
from scipy.stats import norm

# calculates BS analytical price for Euro Call
def BSAnalytical(op_type, S, K, T, r, sigma):
    N = norm.cdf
    if op_type == "call":
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    else:
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma* np.sqrt(T)
        return K*np.exp(-r*T)*N(-d2) - S*N(-d1)