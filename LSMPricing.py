import numpy as np
from MonteCarlo import simulate_gbm_paths

def lsm_pricer(OP_TYPE, N, S0, K, r, sig, T, discrete_freq, degree):
    # simulate gbm paths
    sims = simulate_gbm_paths(N, S0, sig, r, T, discrete_freq).T

    nparts = int(T)*discrete_freq + int(discrete_freq * (T - np.floor(T)))
    # increments
    dt = T / nparts
    # discount factor
    discount = np.exp(-r * dt)
    
    payoffs = None
    if OP_TYPE == "ec":
        return np.exp(-r*T)*np.average(np.maximum(sims[:,-1] - K, 0))
    elif OP_TYPE == "ep":
        return np.exp(-r*T)*np.average(np.maximum(K - sims[:,-1], 0))
    elif OP_TYPE == "ap":
        payoffs = np.maximum(K - sims, np.zeros_like(sims))
    elif OP_TYPE == "ac":
        payoffs = np.maximum(sims - K, np.zeros_like(sims))

    value_matrix = np.zeros_like(payoffs)
    value_matrix[:, -1] = payoffs[:, -1]
    for t in range(nparts - 1, 0 , -1):
        regression = np.polyfit(sims[:, t], value_matrix[:, t + 1] * discount, degree)
        cv = np.polyval(regression, sims[:, t])
        value_matrix[:, t] = np.where(payoffs[:, t] > cv, payoffs[:, t], value_matrix[:, t + 1] * discount)
    option_premium = np.mean(value_matrix[:, 1] * discount)
    return option_premium


