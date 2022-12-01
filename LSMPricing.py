import numpy as np
from MonteCarlo import simulate_gbm_paths

def lsm_pricer(OP_TYPE, N, S0, K, r, sig, T, discrete_freq, POLY_DEGREE):
    # simulate gbm paths
    sims = simulate_gbm_paths(N, S0, sig, r, T, discrete_freq).T

    nparts = int(T)*discrete_freq + int(discrete_freq * (T - np.floor(T)))
    # increments
    dt = T / nparts
    # discount factor
    discount_factor = np.exp(-r * dt)
    
    payoff_matrix = None
    if OP_TYPE == "put":
        payoff_matrix = np.maximum(K - sims, np.zeros_like(sims))
    elif OP_TYPE == "call":
        payoff_matrix = np.maximum(sims - K, np.zeros_like(sims))

    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]
    for t in range(nparts - 1, 0 , -1):
        regression = np.polyfit(sims[:, t], 
                                value_matrix[:, t + 1] *
                                discount_factor, 
                                POLY_DEGREE)
        continuation_value = np.polyval(regression, sims[:, t])
        value_matrix[:, t] = np.where(
            payoff_matrix[:, t] > continuation_value,
            payoff_matrix[:, t],
            value_matrix[:, t + 1] * discount_factor
        )
    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    return option_premium


