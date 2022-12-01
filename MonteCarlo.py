import numpy as np

# simulates GBM paths
def simulate_gbm_paths(nsims, S0, sig, mu, T, discrete_freq):
    '''
    nsims: number of simulations to produce
    S0: initial stock price
    sig: volatility expressed as a percentage
    mu: annualized drift expressed as a percentage
    r: interest rate
    T: time in years
    discrete_freq: number of discrete time intervals per increment of T (252 would be 1 trading year)
    '''
    nparts = int(T)*discrete_freq + int(discrete_freq * (T - np.floor(T)))
    dt = T / nparts
    Xt = np.log(S0) + np.cumsum(( (mu - sig**2/2)*dt + sig*np.sqrt(dt) * np.random.normal(size=(nparts,nsims)) ), axis=0)
    Xt = np.exp(Xt)
    Xt = np.vstack([np.repeat(S0, nsims), Xt])

    return Xt