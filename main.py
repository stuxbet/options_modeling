import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def monte_carlo_option(S0, K, T, r, sigma, n_sims=500_000):
    """Price via risk-neutral simulation (drift = r, not mu)"""
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)
    stderr = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, stderr

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return {
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'theta': -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2),
        'vega':  S * np.sqrt(T) * norm.pdf(d1),
        'rho':   K * T * np.exp(-r*T) * norm.cdf(d2),
    }

# Verify: Monte Carlo converges to Black-Scholes
S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2

bs = black_scholes(S, K, T, r, sigma)
mc, err = monte_carlo_option(S, K, T, r, sigma)
g = greeks(S, K, T, r, sigma)

print(f"Black-Scholes: ${bs:.4f}")
print(f"Monte Carlo:   ${mc:.4f} ± {err:.4f}")
print(f"Difference:    ${abs(bs - mc):.4f}\n")
for name, val in g.items():
    print(f"  {name:>6}: {val:.6f}")