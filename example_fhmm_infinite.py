import numpy as np

from InfiniteFHMMGibbs import InfiniteFHMMGibbs

# Simulate 3 chains with additive Gaussian emissions (arbitrary variance)
T = 2000
true_chains = 3
obs = np.zeros(T)
hidden_states = []

for c in range(true_chains):
    X = np.random.choice([0,1], size=T)
    mu = np.array([0.0, 1.5])
    var = np.array([0.05, 0.2])  # different variance per state
    obs += mu[X] + np.random.normal(0, np.sqrt(var[X]))
    hidden_states.append(X)
hidden_states = np.array(hidden_states).T

# Initialize iFHMM
iFHMM = InfiniteFHMMGibbs(alpha=2.0, n_states=2)
iFHMM.initialize(obs, max_initial_chains=3)

# Run Gibbs sampler
Z, X_samples, mus, vars_, viterbi_paths = iFHMM.gibbs_sample(obs, n_iter=100)

#print("Observations:", obs)
print("Z shape:", Z.shape)
for k, (Xk, path, mu, var) in enumerate(zip(X_samples, viterbi_paths, mus, vars_)):
    print(f"Chain {k}:")
    print("  Sampled states:", Xk)
    print("  Viterbi path  :", path)
    print("  Learned means :", mu)
    print("  Learned vars  :", var)
