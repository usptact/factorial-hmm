import numpy as np

from FHMMVariational import FHMMVariational

# -----------------------------
# Example usage
# -----------------------------
np.random.seed(10012025)
fhmm = FHMMVariational(n_chains=3, n_states=2)

# Simulate hidden states & additive emissions
T = 100
hidden = np.zeros((T, 3), dtype=int)
obs = np.zeros(T)
for c in range(3):
    z = np.zeros(T, dtype=int)
    # simple sticky HMM
    z[0] = np.random.choice(2, p=[0.05, 0.95])
    for t in range(1, T):
        z[t] = np.random.choice(2, p=[0.9 if z[t-1]==0 else 0.1, 0.1 if z[t-1]==0 else 0.9])
    hidden[:, c] = z
    obs += fhmm.means[c][z] + np.random.normal(0, np.sqrt(fhmm.vars[c][z]))

# Run variational inference with Viterbi decoding
posterior, viterbi_paths = fhmm.variational_inference(obs, n_iter=1000)

print("Observations:", obs)
print("True hidden states:\n", hidden)

for c in range(3):
    print(f"\nChain {c}:")
    print("Posterior probabilities (first 5 time steps):\n", posterior[c][:5])
    print("Viterbi path:", viterbi_paths[c])
