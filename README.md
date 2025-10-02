# Factorial Hidden Markov Model (FHMM) Variants

This project implements two approaches to Factorial Hidden Markov Models for modeling time series data as a sum of multiple independent Markov chains.

## Overview

Factorial HMMs decompose observed data into contributions from multiple parallel hidden Markov chains. At each time step, the observation is the sum of emissions from all active chains plus noise:

```
y(t) = Σ_k μ_k[x_k(t)] + ε(t)
```


where:
- `y(t)` is the observation at time `t`
- `x_k(t)` is the hidden state of chain `k` at time `t`
- `μ_k[s]` is the emission mean for chain `k` in state `s`
- `ε(t)` is Gaussian noise

## Models

### 1. Fixed Chains with Variational Inference (`FHMMVariational`)

A standard FHMM with a **fixed number of chains** using mean-field variational inference for approximate posterior estimation.

**Features:**
- Fixed number of chains specified at initialization
- Each chain has fixed number of states
- Efficient variational inference via forward-backward algorithm
- Viterbi decoding for most likely state sequences

### 2. Infinite Indian Buffet Process FHMM (`InfiniteFHMMGibbs`)

A nonparametric Bayesian FHMM that **automatically infers the number of chains** from data using an Indian Buffet Process prior and Gibbs sampling.

**Features:**
- Automatically discovers number of chains
- Binary indicator matrix `Z` determines which chains are active at each time
- Gibbs sampling for full posterior inference
- Can add new chains during inference

---

## Installation

Ensure you have the required packages:
```shell script
pip install numpy scipy tqdm
```


---

## Usage

### 1. Fixed Chains FHMM (Variational Inference)

#### Generate Synthetic Data

```python
import numpy as np
from FHMMVariational import FHMMVariational

np.random.seed(42)

# Initialize model with 3 chains, 2 states per chain
fhmm = FHMMVariational(n_chains=3, n_states=2)

# Generate synthetic data
T = 100  # time steps
hidden = np.zeros((T, 3), dtype=int)
obs = np.zeros(T)

for c in range(3):
    z = np.zeros(T, dtype=int)
    # Simple sticky HMM: tend to stay in same state
    z[0] = np.random.choice(2, p=[0.05, 0.95])
    for t in range(1, T):
        z[t] = np.random.choice(2, p=[0.9 if z[t-1]==0 else 0.1, 
                                       0.1 if z[t-1]==0 else 0.9])
    hidden[:, c] = z
    # Add contribution from this chain
    obs += fhmm.means[c][z] + np.random.normal(0, np.sqrt(fhmm.vars[c][z]))
```


#### Run Inference

```python
# Perform variational inference
posterior, viterbi_paths = fhmm.variational_inference(obs, n_iter=100)

# View results
print("True hidden states:\n", hidden)

for c in range(3):
    print(f"\nChain {c}:")
    print(f"  Learned means: {fhmm.means[c]}")
    print(f"  Learned variances: {fhmm.vars[c]}")
    print(f"  Viterbi path: {viterbi_paths[c]}")
    print(f"  Posterior (first 5 steps):\n{posterior[c][:5]}")
```


**Output:**
- `posterior`: List of posterior probability matrices for each chain (shape: `[T, n_states]`)
- `viterbi_paths`: Most likely state sequence for each chain
- Model parameters are updated during inference

---

### 2. Infinite FHMM (Gibbs Sampling)

#### Generate Synthetic Data

```python
import numpy as np
from InfiniteFHMMGibbs import InfiniteFHMMGibbs

np.random.seed(42)

# Generate data from 3 unknown chains
T = 500
true_chains = 3
obs = np.zeros(T)
hidden_states = []

for c in range(true_chains):
    X = np.random.choice([0, 1], size=T)
    mu = np.array([0.0, 1.5])  # state means
    var = np.array([0.05, 0.2])  # state variances
    obs += mu[X] + np.random.normal(0, np.sqrt(var[X]))
    hidden_states.append(X)

hidden_states = np.array(hidden_states).T
```


#### Run Inference

```python
# Initialize with IBP prior
iFHMM = InfiniteFHMMGibbs(alpha=2.0, n_states=2)
iFHMM.initialize(obs, max_initial_chains=3)

# Run Gibbs sampler
Z, X_samples, mus, vars_, viterbi_paths = iFHMM.gibbs_sample(obs, n_iter=100)

# View results
print(f"Discovered {len(X_samples)} chains")
print(f"Activity matrix Z shape: {Z.shape}")

for k in range(len(X_samples)):
    print(f"\nChain {k}:")
    print(f"  Learned means: {mus[k]}")
    print(f"  Learned variances: {vars_[k]}")
    print(f"  Viterbi path: {viterbi_paths[k]}")
    print(f"  Active timesteps: {np.sum(Z[:, k])}/{T}")
```


**Output:**
- `Z`: Binary activity matrix (shape: `[T, K]`) indicating which chains are active
- `X_samples`: Sampled state sequences for each chain
- `mus`: Learned emission means for each chain
- `vars_`: Learned emission variances for each chain
- `viterbi_paths`: Most likely state sequences

---

## Key Differences

| Feature | Variational FHMM | Infinite FHMM |
|---------|------------------|---------------|
| **Number of chains** | Fixed | Inferred from data |
| **Inference method** | Mean-field variational | Gibbs sampling |
| **Computation** | Faster, approximate | Slower, full posterior |
| **Chain activity** | All chains always active | Chains can be inactive (Z=0) |
| **Prior** | None | Indian Buffet Process |
| **Best for** | Known structure, speed | Discovery, flexibility |

---

## Parameters

### FHMMVariational
- `n_chains`: Number of parallel chains (fixed)
- `n_states`: Number of states per chain
- `n_iter`: Number of variational inference iterations

### InfiniteFHMMGibbs
- `alpha`: IBP concentration parameter (controls number of chains)
- `n_states`: Number of states per chain
- `max_initial_chains`: Initial number of chains to start with
- `n_iter`: Number of Gibbs sampling iterations

---

## References

- Ghahramani, Z., & Jordan, M. I. (1997). Factorial hidden Markov models. *Machine Learning*.
- Griffiths, T. L., & Ghahramani, Z. (2011). The Indian buffet process: An introduction and review. *Journal of Machine Learning Research*.