# Factorial Hidden Markov Models

Two implementations of Factorial Hidden Markov Models (FHMMs) for decomposing time series into contributions from multiple independent Markov chains.

## Overview

A Factorial HMM models an observed sequence $y(1), \ldots, y(T)$ as the sum of emissions from $K$ parallel hidden Markov chains:

$$y(t) = \sum_{k=1}^{K} \mu_k[x_k(t)] + \varepsilon(t)$$

where $x_k(t) \in \{0, \ldots, S-1\}$ is the hidden state of chain $k$ at time $t$, $\mu_k[s]$ is the emission mean for chain $k$ in state $s$, and $\varepsilon(t)$ is Gaussian noise.

Each chain is a Markov process with initial distribution $\pi_k$ and transition matrix $A_k$:

$$p(x_k(t) \mid x_k(t-1)) = A_k[x_k(t-1),\, x_k(t)]$$

The noise at time $t$ is Gaussian with variance that depends on all current hidden states:

$$\varepsilon(t) \mid \mathbf{x}(t) \sim \mathcal{N}\!\left(0,\; \sum_k \sigma^2_k[x_k(t)]\right)$$

---

## Model 1 — Fixed Chains: Variational EM (`FHMMVariational`)

### Mathematics

Exact posterior inference over all $K$ chains is intractable because the chains are coupled through the shared observation. The **mean-field** approximation factorises the posterior:

$$q(\mathbf{x}) = \prod_{k=1}^{K} q_k(x_k)$$

Inference alternates between an E-step (updating each $q_k$) and an M-step (updating parameters).

#### E-step: coordinate-ascent forward-backward

For chain $c$, the coordinate-ascent update runs the standard HMM forward-backward algorithm on an **effective observation** that subtracts the posterior-expected contributions of all other chains:

$$y^{\mathrm{eff}}_c(t) = y(t) - \sum_{d \neq c} \underbrace{\mathbb{E}_{q_d}[\mu_d(x_d(t))]}_{\mathbf{q}_d(t)^\top \boldsymbol{\mu}_d}$$

The effective emission variance for chain $c$ in state $s$ is:

$$\sigma^2_{\mathrm{eff},c}(t, s) = \sigma^2_c[s] + \sum_{d \neq c} \Bigl(\underbrace{\mathbb{E}_{q_d}[\sigma^2_d(x_d(t))]}_{\mathbf{q}_d(t)^\top \boldsymbol{\sigma}^2_d} + \underbrace{\operatorname{Var}_{q_d}[\mu_d(x_d(t))]}_{\mathbf{q}_d(t)^\top \boldsymbol{\mu}_d^{\circ 2} - (\mathbf{q}_d(t)^\top \boldsymbol{\mu}_d)^2}\Bigr)$$

The second variance term accounts for uncertainty about other chains' mean contributions; omitting it would make the effective likelihood overconfident.

The forward and backward recursions (in log space) are:

$$\log \alpha_c(t, j) = \log \mathcal{N}\!\left(y^{\mathrm{eff}}_c(t);\, \mu_c[j],\, \sigma^2_{\mathrm{eff},c}(t,j)\right) + \log \sum_i \alpha_c(t-1, i)\, A_c[i, j]$$

$$\log \beta_c(t, i) = \log \sum_j A_c[i, j]\, \mathcal{N}\!\left(y^{\mathrm{eff}}_c(t+1);\, \mu_c[j],\, \sigma^2_{\mathrm{eff},c}(t+1,j)\right)\, \beta_c(t+1, j)$$

The one-slice and two-slice posterior marginals are:

$$\gamma_c(t, s) = p(x_c(t) = s \mid \mathbf{y}) \propto \alpha_c(t, s)\, \beta_c(t, s)$$

$$\xi_c(t, i, j) = p(x_c(t)=i,\, x_c(t+1)=j \mid \mathbf{y}) \propto \alpha_c(t,i)\, A_c[i,j]\, \mathcal{N}(\cdot)\, \beta_c(t+1,j)$$

#### M-step: parameter updates

All chains are updated simultaneously using residuals computed from the pre-update parameters. Let $r_c(t) = y(t) - \sum_{d \neq c} \mathbf{q}_d(t)^\top \boldsymbol{\mu}_d$ be the residual for chain $c$. Then:

$$\mu_c[s] \leftarrow \frac{\sum_t \gamma_c(t,s)\, r_c(t)}{\sum_t \gamma_c(t,s)}, \qquad \sigma^2_c[s] \leftarrow \frac{\sum_t \gamma_c(t,s)\,(r_c(t) - \mu_c[s])^2}{\sum_t \gamma_c(t,s)}$$

$$A_c[i,j] \leftarrow \frac{\sum_t \xi_c(t,i,j)}{\sum_j \sum_t \xi_c(t,i,j)}, \qquad \pi_c \leftarrow \gamma_c(0, :)$$

### Usage

```python
import numpy as np
from FHMMVariational import FHMMVariational

np.random.seed(42)
fhmm = FHMMVariational(n_chains=3, n_states=2)

# Generate synthetic data from the model's initial parameters
T = 200
obs = np.zeros(T)
hidden = np.zeros((T, 3), dtype=int)
for c in range(3):
    z = np.zeros(T, dtype=int)
    z[0] = np.random.choice(2)
    for t in range(1, T):
        z[t] = np.random.choice(2, p=[0.9 if z[t-1]==0 else 0.1,
                                       0.1 if z[t-1]==0 else 0.9])
    hidden[:, c] = z
    obs += fhmm.means[c][z] + np.random.normal(0, np.sqrt(fhmm.vars[c][z]))

# Run variational EM
posterior, viterbi_paths = fhmm.variational_inference(obs, n_iter=200)

# posterior[c]       — (T, n_states) array of marginal probabilities for chain c
# viterbi_paths[c]   — (T,) most-likely state sequence for chain c
# fhmm.means[c]      — learned emission means for chain c
# fhmm.A[c]          — learned transition matrix for chain c

for c in range(3):
    print(f"Chain {c}: means={fhmm.means[c].round(3)}, "
          f"Viterbi[:5]={viterbi_paths[c][:5]}")
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `n_chains` | Number of parallel Markov chains $K$ |
| `n_states` | Number of states per chain $S$ |
| `n_iter` | Number of variational EM iterations |

---

## Model 2 — Infinite Chains: IBP-FHMM with Gibbs Sampling (`InfiniteFHMMGibbs`)

### Mathematics

This model does not fix $K$ in advance. Instead, a binary **activity matrix** $Z \in \{0,1\}^{T \times K}$ indicates whether chain $k$ is active at time $t$, and the observation model becomes:

$$y(t) = \sum_{k=1}^{K} Z_{t,k}\, \mu_k[x_k(t)] + \varepsilon(t)$$

#### Indian Buffet Process prior

The activity matrix is given an IBP-inspired prior. For each chain $k$ and timestep $t$, conditioned on all other entries in column $k$:

$$p(Z_{t,k} = 1 \mid Z_{-t,k}) = \frac{m_{-t,k}}{T}, \qquad m_{-t,k} = \sum_{t' \neq t} Z_{t',k}$$

New chains are proposed each iteration by drawing $m_{\mathrm{new}} \sim \mathrm{Poisson}(\alpha / T)$ and appending them with $Z_{:,k_{\mathrm{new}}} = \mathbf{1}$.

#### Gibbs sampling

The sampler alternates five steps per iteration.

**Step 1 — Sample $Z_{t,k}$.**  For each $(t, k)$, compute the IBP prior probability $p = m_{-t,k}/T$ and compare the marginal likelihoods under the two hypotheses:

$$\log p(y(t) \mid Z_{t,k}=1) = \log \mathcal{N}\!\left(y(t);\; \tilde{\mu}^{-k}(t) + \mu_k[x_k(t)],\; \sigma^2_k[x_k(t)] + \tilde{\sigma}^2_{-k}(t)\right)$$

$$\log p(y(t) \mid Z_{t,k}=0) = \log \mathcal{N}\!\left(y(t);\; \tilde{\mu}^{-k}(t),\; \tilde{\sigma}^2_{-k}(t)\right)$$

where $\tilde{\mu}^{-k}(t) = \sum_{j \neq k} Z_{t,j}\,\mu_j[x_j(t)]$ and $\tilde{\sigma}^2_{-k}(t) = \sum_{j \neq k} Z_{t,j}\,\sigma^2_j[x_j(t)]$ are the contributions from all other active chains.

**Step 2 — Sample $x_k$ via FFBS.**  For each chain $k$, sample a complete state trajectory using Forward Filtering Backward Sampling. The emission log-likelihood at timestep $t$ is:

$$\ell_k(t, s) = \begin{cases} \log \mathcal{N}\!\left(y(t) - \tilde{\mu}^{-k}(t);\; \mu_k[s],\; \sigma^2_k[s]\right) & \text{if } Z_{t,k} = 1 \\ 0 & \text{if } Z_{t,k} = 0 \end{cases}$$

The forward filter accumulates $\log \alpha_k(t, s)$ using the chain's transition matrix $A_k$, and the backward pass samples:

$$x_k(t) \sim p(x_k(t) = s \mid x_k(t+1),\, \mathbf{y}_{1:t}) \propto A_k[s,\, x_k(t+1)]\, \alpha_k(t, s)$$

**Step 3 — Update emission parameters.**  For each chain $k$ and state $s$, update by MLE on the residuals at timesteps where chain $k$ is active and in state $s$:

$$\mu_k[s] \leftarrow \operatorname{mean}_{t:\, Z_{t,k}=1,\, x_k(t)=s}\!\left[y(t) - \tilde{\mu}^{-k}(t)\right]$$

$$\sigma^2_k[s] \leftarrow \operatorname{var}_{t:\, Z_{t,k}=1,\, x_k(t)=s}\!\left[y(t) - \tilde{\mu}^{-k}(t)\right]$$

**Step 4 — Prune.**  Remove any chain $k$ for which $\sum_t Z_{t,k} = 0$.

**Step 5 — Propose new chains.**  Draw $m_{\mathrm{new}} \sim \mathrm{Poisson}(\alpha/T)$ and add that many new chains, each initialised with $Z_{:,k} = \mathbf{1}$.

### Usage

```python
import numpy as np
from InfiniteFHMMGibbs import InfiniteFHMMGibbs

np.random.seed(42)

# Generate synthetic data from 3 unknown chains
T = 500
obs = np.zeros(T)
for mu_low, mu_high in [(0.0, 2.0), (0.0, 1.5), (0.0, 1.0)]:
    X = np.random.choice([0, 1], size=T, p=[0.5, 0.5])
    obs += np.where(X == 0, mu_low, mu_high)
    obs += np.random.normal(0, 0.2, T)

# Initialise and run the sampler
iFHMM = InfiniteFHMMGibbs(alpha=3.0, n_states=2)
iFHMM.initialize(obs, max_initial_chains=4)

Z, X_samples, mus, vars_, viterbi_paths = iFHMM.gibbs_sample(obs, n_iter=200)

# Z              — (T, K) binary activity matrix for the K discovered chains
# X_samples[k]  — (T,) sampled state sequence for chain k
# mus[k]        — (n_states,) learned emission means for chain k
# vars_[k]      — (n_states,) learned emission variances for chain k
# viterbi_paths[k] — (T,) most-likely state sequence for chain k

print(f"Discovered {len(mus)} chains")
for k, (mu, var) in enumerate(zip(mus, vars_)):
    active_frac = Z[:, k].mean()
    print(f"  Chain {k}: means={mu.round(3)}, active={active_frac:.2f}")
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `alpha` | IBP concentration — higher values encourage more chains |
| `n_states` | Number of states per chain $S$ |
| `max_initial_chains` | Starting number of chains before the sampler adapts |
| `n_iter` | Number of Gibbs sampling iterations |

---

## Comparison

| | `FHMMVariational` | `InfiniteFHMMGibbs` |
|---|---|---|
| **Number of chains** | Fixed $K$ | Inferred from data |
| **Inference** | Variational EM (ELBO maximisation) | Gibbs sampling (MCMC) |
| **Hidden state sampling** | Forward-backward (soft, posterior marginals) | FFBS (hard sample per iteration) |
| **Parameter learning** | M-step closed-form updates | MLE on active residuals |
| **Chain activity** | All chains always active | Per-timestep binary $Z_{t,k}$ |
| **Speed** | Fast | Slower, scales with $K \times T$ |
| **Best for** | Known number of components, speed | Automatic discovery of $K$ |

---

## Installation

```bash
pip install numpy scipy tqdm
```

---

## References

- Ghahramani, Z. & Jordan, M. I. (1997). Factorial hidden Markov models. *Machine Learning*, 29(2–3), 245–273.
- Griffiths, T. L. & Ghahramani, Z. (2011). The Indian buffet process: An introduction and review. *Journal of Machine Learning Research*, 12, 1185–1224.
