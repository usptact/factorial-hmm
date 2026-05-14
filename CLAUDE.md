# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Warning

The README notes: **"This code has been generated using AI and likely contains bugs."** Treat the implementations with appropriate skepticism; numerical edge cases and algorithmic correctness issues may exist.

## Setup

```bash
pip install numpy scipy tqdm
```

## Running the examples

```bash
python example_fhmm_variational.py   # Fixed-chain FHMM with variational inference
python example_fhmm_infinite.py      # Infinite FHMM with Gibbs sampling
```

There is no test suite, linter config, or build system.

## Known limitations

- **FHMMVariational**: Mean-field VI can get stuck in local optima when chains have overlapping or similarly-scaled signals. Chain permutation symmetry (label-switching) is not handled. 1-chain inference is verified to recover means and transition matrices accurately.
- **InfiniteFHMMGibbs**: The nonparametric Z model (per-timestep activity indicators) differs from the standard fixed-K FHMM. With poor initialization or insufficient iterations, chains can collapse or fail to separate. Emission parameters are updated by MLE rather than Bayesian posterior sampling, which underestimates parameter uncertainty.

## Architecture

Two independent, self-contained model classes — no shared base class or utility module.

### `FHMMVariational` (`FHMMVariational.py`)

Fixed-K FHMM using **variational EM**. Observation model: `y(t) = Σ_k μ_k[x_k(t)] + ε(t)`, with per-chain per-state emission variances.

- Parameters per chain: `pi` (initial dist), `A` (transition matrix), `means` (emission means), `vars` (emission variances). All are initialised at construction with offset means per chain to break symmetry.
- `variational_inference(obs, n_iter)`: alternates E-step and M-step.
  - **E-step**: coordinate-ascent over chains. For chain c, `other_means[t] = Σ_{d≠c} q[d]ᵀ μ_d` and `other_vars[t] = Σ_{d≠c} (E[σ²_d] + Var[μ_d])` — the second term (hidden-state uncertainty from other chains) is critical. Calls `_forward_backward` to get one-slice `γ` and two-slice `ξ` marginals.
  - **M-step** (batch): updates `means`, `vars`, `A`, `pi` for all chains simultaneously using the just-computed posteriors and residuals computed from old parameters.
- `_forward_backward`: log-space forward-backward; returns `(gamma, xi)` where xi is the (T-1, S, S) two-slice marginal used for the transition-matrix update.
- `_viterbi`: standard log-space Viterbi.

### `InfiniteFHMMGibbs` (`InfiniteFHMMGibbs.py`)

Nonparametric FHMM with IBP prior using **Gibbs sampling**. The number of chains K grows and shrinks during inference.

- `Chain` inner class: `A`, `pi`, `mu` (per-state means), `var` (per-state variances), `X` (current state sequence).
- `Z`: binary `[T, K]` activity matrix — whether chain k is active at time t.
- `gibbs_sample(obs, n_iter)` runs five steps per iteration:
  1. **Sample Z**: IBP prior `p = m_minus/T`; likelihood compares `N(obs; other + μ_k, var_k + var_others)` vs `N(obs; other, var_others)` using total variance of active chains (not a hardcoded constant).
  2. **Sample X via FFBS** (`_ffbs`): forward-filter using the HMM transition matrix, backward-sample. Emission log-likelihood is 0 at inactive timesteps so the Markov chain still propagates through gaps.
  3. **Update emissions**: MLE on residuals at active timesteps where state == s.
  4. **Prune inactive chains**: remove chains where `Z[:, k]` is all zeros.
  5. **Propose new chains**: Poisson(`alpha/T`) new chains, each initialised with all-ones Z column.
- `_viterbi_all`: Viterbi for all chains; uses emission log-likelihood 0 at inactive timesteps (rather than skipping and leaving delta at −∞).
- Returns `(Z, X_samples, mus, vars_, viterbi_paths)`.
