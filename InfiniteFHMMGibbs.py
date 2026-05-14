import numpy as np
from scipy.stats import norm, bernoulli
from tqdm import tqdm


class InfiniteFHMMGibbs:
    def __init__(self, alpha=1.0, n_states=2):
        self.alpha = alpha
        self.n_states = n_states
        self.chains = []
        self.Z = None
        self.T = 0

    class Chain:
        def __init__(self, n_states):
            self.n_states = n_states
            self.A = np.array([[0.9, 0.1], [0.1, 0.9]])
            self.pi = np.array([0.5, 0.5])
            self.mu = np.random.randn(n_states)
            self.var = np.random.rand(n_states) * 0.5 + 0.01
            self.X = None

    def initialize(self, obs, max_initial_chains=3):
        self.T = len(obs)
        K = max_initial_chains
        self.Z = np.ones((self.T, K), dtype=int)
        self.chains = [self.Chain(self.n_states) for _ in range(K)]
        for chain in self.chains:
            chain.X = np.random.choice(self.n_states, size=self.T)

    # ------------------------------------------------------------------
    # Forward Filtering Backward Sampling
    # ------------------------------------------------------------------

    def _ffbs(self, obs, k):
        """
        Sample a complete hidden-state sequence for chain k via FFBS.

        When Z[t,k]==0, chain k does not contribute to y(t); its state is
        unconstrained by the observation at t, so the emission log-likelihood
        is 0 (uniform over states). The Markov transition still propagates.
        """
        chain = self.chains[k]
        K = len(self.chains)
        T = self.T
        S = self.n_states
        log_A = np.log(np.maximum(chain.A, 1e-300))

        # Other active chains' contributions at each timestep
        other = np.zeros(T)
        for j in range(K):
            if j != k:
                cj = self.chains[j]
                other += self.Z[:, j] * cj.mu[cj.X]               # (T,)

        # Emission log-likelihoods: 0 when inactive (chain k doesn't contribute)
        log_L = np.zeros((T, S))
        active = self.Z[:, k] == 1
        if active.any():
            resid = obs[active] - other[active]
            for s in range(S):
                log_L[active, s] = norm.logpdf(
                    resid, loc=chain.mu[s], scale=np.sqrt(max(chain.var[s], 1e-10))
                )

        # Forward filter (normalize each step for numerical stability)
        log_alpha = np.empty((T, S))
        log_alpha[0] = np.log(np.maximum(chain.pi, 1e-300)) + log_L[0]
        log_alpha[0] -= np.logaddexp.reduce(log_alpha[0])

        for t in range(1, T):
            log_alpha[t] = log_L[t] + np.logaddexp.reduce(
                log_alpha[t - 1, :, None] + log_A, axis=0
            )
            log_alpha[t] -= np.logaddexp.reduce(log_alpha[t])

        # Backward sampling
        X = np.empty(T, dtype=int)
        X[T - 1] = np.random.choice(S, p=np.exp(log_alpha[T - 1]))
        for t in reversed(range(T - 1)):
            log_p = log_alpha[t] + log_A[:, X[t + 1]]
            log_p -= np.logaddexp.reduce(log_p)
            X[t] = np.random.choice(S, p=np.exp(log_p))

        return X

    # ------------------------------------------------------------------
    # Viterbi (all chains, inactive timesteps handled correctly)
    # ------------------------------------------------------------------

    def _viterbi_all(self, obs):
        """
        Viterbi decoding for all chains.

        At inactive timesteps (Z[t,k]==0) the emission log-likelihood is 0
        so the Viterbi still propagates via the transition matrix rather than
        stopping at -inf.
        """
        K = len(self.chains)
        T = self.T
        paths = []

        for k, chain in enumerate(self.chains):
            S = self.n_states
            log_A = np.log(np.maximum(chain.A, 1e-300))

            other = np.zeros(T)
            for j in range(K):
                if j != k:
                    cj = self.chains[j]
                    other += self.Z[:, j] * cj.mu[cj.X]

            # Emission log-likelihoods (0 when inactive)
            log_L = np.zeros((T, S))
            active = self.Z[:, k] == 1
            if active.any():
                resid = obs[active] - other[active]
                for s in range(S):
                    log_L[active, s] = norm.logpdf(
                        resid, loc=chain.mu[s], scale=np.sqrt(max(chain.var[s], 1e-10))
                    )

            delta = np.empty((T, S))
            psi = np.zeros((T, S), dtype=int)

            delta[0] = np.log(np.maximum(chain.pi, 1e-300)) + log_L[0]
            for t in range(1, T):
                scores = delta[t - 1, :, None] + log_A   # (S, S)
                psi[t] = np.argmax(scores, axis=0)
                delta[t] = np.max(scores, axis=0) + log_L[t]

            path = np.empty(T, dtype=int)
            path[T - 1] = np.argmax(delta[T - 1])
            for t in reversed(range(T - 1)):
                path[t] = psi[t + 1, path[t + 1]]

            paths.append(path)

        return paths

    # ------------------------------------------------------------------
    # Main Gibbs sampler
    # ------------------------------------------------------------------

    def gibbs_sample(self, obs, n_iter=20):
        T = self.T

        for _ in tqdm(range(n_iter), desc="Gibbs Sampling"):
            K = len(self.chains)

            # Precompute per-chain per-timestep emission means and variances
            # for efficient reuse across the Z sampling loop.
            chain_mu_t = np.array(
                [c.mu[c.X] for c in self.chains]
            ).T                                                # (T, K)
            chain_var_t = np.array(
                [c.var[c.X] for c in self.chains]
            ).T                                                # (T, K)

            # --------------------------------------------------------
            # Step 1: Sample Z[t, k] for all (t, k)
            # --------------------------------------------------------
            for k in range(K):
                m_k = self.Z[:, k].sum()
                for t in range(T):
                    m_minus = m_k - self.Z[t, k]
                    p_prior = np.clip(m_minus / T, 1e-8, 1 - 1e-8)

                    # Contribution from all chains except k
                    other_t = (self.Z[t] * chain_mu_t[t]).sum() - self.Z[t, k] * chain_mu_t[t, k]

                    # Total variance from other *active* chains
                    var_others = ((self.Z[t] * chain_var_t[t]).sum()
                                  - self.Z[t, k] * chain_var_t[t, k])
                    var_others = max(var_others, 1e-6)

                    var_k = chain_var_t[t, k]
                    mu_k = chain_mu_t[t, k]

                    # Likelihood when k is active vs inactive
                    log_like1 = norm.logpdf(obs[t], loc=other_t + mu_k,
                                            scale=np.sqrt(var_k + var_others))
                    log_like0 = norm.logpdf(obs[t], loc=other_t,
                                            scale=np.sqrt(var_others))

                    logp1 = np.log(p_prior) + log_like1
                    logp0 = np.log(1 - p_prior) + log_like0
                    max_log = max(logp0, logp1)
                    prob1 = np.exp(logp1 - max_log) / (
                        np.exp(logp0 - max_log) + np.exp(logp1 - max_log)
                    )
                    new_z = int(bernoulli.rvs(np.clip(prob1, 1e-8, 1 - 1e-8)))

                    # Keep running sum current for subsequent (t', k) within this k-sweep
                    m_k += new_z - self.Z[t, k]
                    self.Z[t, k] = new_z

            # --------------------------------------------------------
            # Step 2: Sample hidden states X via FFBS (respects Markov structure)
            # --------------------------------------------------------
            for k in range(K):
                self.chains[k].X = self._ffbs(obs, k)

            # Refresh cached values after X update
            chain_mu_t = np.array([c.mu[c.X] for c in self.chains]).T
            chain_var_t = np.array([c.var[c.X] for c in self.chains]).T

            # --------------------------------------------------------
            # Step 3: Update emission parameters (MLE on active timesteps)
            # --------------------------------------------------------
            for k, chain in enumerate(self.chains):
                for s in range(chain.n_states):
                    mask = (self.Z[:, k] == 1) & (chain.X == s)
                    n_obs = mask.sum()
                    if n_obs == 0:
                        continue
                    # Residual = obs - other active chains' contributions
                    other_contrib = (self.Z[mask] * chain_mu_t[mask]).sum(axis=1) - chain_mu_t[mask, k]
                    residuals = obs[mask] - other_contrib
                    chain.mu[s] = residuals.mean()
                    if n_obs >= 2:
                        chain.var[s] = max(residuals.var(), 1e-6)

            # --------------------------------------------------------
            # Step 4: Prune chains that are entirely inactive
            # --------------------------------------------------------
            keep = [k for k in range(K) if self.Z[:, k].any()]
            self.chains = [self.chains[k] for k in keep]
            self.Z = self.Z[:, keep]
            K = len(self.chains)

            # --------------------------------------------------------
            # Step 5: Propose new chains (IBP)
            # --------------------------------------------------------
            m_new = np.random.poisson(self.alpha / T)
            for _ in range(m_new):
                new_chain = self.Chain(self.n_states)
                new_chain.X = np.random.choice(self.n_states, size=T)
                self.chains.append(new_chain)
                self.Z = np.hstack([self.Z, np.ones((T, 1), dtype=int)])

        viterbi_paths = self._viterbi_all(obs)

        return (
            self.Z,
            [chain.X for chain in self.chains],
            [chain.mu for chain in self.chains],
            [chain.var for chain in self.chains],
            viterbi_paths,
        )
