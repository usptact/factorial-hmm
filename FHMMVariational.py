import numpy as np
from tqdm import tqdm
from scipy.stats import norm


class FHMMVariational:
    def __init__(self, n_chains=3, n_states=2):
        self.n_chains = n_chains
        self.n_states = n_states

        self.pi = [np.full(n_states, 1.0 / n_states) for _ in range(n_chains)]
        self.A = [np.full((n_states, n_states), 1.0 / n_states) for _ in range(n_chains)]
        # Offset initial means per chain to break symmetry
        self.means = [np.linspace(c * 0.5, c * 0.5 + 1.0, n_states) for c in range(n_chains)]
        self.vars = [np.full(n_states, 0.1) for _ in range(n_chains)]

    def _forward_backward(self, obs, chain_idx, other_means, other_vars):
        """
        Log-space forward-backward for one chain given contributions from other chains.

        other_means[t] = E_q[sum_{d!=c} mu_d(x_d(t))]
        other_vars[t]  = E_q[sigma^2_{d!=c}] + Var_q[mu_{d!=c}]   (caller must supply both)

        Returns:
            gamma : (T, S) one-slice posterior marginals
            xi    : (T-1, S, S) two-slice posterior marginals (for transition update)
        """
        T = len(obs)
        S = self.n_states
        mu = self.means[chain_idx]
        var = self.vars[chain_idx]
        log_A = np.log(np.maximum(self.A[chain_idx], 1e-300))

        obs_eff = obs - other_means                           # (T,)
        var_eff = var[None, :] + other_vars[:, None]          # (T, S)
        var_eff = np.maximum(var_eff, 1e-10)

        # Emission log-likelihoods: L[t, s] = log p(y_eff(t) | x_c(t)=s)
        L = np.column_stack([
            norm.logpdf(obs_eff, loc=mu[s], scale=np.sqrt(var_eff[:, s]))
            for s in range(S)
        ])                                                     # (T, S)

        # Forward pass
        log_alpha = np.empty((T, S))
        log_alpha[0] = np.log(np.maximum(self.pi[chain_idx], 1e-300)) + L[0]
        for t in range(1, T):
            # log_alpha[t-1,:,None] + log_A  shape (S,S), axis-0 is previous state
            log_alpha[t] = L[t] + np.logaddexp.reduce(
                log_alpha[t - 1, :, None] + log_A, axis=0
            )

        # Backward pass
        log_beta = np.zeros((T, S))
        for t in reversed(range(T - 1)):
            # log_A + L[t+1] + log_beta[t+1]  broadcast to (S,S), axis-1 is next state
            log_beta[t] = np.logaddexp.reduce(
                log_A + L[t + 1] + log_beta[t + 1], axis=1
            )

        # One-slice marginals
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Two-slice marginals for transition update
        # xi[t,i,j] = p(x_t=i, x_{t+1}=j | y)
        log_norm = np.logaddexp.reduce(log_alpha[T - 1])
        log_xi = (
            log_alpha[:-1, :, None]          # (T-1, S, 1)
            + log_A[None]                    # (1,   S, S)
            + L[1:, None, :]                 # (T-1, 1, S)
            + log_beta[1:, None, :]          # (T-1, 1, S)
            - log_norm
        )
        xi = np.exp(log_xi)                  # (T-1, S, S)

        return gamma, xi

    def _viterbi(self, obs, chain_idx, other_means, other_vars):
        T = len(obs)
        S = self.n_states
        mu = self.means[chain_idx]
        var = self.vars[chain_idx]
        log_A = np.log(np.maximum(self.A[chain_idx], 1e-300))

        obs_eff = obs - other_means
        var_eff = np.maximum(var[None, :] + other_vars[:, None], 1e-10)

        L = np.column_stack([
            norm.logpdf(obs_eff, loc=mu[s], scale=np.sqrt(var_eff[:, s]))
            for s in range(S)
        ])

        delta = np.empty((T, S))
        psi = np.zeros((T, S), dtype=int)

        delta[0] = np.log(np.maximum(self.pi[chain_idx], 1e-300)) + L[0]
        for t in range(1, T):
            scores = delta[t - 1, :, None] + log_A     # (S, S)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + L[t]

        states = np.empty(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in reversed(range(T - 1)):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def variational_inference(self, obs, n_iter=10):
        """
        Variational EM for FHMM.

        Alternates:
          E-step: coordinate-ascent over chains (forward-backward with effective obs)
          M-step: update means, variances, transition matrices, initial distributions

        Returns:
            q            : list of (T, n_states) posterior marginals, one per chain
            viterbi_paths: list of (T,) most-likely state sequences, one per chain
        """
        T = len(obs)
        q = [np.full((T, self.n_states), 1.0 / self.n_states) for _ in range(self.n_chains)]

        for _ in tqdm(range(n_iter), desc="Variational EM"):
            # ----------------------------------------------------------------
            # E-step: coordinate ascent — update q[c] for each chain in turn
            # ----------------------------------------------------------------
            xis = []
            for c in range(self.n_chains):
                other_means = np.zeros(T)
                other_vars = np.zeros(T)
                for d in range(self.n_chains):
                    if d == c:
                        continue
                    E_mu_d = q[d] @ self.means[d]                   # (T,)
                    E_mu2_d = q[d] @ (self.means[d] ** 2)           # (T,)
                    other_means += E_mu_d
                    other_vars += q[d] @ self.vars[d]               # E[sigma^2_d]
                    other_vars += E_mu2_d - E_mu_d ** 2             # Var[mu_d]

                gamma, xi = self._forward_backward(obs, c, other_means, other_vars)
                q[c] = gamma
                xis.append(xi)

            # ----------------------------------------------------------------
            # M-step: update parameters using expected sufficient statistics.
            # Compute all residuals using old means BEFORE updating any chain.
            # ----------------------------------------------------------------
            total_E_obs = sum(q[d] @ self.means[d] for d in range(self.n_chains))  # (T,)

            for c in range(self.n_chains):
                gamma = q[c]                                         # (T, S)
                # Effective residual: obs minus other chains' expected contributions
                resid = obs - (total_E_obs - q[c] @ self.means[c])  # (T,)

                new_means = np.empty(self.n_states)
                new_vars = np.empty(self.n_states)
                for s in range(self.n_states):
                    w = gamma[:, s]
                    w_sum = w.sum() + 1e-10
                    new_means[s] = (w * resid).sum() / w_sum
                    new_vars[s] = max(
                        (w * (resid - new_means[s]) ** 2).sum() / w_sum, 1e-6
                    )

                self.means[c] = new_means
                self.vars[c] = new_vars

                # Initial distribution
                self.pi[c] = np.maximum(gamma[0], 1e-10)
                self.pi[c] /= self.pi[c].sum()

                # Transition matrix from two-slice marginals
                xi_sum = xis[c].sum(axis=0)                         # (S, S)
                denom = xi_sum.sum(axis=1, keepdims=True)
                self.A[c] = np.where(denom > 0, xi_sum / (denom + 1e-10), 1.0 / self.n_states)

        # Compute Viterbi paths with final parameters
        viterbi_paths = []
        for c in range(self.n_chains):
            other_means = np.zeros(T)
            other_vars = np.zeros(T)
            for d in range(self.n_chains):
                if d == c:
                    continue
                E_mu_d = q[d] @ self.means[d]
                E_mu2_d = q[d] @ (self.means[d] ** 2)
                other_means += E_mu_d
                other_vars += q[d] @ self.vars[d]
                other_vars += E_mu2_d - E_mu_d ** 2
            viterbi_paths.append(self._viterbi(obs, c, other_means, other_vars))

        return q, viterbi_paths
