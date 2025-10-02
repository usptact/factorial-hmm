import numpy as np
from tqdm import tqdm
from scipy.stats import norm

class FHMMVariational:
    def __init__(self, n_chains=3, n_states=2):
        self.n_chains = n_chains
        self.n_states = n_states

        # Random initialization of HMM parameters
        self.pi = [np.full(n_states, 1/n_states) for _ in range(n_chains)]
        self.A = [np.full((n_states, n_states), 1/n_states) for _ in range(n_chains)]
        self.means = [np.linspace(0, 1, n_states) for _ in range(n_chains)]
        self.vars = [np.full(n_states, 0.05) for _ in range(n_chains)]

    def _forward_backward(self, obs, chain_idx, other_means, other_vars):
        T = len(obs)
        S = self.n_states
        pi = self.pi[chain_idx]
        A = self.A[chain_idx]
        mu = self.means[chain_idx]
        var = self.vars[chain_idx]

        obs_eff = obs - other_means  # shape (T,)
        var_eff = var[None, :] + other_vars[:, None]  # shape (T, S)

        # Likelihood matrix
        L = np.zeros((T, S))
        for s in range(S):
            L[:, s] = norm.logpdf(obs_eff, loc=mu[s], scale=np.sqrt(var_eff[:, s]))

        # Forward pass
        alpha = np.zeros((T, S))
        alpha[0] = np.log(pi) + L[0]
        for t in range(1, T):
            for j in range(S):
                alpha[t, j] = L[t, j] + np.logaddexp.reduce(alpha[t - 1] + np.log(A[:, j]))

        # Backward pass
        beta = np.zeros((T, S))
        for t in reversed(range(T - 1)):
            for i in range(S):
                beta[t, i] = np.logaddexp.reduce(np.log(A[i, :]) + L[t + 1, :] + beta[t + 1, :])

        # Posterior
        log_gamma = alpha + beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        return gamma

    def _viterbi(self, obs, chain_idx, other_means, other_vars):
        """
        Viterbi algorithm for one chain given expected contributions from other chains.
        Returns most likely state sequence for this chain.
        """
        T = len(obs)
        S = self.n_states
        pi = self.pi[chain_idx]
        A = self.A[chain_idx]
        mu = self.means[chain_idx]
        var = self.vars[chain_idx]

        obs_eff = obs - other_means
        var_eff = var[None, :] + other_vars[:, None]  # shape (T, S)

        L = np.zeros((T, S))
        for s in range(S):
            L[:, s] = norm.logpdf(obs_eff, loc=mu[s], scale=np.sqrt(var_eff[:, s]))

        delta = np.zeros((T, S))
        psi = np.zeros((T, S), dtype=int)

        # Initialization
        delta[0] = np.log(pi) + L[0]

        # Recursion
        for t in range(1, T):
            for j in range(S):
                seq_probs = delta[t-1] + np.log(A[:, j])
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) + L[t, j]

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in reversed(range(T-1)):
            states[t] = psi[t+1, states[t+1]]

        return states

    def variational_inference(self, obs, n_iter=10):
        """
        Mean-field variational inference for FHMM.
        Returns posterior gamma for each chain and Viterbi path per chain.
        """
        T = len(obs)
        # Initialize q: uniform
        q = [np.full((T, self.n_states), 1/self.n_states) for _ in range(self.n_chains)]

        for iteration in tqdm(range(n_iter), desc="Variational Inference"):
            for c in range(self.n_chains):
                # Expected contributions from other chains
                other_means = np.zeros(T)
                other_vars = np.zeros(T)
                for d in range(self.n_chains):
                    if d == c:
                        continue
                    other_means += q[d] @ self.means[d]
                    other_vars += q[d] @ self.vars[d]

                # Update posterior for this chain
                q[c] = self._forward_backward(obs, c, other_means, other_vars)

        # After convergence, compute Viterbi paths
        viterbi_paths = []
        for c in range(self.n_chains):
            other_means = np.zeros(T)
            other_vars = np.zeros(T)
            for d in range(self.n_chains):
                if d == c:
                    continue
                other_means += q[d] @ self.means[d]
                other_vars += q[d] @ self.vars[d]
            viterbi_paths.append(self._viterbi(obs, c, other_means, other_vars))

        return q, viterbi_paths
