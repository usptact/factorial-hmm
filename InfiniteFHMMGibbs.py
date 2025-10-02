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
            self.A = np.array([[0.9,0.1],[0.1,0.9]])  # sticky HMM
            self.pi = np.array([0.5,0.5])
            self.mu = np.random.randn(n_states)
            self.var = np.random.rand(n_states) * 0.5 + 0.01  # arbitrary variance
            self.X = None

    def initialize(self, obs, max_initial_chains=3):
        self.T = len(obs)
        K = max_initial_chains
        self.Z = np.ones((self.T, K), dtype=int)
        self.chains = [self.Chain(self.n_states) for _ in range(K)]
        for chain in self.chains:
            chain.X = np.random.choice(self.n_states, size=self.T)

    def gibbs_sample(self, obs, n_iter=20):
        T = self.T
        for it in tqdm(range(n_iter), desc="Gibbs Sampling"):
            K = len(self.chains)
            # Step 1: Sample Z
            for t in range(T):
                for k in range(K):
                    m_minus = np.sum(self.Z[:,k]) - self.Z[t,k]
                    p_prior = np.clip(m_minus/T, 1e-8, 1-1e-8)
                    other = sum(self.Z[t,j]*self.chains[j].mu[self.chains[j].X[t]]
                                for j in range(K) if j!=k)
                    mu_active = self.chains[k].mu[self.chains[k].X[t]]
                    var_active = self.chains[k].var[self.chains[k].X[t]]
                    log_like1 = norm.logpdf(obs[t], loc=other + mu_active, scale=np.sqrt(var_active))
                    log_like0 = norm.logpdf(obs[t], loc=other, scale=np.sqrt(0.01))  # small baseline noise
                    logp1 = np.log(p_prior) + log_like1
                    logp0 = np.log(1-p_prior) + log_like0
                    max_log = max(logp0, logp1)
                    prob1 = np.exp(logp1-max_log) / (np.exp(logp0-max_log)+np.exp(logp1-max_log))
                    prob1 = np.clip(prob1, 1e-8, 1-1e-8)
                    self.Z[t,k] = bernoulli.rvs(prob1)

            # Step 2: Sample hidden states X
            for k, chain in enumerate(self.chains):
                for t in range(T):
                    if self.Z[t,k]==0: continue
                    other = sum(self.Z[t,j]*self.chains[j].mu[self.chains[j].X[t]]
                                for j in range(K) if j!=k)
                    probs = []
                    for s in range(chain.n_states):
                        likelihood = norm.pdf(obs[t], loc=other+chain.mu[s], scale=np.sqrt(chain.var[s]))
                        probs.append(likelihood)
                    probs = np.array(probs)
                    probs /= probs.sum()
                    chain.X[t] = np.random.choice(chain.n_states, p=probs)

            # Step 3: Update emission parameters (mean and variance) per chain
            for k, chain in enumerate(self.chains):
                for s in range(chain.n_states):
                    idx = [t for t in range(T) if self.Z[t,k]==1 and chain.X[t]==s]
                    if idx:
                        residuals = [obs[t]-sum(self.Z[t,j]*self.chains[j].mu[self.chains[j].X[t]]
                                                for j in range(K) if j!=k) for t in idx]
                        chain.mu[s] = np.mean(residuals)
                        chain.var[s] = np.var(residuals) + 1e-6  # avoid zero variance

            # Step 4: Add new chains
            m_new = np.random.poisson(self.alpha/T)
            for _ in range(m_new):
                new_chain = self.Chain(self.n_states)
                new_chain.X = np.random.choice(self.n_states, size=T)
                self.chains.append(new_chain)
                self.Z = np.hstack([self.Z, np.ones((T,1), dtype=int)])

        # Step 5: Compute Viterbi paths
        viterbi_paths = []
        K = len(self.chains)
        for k, chain in enumerate(self.chains):
            T = self.T
            S = self.n_states
            delta = np.full((T, S), -np.inf)
            psi = np.zeros((T, S), dtype=int)
            path = np.full(T, -1)

            # initial step
            for s in range(S):
                if self.Z[0,k]==1:
                    delta[0,s] = np.log(chain.pi[s]) + np.log(norm.pdf(obs[0],
                                             loc=chain.mu[s] + sum(self.Z[0,j]*self.chains[j].mu[self.chains[j].X[0]]
                                                                    for j in range(K) if j!=k),
                                             scale=np.sqrt(chain.var[s])))
            # recursion
            for t in range(1,T):
                if self.Z[t,k]==0: continue
                for s in range(S):
                    log_probs = delta[t-1,:] + np.log(chain.A[:,s])
                    psi[t,s] = np.argmax(log_probs)
                    other = sum(self.Z[t,j]*self.chains[j].mu[self.chains[j].X[t]] for j in range(K) if j!=k)
                    delta[t,s] = np.max(log_probs) + np.log(norm.pdf(obs[t], loc=chain.mu[s]+other,
                                                                    scale=np.sqrt(chain.var[s])))
            # backtrace
            if np.any(self.Z[:,k]==1):
                t_active = np.where(self.Z[:,k]==1)[0]
                last = t_active[-1]
                path[last] = np.argmax(delta[last,:])
                for t in reversed(t_active[:-1]):
                    path[t] = psi[t+1,path[t+1]]
            viterbi_paths.append(path)

        return self.Z, [chain.X for chain in self.chains], [chain.mu for chain in self.chains], \
               [chain.var for chain in self.chains], viterbi_paths
