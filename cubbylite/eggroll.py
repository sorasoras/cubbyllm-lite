"""EGGROLL trainer: backprop-free low-rank evolution strategies.

The validated in-session recipe (converges stably on spike-MoE LMs):
antithetic population, rank-r factorized perturbations (EGGROLL,
arXiv 2605.30361 style), fitness standardization, scale-free update with
decayed step size. No gradients, no surrogate functions, no backward pass.

Because routing is a frozen function (CubbyLLM central bet), the trainer
touches only the weights below the router -- the allocation itself is
never optimized, so it cannot collapse.
"""
import numpy as np


class EggrollTrainer:
    def __init__(self, params, trainable_keys=None, pop=64, sigma=0.05,
                 alpha=0.01, rank=8, seed=0):
        self.W = params
        self.keys = list(trainable_keys) if trainable_keys is not None else list(params.keys())
        self.pop, self.sigma, self.alpha, self.rank = pop, sigma, alpha, rank
        self.rng = np.random.default_rng(seed)
        self.base = {k: params[k].copy() for k in self.keys}

    def _perturbation(self):
        d = {}
        for k in self.keys:
            s = self.W[k].shape
            if len(s) == 1:
                d[k] = self.rng.standard_normal(s).astype(np.float32)
                continue
            r = min(self.rank, min(s))
            u = self.rng.standard_normal((s[0], r)).astype(np.float32)
            v = self.rng.standard_normal((r, s[1])).astype(np.float32)
            d[k] = (u @ v) / np.sqrt(r)
        return d

    def step(self, fitness_fn, gen):
        """One generation. fitness_fn() must evaluate the CURRENT weights."""
        half = self.pop // 2
        plus, fit = [], []
        for _ in range(half):
            d = self._perturbation()
            for k in self.keys:
                self.W[k][...] = self.base[k] + self.sigma * d[k]
            fit.append(fitness_fn())
            for k in self.keys:
                self.W[k][...] = self.base[k] - self.sigma * d[k]
            fit.append(fitness_fn())
            plus.append(d)
        f = np.array(fit, dtype=np.float64)
        f = (f - f.mean()) / (f.std() + 1e-8)
        acc = {k: np.zeros_like(self.base[k]) for k in self.keys}
        for i in range(self.pop):
            s = 1.0 if i % 2 == 0 else -1.0
            for k in self.keys:
                acc[k] += s * f[i] * plus[i // 2][k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)   # decay: stops drift
        for k in self.keys:
            a = acc[k]
            a = (a - a.mean()) / (a.std() + 1e-8)      # scale-free step
            self.base[k] += alpha_t * a
            self.W[k][...] = self.base[k]
        return float(f.mean())
