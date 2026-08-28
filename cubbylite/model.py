"""CubbyLite model: spike-attention LM built on CubbyLLM's central bet.

Faithful mini-implementation of the CubbyLLM (Grillcheese-AI) core ideas,
validated in-session on grilly/Vulkan:

- Frozen router: routing is a fixed function of the token (multiplicative
  hash, Hash-Layers style). Zero routing parameters, zero collapse, zero
  balancing losses. Optional 'learned' mode (quantile-balanced argmax) as
  the comparison arm -- the trainable-router ablation.
- Training follows the router's allocation: each routed expert only ever
  sees the tokens the router assigns to it; experts specialize on their
  stationary slice (CubbyLLM: online-learned routing forgets itself).
- Spike attention: K and V are single-step ternary spike trains; QK^T and
  attention@V are addition-semantics operations.
- Shared expert (always on) absorbs context-independent computation.
- Retrieval head: cosine readout against the (tied) embedding table --
  CubbyLLM's softmax-bypass output head, the pillar that scales to huge
  vocabularies.
"""
import numpy as np
from grilly import functional as F


def _init(shape, scale, rng):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


class SpikeMoELM:
    def __init__(self, vocab, d=80, hid=96, n_exp=7, n_blocks=2,
                 router="hash", head="retrieval", seed=0):
        self.V, self.D, self.HID = vocab, d, hid
        self.NEXP, self.NB = n_exp, n_blocks
        self.router_mode = router
        self.head_mode = head
        self.rng = np.random.default_rng(seed)
        W = {"emb": _init((vocab, d), 0.05, self.rng)}
        if head == "linear":                       # untied output head
            W["head"] = np.zeros((vocab, d), dtype=np.float32)
        for b in range(n_blocks):
            p = f"b{b}."
            W[p + "wq"] = _init((d, d), 1 / np.sqrt(d), self.rng)
            W[p + "wk"] = _init((d, d), 1 / np.sqrt(d), self.rng)
            W[p + "wv"] = _init((d, d), 1 / np.sqrt(d), self.rng)
            W[p + "wo"] = np.zeros((d, d), dtype=np.float32)
            if router == "learned":
                W[p + "router"] = _init((n_exp, d), 1 / np.sqrt(d), self.rng)
            W[p + "sh.1"] = _init((hid, d), 1 / np.sqrt(d), self.rng)
            W[p + "sh.2"] = np.zeros((d, hid), dtype=np.float32)
            for e in range(n_exp):
                W[f"{p}e{e}.1"] = _init((hid, d), 1 / np.sqrt(d), self.rng)
                W[f"{p}e{e}.2"] = np.zeros((d, hid), dtype=np.float32)
        self.W = W
        self._lif_reset = None

    # ------------------------------------------------------------------
    def trainable(self):
        """Keys the optimizer may touch. Frozen router => no router keys exist."""
        return list(self.W.keys())

    # ------------------------------------------------------------------
    @staticmethod
    def _spike(x, thr=0.5):
        s = np.mean(np.abs(x)) + 1e-8
        return np.where(x > thr * s, 1.0,
                        np.where(x < -thr * s, -1.0, 0.0)).astype(np.float32)

    @staticmethod
    def _softmax(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    @staticmethod
    def _lin(x, w):
        return np.asarray(F.linear(x, w))          # grilly Vulkan GEMM

    # ------------------------------------------------------------------
    def _route_hash(self, idx, x):
        """Frozen hash router: allocation is a fixed function of token id.
        Multiplicative (Knuth) hash -> expert. No parameters, no drift."""
        h = (idx.astype(np.int64) * 2654435761) % self.NEXP
        return h

    def _route_learned(self, x, p):
        """Quantile-balanced learned router (K3-style): per-expert CDF rank,
        then top-1. Balanced by construction, but weights are trainable and
        can therefore drift under continued training (the ablation arm)."""
        scores = self._lin(x, self.W[p + "router"])
        flat = scores.reshape(-1, self.NEXP)
        n = flat.shape[0]
        ranks = flat.argsort(0).argsort(0) / max(n - 1, 1)
        return ranks.argmax(-1).reshape(x.shape[0], x.shape[1])

    # ------------------------------------------------------------------
    def block(self, x, idx, p):
        B, T, _ = x.shape
        q = self._lin(x, self.W[p + "wq"])
        k = self._spike(self._lin(x, self.W[p + "wk"]))
        v = self._spike(self._lin(x, self.W[p + "wv"]))
        att = q @ k.transpose(0, 2, 1) / np.sqrt(self.D)
        att = np.where(np.triu(np.ones((T, T), bool), 1), -1e9, att)
        x = x + self._lin(self._softmax(att) @ v, self.W[p + "wo"])

        # shared expert: always on
        hs = self._spike(self._lin(x, self.W[p + "sh.1"]))
        shared = self._lin(hs, self.W[p + "sh.2"])

        # routed experts: training follows the router's allocation
        top = (self._route_hash(idx, x) if self.router_mode == "hash"
               else self._route_learned(x, p))
        out = np.zeros_like(x)
        counts = np.bincount(top.ravel(), minlength=self.NEXP)
        for e in range(self.NEXP):
            m = (top == e)
            if not m.any():
                continue
            h = self._spike(self._lin(x[m], self.W[f"{p}e{e}.1"]))
            out[m] = self._lin(h, self.W[f"{p}e{e}.2"])
        return x + out + shared, counts

    # ------------------------------------------------------------------
    def forward(self, idx):
        x = self.W["emb"][idx]
        bal = []
        for b in range(self.NB):
            x, counts = self.block(x, idx, f"b{b}.")
            bal.append(counts / max(counts.sum(), 1))
        if self.head_mode == "retrieval":
            # cosine readout against the tied embedding table
            h = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
            e = self.W["emb"] / (np.linalg.norm(self.W["emb"], axis=-1, keepdims=True) + 1e-8)
            logits = h @ e.T * 8.0
        else:
            logits = self._lin(x, self.W["head"])
        return logits, bal

    # ------------------------------------------------------------------
    def ce(self, idx, tgt):
        logits, _ = self.forward(idx)
        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        return -np.take_along_axis(logp, tgt[..., None], axis=-1).mean()
