import torch
import torch.nn as nn
import math

class BSM(nn.Module):
    def __init__(self, k: torch.Tensor, r: int):
        super().__init__()
        self.k = k
        self.r = r
        self.a, self.b = k[..., ::2, :], k[..., 1::2, :]
        self.scores = self.a @ self.b.transpose(-1, -2)

        self.scores[..., 0, :] = -math.inf  # don’t merge cls token
        self.node_max, self.node_idx = self.scores.max(dim=-1)
        self.edge_idx = self.node_max.argsort(dim=-1, descending=True)[..., None]
        self.unm_idx = self.edge_idx[..., r:, :]  # Unmerged Tokens
        self.src_idx = self.edge_idx[..., :r, :]  # Merged Tokens
        self.dst_idx = self.node_idx[..., None].gather(dim=-2, index=self.src_idx)

    def forward(self, x):
        """ Input is of shape [batch, tokens, channels]. """
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=self.unm_idx.expand(n, t1 - self.r, c))
        src = src.gather(dim=-2, index=self.src_idx.expand(n, self.r, c))
        dst = dst.scatter_add(-2, self.dst_idx.expand(n, self.r, c), src)
        return torch.cat([unm, dst], dim=-2)
