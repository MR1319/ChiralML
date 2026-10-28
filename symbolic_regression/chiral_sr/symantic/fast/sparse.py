"""Exhaustive l0 subset search in a QR-reduced space, and Pareto utilities.

For a pool Z (n x s) with thin QR Z = QR and c = Q^T y, every subset S obeys

    ||y - Z_S b||^2 = ||(I - QQ^T) y||^2 + ||c - R_S b||^2,

so subset fits cost O(s d^2) instead of O(n d^2) and need no n-sized batch
tensors. Subsets are screened with batched Cholesky on R^T R; candidates that
may lie on the complexity/error frontier are refit exactly by least squares.
"""
from itertools import combinations, islice
import math

import numpy as np
import torch


def pareto_mask(complexity, loss):
    """Non-dominated points for minimizing both; O(N log N); ties kept."""
    complexity = np.asarray(complexity, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    mask = np.zeros(len(loss), dtype=bool)
    ok = np.nonzero(np.isfinite(loss) & np.isfinite(complexity))[0]
    if len(ok) == 0:
        return mask
    order = ok[np.lexsort((loss[ok], complexity[ok]))]
    c, l = complexity[order], loss[order]
    starts = np.r_[0, np.nonzero(np.diff(c))[0] + 1]
    group_min = l[starts]
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(c)]))
    previous_best = np.r_[np.inf, np.minimum.accumulate(group_min)[:-1]]
    efficient = (l == group_min[group]) & (group_min[group] < previous_best[group])
    mask[order[efficient]] = True
    return mask


EXHAUSTIVE_LIMIT = 2_000_000  # subsets per pool searched exhaustively; larger pools use beam + swap


def _extension_gains(grams, rhs, S):
    """Explained-sum increase from adding each feature f to each subset S (B x j).

    For a Gram system (G, b) the increase is the Schur-complement gain
    (b_f - G_fS G_SS^-1 b_S)^2 / (G_ff - G_fS G_SS^-1 G_Sf), summed over systems.
    Returns (B, s) gains (-inf for f in S or degenerate f) and current explained (B,).
    """
    B, j = S.shape
    s = grams[0].shape[0]
    gains = torch.zeros((B, s), dtype=grams[0].dtype, device=grams[0].device)
    explained = torch.zeros(B, dtype=grams[0].dtype, device=grams[0].device)
    bad = torch.zeros((B, s), dtype=torch.bool, device=grams[0].device)
    for G, b in zip(grams, rhs):
        diag = G.diagonal()
        if j == 0:
            num, den = b[None, :].expand(B, -1), diag[None, :].expand(B, -1)
        else:
            L, info = torch.linalg.cholesky_ex(G[S[:, :, None], S[:, None, :]])
            bS = b[S]
            u = torch.cholesky_solve(bS[..., None], L)[..., 0]
            explained += torch.where(info == 0, (bS * u).sum(1), -math.inf)
            GSf = G[S]                                        # (B, j, s)
            W = torch.cholesky_solve(GSf, L)                  # G_SS^-1 G_Sf
            num = b[None, :] - torch.einsum("bjs,bj->bs", GSf, u)
            den = diag[None, :] - (GSf * W).sum(1)
        ok = den > 1e-10 * diag[None, :].clamp_min(1e-300)
        gains += torch.where(ok, num.square() / den.clamp_min(1e-300), 0.0)
        bad |= ~ok
    gains[bad] = -math.inf
    if j:
        gains.scatter_(1, S, -math.inf)
    return gains, explained


def beam_subsets(grams, rhs, complexity, d, width=512, keep_top=64, swap_top=32, max_sweeps=8):
    """Approximate best d-subsets maximizing the summed explained sum b_S^T G_S^-1 b_S.

    Beam search over subset sizes using exact Schur-complement gains, then
    1-swap local search from the best sets. Returns candidate index sets
    (C x d): the top ``keep_top`` plus the approximate complexity/explained
    frontier among all evaluated size-d sets.
    """
    device = grams[0].device
    S = torch.empty((1, 0), dtype=torch.long, device=device)
    for size in range(1, d + 1):
        gains, explained = _extension_gains(grams, rhs, S)
        total = explained[:, None] + gains
        flat = torch.nonzero(torch.isfinite(total))
        if len(flat) == 0:
            return torch.empty((0, d), dtype=torch.long, device=device)
        sets = torch.sort(torch.cat([S[flat[:, 0]], flat[:, 1:2]], dim=1), dim=1).values
        values = total[flat[:, 0], flat[:, 1]]
        sets, inverse = torch.unique(sets, dim=0, return_inverse=True)
        best = torch.full((len(sets),), -math.inf, dtype=values.dtype, device=device)
        best.scatter_reduce_(0, inverse, values, reduce="amax")
        if size == d:
            cplx = complexity[sets].sum(1)
            front = torch.as_tensor(pareto_mask(cplx.cpu().numpy(), (-best).cpu().numpy()), device=device)
            top = torch.topk(best, min(keep_top, len(best))).indices
            pool = sets[torch.unique(torch.cat([top, torch.nonzero(front).flatten()]))]
            refined = _swap_refine(grams, rhs, sets[torch.topk(best, min(swap_top, len(best))).indices], max_sweeps)
            return torch.unique(torch.cat([pool, refined]), dim=0)
        S = sets[torch.topk(best, min(width, len(best))).indices]


def _swap_refine(grams, rhs, S, max_sweeps):
    """1-swap local search: replace one member with the best outsider while it helps."""
    S = S.clone()
    d = S.shape[1]
    current = _extension_gains(grams, rhs, S)[1] if d else torch.zeros(len(S))
    for _ in range(max_sweeps):
        improved = False
        for i in range(d):
            rest = torch.cat([S[:, :i], S[:, i + 1:]], dim=1)
            gains, explained = _extension_gains(grams, rhs, rest)
            value, f = (explained[:, None] + gains).max(dim=1)
            better = value > current + 1e-12 * current.abs().clamp_min(1.0)
            if better.any():
                S[better] = torch.sort(torch.cat([rest[better], f[better, None]], dim=1), dim=1).values
                current = torch.where(better, value, current)
                improved = True
        if not improved:
            break
    return S


def subset_search(Z, y, d, complexity, chunk=262144, tolerance=1e-9, exhaustive_limit=None, keep_top=0):
    """Best d-subsets of the pool columns; returns frontier fits and the best fit.

    Z: (n, s) standardized pool; y: centered target; complexity: (s,) tensor.
    Pools with at most ``exhaustive_limit`` subsets are searched exhaustively;
    larger ones by beam + swap search. Returns (indices, betas, rss, cplx) for
    frontier candidates (exact refits) and the index of the minimum-RSS one.
    With keep_top > 0 a sixth element holds the exact fits of the keep_top
    lowest-RSS subsets regardless of complexity (candidates for refinement).
    """
    n, s = Z.shape
    if s < d:
        return None
    Q, R = torch.linalg.qr(Z)
    c = Q.T @ y
    rss_perp = (y - Q @ c).square().sum()
    G = R.T @ R
    g = R.T @ c
    cc = c.square().sum()
    tol = tolerance * float(cc + rss_perp) + 1e-300
    kept_idx, kept_beta, kept_rss, kept_cplx = [], [], [], []
    top_idx = torch.empty((0, d), dtype=torch.long, device=Z.device)
    top_rss = torch.empty(0, dtype=Z.dtype, device=Z.device)
    limit = EXHAUSTIVE_LIMIT if exhaustive_limit is None else exhaustive_limit
    if math.comb(s, d) > limit:
        blocks = iter([beam_subsets([G], [g], complexity, d).cpu().numpy()])
    else:
        combos = combinations(range(s), d)
        blocks = (np.fromiter(islice((v for combo in combos for v in combo), chunk * d), dtype=np.int64).reshape(-1, d)
                  for _ in iter(int, 1))
    for block in blocks:
        if block.size == 0:
            break
        idx = torch.as_tensor(block, device=Z.device)
        Gs = G[idx[:, :, None], idx[:, None, :]]
        gs = g[idx]
        L, info = torch.linalg.cholesky_ex(Gs)
        beta = torch.cholesky_solve(gs[..., None], L)[..., 0]
        rss = rss_perp + cc - (gs * beta).sum(dim=1)
        ok = (info == 0) & torch.isfinite(rss)
        cplx = complexity[idx].sum(dim=1)
        if keep_top:
            all_idx = torch.cat([top_idx, idx])
            all_rss = torch.cat([top_rss, torch.where(ok, rss, math.inf)])
            pick = torch.topk(-all_rss, min(keep_top, len(all_rss))).indices
            top_idx, top_rss = all_idx[pick], all_rss[pick]
        candidates = _near_frontier(cplx, rss, ok, tol)
        if len(candidates):
            sub = idx[candidates]
            Rs = R[:, sub].permute(1, 0, 2)
            b = torch.linalg.lstsq(Rs, c.expand(len(sub), -1)[..., None]).solution[..., 0]
            exact = rss_perp + (c[None, :] - (Rs @ b[..., None])[..., 0]).square().sum(dim=1)
            kept_idx.append(sub)
            kept_beta.append(b)
            kept_rss.append(exact)
            kept_cplx.append(cplx[candidates])
            # Keep the running store small: prune to the (tolerant) frontier.
            if len(kept_idx) > 8:
                kept_idx, kept_beta, kept_rss, kept_cplx = _compact(kept_idx, kept_beta, kept_rss, kept_cplx, tol)
    if not kept_idx:
        return None
    kept_idx, kept_beta, kept_rss, kept_cplx = _compact(kept_idx, kept_beta, kept_rss, kept_cplx, 0.0)
    idx, beta, rss, cplx = kept_idx[0], kept_beta[0], kept_rss[0], kept_cplx[0]
    best = int(torch.argmin(rss))
    if not keep_top:
        return idx, beta, rss, cplx, best
    top_idx = top_idx[torch.isfinite(top_rss)]
    Rs = R[:, top_idx].permute(1, 0, 2)
    b = torch.linalg.lstsq(Rs, c.expand(len(top_idx), -1)[..., None]).solution[..., 0]
    exact = rss_perp + (c[None, :] - (Rs @ b[..., None])[..., 0]).square().sum(dim=1)
    return idx, beta, rss, cplx, best, (top_idx, b, exact, complexity[top_idx].sum(dim=1))


def _near_frontier(cplx, rss, ok, tol):
    """Indices whose Gram RSS is within tol of the frontier at their complexity."""
    sel = torch.nonzero(ok).flatten()
    if len(sel) == 0:
        return sel
    c, r = cplx[sel], rss[sel]
    order = torch.argsort(c, stable=True)
    c_sorted, r_sorted = c[order], r[order]
    frontier = torch.cummin(r_sorted, dim=0).values
    # Points with equal complexity share the frontier value of the last of them.
    last = torch.searchsorted(c_sorted, c_sorted, right=True) - 1
    near = r_sorted <= frontier[last] + tol
    return sel[order[near]]


def _compact(idx, beta, rss, cplx, tol):
    idx, beta, rss, cplx = torch.cat(idx), torch.cat(beta), torch.cat(rss), torch.cat(cplx)
    keep = _near_frontier(cplx, rss, torch.ones_like(rss, dtype=torch.bool), tol)
    return [idx[keep]], [beta[keep]], [rss[keep]], [cplx[keep]]


class ParetoStore:
    """Running complexity/loss frontier over all models seen (loss key: 'rmse' or 'loss')."""

    def __init__(self, loss_key="rmse"):
        self.rows = []
        self.loss_key = loss_key

    def add(self, rows):
        self.rows.extend(rows)
        if len(self.rows) > 64:
            self.prune()

    def prune(self):
        if not self.rows:
            return
        mask = pareto_mask([r["complexity"] for r in self.rows], [r[self.loss_key] for r in self.rows])
        self.rows = [r for r, m in zip(self.rows, mask) if m]

    def frontier(self):
        self.prune()
        rows = sorted(self.rows, key=lambda r: (r["complexity"], r[self.loss_key]))
        seen, out = set(), []
        for row in rows:
            if row["complexity"] not in seen:
                seen.add(row["complexity"])
                out.append(row)
        return out


def format_equation(row, digits=4):
    """Readable equation in the format of SymanticModel.combine_equation."""
    parts = []
    for term, coeff in zip(row["terms"], row["coefficients"]):
        if coeff == 1:
            parts.append(term)
        elif coeff == -1:
            parts.append(f"-{term}")
        else:
            parts.append(f"{coeff:.{digits}f}*{term}")
    equation = " + ".join(parts)
    intercept = row["intercept"]
    if intercept != 0:
        equation = f"{equation} + {intercept:.{digits}f}" if intercept > 0 else f"{equation} - {abs(intercept):.{digits}f}"
    return equation


def full_precision_equation(terms, coefficients, intercept):
    body = " + ".join(f"{c:.20f}*{t}" for t, c in zip(terms, coefficients))
    return f"{body} {intercept:+.16g}"
