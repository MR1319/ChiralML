"""Streaming sure-independence screening (SIS) over an unmaterialized level.

The children of level L are binary(new x all) and unary(new), where ``new``
is the last materialized level. Unary children are evaluated in chunks. Binary
children are never formed: their SIS score

    |r . f| / sd(f)    (r centered; sd unbiased, as in the original Regressor)

is computed from matrix products of standardized parents. With population
z-scores z = (x - mu)/sigma and rho = mu/sigma, up to a positive scale factor

    a*b  ~  rho_a z_b + rho_b z_a + z_a z_b
    a+b  ~  z_a + (sigma_b/sigma_a) z_b

so every needed moment is one of z_a.z_b, z_a^2.z_b^2, z_a.z_b^2, z_a^2.z_b,
or (r z_a).z_b. Division uses a*(1/b). Working with centered, scaled parents
avoids the E[f^2] - E[f]^2 cancellation of raw moments. A constant parent
(physical constant) has z = 0, sigma = |mu| and sum(z^2) = 0, which keeps the
same formulas exact for children such as c*x.

Units: '+'/'-' pairs must share units, unary rules come from the operator
registry, and when a target unit is given only children with that unit are
scored. Unit masks are evaluated before any matrix product, so tiles without
unit-consistent pairs cost almost nothing.

Memory is O(n * tile + tile^2) per step instead of O(n * #children).
Scores of buffered candidates are recomputed exactly from materialized columns
before selection, so moment round-off cannot change which features are chosen
among the buffered ones.
"""
import math

import numpy as np
import torch

from .bank import CONSTANT_TOLERANCE
from .units import unit_keys

_TINY = 1e-300
_CONSTANT_CHILD = 1e-9


def log2_table(size, device):
    table = np.full(size, -np.inf)
    table[1:] = np.log2(np.arange(1, size, dtype=np.float64))
    return torch.as_tensor(table, device=device)


def quantiles_from_counts(values, counts, qs):
    """Exactly reproduce torch.quantile (linear) from a value histogram."""
    order = np.argsort(values)
    values, counts = np.asarray(values)[order], np.asarray(counts)[order]
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])

    def order_statistic(k):
        return values[np.searchsorted(cumulative, k, side="right")]

    out = []
    for q in qs:
        position = q * (total - 1)
        lo, hi = math.floor(position), math.ceil(position)
        v_lo, v_hi = order_statistic(lo), order_statistic(hi)
        out.append(v_lo + (v_hi - v_lo) * (position - lo))
    return np.asarray(out, dtype=np.float64)


def explicit_scores(V, geo, valid=None):
    """Second-order SIS scores of materialized features (rows of V).

    score = sqrt((n - 1) * sum_k U_k^2 / I_k), U_k = sum g_k (v - vbar), I_k =
    sum h_k (v - vbar)^2 with vbar the h_k-weighted mean. Invalid rows get -inf.
    """
    n = V.shape[1]
    gain = torch.zeros(V.shape[0], dtype=V.dtype, device=V.device)
    ok = torch.isfinite(V).all(dim=1) if valid is None else valid & torch.isfinite(V).all(dim=1)
    V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    for cl in geo:
        centered = V - ((V @ cl["h"]) / cl["H"])[:, None]
        info = centered.square() @ cl["h"]
        U = centered @ cl["g"]
        good = info > 0
        gain += torch.where(good, U.square() / info.clamp_min(_TINY), 0.0)
        ok &= good
    return torch.where(ok, (gain * (n - 1)).sqrt(), -math.inf)


class TopBuffers:
    """Per-complexity-bin top-K buffers with running admission thresholds.

    Bins 0..nb-1 are the quantile range bins; slot nb is "above the last
    range boundary" (no range bin); the extra buffer nb is the all-features bin.
    ``floor`` is the lowest admission threshold over bins that can still
    receive features; scores at or below it can be discarded without
    computing their complexity.
    """

    def __init__(self, n_range_bins, capacity, device, active=None):
        self.nb = n_range_bins
        self.capacity = capacity
        empty = torch.empty(0, dtype=torch.float64, device=device)
        self.score = [empty] * (n_range_bins + 1)
        self.cplx = [empty] * (n_range_bins + 1)
        self.code = [torch.empty((0, 3), dtype=torch.int64, device=device)] * (n_range_bins + 1)
        self.tau = torch.full((n_range_bins + 2,), -math.inf, dtype=torch.float64, device=device)
        self.tau[n_range_bins] = math.inf
        self.active = np.ones(n_range_bins, bool) if active is None else np.asarray(active, bool)
        self._tau = np.full(n_range_bins + 2, -math.inf)
        self._tau[n_range_bins] = math.inf
        self.floor = -math.inf

    def survivors(self, score, bins):
        return (score > self.tau[bins]) | (score > self.tau[self.nb + 1])

    def insert(self, score, cplx, bins, code):
        for j in torch.unique(bins).tolist():
            if j < self.nb:
                keep = (bins == j) & (score > self.tau[j])
                if keep.any():
                    self._merge(j, score[keep], cplx[keep], code[keep], j)
        keep = score > self.tau[self.nb + 1]
        if keep.any():
            self._merge(self.nb, score[keep], cplx[keep], code[keep], self.nb + 1)

    def _merge(self, slot, score, cplx, code, tau_index):
        score = torch.cat([self.score[slot], score])
        cplx = torch.cat([self.cplx[slot], cplx])
        code = torch.cat([self.code[slot], code])
        if len(score) > self.capacity:
            top = torch.topk(score, self.capacity, sorted=False).indices
            score, cplx, code = score[top], cplx[top], code[top]
        self.score[slot], self.cplx[slot], self.code[slot] = score, cplx, code
        if len(score) >= self.capacity:
            self.tau[tau_index] = self._tau[tau_index] = float(score.min())
            self.floor = min(self._tau[:self.nb][self.active].min(initial=math.inf), self._tau[self.nb + 1])


class LevelScreen:
    """SIS over materialized bank features plus the streamed next level."""

    def __init__(self, bank, stream_level, quantiles, capacity, tile=None, memory_bytes=512 << 20,
                 bins_from="all", target_units=None):
        """bins_from: 'all' (original: quantiles over every eligible feature) or
        'materialized' (quantiles over levels < L only, so low bins keep
        simple features when the deepest level dominates the count).
        target_units: canonical unit vector every scored feature must have."""
        self.bins_from = bins_from
        self.bank = bank
        self.stream = stream_level is not None
        if self.stream:
            self.a0, self.a1 = bank.block(stream_level - 1)
        device = bank.device
        self.device = device
        ops = bank.ops
        self.binary = [(op, code) for op, code in ops.binary.items()]
        self.unary = list(ops.unary)
        self.capacity = capacity
        self.nsym = torch.as_tensor(bank.nsym, dtype=torch.int64, device=device)
        self.V = torch.as_tensor(bank.varset, dtype=torch.float32, device=device)
        self.O = torch.as_tensor(bank.opset, dtype=torch.float32, device=device)
        self.Vc, self.Oc = self.V.sum(1), self.O.sum(1)
        self.bank_cplx = torch.as_tensor(bank.complexity, device=device)
        self.log2 = log2_table(bank.varset.shape[1] + len(ops) + 2, device)
        self._units(bank, target_units)
        n = bank.n
        if tile is None:
            # ~16 tile^2 buffers + ~14 n x tile side matrices (float64)
            tile = int((-14 * n + math.sqrt((14 * n) ** 2 + 4 * 16 * memory_bytes / 8)) / (2 * 16))
            tile = max(16, min(4096, tile))
        self.tile = tile
        qs = [float(q) for q in quantiles if float(q) < 1.0]
        boundaries, self.active = self._boundaries(qs)
        self.boundaries = torch.as_tensor(boundaries, dtype=torch.float64, device=device)
        self.n_candidates = self._count

    # ------------------------------------------------------------- units
    def _units(self, bank, target):
        t = lambda a: torch.as_tensor(a, device=self.device)
        self.units_active = bank.unit_size > 0
        self.ukey = t(bank.ukey)
        self.dimensionless = bank.dimensionless_key
        self.target = None if target is None or not self.units_active else int(unit_keys(target)[0])
        if self.target is not None:
            # Partner unit keys that make a child land on the target unit.
            self.partner = {"*": t(unit_keys(target[None, :] - bank.units)),
                            "/": t(unit_keys(bank.units - target[None, :])),
                            "\\": t(unit_keys(bank.units + target[None, :]))}
        self.unary_key = {code: t(unit_keys(bank.ops.unit_factor[code] * bank.units)) for code in self.unary}
        # Domain safety (interval arithmetic over each feature's input-domain bounds).
        self.safe = bank.safe_domain
        if self.safe:
            from .bank import _OVERFLOW, child_interval
            magnitude = np.maximum(np.abs(bank.lo), np.abs(bank.hi))
            recip_ok = (bank.lo > 0) | (bank.hi < 0)
            with np.errstate(divide="ignore"):
                recip = np.where(recip_ok, 1.0 / np.minimum(np.abs(bank.lo), np.abs(bank.hi)), np.inf)
            self.magnitude, self.recip_ok, self.recip = t(magnitude), t(recip_ok), t(recip)
            self.overflow = _OVERFLOW
            self.unary_safe = {code: t(child_interval(bank.ops.names, code, bank.lo, bank.hi)[2]) for code in self.unary}

    def _bank_eligible(self):
        eligible = ~self.bank.const
        if self.target is not None:
            eligible &= self.ukey == self.target
        return eligible

    def _unary_masks(self, code, lo, hi):
        """Structural unit validity and target eligibility of unary children."""
        ok = torch.ones(hi - lo, dtype=torch.bool, device=self.device)
        if self.safe:
            ok &= self.unary_safe[code][lo:hi]
        if self.bank.ops.requires_dimensionless[code]:
            ok &= self.ukey[lo:hi] == self.dimensionless
        if self.target is not None:
            ok &= self.unary_key[code][lo:hi] == self.target
        return ok

    def _pair_mask(self, A, B, op):
        """Unit mask for children A_i op B_j (None means all pairs allowed).

        op is '+', '-', '*', '/' (a/b) or '\\' (b/a)."""
        if not self.units_active:
            return None
        ka, kb = self.ukey[A["lo"]:A["hi"]], self.ukey[B["lo"]:B["hi"]]
        if op in ("+", "-"):
            if self.target is None:
                return ka[:, None] == kb[None, :]
            return (ka == self.target)[:, None] & (kb == self.target)[None, :]
        if self.target is None:
            return None
        return self.partner[op][A["lo"]:A["hi"]][:, None] == kb[None, :]

    # ------------------------------------------------------------ geometry
    def _tile_pairs(self, a0):
        t = self.tile
        pairs = [(b0, min(b0 + t, self.a0)) for b0 in range(0, self.a0, t)]
        pairs += [(b0, min(b0 + t, self.a1)) for b0 in range(a0, self.a1, t)]
        return pairs

    def _meta(self, lo, hi):
        return {"lo": lo, "hi": hi, "nsym": self.nsym[lo:hi], "V": self.V[lo:hi], "O": self.O[lo:hi],
                "Vc": self.Vc[lo:hi], "Oc": self.Oc[lo:hi]}

    def _unique(self, A, B):
        """#distinct symbols of every child (without its own operator), and #symbols."""
        inter_v = A["V"] @ B["V"].T
        inter_o = A["O"] @ B["O"].T
        base = (A["Vc"][:, None] + B["Vc"][None, :] - inter_v) + (A["Oc"][:, None] + B["Oc"][None, :] - inter_o)
        return base.to(torch.int64), (A["nsym"][:, None] + B["nsym"][None, :] + 1)

    def _with_op(self, A, B, base, code):
        return base + ((1 - A["O"][:, code])[:, None] * (1 - B["O"][:, code])[None, :]).to(torch.int64)

    def _entry_complexity(self, A, B, rl, cl, code):
        """Complexity of selected children only (rows rl of A, columns cl of B)."""
        Va, Vb, Oa, Ob = A["V"][rl], B["V"][cl], A["O"][rl], B["O"][cl]
        unique = (A["Vc"][rl] + B["Vc"][cl] - (Va * Vb).sum(1) + A["Oc"][rl] + B["Oc"][cl] - (Oa * Ob).sum(1)
                  + (1 - Oa[:, code]) * (1 - Ob[:, code]))
        return (A["nsym"][rl] + B["nsym"][cl] + 1) * self.log2[unique.to(torch.int64)]

    def _masks(self, A, B):
        rows = torch.arange(A["lo"], A["hi"], device=self.device)[:, None]
        cols = torch.arange(B["lo"], B["hi"], device=self.device)[None, :]
        return cols > rows, cols != rows

    @staticmethod
    def _combine(*masks):
        """AND of optional masks (None = all True); returns None if all are None."""
        out = None
        for m in masks:
            if m is None:
                continue
            out = m if out is None else out & m
        return out

    def _tile_ops(self, A, B, diag, upper, offdiag):
        """(op symbol, code, mask, orientation) for every binary child family of a tile."""
        w_a = self.bank.w_ok[A["lo"]:A["hi"]]
        w_b = self.bank.w_ok[B["lo"]:B["hi"]]
        safe = {"*": None, "/": None, "\\": None}
        if self.safe:
            ma, mb = self.magnitude[A["lo"]:A["hi"]], self.magnitude[B["lo"]:B["hi"]]
            safe["*"] = ma[:, None] * mb[None, :] < self.overflow
            safe["/"] = self.recip_ok[B["lo"]:B["hi"]][None, :] & (ma[:, None] * self.recip[B["lo"]:B["hi"]][None, :] < self.overflow)
            safe["\\"] = self.recip_ok[A["lo"]:A["hi"]][:, None] & (self.recip[A["lo"]:A["hi"]][:, None] * mb[None, :] < self.overflow)
        out = []
        for op, code in self.binary:
            if op in ("+", "-"):
                out.append((op, code, self._combine(upper, self._pair_mask(A, B, op)), "sym"))
            elif op == "*":
                out.append((op, code, self._combine(upper, self._pair_mask(A, B, op), safe["*"]), "sym"))
            elif diag:
                out.append(("/", code, self._combine(offdiag, w_b[None, :], self._pair_mask(A, B, "/"), safe["/"]), "rc"))
            else:
                out.append(("/", code, self._combine(w_b[None, :], self._pair_mask(A, B, "/"), safe["/"]), "rc"))
                out.append(("\\", code, self._combine(w_a[:, None], self._pair_mask(A, B, "\\"), safe["\\"]), "cr"))
        return out

    # ---------------------------------------------------- complexity bins
    def _boundaries(self, qs):
        """Quantile boundaries over eligible features, and which range bins are populated."""
        bank = self.bank
        eligible = self._bank_eligible().cpu().numpy()
        values, counts = np.unique(bank.complexity[eligible], return_counts=True)
        bank_hist = dict(zip(values.tolist(), counts.tolist()))
        hist = dict(bank_hist)
        self._count = int(eligible.sum())
        if self.stream:
            width = len(self.log2)
            acc = torch.zeros((2 * int(bank.nsym.max()) + 2) * width, dtype=torch.int64, device=self.device)

            def add(nsym, unique, mask=None):
                keys = nsym * width + unique
                keys = keys.flatten() if mask is None else keys[mask.expand_as(keys)]
                acc.add_(torch.bincount(keys, minlength=len(acc)))

            for code in self.unary:
                for lo in range(self.a0, self.a1, bank.chunk_rows):
                    hi = min(lo + bank.chunk_rows, self.a1)
                    valid, _ = self._unary_values(code, lo, hi)
                    m = self._meta(lo, hi)
                    unique = m["Vc"] + m["Oc"] + (1 - m["O"][:, code])
                    add(m["nsym"] + 1, unique.to(torch.int64), valid & self._unary_masks(code, lo, hi))
            for a0 in range(self.a0, self.a1, self.tile):
                A = self._meta(a0, min(a0 + self.tile, self.a1))
                for b0, b1 in self._tile_pairs(a0):
                    B = self._meta(b0, b1)
                    diag = b0 == a0
                    upper, offdiag = self._masks(A, B) if diag else (None, None)
                    families = self._tile_ops(A, B, diag, upper, offdiag)
                    if all(mask is not None and not mask.any() for _, _, mask, _ in families):
                        continue
                    base, nsym = self._unique(A, B)
                    for _, code, mask, _ in families:
                        add(nsym, self._with_op(A, B, base, code), mask)
            keys = torch.nonzero(acc).flatten()
            for key, count in zip(keys.tolist(), acc[keys].tolist()):
                value = float((key // width) * self.log2[key % width])
                hist[value] = hist.get(value, 0) + count
                self._count += count
        if not qs or not hist:
            return np.zeros(len(qs)) if qs else [], np.zeros(len(qs), bool)
        source = bank_hist if (self.bins_from == "materialized" and bank_hist) else hist
        boundaries = quantiles_from_counts(np.array(list(source)), np.array(list(source.values())), qs)
        # Bins with no features anywhere never fill and must not block the floor.
        bins = np.searchsorted(boundaries, np.array(list(hist)), side="left")
        population = np.bincount(bins, weights=np.array(list(hist.values())), minlength=len(qs) + 1)
        return boundaries, population[:len(qs)] > 0

    # --------------------------------------------------------- one round
    @staticmethod
    def geometry(G, W=None):
        """Per-class loss geometry: gradients g (e.g. residual or y - p) and curvatures h.

        Regression is a single class with g = residual and h = 1. Classification
        uses one class per logit with g = y - p and h = p(1 - p) (the diagonal
        Hessian approximation used by second-order boosting).
        """
        G = G[None, :] if G.dim() == 1 else G
        W = torch.ones_like(G) if W is None else (W[None, :] if W.dim() == 1 else W)
        return [{"g": g, "h": h, "H": float(h.sum()), "G": float(g.sum())} for g, h in zip(G, W)]

    def run_round(self, geo, W=None):
        """Fill per-bin top-K buffers with second-order SIS scores.

        ``geo`` is ``geometry(G, W)`` (or a residual vector, for regression).
        A feature's score is sqrt((n - 1) * sum_k U_k^2 / I_k) with the efficient
        score U_k = sum g (f - fbar_h) and information I_k = sum h (f - fbar_h)^2;
        for regression this is exactly |residual . standardized feature|.
        """
        if isinstance(geo, torch.Tensor):
            geo = self.geometry(geo, W)
        buffers = TopBuffers(len(self.boundaries), self.capacity, self.device, self.active)
        bank = self.bank
        n = bank.n
        eligible = self._bank_eligible()
        for lo in range(0, len(bank), bank.chunk_rows):
            hi = min(lo + bank.chunk_rows, len(bank))
            score = explicit_scores(bank.X[lo:hi], geo, eligible[lo:hi])
            ids = torch.arange(lo, hi, device=self.device)
            self._offer(buffers, score, self.bank_cplx[lo:hi], -1, ids, torch.full_like(ids, -1))
        if not self.stream:
            return buffers
        for code in self.unary:
            for lo in range(self.a0, self.a1, bank.chunk_rows):
                hi = min(lo + bank.chunk_rows, self.a1)
                ok = self._unary_masks(code, lo, hi)
                if not ok.any():
                    continue
                valid, values = self._unary_values(code, lo, hi)
                s = explicit_scores(values, geo, valid & ok)
                m = self._meta(lo, hi)
                unique = (m["Vc"] + m["Oc"] + (1 - m["O"][:, code])).to(torch.int64)
                cplx = (m["nsym"] + 1) * self.log2[unique]
                rows = torch.arange(lo, hi, device=self.device)
                self._offer(buffers, s, cplx, code, rows, torch.full_like(rows, -1))
        for a0 in range(self.a0, self.a1, self.tile):
            A = None
            for b0, b1 in self._tile_pairs(a0):
                Am = self._meta(a0, min(a0 + self.tile, self.a1))
                Bm = self._meta(b0, b1)
                diag = b0 == a0
                upper, offdiag = self._masks(Am, Bm) if diag else (None, None)
                families = self._tile_ops(Am, Bm, diag, upper, offdiag)
                families = [f for f in families if f[2] is None or f[2].any()]
                if not families:
                    continue
                if A is None:
                    A = self._side(a0, Am["hi"])
                B = A if diag else self._side(b0, b1)
                self._tile(A, B, families, buffers, n, geo)
        return buffers

    def _unary_values(self, code, lo, hi):
        values = self.bank.ops.apply(code, self.bank.X[lo:hi])
        mu = values.mean(dim=1)
        sd = (values - mu[:, None]).square().mean(dim=1).sqrt()
        valid = torch.isfinite(values).all(dim=1) & torch.isfinite(sd) & (sd > CONSTANT_TOLERANCE * mu.abs()) & (sd > 0)
        return valid, torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _side(self, lo, hi):
        bank = self.bank
        X = bank.X[lo:hi]
        const = bank.const[lo:hi]
        side = self._meta(lo, hi)
        side["sd"] = bank.sd[lo:hi]
        side.update(self._standardized(X, bank.mu[lo:hi], bank.sd[lo:hi], const))
        ok = bank.w_ok[lo:hi]
        if ok.any():
            W = torch.where(ok[:, None], torch.reciprocal(X), 0.0)
            side["w"] = self._standardized(W, torch.where(ok, bank.mu_w[lo:hi], 0.0),
                                           torch.where(ok, bank.sd_w[lo:hi], 1.0), const)
        return side

    @staticmethod
    def _standardized(X, mu, sd, const):
        z = (X - mu[:, None]) / sd[:, None]
        if const.any():
            z[const] = 0.0
        return {"z": z, "z2": z.square(), "rho": mu / sd}

    @staticmethod
    def _weighted(S, cl):
        """A side (row operand) of the products, weighted for one class."""
        h, g = cl["h"], cl["g"]
        return {"zh": S["z"] * h, "z2h": S["z2"] * h, "zg": S["z"] * g,
                "hz": S["z"] @ h, "s2h": S["z2"] @ h, "gz": S["z"] @ g}

    @staticmethod
    def _vectors(S, cl):
        """B side (column operand): only per-feature weighted sums are needed."""
        return {"hz": S["z"] @ cl["h"], "s2h": S["z2"] @ cl["h"], "gz": S["z"] @ cl["g"]}

    def _tile(self, A, B, families, buffers, n, geo):
        lazy = {}
        gains = [torch.zeros((A["hi"] - A["lo"], B["hi"] - B["lo"]), dtype=torch.float64, device=self.device)
                 for _ in families]
        valid = [torch.ones_like(g, dtype=torch.bool) for g in gains]
        for cl in geo:
            Aw = self._weighted(A, cl)
            Bv = self._vectors(B, cl)
            C_h = None
            for f, (op, _, _, _) in enumerate(families):
                if op in ("+", "-", "*") and C_h is None:
                    C_h = Aw["zh"] @ B["z"].T
                if op in ("+", "-"):
                    gain, ok = self._sum_gain(A, Aw, B, Bv, cl, C_h, 1.0 if op == "+" else -1.0)
                elif op == "*":
                    gain, ok = self._product_gain(A, Aw, B, Bv, cl, C_h)
                elif op == "/":
                    if "w" not in B:
                        continue
                    gain, ok = self._product_gain(A, Aw, B["w"], self._vectors(B["w"], cl), cl)
                else:
                    if "w" not in A:
                        continue
                    gain, ok = self._product_gain(A["w"], self._weighted(A["w"], cl), B, Bv, cl)
                gains[f].add_(gain.masked_fill_(~ok, 0.0))
                valid[f] &= ok
        for f, (op, code, mask, orientation) in enumerate(families):
            if (op == "/" and "w" not in B) or (op == "\\" and "w" not in A):
                continue
            score = gains[f].mul_(n - 1).sqrt_().masked_fill_(~valid[f], -math.inf)
            if mask is not None:
                score.masked_fill_(~mask, -math.inf)
            self._offer_tile(buffers, score, lazy, code, A, B, orientation)

    @staticmethod
    def _sum_gain(A, Aw, B, Bv, cl, C_h, sign):
        """Gain of a_i + sign * b_j:  f ~ z_a + sign * t z_b with t = sigma_b / sigma_a."""
        t = B["sd"][None, :] / A["sd"][:, None]
        st = sign * t
        m1 = torch.addcmul(Aw["hz"][:, None], st, Bv["hz"][None, :])
        scale = torch.addcmul(Aw["s2h"][:, None], t.square(), Bv["s2h"][None, :])
        info = torch.addcmul(scale, st, C_h, value=2.0).addcmul_(m1, m1, value=-1.0 / cl["H"])
        U = torch.addcmul(Aw["gz"][:, None], st, Bv["gz"][None, :])
        if cl["G"] != 0.0:
            U.add_(m1, alpha=-cl["G"] / cl["H"])
        ok = info > _CONSTANT_CHILD * scale
        gain = U.square_().div_(info.clamp_min_(_TINY))
        return gain, ok & torch.isfinite(gain)

    @staticmethod
    def _product_gain(A, Aw, B, Bv, cl, C_h=None):
        """Gain of a_i * b_j:  f ~ rho_a z_b + rho_b z_a + z_a z_b (weighted moments)."""
        if C_h is None:
            C_h = Aw["zh"] @ B["z"].T
        ra, rb = A["rho"][:, None], B["rho"][None, :]
        s2a, s2b = Aw["s2h"][:, None], Bv["s2h"][None, :]
        Q = Aw["z2h"] @ B["z2"].T
        scale = Q + ra.square() * s2b + rb.square() * s2a
        m1 = C_h + ra * Bv["hz"][None, :] + rb * Aw["hz"][:, None]
        info = Q.addcmul_(C_h * ra, rb, value=2.0)
        info.addcmul_(Aw["zh"] @ B["z2"].T, ra, value=2.0)
        info.addcmul_(Aw["z2h"] @ B["z"].T, rb, value=2.0)
        info.addcmul_(ra.square(), s2b).addcmul_(rb.square(), s2a)
        info.addcmul_(m1, m1, value=-1.0 / cl["H"])
        U = Aw["zg"] @ B["z"].T
        U.addcmul_(ra, Bv["gz"][None, :]).addcmul_(Aw["gz"][:, None], rb)
        if cl["G"] != 0.0:
            U.add_(m1, alpha=-cl["G"] / cl["H"])
        ok = info > _CONSTANT_CHILD * scale
        gain = U.square_().div_(info.clamp_min_(_TINY))
        return gain, ok & torch.isfinite(gain)

    def _offer_tile(self, buffers, score, lazy, code, A, B, orientation):
        if buffers.floor > -math.inf:
            # Fast path: complexity only for entries that can enter some buffer.
            rl, cl = torch.nonzero(score > buffers.floor, as_tuple=True)
            if len(rl) == 0:
                return
            s = score[rl, cl]
            cplx = self._entry_complexity(A, B, rl, cl, code)
            bins = torch.bucketize(cplx, self.boundaries)
            keep = buffers.survivors(s, bins)
            rl, cl, s, cplx, bins = rl[keep], cl[keep], s[keep], cplx[keep], bins[keep]
        else:
            if "base" not in lazy:
                lazy["base"], lazy["nsym"] = self._unique(A, B)
            cplx_tile = lazy["nsym"] * self.log2[self._with_op(A, B, lazy["base"], code)]
            bins_tile = torch.bucketize(cplx_tile, self.boundaries)
            keep = buffers.survivors(score, bins_tile)
            rl, cl = torch.nonzero(keep, as_tuple=True)
            s, cplx, bins = score[rl, cl], cplx_tile[rl, cl], bins_tile[rl, cl]
        if len(rl) == 0:
            return
        rows, cols = rl + A["lo"], cl + B["lo"]
        if orientation == "sym":
            left, right = torch.minimum(rows, cols), torch.maximum(rows, cols)
        elif orientation == "rc":
            left, right = rows, cols
        else:
            left, right = cols, rows
        codes = torch.stack([torch.full_like(left, code), left, right], dim=1)
        buffers.insert(s, cplx, bins, codes)

    def _offer(self, buffers, score, cplx, code, left, right):
        score = torch.where(torch.isfinite(score), score, -math.inf)
        bins = torch.bucketize(cplx, self.boundaries)
        keep = buffers.survivors(score, bins)
        if keep.any():
            codes = torch.stack([torch.full_like(left[keep], code), left[keep], right[keep]], dim=1)
            buffers.insert(score[keep], cplx[keep], bins[keep], codes)
