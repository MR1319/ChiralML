"""Materialized feature levels: values, statistics, metadata, and deduplication.

Values are stored feature-major (one contiguous row per feature) so that
column blocks used by the streaming scorer are contiguous slices.

Complexity follows the SyMANTIC definition, computed from each expression
tree instead of padded index tensors:

    complexity = (#symbols) * log2(#distinct symbols),

where symbols are variable occurrences plus operator occurrences. Primitive
variables have complexity 1, as in the original implementation.

Units (optional) are canonical exponent vectors (see units.py). Children that
violate unit rules -- '+'/'-' of different units, or (with strict units) a
transcendental function or +-1 shift of a dimensional quantity -- are never
generated. In dimensional mode constant columns (physical constants) are kept
as building blocks: they enter products and quotients but are never selected
on their own, and expressions built only from constants stay constants.
"""
import math

import numpy as np
import torch

from .units import unit_keys

# Relative spread below which a feature is treated as constant (e.g.
# (x3/x4)*(x4/x3), which is 1 up to round-off).
CONSTANT_TOLERANCE = 1e-10
_FINGERPRINT_SEED = 20240809
_MIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9))
_META = ("op", "left", "right", "nsym", "varset", "opset", "complexity", "units", "ukey", "lo", "hi")
_OVERFLOW = 1e150


def child_interval(names, op, lo_l, hi_l, lo_r=None, hi_r=None):
    """Interval arithmetic for one operator over arrays of parent intervals.

    Returns (lo, hi, ok): ok is False where the child may be undefined (log of a
    non-positive value, division by an interval containing 0, ...) or may
    overflow. ``names`` maps operator codes to operator strings.
    """
    name = names[op]
    with np.errstate(all="ignore"):
        if lo_r is not None:
            if name == "+":
                lo, hi = lo_l + lo_r, hi_l + hi_r
            elif name == "-":
                lo, hi = lo_l - hi_r, hi_l - lo_r
            else:
                ok_div = np.ones_like(lo_l, dtype=bool)
                if name == "/":
                    ok_div = (lo_r > 0) | (hi_r < 0)
                    lo_r, hi_r = 1.0 / hi_r, 1.0 / lo_r
                corners = np.stack([lo_l * lo_r, lo_l * hi_r, hi_l * lo_r, hi_l * hi_r])
                lo, hi = corners.min(0), corners.max(0)
                lo, hi = np.where(ok_div, lo, np.nan), np.where(ok_div, hi, np.nan)
            return _finish(lo, hi)
        x0, x1 = lo_l, hi_l
        spans_zero = (x0 <= 0) & (x1 >= 0)
        ok = np.ones_like(x0, dtype=bool)
        if name == "exp":
            lo, hi = np.exp(x0), np.exp(x1)
        elif name == "exp(-1)":
            lo, hi = np.exp(-x1), np.exp(-x0)
        elif name in ("ln", "log"):
            ok = x0 > 0
            f = np.log if name == "ln" else np.log10
            lo, hi = f(x0), f(x1)
        elif name in ("sin", "cos"):
            lo, hi = np.full_like(x0, -1.0), np.full_like(x0, 1.0)
        elif name in ("sinh", "tanh"):
            f = np.sinh if name == "sinh" else np.tanh
            lo, hi = f(x0), f(x1)
        elif name == "cosh":
            hi = np.maximum(np.cosh(x0), np.cosh(x1))
            lo = np.where(spans_zero, 1.0, np.minimum(np.cosh(x0), np.cosh(x1)))
        elif name == "^-1":
            ok = ~spans_zero
            lo, hi = 1.0 / x1, 1.0 / x0
        elif name in ("+1", "-1"):
            shift = 1.0 if name == "+1" else -1.0
            lo, hi = x0 + shift, x1 + shift
        elif name in ("/2", "/2pi", "*2pi"):
            scale = {"/2": 0.5, "/2pi": 1 / (2 * math.pi), "*2pi": 2 * math.pi}[name]
            lo, hi = x0 * scale, x1 * scale
        elif name.startswith("hinge"):
            # the knot lies inside the parent's training range, hence inside [x0, x1]
            lo, hi = np.zeros_like(x0), np.maximum(x1 - x0, 0.0)
        else:  # pow(k)
            from fractions import Fraction
            k = float(Fraction(name[4:-1]))
            if float(k).is_integer():
                # Integer powers are monotone on intervals that exclude 0.
                ok = ~spans_zero if k < 0 else ok
                a, b = x0 ** k, x1 ** k
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                if k > 0 and int(k) % 2 == 0:
                    lo = np.where(spans_zero, 0.0, lo)
            else:
                ok = x0 > 0 if k < 0 else x0 >= 0
                a, b = np.abs(x0) ** k, np.abs(x1) ** k
                lo, hi = np.minimum(a, b), np.maximum(a, b)
        lo, hi = np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)
        return _finish(lo, hi)


def _finish(lo, hi):
    ok = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo) < _OVERFLOW) & (np.abs(hi) < _OVERFLOW)
    return np.where(ok, lo, np.nan), np.where(ok, hi, np.nan), ok


def complexity_of(nsym, unique):
    """Vectorized complexity; inputs are integer arrays/tensors."""
    if isinstance(nsym, torch.Tensor):
        return nsym * torch.log2(unique)
    return nsym * np.log2(unique)


def _hash_columns(columns):
    """Combine integer key columns into one uint64 hash per row."""
    h = np.zeros(columns.shape[0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(columns.shape[1]):
            h ^= columns[:, j].astype(np.uint64) * _MIX[j % 3]
            h = (h << np.uint64(13)) | (h >> np.uint64(51))
    return h


class FeatureBank:
    """Materialized features (levels 0..L) with per-feature statistics."""

    def __init__(self, values, names, operators, device="cpu", chunk_bytes=256 << 20, units=None,
                 allow_constants=False, bounds=None, safe_domain=False):
        """bounds: (lo, hi) arrays giving each primitive's input domain (default:
        the data range). With safe_domain, children that may be undefined or
        overflow anywhere in that domain are never generated."""
        X = torch.tensor(np.asarray(values, dtype=np.float64).T, dtype=torch.float64, device=device)
        self.device = torch.device(device)
        self.ops = operators
        self.n = X.shape[1]
        self.chunk_rows = max(1, chunk_bytes // (8 * self.n))
        generator = torch.Generator(device="cpu").manual_seed(_FINGERPRINT_SEED)
        self.projections = torch.randn(self.n, 2, generator=generator, dtype=torch.float64).to(self.device)
        self.projections /= self.projections.norm(dim=0)
        p = X.shape[0]
        units = np.zeros((p, 0)) if units is None else np.asarray(units, dtype=np.float64).reshape(p, -1)
        self.unit_size = units.shape[1]
        self.dimensionless_key = int(unit_keys(np.zeros((1, self.unit_size)))[0])
        self.allow_constants = allow_constants
        self.safe_domain = safe_domain
        raw = np.asarray(values, dtype=np.float64)
        lo, hi = (raw.min(axis=0), raw.max(axis=0)) if bounds is None else map(np.asarray, bounds)
        self.lo, self.hi = np.empty(0), np.empty(0)
        self.X = X[:0]
        self.names = []
        self.op = np.empty(0, np.int16)
        self.left = np.empty(0, np.int64)
        self.right = np.empty(0, np.int64)
        self.nsym = np.empty(0, np.int32)
        self.varset = np.zeros((0, p), bool)
        self.opset = np.zeros((0, len(operators)), bool)
        self.complexity = np.empty(0)
        self.units = np.zeros((0, self.unit_size))
        self.ukey = np.empty(0, np.int64)
        self.level = np.empty(0, np.int16)
        self.keys = np.empty(0, np.uint64)
        empty = X.new_empty(0)
        self.mu = self.sd = self.sd_unbiased = self.s2 = empty
        self.mu_w = self.sd_w = self.s2_w = empty
        self.w_ok = self.const = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.level_start = [0]
        self.p = p
        stats = self.column_stats(X)
        self.const_primitive = (stats["const"] & allow_constants).cpu().numpy()
        meta = {
            "op": np.full(p, -1, np.int16), "left": np.arange(p), "right": np.full(p, -1),
            "nsym": np.ones(p, np.int32), "varset": np.eye(p, dtype=bool),
            "opset": np.zeros((p, len(operators)), bool), "complexity": np.ones(p),
            "units": units, "ukey": unit_keys(units),
            "lo": np.asarray(lo, dtype=np.float64), "hi": np.asarray(hi, dtype=np.float64),
        }
        self._append(X, list(names), meta, level=0)
        if not (~self.const).any():
            raise ValueError("No finite, non-constant primitive features")

    # ------------------------------------------------------------------ sizes
    def __len__(self):
        return self.X.shape[0]

    @property
    def levels(self):
        return len(self.level_start) - 1

    def block(self, level):
        return self.level_start[level], self.level_start[level + 1]

    # ------------------------------------------------------------ statistics
    def column_stats(self, X):
        """Validity, constancy, and moments of feature rows; reciprocals included.

        ``valid`` means finite and non-constant (a regressor). For constants the
        effective spread is |mean|, which makes the streaming moment formulas
        exact for children such as c*x (see screening.py).
        """
        mu = X.mean(dim=1)
        sd = (X - mu[:, None]).square().mean(dim=1).sqrt()
        finite = torch.isfinite(X).all(dim=1) & torch.isfinite(sd)
        spread = (sd > CONSTANT_TOLERANCE * mu.abs()) & (sd > 0)
        valid = finite & spread
        const = finite & ~spread & (mu != 0)
        W = torch.reciprocal(X)
        mu_w = W.mean(dim=1)
        sd_w = (W - mu_w[:, None]).square().mean(dim=1).sqrt()
        w_finite = torch.isfinite(W).all(dim=1) & (X != 0).all(dim=1) & torch.isfinite(sd_w)
        w_spread = (sd_w > CONSTANT_TOLERANCE * mu_w.abs()) & (sd_w > 0)
        w_ok = w_finite & ((valid & w_spread) | (const & (mu_w != 0)))
        return {"valid": valid, "const": const, "mu": mu, "sd": torch.where(const, mu.abs(), sd),
                "mu_w": mu_w, "sd_w": torch.where(const, mu_w.abs(), sd_w), "w_ok": w_ok}

    def exact_keys(self, X):
        """Scale-sensitive fingerprints: equal keys <=> equal values (to ~1e-9)."""
        norm = X.norm(dim=1)
        proj = (X @ self.projections) / norm[:, None].clamp_min(1e-300)
        cols = torch.stack([torch.round(proj[:, 0] * 2**30), torch.round(proj[:, 1] * 2**30),
                            torch.round(torch.log2(norm.clamp_min(1e-300)) * 2**30)], dim=1)
        return _hash_columns(cols.to(torch.int64).cpu().numpy())

    def affine_keys(self, X):
        """Fingerprints invariant to x -> a*x + b (a != 0): equivalent regressors."""
        mu = X.mean(dim=1, keepdim=True)
        Z = X - mu
        Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-300)
        proj = Z @ self.projections
        sign = torch.where(proj[:, :1] < 0, -1.0, 1.0)
        proj = proj * sign
        cols = torch.round(proj * 2**26).to(torch.int64).cpu().numpy()
        return _hash_columns(cols)

    # -------------------------------------------------------------- children
    def child_meta(self, op, left, right):
        """Metadata and unit validity of children (op, left, right); right < 0 means unary."""
        op = np.asarray(op, np.int64)
        left = np.asarray(left, np.int64)
        right = np.asarray(right, np.int64)
        binary = right >= 0
        r = np.where(binary, right, left)
        nsym = self.nsym[left] + np.where(binary, self.nsym[r] + 1, 1)
        varset = self.varset[left] | np.where(binary[:, None], self.varset[r], False)
        opset = self.opset[left] | np.where(binary[:, None], self.opset[r], False)
        opset[np.arange(len(op)), op] = True
        unique = varset.sum(axis=1) + opset.sum(axis=1)
        units, valid = self._child_units(op, left, r, binary)
        lo, hi = np.full(len(op), np.nan), np.full(len(op), np.nan)
        for code in np.unique(op):
            rows = op == code
            if binary[rows][0]:
                lo[rows], hi[rows], ok = child_interval(self.ops.names, code, self.lo[left[rows]], self.hi[left[rows]],
                                                        self.lo[r[rows]], self.hi[r[rows]])
            else:
                lo[rows], hi[rows], ok = child_interval(self.ops.names, code, self.lo[left[rows]], self.hi[left[rows]])
            if self.safe_domain:
                valid[rows] &= ok
        return {"op": op.astype(np.int16), "left": left, "right": right, "nsym": nsym.astype(np.int32),
                "varset": varset, "opset": opset, "complexity": complexity_of(nsym, unique).astype(np.float64),
                "units": units, "ukey": unit_keys(units), "lo": lo, "hi": hi, "valid": valid}

    def _child_units(self, op, left, right, binary):
        ul, ur = self.units[left], self.units[right]
        units = ul.copy()
        valid = np.ones(len(op), dtype=bool)
        for code in np.unique(op):
            rows = op == code
            name = self.ops.names[code]
            if name in ("+", "-"):
                valid[rows] = self.ukey[left[rows]] == self.ukey[right[rows]]
            elif name == "*":
                units[rows] = ul[rows] + ur[rows]
            elif name == "/":
                units[rows] = ul[rows] - ur[rows]
            else:
                units[rows] = self.ops.unit_factor[code] * ul[rows]
                if self.ops.requires_dimensionless[code]:
                    valid[rows] = self.ukey[left[rows]] == self.dimensionless_key
        return units, valid

    def child_values(self, op, left, right):
        """Evaluate children row-wise; returns a (len(op), n) tensor."""
        op = np.asarray(op, np.int64)
        left = np.asarray(left, np.int64)
        right = np.asarray(right, np.int64)
        out = self.X.new_empty((len(op), self.n))
        for code in np.unique(op):
            rows = np.nonzero(op == code)[0]
            li = torch.as_tensor(left[rows], device=self.device)
            if right[rows[0]] < 0:
                values = self.ops.apply(int(code), self.X[li])
            else:
                values = self.ops.apply(int(code), self.X[li], self.X[torch.as_tensor(right[rows], device=self.device)])
            out[torch.as_tensor(rows, device=self.device)] = values
        return out

    def child_name(self, op, left, right):
        if op < 0:
            return self.names[left]
        if right < 0:
            knot = float(self.ops.knot(int(op), self.X[left])) if self.ops.is_hinge(int(op)) else None
            return self.ops.format(int(op), self.names[left], knot=knot)
        return self.ops.format(int(op), self.names[left], self.names[right])

    # ------------------------------------------------------------ appending
    def add_level(self, op, left, right):
        """Materialize children as a new level: drop invalid, dedup, append.

        Among duplicates (equal values up to round-off, including duplicates of
        lower levels) the lowest-complexity expression is kept.
        """
        meta = self.child_meta(op, left, right)
        keep = np.nonzero(meta.pop("valid"))[0]
        meta = {k: v[keep] for k, v in meta.items()}
        level = self.levels
        kept_total = 0
        order = np.argsort(meta["complexity"], kind="stable")
        meta = {k: v[order] for k, v in meta.items()}
        for start in range(0, len(order), self.chunk_rows):
            part = {k: v[start:start + self.chunk_rows] for k, v in meta.items()}
            values = self.child_values(part["op"], part["left"], part["right"])
            names = [self.child_name(o, l, r) for o, l, r in zip(part["op"], part["left"], part["right"])]
            kept_total += len(self._append(values, names, part, level))
        if level == self.levels:  # nothing survived; keep an empty level for bookkeeping
            self.level_start.append(len(self))
        return kept_total

    def _append(self, X, names, meta, level):
        stats = self.column_stats(X)
        # Constants survive only when built entirely from constant primitives.
        only_constants = ~(meta["varset"] & ~self.const_primitive[None, :]).any(axis=1)
        allowed = torch.as_tensor(only_constants, device=self.device) & self.allow_constants
        keep = stats["valid"] | (stats["const"] & allowed)
        idx = torch.nonzero(keep).flatten()
        X = X[idx]
        stats = {k: v[idx] for k, v in stats.items()}
        sel = idx.cpu().numpy()
        keys = self.exact_keys(X)
        _, first = np.unique(keys, return_index=True)
        first = np.sort(first)
        first = first[~np.isin(keys[first], self.keys)]
        sel = sel[first]
        t = torch.as_tensor(first, device=self.device)
        X = X[t]
        stats = {k: v[t] for k, v in stats.items()}
        if level == self.levels:
            self.level_start.append(len(self))
        self.X = torch.cat([self.X, X])
        self.names.extend(names[i] for i in sel)
        for field in _META:
            setattr(self, field, np.concatenate([getattr(self, field), meta[field][sel]]))
        self.level = np.concatenate([self.level, np.full(len(sel), level, np.int16)])
        self.keys = np.concatenate([self.keys, keys[first]])
        n = self.n
        const = stats["const"]
        self.const = torch.cat([self.const, const])
        self.mu = torch.cat([self.mu, stats["mu"]])
        self.sd = torch.cat([self.sd, stats["sd"]])
        self.sd_unbiased = torch.cat([self.sd_unbiased, torch.where(const, math.inf, stats["sd"] * math.sqrt(n / max(n - 1, 1)))])
        self.s2 = torch.cat([self.s2, torch.where(const, 0.0, float(n))])
        self.mu_w = torch.cat([self.mu_w, stats["mu_w"]])
        self.sd_w = torch.cat([self.sd_w, stats["sd_w"]])
        self.w_ok = torch.cat([self.w_ok, stats["w_ok"]])
        self.s2_w = torch.cat([self.s2_w, torch.where(const, 0.0, float(n))])
        self.level_start[-1] = len(self)
        return sel

    def memory_bytes(self):
        return self.X.numel() * 8
