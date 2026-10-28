"""Memory-bounded SyMANTIC: streamed last level, exact SIS, reduced l0 search.

Search semantics follow the original pipeline (non-dimensional and, with
``dimensionality``, dimensional):

* level L contains binary(new x all) and unary(new) children of the previous
  level (the original recomputes old x old pairs and removes duplicates);
* each sparse-regression round screens features by |corr(residual, f)|, keeps
  the top ``sis_features`` per complexity-quantile bin plus overall, and fits
  every ``d``-subset of (previously selected) U (newly screened) features;
* the overall best ``d``-term model sets the next residual.

The deepest level is never materialized; see ``screening.py``. Lower levels
are materialized once, without invalid, constant, or duplicate features.

Dimensional mode uses exponent-vector units (``units.py``): '+'/'-' combine
only equal units, products/quotients add/subtract exponents, and with
``output_dim`` only features carrying the target unit enter the regression.
Physical constants (constant columns) are kept as building blocks.
"""
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .._data import prepare_dataframe
from .bank import FeatureBank
from .operators import OperatorSet
from .screening import LevelScreen, explicit_scores
from .sparse import ParetoStore, format_equation, full_precision_equation, subset_search
from .units import UnitSystem

DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class FastSymanticModel:
    """Drop-in alternative to ``SymanticModel`` (non-dimensional and dimensional).

    Dimensional arguments (as in SymanticModel)
    -------------------------------------------
    dimensionality : one unit per input column, e.g. ['L', 'L/T', '1'] or
        SymPy expressions; '1' means dimensionless.
    output_dim : unit of the target; only features with this unit are fitted.
    relational_units : equivalences such as [('u1*u2', 'u3')].
    strict_units : True (default) allows transcendental functions and +-1
        shifts only on dimensionless quantities. False reproduces the original
        implementation, which applies them to dimensional quantities too.

    Additional arguments
    --------------------
    max_level : deepest expansion level tried in adaptive mode.
    memory_limit_gb : budget for materializing a level; the deepest level is
        always streamed and needs only O(n * #parents) memory.
    safe_domain : only build descriptors that are finite everywhere in the input
        domain (interval arithmetic): no log/sqrt of possibly non-positive
        values, no division by quantities that may cross zero, no overflow.
    domain_margin : the default domain is the training range of each input,
        widened by this fraction of its span on each side.
    domain : explicit {column: (low, high)} bounds overriding the default.
    optimize_constants : after the search, fit nonlinear constants inside each
        Pareto model's descriptors (e.g. exp(x) -> exp(a*x+b), x**2 -> x**p) by
        variable projection + Levenberg-Marquardt; refinements that improve
        BIC join the Pareto front as extra (more complex) models.
    constant_restarts : random restarts of the constant optimizer.
    max_parents : if a level is too large to materialize, keep only this many
        of its best-screened features as parents for the next level (a beam).
        ``None`` keeps the search exact and stops instead.
    oversample : candidate buffer size per bin, as a multiple of sis_features,
        before exact rescoring and duplicate removal.
    bins_from : 'all' (original) computes complexity quantiles over every
        feature; 'materialized' uses levels below the deepest one, so the low
        bins still hold simple features when the deepest level dominates.
    share_bins : if set, the top ``share_bins`` features of every complexity
        bin also join the selected set (and one shared l0 pool), letting simple
        and compound features combine. Pool sizes grow by about
        ``sis_features + n_bins * share_bins`` per term.
    """

    def __init__(self, df, operators=None, n_expansion=None, n_term=None, sis_features=20, device="cpu",
                 initial_screening=None, metrics=(0.06, 0.995), disp=False, quantiles=DEFAULT_QUANTILES,
                 max_level=3, memory_limit_gb=8.0, max_parents=None, oversample=4, tile=None,
                 tile_memory_mb=512, bins_from="all", share_bins=None, dimensionality=None, output_dim=None,
                 relational_units=None, strict_units=True, safe_domain=False, domain_margin=0.1, domain=None,
                 optimize_constants=False, constant_restarts=3):
        if bins_from not in ("all", "materialized"):
            raise ValueError("bins_from must be 'all' or 'materialized'")
        if not operators:
            raise ValueError("Provide the operator set")
        if dimensionality is not None and len(dimensionality) != df.shape[1] - 1:
            raise ValueError("Provide one dimensionality entry per input feature column")
        self.dimensionality = None if dimensionality is None else list(dimensionality)
        self.safe_domain = safe_domain
        self.domain_margin = domain_margin
        self.domain = dict(domain or {})
        self.optimize_constants = optimize_constants
        self.constant_restarts = constant_restarts
        self.refine_top = 8
        self.output_dim = output_dim
        self.relational_units = relational_units
        self.strict_units = strict_units
        self.df = df
        self.operators = list(operators)
        self.n_expansion = n_expansion
        self.n_term = 3 if n_term is None else n_term
        self.k = 10 if sis_features is None else sis_features
        self.device = torch.device(device)
        self.initial_screening = initial_screening
        self.metrics = tuple(metrics)
        self.disp = disp
        self.quantiles = tuple(quantiles)
        self.max_level = max_level
        self.memory_limit = memory_limit_gb * 1e9
        self.max_parents = max_parents
        self.capacity = max(oversample * self.k, self.k + 32)
        self.tile = tile
        self.tile_memory = tile_memory_mb << 20
        self.bins_from = bins_from
        self.share_bins = share_bins
        self.final_df = None
        self.history = []

    # ------------------------------------------------------------------ fit
    def _build(self, df):
        """Prepare data, units, and the level-0 feature bank; returns the target array."""
        dimensional = self.dimensionality is not None
        features, target, positions = prepare_dataframe(df, drop_constant=not dimensional)
        units, self.target_units, self.unit_system = None, None, None
        if dimensional:
            self.unit_system = UnitSystem([self.dimensionality[i] for i in positions],
                                          self.relational_units, self.output_dim)
            units, self.target_units = self.unit_system.primitive, self.unit_system.target
        if self.initial_screening is not None:
            features, kept = _screen_inputs(features, target, *self.initial_screening)
            units = None if units is None else units[kept]
            positions = [positions[i] for i in kept]
        self.input_columns_ = [df.columns[1 + i] for i in positions]  # original input labels used
        values = features.to_numpy(np.float64)
        lo, hi = values.min(axis=0), values.max(axis=0)
        span = hi - lo
        lo, hi = lo - self.domain_margin * span, hi + self.domain_margin * span
        for label, (a, b) in self.domain.items():
            if label in self.input_columns_:
                j = self.input_columns_.index(label)
                lo[j], hi[j] = a, b
        self.domain_lo_, self.domain_hi_ = lo, hi
        self.ops = OperatorSet(self.operators, strict_units=self.strict_units)
        self.bank = FeatureBank(values, features.columns, self.ops, self.device, units=units,
                                allow_constants=dimensional, bounds=(lo, hi), safe_domain=self.safe_domain)
        self.n = len(target)
        return target

    def fit(self):
        target = self._build(self.df)
        y = torch.as_tensor(target, dtype=torch.float64, device=self.device)
        self.y_mean = float(y.mean())
        self.yc = y - self.y_mean
        self.tss = float(self.yc.square().sum())
        self.refine_pool = []
        self.store = ParetoStore()
        self.store.add([{"complexity": 0.0, "rmse": math.sqrt(self.tss / self.n), "r2": 0.0,
                         "terms": [str(self.y_mean)], "coefficients": [1.0], "intercept": 0.0}])

        if self.n_expansion is not None:
            depth = self.n_expansion - 1
            for level in range(1, depth):
                if not self._materialize(level, parents=None):
                    raise MemoryError(f"Level {level} exceeds memory_limit_gb; increase it or set max_parents")
            best = self._regress(depth if depth >= 1 else None)
            return best["rmse"], best["equation"], best["r2"]

        for level in range(1, self.max_level + 1):
            best, parents = self._regress(level, collect_parents=True)
            if self._met(best) or level == self.max_level:
                break
            if not self._materialize(level, parents):
                break
        if self.optimize_constants:
            self._refine_constants()
        return self._frontier()

    # ------------------------------------------------------- expansion
    def _children(self, level):
        """Exact specification of every child of block level-1 (for materialization)."""
        bank = self.bank
        a0, a1 = bank.block(level - 1)
        new = np.arange(a0, a1)
        old = np.arange(0, a0)
        ops, lefts, rights = [], [], []
        ii, jj = np.triu_indices(len(new), k=1)
        pairs_new = (new[ii], new[jj])
        pairs_old = (np.repeat(old, len(new)), np.tile(new, len(old)))
        for op, code in self.ops.binary.items():
            for left, right in (pairs_old, pairs_new):
                lefts.append(left)
                rights.append(right)
                ops.append(np.full(len(left), code))
                if op == "/":
                    lefts.append(right)
                    rights.append(left)
                    ops.append(np.full(len(left), code))
        for code in self.ops.unary:
            lefts.append(new)
            rights.append(np.full(len(new), -1))
            ops.append(np.full(len(new), code))
        return np.concatenate(ops), np.concatenate(lefts), np.concatenate(rights)

    def _count_children(self, level):
        """Number of unit-consistent children of block level-1 (before value checks)."""
        bank = self.bank
        a0, a1 = bank.block(level - 1)
        m, old = a1 - a0, a0
        pairs = m * old + m * (m - 1) // 2
        new_keys, old_keys = bank.ukey[a0:a1], bank.ukey[:a0]
        keys, new_counts = np.unique(new_keys, return_counts=True)
        old_counts = np.array([(old_keys == k).sum() for k in keys])
        same_unit_pairs = int((new_counts * old_counts + new_counts * (new_counts - 1) // 2).sum())
        total = 0
        for op in self.ops.binary:
            total += same_unit_pairs if op in ("+", "-") else pairs * (2 if op == "/" else 1)
        dimensionless = int((new_keys == bank.dimensionless_key).sum())
        for code in self.ops.unary:
            total += dimensionless if self.ops.requires_dimensionless[code] else m
        return total

    def _materialize(self, level, parents):
        count = self._count_children(level)
        need = count * self.n * 8 * 2  # values plus transient copy during append
        start = time.perf_counter()
        if need <= self.memory_limit:
            kept = self.bank.add_level(*self._children(level))
            how = "exact"
        elif self.max_parents and parents is not None:
            kept = self.bank.add_level(*parents)
            how = f"beam (top {len(parents[0])} of {count:,} by screening score)"
        else:
            warnings.warn(f"Level {level} has {count:,} features (~{need / 1e9:.1f} GB); it exceeds "
                          f"memory_limit_gb={self.memory_limit / 1e9:g}. Stopping expansion. "
                          "Raise memory_limit_gb or set max_parents to continue with a beam.", stacklevel=3)
            return False
        if self.disp:
            print(f"[level {level}] materialized {kept:,} distinct valid features ({how}) "
                  f"in {time.perf_counter() - start:.2f}s; bank = {len(self.bank):,} features, "
                  f"{self.bank.memory_bytes() / 1e9:.3f} GB")
        return True

    # ------------------------------------------------- sparse regression
    def _regress(self, level, collect_parents=False):
        start = time.perf_counter()
        screen = LevelScreen(self.bank, level, self.quantiles, self.capacity, tile=self.tile,
                             memory_bytes=self.tile_memory, bins_from=self.bins_from,
                             target_units=self.target_units)
        if self.disp:
            print(f"[level {level}] screening {screen.n_candidates:,} candidate features "
                  f"({len(self.bank):,} materialized, tile={screen.tile})")
        parent_pool = {}
        selected = []
        r = self.yc.clone()
        best = None
        nb = len(screen.boundaries)
        for d in range(1, self.n_term + 1):
            geo = LevelScreen.geometry(r)
            buffers = screen.run_round(geo)
            picks = self._select(buffers, geo, selected)
            if self.optimize_constants:
                picks[nb] = self._refine_picks(picks[nb], geo)
            if self.share_bins:
                picks = _share(picks, self.share_bins)
            for slot in range(nb + 1):
                if not picks[slot] and slot != nb:
                    continue
                pool = selected + picks[slot]
                out = self._fit_pool(pool, d, keep=slot == nb)
                if out is None:
                    continue
                rows, best_row = out
                self.store.add(rows)
                if slot == nb:
                    best = best_row
            if collect_parents and self.max_parents:
                self._collect_parents(screen, parent_pool, geo)
            if best is None:
                break
            r = best["residual"]
            selected = selected + picks[nb]
            if self.disp:
                print(f"[level {level}] {d}-term: rmse={best['rmse']:.6g} r2={best['r2']:.6f}  {best['equation']}")
            if self._met(best):
                break
        self.history.append({"level": level, "candidates": screen.n_candidates, "bank": len(self.bank),
                             "seconds": time.perf_counter() - start, "rmse": best["rmse"] if best else None})
        if not collect_parents:
            return best
        parents = None
        if parent_pool:
            items = sorted(parent_pool.items(), key=lambda kv: -kv[1])[: self.max_parents]
            codes = np.array([code for code, _ in items])
            parents = (codes[:, 0], codes[:, 1], codes[:, 2])
        return best, parents

    def _collect_parents(self, screen, pool, geo):
        """Best streamed children for this residual (used only for beam expansion)."""
        saved = screen.capacity
        screen.capacity = self.max_parents
        buffers = screen.run_round(geo)
        screen.capacity = saved
        for slot in range(len(buffers.score)):
            for score, code in zip(buffers.score[slot].tolist(), buffers.code[slot].tolist()):
                if code[0] >= 0:
                    key = tuple(code)
                    pool[key] = max(pool.get(key, -math.inf), score)

    def _select(self, buffers, geo, selected):
        """Exact rescoring, duplicate removal, and top-k per bin (geo: LevelScreen.geometry)."""
        bank = self.bank
        codes = torch.cat(buffers.code)
        slots = torch.cat([torch.full((len(c),), i, device=self.device) for i, c in enumerate(buffers.code)])
        cplx = torch.cat(buffers.cplx)
        picks = [[] for _ in buffers.code]
        if len(codes) == 0:
            return picks
        unique, inverse = torch.unique(codes, dim=0, return_inverse=True)
        ops, lefts, rights = (unique[:, j].cpu().numpy() for j in range(3))
        values = bank.X.new_empty((len(unique), self.n))
        is_bank = ops < 0
        if is_bank.any():
            values[torch.as_tensor(np.nonzero(is_bank)[0], device=self.device)] = bank.X[torch.as_tensor(lefts[is_bank], device=self.device)]
        if (~is_bank).any():
            rows = torch.as_tensor(np.nonzero(~is_bank)[0], device=self.device)
            values[rows] = bank.child_values(ops[~is_bank], lefts[~is_bank], rights[~is_bank])
        stats = bank.column_stats(values)
        valid, mu = stats["valid"], stats["mu"]
        sd_unbiased = stats["sd"] * math.sqrt(self.n / (self.n - 1))
        score = explicit_scores(values, geo, valid)
        valid &= torch.isfinite(score)
        keys = bank.affine_keys(torch.nan_to_num(values))
        taken = {item["akey"] for item in selected}
        score_np, valid_np, cplx_np = score.cpu().numpy(), valid.cpu().numpy(), cplx.cpu().numpy()
        inverse_np, slots_np = inverse.cpu().numpy(), slots.cpu().numpy()
        for slot in range(len(picks)):
            entries = np.nonzero(slots_np == slot)[0]
            groups = {}
            for e in entries:
                u = inverse_np[e]
                if not valid_np[u] or keys[u] in taken:
                    continue
                best = groups.get(keys[u])
                # Prefer the simplest expression of equivalent features.
                if best is None or (cplx_np[e], -score_np[u]) < (cplx_np[best[0]], -score_np[best[1]]):
                    groups[keys[u]] = (e, u)
            ranked = sorted(groups.values(), key=lambda eu: (-score_np[eu[1]], cplx_np[eu[0]]))[: self.k]
            for e, u in ranked:
                picks[slot].append({"name": bank.child_name(ops[u], lefts[u], rights[u]),
                                    "complexity": float(cplx_np[e]), "values": values[u],
                                    "mu": mu[u], "sd": sd_unbiased[u], "akey": keys[u],
                                    "code": (int(ops[u]), int(lefts[u]), int(rights[u]))})
        return picks

    def _fit_pool(self, pool, d, keep=False):
        if len(pool) < d:
            return None
        Z = torch.stack([(p["values"] - p["mu"]) / p["sd"] for p in pool], dim=1)
        cplx = torch.tensor([p["complexity"] for p in pool], dtype=torch.float64, device=self.device)
        top = self.refine_top if (keep and self.optimize_constants) else 0
        result = subset_search(Z, self.yc, d, cplx, keep_top=top)
        if result is None:
            return None
        idx, beta, rss, cplx_sub, best = result[:5]
        mus = torch.stack([p["mu"] for p in pool])
        sds = torch.stack([p["sd"] for p in pool])
        rows = []
        for i in range(len(idx)):
            rows.append(self._row(pool, idx[i], beta[i], float(rss[i]), float(cplx_sub[i]), mus, sds))
        best_row = rows[best]
        best_row["residual"] = self.yc - Z[:, idx[best]] @ beta[best]
        best_row["equation"] = full_precision_equation(best_row["terms"], best_row["coefficients"], best_row["intercept"])
        if top:  # lowest-RSS subsets of any complexity: candidates for constant refinement
            t_idx, t_beta, t_rss, t_cplx = result[5]
            self.refine_pool.extend(self._row(pool, t_idx[i], t_beta[i], float(t_rss[i]), float(t_cplx[i]), mus, sds)
                                    for i in range(len(t_idx)))
        return rows, best_row

    def _row(self, pool, idx, beta, rss, cplx, mus, sds):
        idx_list = idx.tolist()
        coef = beta / sds[idx]
        intercept = self.y_mean - float((beta * mus[idx] / sds[idx]).sum())
        return {"complexity": cplx, "rmse": math.sqrt(max(rss, 0.0) / self.n), "r2": 1.0 - rss / self.tss,
                "terms": [pool[j]["name"] for j in idx_list], "coefficients": coef.tolist(),
                "intercept": intercept, "codes": [pool[j]["code"] for j in idx_list],
                "term_params": [pool[j].get("param") for j in idx_list]}

    # ------------------------------------------------------ inner constants
    def _primitive_values(self):
        """Feature-major primitive inputs indexed by input-column position."""
        bank = self.bank
        prim = [i for i in range(len(bank)) if bank.op[i] < 0]
        values = bank.X.new_zeros((max(int(bank.left[i]) for i in prim) + 1, self.n))
        for i in prim:
            values[int(bank.left[i])] = bank.X[i]
        return values

    def _refine_picks(self, picks, geo):
        """Screening-time constants: fit each new descriptor's inner constants to the current residual.

        The target is the working response g/h with weights h of every logit (for
        squared loss: the residual with unit weights). BIC-accepted refinements
        are added next to the original descriptors, so the subset search can
        choose them; e.g. exp(x) becomes exp(0.44*x) when that fits better.
        """
        from .constants import ParamModel, domain_finite, fit_weighted, projection_rss, refined_complexity
        if not picks:
            return picks
        values = self._primitive_values()
        W = torch.stack([c["h"] for c in geo], 1)
        Y = torch.stack([c["g"] for c in geo], 1) / W
        taken = {p["akey"] for p in picks}
        out = list(picks)
        for pick in picks:
            if pick.get("param") is not None:
                continue
            model = ParamModel(self.bank, [pick["code"]])
            if model.size == 0:
                continue
            fitted = fit_weighted(model, values, Y, W, restarts=1, iterations=40)
            if fitted is None:
                continue
            theta, rss = fitted
            base = projection_rss(model.features(model.init, values), Y, W)
            m = model.changed(theta)
            if m == 0 or rss <= 0 or self.n * math.log(rss / max(base, 1e-300)) + m * math.log(self.n) >= 0:
                continue
            if self.safe_domain and not domain_finite(model, theta, self.bank):
                continue
            with torch.no_grad():
                vals = model.features(theta, values)[:, 0]
            akey = self.bank.affine_keys(vals[None])[0]
            if akey in taken or not torch.isfinite(vals).all() or float(vals.std()) <= 0:
                continue
            taken.add(akey)
            out.append({"name": model.terms(theta)[0], "values": vals, "mu": vals.mean(), "sd": vals.std(),
                        "complexity": refined_complexity(self.bank, [pick["code"]], model, theta),
                        "akey": akey, "code": pick["code"], "param": (model, theta)})
        return out

    def _refine_constants(self):
        """Fit nonlinear constants inside each Pareto model's descriptors (BIC-accepted)."""
        from .constants import ParamModel, domain_finite, fit_regression, refined_complexity, start_vector
        values = self._primitive_values()
        y = self.yc + self.y_mean
        accepted = []
        for row in _refinement_candidates(self.store.frontier(), self.refine_pool):
            model = ParamModel(self.bank, row["codes"])
            if model.size == 0:
                continue
            start = start_vector(model, row.get("term_params"))
            fitted = fit_regression(model, values, y, restarts=self.constant_restarts, theta0=start)
            if fitted is None:
                continue
            theta, coef, intercept, rss = fitted
            rss_old = row["rmse"] ** 2 * self.n
            m, extra = model.changed(theta), max(model.changed(theta) - model.changed(start), 0)
            if (m == 0 or rss <= 0 or rss >= rss_old
                    or self.n * math.log(rss / max(rss_old, 1e-300)) + extra * math.log(self.n) >= 0):
                continue
            if self.safe_domain and not domain_finite(model, theta, self.bank):
                continue
            accepted.append({"complexity": refined_complexity(self.bank, row["codes"], model, theta),
                             "rmse": math.sqrt(rss / self.n), "r2": 1.0 - rss / self.tss,
                             "terms": model.terms(theta), "coefficients": coef.tolist(), "intercept": intercept,
                             "constants": theta.tolist(), "polished": True})
        self.store.add(accepted)
        return accepted

    def _met(self, best):
        return best is not None and best["rmse"] <= self.metrics[0] and best["r2"] >= self.metrics[1]

    # ---------------------------------------------------------- output
    def _frontier(self):
        rows = self.store.frontier()
        final = pd.DataFrame({
            "Loss": [r["rmse"] for r in rows], "Complexity": [r["complexity"] for r in rows],
            "Equations": [r["terms"] for r in rows], "Intercepts": [r["intercept"] for r in rows],
            "Coefficients": [r["coefficients"] for r in rows], "Score": [r["r2"] for r in rows]})
        self.final_df = final
        loss, comp = final["Loss"], final["Complexity"]
        norm_loss = (loss - loss.min()) / ((loss.max() - loss.min()) or 1.0)
        norm_comp = (comp - comp.min()) / ((comp.max() - comp.min()) or 1.0)
        utopia = int(np.argmin(np.sqrt(norm_loss ** 2 + norm_comp ** 2)))
        edited = pd.DataFrame({"Loss": loss, "Complexity": comp, "R2": final["Score"],
                               "Equation": [format_equation(r) for r in rows]})
        res = {"utopia": {"expression": edited.loc[utopia, "Equation"], "rmse": edited.loc[utopia, "Loss"],
                          "r2": edited.loc[utopia, "R2"], "complexity": edited.loc[utopia, "Complexity"]}}
        return res, edited


def _refinement_candidates(frontier, extra):
    """Frontier models plus the extra low-loss models, without duplicates (by descriptor codes)."""
    seen, out = set(), []
    for row in list(frontier) + list(extra):
        codes = row.get("codes")
        if not codes or row.get("polished"):
            continue
        key = tuple(sorted(map(tuple, codes)))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _share(picks, per_bin):
    """Merge overall picks with the top ``per_bin`` of each bin into one pool."""
    union, seen = [], set()
    for slot, limit in [(len(picks) - 1, None)] + [(s, per_bin) for s in range(len(picks) - 1)]:
        for item in picks[slot][:limit]:
            if item["akey"] not in seen:
                seen.add(item["akey"])
                union.append(item)
    return [[] for _ in picks[:-1]] + [union]


def _screen_inputs(features, target, method, quantile):
    """Initial screening of primitive features (as in the original).

    Constant columns (physical constants in dimensional mode) are always kept.
    Returns the screened frame and the kept column positions.
    """
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_regression
    values = features.to_numpy(dtype=np.float64)
    constant = np.ptp(values, axis=0) == 0
    varying = np.nonzero(~constant)[0]
    if method == "spearman":
        stat = np.abs(spearmanr(values[:, varying], target, axis=0).statistic)
        score = stat[:-1, -1] if stat.ndim > 1 else np.atleast_1d(stat)
    elif method == "mi":
        score = mutual_info_regression(values[:, varying], target)
    else:
        raise ValueError("initial_screening method must be 'spearman' or 'mi'")
    order = np.argsort(-score, kind="stable")
    keep = order[score[order] > np.quantile(score, quantile)]
    if len(keep) == 0:
        keep = order[: max(1, len(varying) // 2)]
    kept = np.sort(np.concatenate([varying[keep], np.nonzero(constant)[0]]))
    out = features.iloc[:, kept].copy()
    out.columns = out.columns.str.replace("-", "_")
    return out, kept
