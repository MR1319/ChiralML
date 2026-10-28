"""Symbolic classification on the streaming SyMANTIC engine.

Model (criterion='logistic', the default): logits are linear in a small set of
symbolic descriptors shared by all classes,

    binary:      P(y=1|x) = sigmoid(b + sum_j beta_j f_j(x))
    multiclass:  P(y=k|x) = softmax_k(b_k + sum_j beta_jk f_j(x)),

fitted by ridge-penalized maximum likelihood (cross-entropy), as in logistic
regression, XGBoost, and neural-network output layers. The ridge keeps fits
finite when a descriptor separates the classes perfectly.

Search mirrors regression SyMANTIC with the loss geometry of second-order
boosting: descriptors are screened by the score-test gain sum_k U_k^2 / I_k
(g = y - p, h = p(1 - p) at the current model; exactly correlation SIS for
squared loss), and every d-subset of the screened pool is ranked by a
weighted-least-squares (one Newton step) approximation before the best
candidates are refit exactly with batched Newton iterations.

criterion='overlap' (SISSO) ranks descriptors by the number of samples lying
in the convex-hull domain of another class, and criterion='svm' (SISSO++) uses
the same descriptors with linear-SVM hyperplanes. Both use the logistic gain
to shortlist candidates before the exact geometric ranking.
"""
from itertools import combinations, islice
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .model import FastSymanticModel, _refinement_candidates, _share
from .screening import LevelScreen
from .sparse import EXHAUSTIVE_LIMIT, ParetoStore, beam_subsets, pareto_mask

CRITERIA = ("logistic", "overlap", "svm")


# ----------------------------------------------------------------------------- likelihood
def probabilities(eta, binary):
    return torch.sigmoid(eta) if binary else torch.softmax(eta, dim=-1)


def nll(eta, Y, binary, weight):
    """Summed (weighted) negative log-likelihood; eta, Y: (..., n, K')."""
    if binary:
        per = torch.nn.functional.softplus(eta) - Y * eta
    else:
        per = torch.logsumexp(eta, dim=-1, keepdim=True) - (Y * eta).sum(dim=-1, keepdim=True)
    return (per.sum(dim=-1) * weight).sum(dim=-1)


def geometry(eta, Y, binary, weight):
    """Per-logit gradients g = w (y - p) and curvatures h = w p (1 - p)."""
    P = probabilities(eta, binary)
    G = (weight[:, None] * (Y - P)).T
    H = (weight[:, None] * P * (1 - P)).T.clamp_min(1e-12)
    return LevelScreen.geometry(G.contiguous(), H.contiguous())


def fit_glm(Z, Y, binary, weight, ridge, theta0=None, iterations=60, tolerance=1e-10, memory=1 << 27,
            offset=None):
    """Batched ridge-penalized logistic/softmax fits by damped Newton steps.

    Z: (C, n, d) standardized descriptors (one problem per leading index);
    Y: (n, K') targets (K' = 1 for binary). The intercept is not penalized.
    ridge: one penalty for all d columns, or a length-d tensor (per column).
    offset: optional fixed (n, K') logit offset (e.g. group effects).
    Returns theta (C, d+1, K'), penalized objective, and NLL, all per problem.
    """
    C, n, d = Z.shape
    K = Y.shape[1]
    D = d + 1
    step_rows = max(1, int(memory // (8 * n * D * max(K * K, 2))))
    if C > step_rows:
        parts = [fit_glm(Z[i:i + step_rows], Y, binary, weight, ridge,
                         None if theta0 is None else theta0[i:i + step_rows], iterations, tolerance, memory, offset)
                 for i in range(0, C, step_rows)]
        return tuple(torch.cat([p[j] for p in parts]) for j in range(3))
    X = torch.cat([torch.ones((C, n, 1), dtype=Z.dtype, device=Z.device), Z], dim=2)
    if torch.is_tensor(ridge) and ridge.dim() == 1:
        penalty = torch.cat([torch.zeros(1, dtype=Z.dtype, device=Z.device), ridge.to(Z.dtype).to(Z.device)])
    else:
        penalty = torch.full((D,), float(ridge), dtype=Z.dtype, device=Z.device)
        penalty[0] = 0.0
    base = 0.0 if offset is None else offset
    if theta0 is None:
        prior = (weight[:, None] * Y).sum(0) / weight.sum()
        theta = torch.zeros((C, D, K), dtype=Z.dtype, device=Z.device)
        if binary:
            theta[:, 0, 0] = torch.logit(prior.clamp(1e-6, 1 - 1e-6))[0]
        else:
            theta[:, 0, :] = prior.clamp_min(1e-6).log()
    else:
        theta = theta0.clone()

    def objective(t):
        eta = X @ t + base
        return nll(eta, Y, binary, weight) + 0.5 * (penalty[None, :, None] * t.square()).sum(dim=(1, 2))

    current = objective(theta)
    active = torch.ones(C, dtype=torch.bool, device=Z.device)
    for _ in range(iterations):
        idx = torch.nonzero(active).flatten()
        if len(idx) == 0:
            break
        Xa, ta = X[idx], theta[idx]
        P = probabilities(Xa @ ta + base, binary)
        grad = Xa.transpose(1, 2) @ (weight[:, None] * (P - Y)) + penalty[None, :, None] * ta
        hess = torch.zeros((len(idx), D, K, D, K), dtype=Z.dtype, device=Z.device)
        for k in range(K):
            for l in range(k, K):
                w = P[..., k] * ((1.0 if k == l else 0.0) - P[..., l]) if not binary else P[..., 0] * (1 - P[..., 0])
                block = (Xa * (weight * w)[..., None]).transpose(1, 2) @ Xa
                hess[:, :, k, :, l] = block
                hess[:, :, l, :, k] = block
        hess = hess.reshape(len(idx), D * K, D * K)
        diag = penalty.repeat_interleave(K) + 1e-9 * (1.0 + hess.diagonal(dim1=1, dim2=2).abs().mean(dim=1, keepdim=True))
        hess = hess + torch.diag_embed(diag)
        step = torch.linalg.solve(hess, grad.reshape(len(idx), D * K)).reshape(len(idx), D, K)
        accepted = torch.zeros(len(idx), dtype=torch.bool, device=Z.device)
        new_theta, new_value = ta.clone(), current[idx].clone()
        alpha = 1.0
        for _ in range(30):
            trial = ta - alpha * step
            value = objective_rows(X, idx, trial, Y, binary, weight, penalty, base)
            better = (~accepted) & (value <= current[idx] + 1e-12 * current[idx].abs())
            new_theta[better], new_value[better] = trial[better], value[better]
            accepted |= better
            if accepted.all():
                break
            alpha *= 0.5
        theta[idx] = new_theta
        moved = (new_theta - ta).abs().amax(dim=(1, 2))
        current[idx] = new_value
        active[idx] = accepted & (moved > tolerance * (1.0 + ta.abs().amax(dim=(1, 2))))
    loss = nll(X @ theta + base, Y, binary, weight)
    return theta, current, loss


def objective_rows(X, idx, theta, Y, binary, weight, penalty, base=0.0):
    eta = X[idx] @ theta + base
    return nll(eta, Y, binary, weight) + 0.5 * (penalty[None, :, None] * theta.square()).sum(dim=(1, 2))


# ----------------------------------------------------------------------------- subsets
def _combination_chunks(s, d, chunk):
    combos = combinations(range(s), d)
    while True:
        block = np.fromiter(islice((v for combo in combos for v in combo), chunk * d), dtype=np.int64)
        if block.size == 0:
            return
        yield block.reshape(-1, d)


def screen_subsets(Z, cplx, d, eta0, Y, binary, weight, ridge, keep_top=64, chunk=131072, exhaustive_limit=None,
                   offset=None):
    """Rank all d-subsets of the pool by a one-Newton-step (weighted least squares) model.

    For each logit k the loss is approximated around eta0 by
    0.5 * sum h_k (z_k - b_k - Z_S beta_k)^2 with working response
    z_k = eta0_k - offset_k + g_k / h_k; subsets are scored by the summed explained
    weighted sum of squares. Returns candidate index sets: the top `keep_top`
    plus the approximate complexity/loss frontier.
    """
    n, s = Z.shape
    if s < d:
        return torch.empty((0, d), dtype=torch.long, device=Z.device)
    P0 = probabilities(eta0, binary)
    grams, rhs = [], []
    for k in range(Y.shape[1]):
        h = (weight * P0[:, k] * (1 - P0[:, k])).clamp_min(1e-12)
        z = eta0[:, k] + weight * (Y[:, k] - P0[:, k]) / h
        if offset is not None:
            z = z - offset[:, k]
        mean_z = (h * z).sum() / h.sum()
        Zc = Z - ((h @ Z) / h.sum())[None, :]
        grams.append(Zc.T @ (h[:, None] * Zc) + ridge * torch.eye(s, dtype=Z.dtype, device=Z.device))
        rhs.append(Zc.T @ (h * (z - mean_z)))
    if math.comb(s, d) > (EXHAUSTIVE_LIMIT if exhaustive_limit is None else exhaustive_limit):
        return beam_subsets(grams, rhs, cplx, d, keep_top=keep_top)
    best_idx = torch.empty((0, d), dtype=torch.long, device=Z.device)
    best_val = torch.empty(0, dtype=Z.dtype, device=Z.device)
    front_idx, front_val, front_c = best_idx, best_val, best_val
    for block in _combination_chunks(s, d, chunk):
        idx = torch.as_tensor(block, device=Z.device)
        explained = torch.zeros(len(idx), dtype=Z.dtype, device=Z.device)
        for G, b in zip(grams, rhs):
            Gs = G[idx[:, :, None], idx[:, None, :]]
            bs = b[idx]
            L, info = torch.linalg.cholesky_ex(Gs)
            sol = torch.cholesky_solve(bs[..., None], L)[..., 0]
            explained += torch.where(info == 0, (bs * sol).sum(dim=1), -math.inf)
        c = cplx[idx].sum(dim=1)
        all_idx, all_val = torch.cat([best_idx, idx]), torch.cat([best_val, explained])
        top = torch.topk(all_val, min(keep_top, len(all_val))).indices
        best_idx, best_val = all_idx[top], all_val[top]
        cand_idx = torch.cat([front_idx, idx])
        cand_val = torch.cat([front_val, explained])
        cand_c = torch.cat([front_c, c])
        keep = torch.as_tensor(pareto_mask(cand_c.cpu().numpy(), (-cand_val).cpu().numpy()), device=Z.device)
        front_idx, front_val, front_c = cand_idx[keep], cand_val[keep], cand_c[keep]
    out = torch.cat([best_idx, front_idx])
    return torch.unique(out, dim=0)


# ----------------------------------------------------------------------------- SISSO overlap
def overlap_1d(V, labels, classes, active):
    """Per feature (rows of V): #active samples inside another class's [min, max] domain,
    and the summed pairwise domain-overlap length relative to the feature range."""
    m = V.shape[0]
    lo, hi = [], []
    for c in classes:
        mask = active & (labels == c)
        vals = V[:, mask] if mask.any() else V[:, :0]
        lo.append(vals.amin(dim=1) if vals.shape[1] else torch.full((m,), math.inf, dtype=V.dtype))
        hi.append(vals.amax(dim=1) if vals.shape[1] else torch.full((m,), -math.inf, dtype=V.dtype))
    lo, hi = torch.stack(lo, 1), torch.stack(hi, 1)
    count = torch.zeros(m, dtype=torch.long)
    for j, c in enumerate(classes):
        inside = torch.zeros_like(V, dtype=torch.bool)
        for k, _ in enumerate(classes):
            if k != j:
                inside |= (V >= lo[:, k:k + 1]) & (V <= hi[:, k:k + 1])
        count += (inside & (active & (labels == c))[None, :]).sum(dim=1)
    span = (V.amax(dim=1) - V.amin(dim=1)).clamp_min(1e-300)
    length = torch.zeros(m, dtype=V.dtype)
    for j in range(len(classes)):
        for k in range(j + 1, len(classes)):
            length += (torch.minimum(hi[:, j], hi[:, k]) - torch.maximum(lo[:, j], lo[:, k])).clamp_min(0)
    return count, length / span


def hull_overlap(F, labels):
    """Number of samples lying inside the convex hull of another class (SISSO criterion)."""
    from scipy.optimize import linprog
    from scipy.spatial import Delaunay, QhullError
    n, d = F.shape
    if d == 1:
        V = torch.as_tensor(F.T)
        count, _ = overlap_1d(V, torch.as_tensor(labels), np.unique(labels).tolist(), torch.ones(n, dtype=torch.bool))
        return int(count[0])
    inside = np.zeros(n, dtype=bool)
    for c in np.unique(labels):
        pts, others = F[labels == c], np.nonzero(labels != c)[0]
        if len(pts) <= d:
            continue
        try:
            if d <= 3:
                inside[others] |= Delaunay(pts).find_simplex(F[others]) >= 0
                continue
        except QhullError:
            pass
        A_eq = np.vstack([pts.T, np.ones((1, len(pts)))])
        for i in others:
            if not inside[i]:
                res = linprog(np.zeros(len(pts)), A_eq=A_eq, b_eq=np.r_[F[i], 1.0], bounds=(0, None), method="highs")
                inside[i] = res.status == 0
    return int(inside.sum())


# ----------------------------------------------------------------------------- estimator
class SymanticClassifier(FastSymanticModel):
    """Symbolic classifier (binary and multiclass) with a scikit-learn style API.

    Parameters (beyond FastSymanticModel's search options)
    ---------------------------------------------------------
    criterion : 'logistic' (default), 'overlap' (SISSO) or 'svm' (SISSO++).
    ridge : L2 penalty on standardized descriptor coefficients (summed-loss
        scale, like XGBoost's lambda). Keeps separable problems finite; small
        values approach the max-margin boundary, which recovers deterministic
        decision boundaries best (1.0 visibly blurs them).
    class_weight : None, 'balanced', or {label: weight}.
    accuracy_target : stop expanding once training accuracy reaches it.
    refit_candidates : subsets per pool refit exactly after approximate ranking.
    svm_C : linear-SVM regularization for criterion='svm' (large = near hard margin).
    selection : 'validation' (default) picks the Pareto model with the lowest
        held-out log-loss (error rate for 'overlap'/'svm'); 'utopia' uses the
        original SyMANTIC knee point of the training frontier.
    validation_fraction : stratified share held out for selection.
    refit : after selection, refit every Pareto model's coefficients on all
        data (descriptors stay as found).
    group_ridge : shrinkage of the per-group intercepts used when ``fit`` gets
        ``groups`` (summed-loss scale; larger = closer to the shared law).

    Group terms (``fit(X, y, groups=...)``, logistic criterion): the logit gets a
    per-group intercept, eta = b + sum_j beta_j f_j(x) + u_group, with the u
    shrunk by ``group_ridge`` -- a symbolic law shared by all groups plus group
    offsets (batches, labs, subjects, material families). ``groups`` is one
    label per row, or a DataFrame of several crossed factors. During search
    the u act as a fixed offset (no extra columns in the subset search); they
    are re-estimated jointly with the descriptors after every round and for
    every final model. Prediction adds u for known groups when ``groups`` is
    passed; unseen groups (or groups=None) use the shared law (u = 0).
    """

    def __init__(self, operators, n_expansion=None, n_term=3, sis_features=20, criterion="logistic", ridge=0.1,
                 class_weight=None, accuracy_target=1.0, max_level=2, refit_candidates=64, svm_C=1000.0,
                 selection="validation", validation_fraction=0.2, refit=True, random_state=0,
                 device="cpu", disp=False, dimensionality=None, output_dim=None, relational_units=None,
                 strict_units=True, group_ridge=1.0, **search):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        if selection not in ("validation", "utopia"):
            raise ValueError("selection must be 'validation' or 'utopia'")
        self.selection = selection
        self.validation_fraction = validation_fraction
        self.refit = refit
        self.random_state = random_state
        search.setdefault("safe_domain", True)  # predictions must stay finite on new data
        self._init_args = dict(operators=operators, n_expansion=n_expansion, n_term=n_term,
                               sis_features=sis_features, device=device, disp=disp, max_level=max_level,
                               dimensionality=dimensionality, output_dim=output_dim,
                               relational_units=relational_units, strict_units=strict_units, **search)
        self.criterion = criterion
        self.ridge = ridge
        self.class_weight = class_weight
        self.accuracy_target = accuracy_target
        self.refit_candidates = refit_candidates
        self.svm_C = svm_C
        self.group_ridge = group_ridge

    # ---------------------------------------------------------------- groups
    @staticmethod
    def _group_frame(groups, n):
        frame = groups.copy() if isinstance(groups, pd.DataFrame) else pd.DataFrame(np.asarray(groups, dtype=object))
        if frame.ndim != 2 or len(frame) != n:
            raise ValueError("groups needs one label per row (or a DataFrame with one row per sample)")
        if not isinstance(groups, pd.DataFrame):
            frame.columns = ["group"] if frame.shape[1] == 1 else [f"group{j}" for j in range(frame.shape[1])]
        return frame.reset_index(drop=True).astype(str)

    def _group_design(self, groups, n):
        """One-hot design over the fitted levels; rows of unseen levels are all zero."""
        frame = self._group_frame(groups, n)
        if list(frame.columns) != self.group_columns_:
            raise ValueError(f"groups must have columns {self.group_columns_}")
        G = np.zeros((n, len(self.group_levels_)))
        index = {lv: j for j, lv in enumerate(self.group_levels_)}
        for col in frame.columns:
            for i, v in enumerate(frame[col]):
                j = index.get((col, v))
                if j is not None:
                    G[i, j] = 1.0
        device = getattr(self, "device", None) or self._init_args.get("device", "cpu")  # set before the first search
        return torch.as_tensor(G, dtype=torch.float64, device=device)

    def _set_groups(self, groups, n):
        if groups is None:
            self.group_columns_, self.group_levels_, self._G_all = None, None, None
            return
        if self.criterion != "logistic":
            raise ValueError("groups are supported with criterion='logistic'")
        frame = self._group_frame(groups, n)
        self.group_columns_ = list(frame.columns)
        self.group_levels_ = [(c, v) for c in frame.columns for v in sorted(frame[c].unique())]
        self._G_all = self._group_design(frame, n)

    # ---------------------------------------------------------------- fitting
    def fit(self, X, y, sample_weight=None, groups=None):
        X = _frame(X)
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classification needs at least two classes")
        self._set_groups(groups, len(codes))
        counts = np.bincount(codes)
        n_val = int(round(self.validation_fraction * len(codes)))
        use_val = (self.selection == "validation" and n_val >= len(counts) and counts.min() >= 2)
        if not use_val:
            self._G = self._G_all
            self._search(X, codes, sample_weight)
            self._maybe_refine()
            return self
        from sklearn.model_selection import train_test_split
        fit_idx, val_idx = train_test_split(np.arange(len(codes)), test_size=n_val, stratify=codes,
                                            random_state=self.random_state)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self._G = None if self._G_all is None else self._G_all[torch.as_tensor(fit_idx, device=self._G_all.device)]
        self._search(X.iloc[fit_idx], codes[fit_idx], None if sw is None else sw[fit_idx])
        self._maybe_refine()  # constants are fit on the search split; validation decides if they help
        X_val, y_val = X.iloc[val_idx], self.classes_[codes[val_idx]]
        w_val = self._sample_weight(codes[val_idx], None if sw is None else sw[val_idx]).cpu().numpy()
        g_val = None if groups is None else self._group_frame(groups, len(codes)).iloc[val_idx]
        scores = [self._validation_score(X_val, y_val, w_val, i, g_val) for i in range(len(self.models_))]
        self.equations_.insert(1, "Validation", scores)
        self.best_index_ = int(min(range(len(scores)), key=lambda i: (scores[i], self.models_[i]["complexity"])))
        if self.refit:
            self._G = self._G_all
            self._refit_all(X, codes, sw)
        return self

    def _maybe_refine(self):
        if self.optimize_constants and self.criterion == "logistic":
            if self._refine_constants_glm():
                self._finish()

    def _refine_constants_glm(self):
        """Fit nonlinear constants inside Pareto descriptors with the linear head (BIC-accepted)."""
        from .constants import ParamModel, domain_finite, fit_classification, refined_complexity, start_vector
        values = self._primitive_values()
        total = float(self.weight.sum())
        accepted = []
        for row in _refinement_candidates(self.store.frontier(), self.refine_pool):
            model = ParamModel(self.bank, row["codes"])
            if model.size == 0:
                continue
            start = start_vector(model, row.get("term_params"))
            fitted = fit_classification(model, values, self.Y, self.binary, self.weight, self.ridge,
                                        row["coef"], row["intercept"], restarts=self.constant_restarts, theta0=start,
                                        offset=self.offset0)
            if fitted is None:
                continue
            theta, coef, intercept, loss = fitted
            m, extra = model.changed(theta), max(model.changed(theta) - model.changed(start), 0)
            if m == 0 or loss >= row["log_loss"] * total or 2 * (row["log_loss"] * total - loss) <= extra * math.log(self.n):
                continue
            if self.safe_domain and not domain_finite(model, theta, self.bank):
                continue
            with torch.no_grad():
                F = model.features(theta, values)
                eta = intercept[None, :] + F @ coef + (0.0 if self.offset0 is None else self.offset0)
            P = probabilities(eta, self.binary)
            pred = (P[:, 0] > 0.5).long() if self.binary else P.argmax(dim=1)
            new = {"complexity": refined_complexity(self.bank, row["codes"], model, theta),
                   "loss": loss / total, "log_loss": loss / total,
                   "accuracy": float(((pred == self.labels).to(torch.float64) * self.weight).sum() / total),
                   "terms": model.terms(theta), "codes": row["codes"], "coef": coef.cpu().numpy(),
                   "intercept": intercept.cpu().numpy(), "eta": eta, "overlap": None,
                   "mus": F.mean(0), "sds": F.std(0).clamp_min(1e-300), "param": (model, theta), "polished": True}
            new["equation"] = self._equation(new)
            accepted.append(new)
        self.store.add(accepted)
        return accepted

    def _validation_score(self, X, y, w, index, groups=None):
        if self.criterion == "svm" and "svm" in self.models_[index]:
            return float(np.average(self.predict(X, index) != y, weights=w))
        proba = np.clip(self.predict_proba(X, index, groups=groups), 1e-15, 1.0)
        column = np.searchsorted(self.classes_, y)
        return float(-np.average(np.log(proba[np.arange(len(y)), column]), weights=w))

    def _refit_all(self, X, codes, sample_weight):
        """Refit each Pareto model's coefficients on all data (descriptors fixed)."""
        self.labels = torch.as_tensor(codes, device=self.device)
        self.n = len(codes)
        K = self.Y.shape[1]
        self.Y = torch.zeros((self.n, K), dtype=torch.float64, device=self.device)
        if self.binary:
            self.Y[:, 0] = self.labels.to(torch.float64)
        else:
            self.Y[torch.arange(self.n), self.labels] = 1.0
        self.weight = self._sample_weight(codes, sample_weight)
        for i, row in enumerate(self.models_):
            F = self._features(X, row)
            terms = [{"name": t, "complexity": 0.0, "values": F[:, j], "mu": row["mus"][j], "sd": row["sds"][j],
                      "code": row["codes"][j]} for j, t in enumerate(row["terms"])]
            Zs = (F - row["mus"]) / row["sds"] if len(terms) else F
            coef = torch.as_tensor(row["coef"], device=self.device)
            theta0 = torch.cat([torch.as_tensor(row["intercept"], device=self.device)[None, :]
                                + (coef * row["mus"][:, None]).sum(0, keepdim=True) if len(terms) else
                                torch.as_tensor(row["intercept"], device=self.device)[None, :],
                                coef * row["sds"][:, None]])
            theta = fit_glm(Zs[None], self.Y, self.binary, self.weight, self.ridge, theta0=theta0[None])[0][0]
            eta = torch.cat([torch.ones(self.n, 1, dtype=torch.float64, device=self.device), Zs], 1) @ theta
            new = self._row(terms, eta, theta, list(range(len(terms))), row["mus"], row["sds"])
            new["complexity"], new["overlap"] = row["complexity"], row["overlap"]
            if "param" in row:
                new["param"] = row["param"]
            if row["overlap"] is not None:
                new["loss"] = row["loss"]
            if "svm" in row:
                new["svm"] = self._fit_svm(Zs, row["mus"], row["sds"])
            if self._G is not None:
                new = self._grouped(new, Zs)
            self.models_[i] = new
        self.equations_["Loss"] = [r["log_loss"] for r in self.models_]
        self.equations_["Accuracy"] = [r["accuracy"] for r in self.models_]
        self.equations_["Equation"] = [r["equation"] for r in self.models_]

    def _search(self, X, codes, sample_weight):
        df = X.copy()
        df.insert(0, "__target__", codes.astype(np.float64))
        FastSymanticModel.__init__(self, df, **self._init_args)
        self.refine_pool = []
        if self.criterion != "logistic":
            self.capacity = max(10 * self.k, self.k + 64)
        self._build(df)
        self.binary = len(self.classes_) == 2
        K = 1 if self.binary else len(self.classes_)
        self.labels = torch.as_tensor(codes, device=self.device)
        Y = torch.zeros((self.n, K), dtype=torch.float64, device=self.device)
        if self.binary:
            Y[:, 0] = self.labels.to(torch.float64)
        else:
            Y[torch.arange(self.n), self.labels] = 1.0
        self.Y = Y
        self.weight = self._sample_weight(codes, sample_weight)
        self.store = ParetoStore(loss_key="loss")
        if self._G is not None:  # start from the group-only model; its group effects are the first offset
            theta_g = fit_glm(self._G[None], Y, self.binary, self.weight, self.group_ridge)[0][0]
            self.offset0 = self._G @ theta_g[1:]
            theta0 = theta_g[:1]
            self.eta0 = theta0[0][None, :] + self.offset0
        else:
            self.offset0 = None
            theta0 = fit_glm(torch.zeros((1, self.n, 0), dtype=torch.float64, device=self.device), Y, self.binary,
                             self.weight, self.ridge)[0][0]
            self.eta0 = theta0[0][None, :].expand(self.n, -1).clone()
        self.offset = self.offset0
        self.store.add([self._row([], self.eta0, theta0, [], torch.empty(0), torch.empty(0))])
        if self.n_expansion is not None:
            depth = self.n_expansion - 1
            for level in range(1, depth):
                if not self._materialize(level, parents=None):
                    raise MemoryError(f"Level {level} exceeds memory_limit_gb; increase it or set max_parents")
            self._classify(depth if depth >= 1 else None)
        else:
            for level in range(1, self.max_level + 1):
                best = self._classify(level)
                if (best is not None and best["accuracy"] >= self.accuracy_target) or level == self.max_level:
                    break
                if not self._materialize(level, parents=None):
                    break
        self._finish()

    def _sample_weight(self, codes, sample_weight):
        w = np.ones(len(codes)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(codes)
            w = w * (len(codes) / (len(counts) * counts[codes]))
        elif isinstance(self.class_weight, dict):
            w = w * np.array([self.class_weight.get(c, 1.0) for c in self.classes_[codes]])
        return torch.as_tensor(w, dtype=torch.float64, device=self.device)

    def _classify(self, level):
        start = time.perf_counter()
        screen = LevelScreen(self.bank, level, self.quantiles, self.capacity, tile=self.tile,
                             memory_bytes=self.tile_memory, bins_from=self.bins_from,
                             target_units=self.target_units)
        if self.disp:
            print(f"[level {level}] screening {screen.n_candidates:,} candidate descriptors")
        nb = len(screen.boundaries)
        selected, best, eta = [], None, self.eta0
        self.offset = self.offset0
        for d in range(1, self.n_term + 1):
            geo = geometry(eta, self.Y, self.binary, self.weight)
            picks = self._select(screen.run_round(geo), geo, selected)
            if self.criterion != "logistic":
                picks = [self._rank_by_overlap(p, eta, best) for p in picks]
            elif self.optimize_constants:
                picks[nb] = self._refine_picks(picks[nb], geo)
            if self.share_bins:
                picks = _share(picks, self.share_bins)
            for slot in range(nb + 1):
                if not picks[slot] and slot != nb:
                    continue
                rows = self._fit_pool(selected + picks[slot], d, eta, keep=slot == nb)
                if not rows:
                    continue
                self.store.add(rows)
                if slot == nb:
                    best = min(rows, key=lambda r: r["objective"])
            if best is None:
                break
            eta = best["eta"]
            if self._G is not None:  # backfitting: re-estimate group effects with this round's descriptors
                joint = self._grouped(best, best["Zs"])
                eta, self.offset = joint["eta"], self._G @ torch.as_tensor(joint["group_effects"], device=self.device)
            selected = selected + picks[nb]
            if self.disp:
                print(f"[level {level}] {d}-term: logloss={best['loss']:.5g} acc={best['accuracy']:.4f}  {best['equation']}")
            if best["accuracy"] >= self.accuracy_target:
                break
        self.history.append({"level": level, "candidates": screen.n_candidates, "bank": len(self.bank),
                             "seconds": time.perf_counter() - start, "loss": best["loss"] if best else None})
        return best

    def _fit_pool(self, pool, d, eta0, keep=False):
        if len(pool) < d:
            return []
        Z = torch.stack([(p["values"] - p["mu"]) / p["sd"] for p in pool], dim=1)
        cplx = torch.tensor([p["complexity"] for p in pool], dtype=torch.float64, device=self.device)
        offset = self.offset
        base = 0.0 if offset is None else offset
        candidates = screen_subsets(Z, cplx, d, eta0, self.Y, self.binary, self.weight, self.ridge,
                                    keep_top=self.refit_candidates, offset=offset)
        fit = lambda idx: fit_glm(Z[:, idx].permute(1, 0, 2), self.Y, self.binary, self.weight, self.ridge,
                                  offset=offset)
        theta, objective, _ = fit(candidates)
        # Second pass: re-linearize around the best exact fit and refit any new candidates.
        best = int(torch.argmin(objective))
        X1 = torch.ones(self.n, 1, dtype=Z.dtype, device=Z.device)
        eta_best = torch.cat([X1, Z[:, candidates[best]]], 1) @ theta[best] + base
        more = screen_subsets(Z, cplx, d, eta_best, self.Y, self.binary, self.weight, self.ridge,
                              keep_top=self.refit_candidates, offset=offset)
        seen = {tuple(row) for row in candidates.tolist()}
        new = [row for row in more.tolist() if tuple(row) not in seen]
        if new:
            new = torch.as_tensor(new, device=Z.device)
            theta_new, objective_new, _ = fit(new)
            candidates = torch.cat([candidates, new])
            theta, objective = torch.cat([theta, theta_new]), torch.cat([objective, objective_new])
        if self.criterion != "logistic":
            candidates = self._overlap_filter(Z, candidates, objective)
            theta, objective, _ = fit(candidates)
        rows = []
        for i, idx in enumerate(candidates.tolist()):
            eta = torch.cat([X1, Z[:, idx]], 1) @ theta[i] + base
            terms = [pool[j] for j in idx]
            row = self._row(terms, eta, theta[i], idx, torch.stack([pool[j]["mu"] for j in idx]),
                            torch.stack([pool[j]["sd"] for j in idx]))
            row["objective"] = float(objective[i])
            if self._G is not None:
                row["Zs"] = Z[:, idx]
            rows.append(row)
        if keep and self.optimize_constants and self.criterion == "logistic":
            self.refine_pool.extend(sorted(rows, key=lambda r: r["objective"])[: self.refine_top])
        if self.criterion == "logistic":
            mask = pareto_mask([r["complexity"] for r in rows], [r["loss"] for r in rows])
            best = int(np.argmin([r["objective"] for r in rows]))
            mask[best] = True
            rows = [r for r, m in zip(rows, mask) if m]
        return rows

    def _row(self, terms, eta, theta, idx, mus, sds):
        """Model summary in original feature units (coefficients undo standardization)."""
        beta = theta[1:]
        if len(terms):
            coef = beta / sds[:, None]
            intercept = theta[0] - (beta * (mus / sds)[:, None]).sum(dim=0)
        else:
            coef, intercept = beta, theta[0]
        P = probabilities(eta, self.binary)
        pred = (P[:, 0] > 0.5).long() if self.binary else P.argmax(dim=1)
        loss = float(nll(eta, self.Y, self.binary, self.weight) / self.weight.sum())
        correct = (pred == self.labels).to(torch.float64)
        overlap = None
        if self.criterion != "logistic" and len(terms):
            F = torch.stack([t["values"] for t in terms], dim=1).cpu().numpy()
            overlap = hull_overlap(F, self.labels.cpu().numpy())
        row = {"complexity": float(sum(t["complexity"] for t in terms)),
               "loss": loss if overlap is None else overlap / self.n + 1e-9 * loss,
               "log_loss": loss, "accuracy": float((correct * self.weight).sum() / self.weight.sum()),
               "terms": [t["name"] for t in terms], "codes": [t["code"] for t in terms],
               "coef": coef.cpu().numpy(), "intercept": intercept.cpu().numpy(), "eta": eta,
               "overlap": overlap, "mus": mus, "sds": sds}
        if any(t.get("param") is not None for t in terms):
            from .constants import ParamModel, start_vector
            model = ParamModel(self.bank, row["codes"])
            row["param"] = (model, start_vector(model, [t.get("param") for t in terms]))
            row["term_params"] = [t.get("param") for t in terms]
        row["equation"] = self._equation(row)
        return row

    def _equation(self, row):
        groups = f" + u[{', '.join(self.group_columns_)}]" if "group_effects" in row else ""

        def linear(coef, intercept):
            parts = [f"{c:.6g}*{t}" for c, t in zip(coef, row["terms"])]
            return " + ".join(parts + [f"{intercept:.6g}"]) + groups
        if self.binary:
            return f"logit P({self.classes_[1]}) = " + linear(row["coef"][:, 0], row["intercept"][0])
        center = row["intercept"].mean()
        return " | ".join(f"z[{c}] = " + linear(row["coef"][:, k], row["intercept"][k] - center)
                          for k, c in enumerate(self.classes_))

    # ---------------------------------------------------------------- group terms
    def _train_values(self, row):
        """Training descriptor values of a model (n, d)."""
        if "param" in row:
            model, theta = row["param"]
            with torch.no_grad():
                return model.features(theta, self._primitive_values())
        if not row["codes"]:
            return torch.empty((self.n, 0), dtype=torch.float64, device=self.device)
        return torch.stack([self._values_of(c) for c in row["codes"]], 1)

    def _grouped(self, row, Zs=None):
        """Refit one model's head jointly with the group intercepts (descriptors fixed)."""
        d = len(row["terms"])
        if Zs is None:
            F = self._train_values(row)
            Zs = (F - row["mus"]) / row["sds"] if d else F
        L = self._G.shape[1]
        penalty = torch.cat([torch.full((d,), float(self.ridge)), torch.full((L,), float(self.group_ridge))])
        X = torch.cat([Zs, self._G], 1)
        theta = fit_glm(X[None], self.Y, self.binary, self.weight, penalty.to(self.device))[0][0]
        u = theta[1 + d:]
        ones = torch.ones(self.n, 1, dtype=torch.float64, device=self.device)
        eta = torch.cat([ones, Zs], 1) @ theta[:1 + d] + self._G @ u
        terms = [{"name": t, "complexity": 0.0, "code": c} for t, c in zip(row["terms"], row["codes"])]
        new = self._row(terms, eta, theta[:1 + d], list(range(d)), row["mus"], row["sds"])
        for key in ("complexity", "overlap", "param", "term_params", "polished"):
            if key in row:
                new[key] = row[key]
        new["group_effects"] = u.cpu().numpy()
        new["equation"] = self._equation(new)
        return new

    def group_effects(self, index=None):
        """Fitted group intercepts of a model as a DataFrame (one column per logit)."""
        row = self.models_[self.best_index_ if index is None else index]
        if "group_effects" not in row:
            return None
        index = pd.MultiIndex.from_tuples(self.group_levels_, names=["factor", "level"])
        cols = [f"u[{self.classes_[1]}]"] if self.binary else [f"u[{c}]" for c in self.classes_]
        return pd.DataFrame(row["group_effects"], index=index, columns=cols)

    # --------------------------------------------------------- SISSO / SISSO++
    def _rank_by_overlap(self, picks, eta, best):
        """Re-rank screened descriptors by 1-D domain overlap on samples still in the overlap region."""
        if not picks:
            return picks
        labels = self.labels.cpu()
        active = torch.ones(self.n, dtype=torch.bool)
        if best is not None and best.get("overlap"):
            F = np.stack([t for t in self._row_values(best)], 1)
            active = torch.as_tensor(_in_other_hulls(F, labels.numpy()))
            if not active.any():
                active = torch.ones(self.n, dtype=torch.bool)
        V = torch.stack([p["values"] for p in picks]).cpu()
        count, length = overlap_1d(V, labels, list(range(len(self.classes_))), active)
        order = sorted(range(len(picks)), key=lambda i: (int(count[i]), float(length[i])))
        return [picks[i] for i in order[: self.k]]

    def _row_values(self, row):
        return [self._values_of(code).cpu().numpy() for code in row["codes"]]

    def _values_of(self, code):
        op, left, right = code
        if op < 0:
            return self.bank.X[left]
        return self.bank.child_values([op], [left], [right])[0]

    def _overlap_filter(self, Z, candidates, objective):
        labels = self.labels.cpu().numpy()
        counts = np.array([hull_overlap(Z[:, idx].cpu().numpy(), labels) for idx in candidates.tolist()])
        order = np.lexsort((objective.cpu().numpy(), counts))
        return candidates[torch.as_tensor(order[: max(1, self.refit_candidates // 4)], device=Z.device)]

    def _fit_svm(self, Zs, mus, sds):
        from sklearn.svm import SVC
        svm = SVC(kernel="linear", C=self.svm_C).fit(Zs.cpu().numpy(), self.labels.cpu().numpy(),
                                                       sample_weight=self.weight.cpu().numpy())
        return {"model": svm, "mu": mus.cpu().numpy(), "sd": sds.cpu().numpy()}

    # ---------------------------------------------------------------- results
    def _finish(self):
        rows = self.store.frontier()
        if getattr(self, "_G", None) is not None:  # every final model gets its own joint group intercepts
            rows = [self._grouped(r) for r in rows]
        self.models_ = rows
        if self.criterion == "svm":  # SISSO++: hyperplanes on the selected descriptors
            for row in rows:
                if row["codes"]:
                    values = torch.stack([self._values_of(c) for c in row["codes"]], 1)
                    row["svm"] = self._fit_svm((values - row["mus"]) / row["sds"], row["mus"], row["sds"])
        frame = pd.DataFrame({
            "Loss": [r["log_loss"] for r in rows], "Complexity": [r["complexity"] for r in rows],
            "Accuracy": [r["accuracy"] for r in rows], "Terms": [len(r["terms"]) for r in rows],
            "Equation": [r["equation"] for r in rows]})
        if self.criterion != "logistic":
            frame.insert(1, "Overlap", [r["overlap"] for r in rows])
        self.equations_ = frame
        key = frame["Overlap"].fillna(self.n).astype(float) if self.criterion != "logistic" else frame["Loss"]
        norm_loss = (key - key.min()) / ((key.max() - key.min()) or 1.0)
        comp = frame["Complexity"]
        norm_comp = (comp - comp.min()) / ((comp.max() - comp.min()) or 1.0)
        self.best_index_ = int(np.argmin(np.sqrt(norm_loss ** 2 + norm_comp ** 2)))

    def _features(self, X, row):
        X = _frame(X)
        values = torch.tensor(X[self.input_columns_].to_numpy(np.float64, copy=True).T, device=self.device)
        F = self._row_features(row, values)
        bad = ~torch.isfinite(F).all(dim=1)
        if bad.any():
            # Rows outside the input domain: re-evaluate them with inputs clipped into the domain.
            lo = torch.as_tensor(self.domain_lo_, device=self.device)[:, None]
            hi = torch.as_tensor(self.domain_hi_, device=self.device)[:, None]
            F[bad] = self._row_features(row, torch.minimum(torch.maximum(values[:, bad], lo), hi))
            warnings.warn(f"{int(bad.sum())} sample(s) gave non-finite descriptor values (inputs outside the "
                          "fitted domain); they were evaluated with inputs clipped to the domain.", stacklevel=3)
        return F

    def _row_features(self, row, values):
        if "param" in row:
            model, theta = row["param"]
            with torch.no_grad():
                return model.features(theta, values)
        return self._evaluate(row["codes"], values)

    def _evaluate(self, codes, values):
        """Descriptor values from expression codes on feature-major inputs (no SymPy)."""
        memo = {}
        bank, ops = self.bank, self.ops

        def unary(op, left):
            # hinge knots always come from the parent's training values (bank), not the new data
            knot = ops.knot(op, bank.X[left]) if ops.is_hinge(op) else None
            return ops.apply(op, feature(left), knot=knot)

        def feature(i):
            if i not in memo:
                op, left, right = int(bank.op[i]), int(bank.left[i]), int(bank.right[i])
                if op < 0:
                    memo[i] = values[left]
                elif right < 0:
                    memo[i] = unary(op, left)
                else:
                    memo[i] = ops.apply(op, feature(left), feature(right))
            return memo[i]

        cols = []
        for op, left, right in codes:
            if op < 0:
                cols.append(feature(left))
            elif right < 0:
                cols.append(unary(op, left))
            else:
                cols.append(ops.apply(op, feature(left), feature(right)))
        n = values.shape[1]
        return torch.stack(cols, 1) if cols else torch.empty((n, 0), dtype=torch.float64, device=self.device)

    def transform(self, X, index=None):
        """Values of a model's symbolic descriptors (columns named by expression)."""
        row = self.models_[self.best_index_ if index is None else index]
        F = self._features(X, row).cpu().numpy()
        frame = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(F, columns=row["terms"], index=frame)

    def decision_function(self, X, index=None, groups=None):
        """Logits. With ``groups`` (same columns as in fit), known groups add their
        fitted intercepts; unseen groups and groups=None use the shared law."""
        row = self.models_[self.best_index_ if index is None else index]
        F = self._features(X, row)
        eta = torch.as_tensor(row["intercept"], device=self.device)[None, :] + F @ torch.as_tensor(row["coef"], device=self.device)
        if groups is not None and "group_effects" in row:
            eta = eta + self._group_design(groups, F.shape[0]) @ torch.as_tensor(row["group_effects"], device=self.device)
        return eta[:, 0].cpu().numpy() if self.binary else eta.cpu().numpy()

    def predict_proba(self, X, index=None, groups=None):
        row = self.models_[self.best_index_ if index is None else index]
        if self.criterion == "svm" and "svm" in row:
            raise NotImplementedError("criterion='svm' gives hyperplanes, not probabilities; use predict()")
        eta = torch.as_tensor(self.decision_function(X, index, groups), device=self.device)
        if self.binary:
            p = torch.sigmoid(eta)
            return torch.stack([1 - p, p], 1).cpu().numpy()
        return torch.softmax(eta, dim=1).cpu().numpy()

    def predict(self, X, index=None, groups=None):
        row = self.models_[self.best_index_ if index is None else index]
        if self.criterion == "svm" and "svm" in row:
            F = self._features(X, row).cpu().numpy()
            svm = row["svm"]
            return self.classes_[svm["model"].predict((F - svm["mu"]) / svm["sd"])]
        return self.classes_[np.argmax(self.predict_proba(X, index, groups), axis=1)]

    def score(self, X, y, index=None, groups=None):
        return float(np.mean(self.predict(X, index, groups) == np.asarray(y)))


def _frame(X):
    """Inputs as a DataFrame (arrays get columns x1, x2, ...)."""
    if isinstance(X, pd.DataFrame):
        return X
    values = np.asarray(X)
    return pd.DataFrame(values, columns=[f"x{i + 1}" for i in range(values.shape[1])])


def _in_other_hulls(F, labels):
    """Mask of samples lying in the domain of another class (current overlap region)."""
    from scipy.spatial import Delaunay, QhullError
    n, d = F.shape
    inside = np.zeros(n, dtype=bool)
    for c in np.unique(labels):
        pts, others = F[labels == c], np.nonzero(labels != c)[0]
        if len(pts) <= d:
            continue
        if d == 1:
            inside[others] |= (F[others, 0] >= pts.min()) & (F[others, 0] <= pts.max())
            continue
        try:
            inside[others] |= Delaunay(pts).find_simplex(F[others]) >= 0
        except QhullError:
            continue
    return inside
