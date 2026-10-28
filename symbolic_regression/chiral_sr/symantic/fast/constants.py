"""Nonlinear constants inside symbolic descriptors (post-search refinement).

SyMANTIC's search finds structures with fixed inner constants, e.g. exp(x).
This module refines a selected model's descriptors into parametric forms such
as exp(a*x + b), sin(a*x + b), ln(a*x + b), 1/(a*x + b), or x**p, and fits the
constants by local optimization, as constant-fitting SR packages do (PySR:
BFGS; Operon: Levenberg-Marquardt).

Constants are placed only where they are not absorbed by the linear model:
inside nonlinear unary operators (an affine argument) and in exponents. With a
linear head, exp(a*x + b) = e^b * exp(a*x), so redundant directions exist; a
small ridge on deviations from the symbolic starting values keeps them at the
start. Regression uses variable projection (the linear coefficients are solved
exactly at every step) with Levenberg-Marquardt; classification optimizes the
penalized likelihood jointly with L-BFGS. A refinement is accepted only if it
improves BIC and stays finite on the data and across the input domain.
"""
from fractions import Fraction
import math

import numpy as np
import torch

from .operators import TRANSCENDENTAL

NONLINEAR_UNARY = TRANSCENDENTAL + ("^-1",)


class ParamExpr:
    """One descriptor as an expression tree with optional fitted constants."""

    POWER_BOUND = 4.0   # |exponent| limit (widened to include the symbolic exponent)
    SCALE_BOUND = 10.0  # |a| limit for inner arguments a*x + b

    def __init__(self, bank, code):
        self.bank, self.ops = bank, bank.ops
        self.init = []
        self.bounds = []
        self.names = {int(bank.left[i]): bank.names[i] for i in range(len(bank)) if bank.op[i] < 0}
        op, left, right = code
        self.tree = self._feature(left) if op < 0 else self._compose(op, self._feature(left),
                                                                       None if right < 0 else self._feature(right),
                                                                       parent=left)

    def _feature(self, i):
        op, left, right = int(self.bank.op[i]), int(self.bank.left[i]), int(self.bank.right[i])
        if op < 0:
            return ("var", left)
        return self._compose(op, self._feature(left), None if right < 0 else self._feature(right), parent=left)

    def _param(self, value, bounds=(-math.inf, math.inf)):
        self.init.append(float(value))
        self.bounds.append(bounds)
        return len(self.init) - 1

    def _compose(self, op, a, b, parent=None):
        name = self.ops.names[op]
        if b is not None:
            return ("bin", op, a, b)
        if self.ops.is_hinge(op):
            # relu(a*x + b) starting at the training knot: a = 1, b = -knot; the knot moves freely
            knot = float(self.ops.knot(op, self.bank.X[parent]))
            return ("un", op, a, self._param(1.0, (-self.SCALE_BOUND, self.SCALE_BOUND)), self._param(-knot))
        if name.startswith("pow"):
            k = float(Fraction(name[4:-1]))
            return ("pow", a, self._param(k, (min(k, -self.POWER_BOUND), max(k, self.POWER_BOUND))))
        if name in NONLINEAR_UNARY:
            return ("un", op, a, self._param(1.0, (-self.SCALE_BOUND, self.SCALE_BOUND)), self._param(0.0))
        return ("fixed", op, a)

    def evaluate(self, theta, values):
        def ev(node):
            kind = node[0]
            if kind == "var":
                return values[node[1]]
            if kind == "bin":
                return self.ops.apply(node[1], ev(node[2]), ev(node[3]))
            if kind == "pow":
                return torch.pow(ev(node[1]), theta[node[2]])
            if kind == "un":  # (knot=0: for a hinge the shift is the fitted constant b)
                return self.ops.apply(node[1], theta[node[3]] * ev(node[2]) + theta[node[4]], knot=0.0)
            return self.ops.apply(node[1], ev(node[2]))
        return ev(self.tree)

    def format(self, theta):
        theta = [float(t) for t in theta]

        def same(k, value):
            return abs(theta[k] - value) <= 1e-9 * (1 + abs(value))

        def fm(node):
            kind = node[0]
            if kind == "var":
                return self.names[node[1]]
            if kind == "bin":
                return self.ops.format(node[1], fm(node[2]), fm(node[3]))
            if kind == "pow":
                return f"({fm(node[1])})**({theta[node[2]]:.6g})"
            if kind == "un" and self.ops.is_hinge(node[1]):  # relu(a*x - knot), printed like the bank
                inner, a, b = fm(node[2]), node[3], node[4]
                inner = inner if same(a, 1.0) else f"{theta[a]:.6g}*{inner}"
                return self.ops.format(node[1], inner, knot=-theta[b])
            if kind == "un":
                inner, a, b = fm(node[2]), node[3], node[4]
                if not same(a, 1.0):
                    inner = f"{theta[a]:.6g}*{inner}"
                if not same(b, 0.0):
                    inner = f"{inner}{theta[b]:+.6g}"
                return self.ops.format(node[1], inner if (same(a, 1.0) and same(b, 0.0)) else f"({inner})")
            return self.ops.format(node[1], fm(node[2]))
        return fm(self.tree)


class ParamModel:
    """Descriptors of one model with a shared constant vector."""

    def __init__(self, bank, codes):
        self.exprs = [ParamExpr(bank, code) for code in codes]
        self.offsets = np.cumsum([0] + [len(e.init) for e in self.exprs])
        self.init = torch.tensor([v for e in self.exprs for v in e.init], dtype=torch.float64, device=bank.device)
        bounds = [b for e in self.exprs for b in e.bounds]
        self.lower = torch.tensor([b[0] for b in bounds], dtype=torch.float64, device=bank.device)
        self.upper = torch.tensor([b[1] for b in bounds], dtype=torch.float64, device=bank.device)

    def clamp(self, theta):
        return torch.minimum(torch.maximum(theta, self.lower), self.upper)

    @property
    def size(self):
        return int(self.offsets[-1])

    def features(self, theta, values):
        cols = [e.evaluate(theta[self.offsets[i]:self.offsets[i + 1]], values) for i, e in enumerate(self.exprs)]
        return torch.stack(cols, 1)

    def terms(self, theta):
        return [e.format(theta[self.offsets[i]:self.offsets[i + 1]]) for i, e in enumerate(self.exprs)]

    def changed(self, theta):
        return int(((theta - self.init).abs() > 1e-6 * (1 + self.init.abs())).sum())


def starts(model, restarts, generator):
    """Symbolic start plus random perturbations (relative for exponents and scales)."""
    out = [model.init.clone()]
    for _ in range(restarts):
        noise = torch.randn(model.size, generator=generator, dtype=torch.float64).to(model.init.device)
        out.append(model.clamp(model.init + 0.5 * noise * (1 + model.init.abs())))
    return out


def _levenberg_marquardt(evaluate, starts_list, init, ridge, iterations, tolerance=1e-10, project=None):
    """Minimize ||r(theta)||^2 + ridge ||theta - init||^2 from several starts.

    ``evaluate(theta, jacobian)`` returns r, or (r, J) when jacobian is True.
    """
    best = None
    m = len(init)
    eye = torch.eye(m, dtype=init.dtype, device=init.device)
    for theta in starts_list:
        r = evaluate(theta, False)
        if not torch.isfinite(r).all():
            continue
        value = float(r.square().sum() + ridge * (theta - init).square().sum())
        lam = 1e-3
        for _ in range(iterations):
            r, J = evaluate(theta, True)
            if not torch.isfinite(J).all():
                break
            grad = J.T @ r + ridge * (theta - init)
            H = J.T @ J + ridge * eye
            improved, relative = False, 0.0
            while lam < 1e10:
                trial = theta - torch.linalg.solve(H + lam * torch.diag(H.diagonal().clamp_min(1e-12)), grad)
                if project is not None:
                    trial = project(trial)
                r_trial = evaluate(trial, False)
                if torch.isfinite(r_trial).all():
                    value_trial = float(r_trial.square().sum() + ridge * (trial - init).square().sum())
                    if value_trial < value:
                        relative = (value - value_trial) / max(value, 1e-300)
                        theta, value, lam, improved = trial, value_trial, max(lam / 3, 1e-12), True
                        break
                lam *= 4
            if not improved or relative < tolerance:
                break
        if best is None or value < best[0]:
            best = (value, theta.detach())
    return best


def fit_weighted(model, values, Y, W, ridge=1e-8, iterations=60, restarts=2, seed=0, theta0=None):
    """Weighted multi-target variable projection for the constants of some descriptors.

    Minimizes sum_k sum_i W_ik (Y_ik - a_k - F_i(theta) b_k)^2 over theta, with the
    linear coefficients (a_k, b_k) solved exactly by weighted least squares for
    every theta. Regression residual fitting is K = 1, W = 1; classification
    uses the working response g/h with weights h of each logit. The Jacobian is
    Kaufman's variable-projection form J = -P_perp (dA/dtheta) b, which needs
    only forward-mode feature derivatives (no differentiation through QR).
    Returns (theta, weighted RSS) or None.
    """
    n, K = Y.shape
    sw = W.clamp_min(0).sqrt()
    ones = torch.ones(n, 1, dtype=torch.float64, device=Y.device)
    features = lambda t: model.features(t, values)

    def evaluate(theta, jacobian):
        with torch.no_grad():
            A = torch.cat([ones, features(theta)], 1)
        dF = torch.func.jacfwd(features)(theta) if jacobian else None  # (n, d, m)
        residuals, jacobians = [], []
        for k in range(K):
            Aw, yw = sw[:, k:k + 1] * A, sw[:, k] * Y[:, k]
            Q, R = torch.linalg.qr(Aw)
            proj = Q.T @ yw
            residuals.append(yw - Q @ proj)
            if jacobian:
                coef = torch.linalg.solve_triangular(R, proj[:, None], upper=True)[:, 0]
                dA_coef = sw[:, k:k + 1] * (dF * coef[1:][None, :, None]).sum(1)  # (n, m)
                jacobians.append(-(dA_coef - Q @ (Q.T @ dA_coef)))
        r = torch.cat(residuals)
        return (r, torch.cat(jacobians)) if jacobian else r

    generator = torch.Generator().manual_seed(seed)
    begin = starts(model, restarts, generator)
    if theta0 is not None:
        begin.insert(0, theta0.clone())
    best = _levenberg_marquardt(evaluate, begin, model.init, ridge, iterations, project=model.clamp)
    return None if best is None else (best[1], best[0])


def fit_regression(model, values, y, ridge=1e-8, iterations=100, restarts=3, seed=0, theta0=None):
    """Variable projection + Levenberg-Marquardt for the constants of one regression model.

    Returns (theta, coefficients, intercept, rss) of the best restart, or None.
    """
    fitted = fit_weighted(model, values, y[:, None], torch.ones_like(y)[:, None], ridge, iterations,
                          restarts, seed, theta0)
    if fitted is None:
        return None
    theta = fitted[0]
    ones = torch.ones(len(y), 1, dtype=torch.float64, device=y.device)
    A = torch.cat([ones, model.features(theta, values)], 1)
    solution = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    rss = float((y - A @ solution).square().sum())
    return theta, solution[1:], float(solution[0]), rss


def fit_classification(model, values, Y, binary, weight, ridge, coef0=None, intercept0=None,
                       iterations=100, restarts=2, seed=0, theta0=None, offset=None):
    """Profiled likelihood: the linear head is solved exactly for every constant vector.

    For constants theta the descriptors are standardized and the ridge-penalized
    logistic/softmax head is fit by Newton's method (convex); the profiled
    objective is minimized over theta with L-BFGS-B using the envelope-theorem
    gradient (derivative at the fixed optimal head). This mirrors variable
    projection in regression. Returns (theta, coef, intercept, nll) with the head
    in original descriptor units, or None.
    """
    from scipy.optimize import minimize
    from .classification import fit_glm, nll
    device = Y.device
    ones = torch.ones(Y.shape[0], 1, dtype=torch.float64, device=device)
    state = {"head": None}
    base = 0.0 if offset is None else offset  # fixed logit offset (group effects)

    def standardized(theta):
        F = model.features(theta, values)
        mu, sd = F.mean(0), F.std(0)
        return (F - mu) / sd, mu, sd

    def head(Z):
        warm = None if state["head"] is None else state["head"][None]
        theta_head, objective, _ = fit_glm(Z[None], Y, binary, weight, ridge, theta0=warm, offset=offset)
        state["head"] = theta_head[0]
        return theta_head[0], float(objective[0])

    def profiled(theta_np):
        theta = torch.tensor(theta_np, dtype=torch.float64, device=device, requires_grad=True)
        Z, _, sd = standardized(theta)
        if not (torch.isfinite(Z).all() and (sd > 0).all()):
            return 1e300, np.zeros_like(theta_np)
        coef, _ = head(Z.detach())
        eta = torch.cat([ones, Z], 1) @ coef + base
        value = (nll(eta, Y, binary, weight) + 0.5 * ridge * coef[1:].square().sum()
                 + 1e-8 * (theta - model.init).square().sum())
        value.backward()
        grad = theta.grad.detach().cpu().numpy()
        if not (math.isfinite(float(value.detach())) and np.isfinite(grad).all()):
            return 1e300, np.zeros_like(theta_np)
        return float(value.detach()), grad

    best = None
    generator = torch.Generator().manual_seed(seed)
    begin = starts(model, restarts, generator)
    if theta0 is not None:
        begin.insert(0, theta0.clone())
    for start in begin:
        state["head"] = None
        bounds = [(None if not math.isfinite(lo) else lo, None if not math.isfinite(hi) else hi)
                  for lo, hi in zip(model.lower.tolist(), model.upper.tolist())]
        result = minimize(profiled, start.cpu().numpy(), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": iterations, "ftol": 1e-12, "gtol": 1e-9})
        if math.isfinite(result.fun) and result.fun < 1e299 and (best is None or result.fun < best[0]):
            best = (result.fun, torch.tensor(result.x, dtype=torch.float64, device=device))
    if best is None:
        return None
    theta = best[1]
    with torch.no_grad():
        Z, mu, sd = standardized(theta)
        state["head"] = None
        coef, _ = head(Z)
        loss = float(nll(torch.cat([ones, Z], 1) @ coef + base, Y, binary, weight))
        beta = coef[1:] / sd[:, None]
        intercept = coef[0] - (coef[1:] * (mu / sd)[:, None]).sum(0)
    return theta, beta, intercept, loss


def domain_finite(model, theta, bank, samples=512, seed=0):
    """True if all descriptors stay finite on random points of the input domain box."""
    rng = np.random.default_rng(seed)
    prim = [i for i in range(len(bank)) if bank.op[i] < 0]
    width = max(int(bank.left[i]) for i in prim) + 1
    lo = np.zeros(width)
    hi = np.zeros(width)
    for i in prim:
        lo[bank.left[i]], hi[bank.left[i]] = bank.lo[i], bank.hi[i]
    points = rng.uniform(lo[:, None], hi[:, None], (width, samples))
    points[:, :2] = np.stack([lo, hi], 1)  # include the box corners per input
    values = torch.as_tensor(points, dtype=torch.float64, device=bank.device)
    with torch.no_grad():
        return bool(torch.isfinite(model.features(theta, values)).all())


def refined_complexity(bank, codes, model, theta):
    """Complexity with each fitted constant counted as one distinct symbol."""
    total = 0.0
    for i, (op, left, right) in enumerate(codes):
        if op < 0:
            nsym = int(bank.nsym[left])
            unique = int(bank.varset[left].sum() + bank.opset[left].sum())
        else:
            meta = bank.child_meta([op], [left], [right])
            nsym = int(meta["nsym"][0])
            unique = int(meta["varset"][0].sum() + meta["opset"][0].sum())
        m = int(((theta[model.offsets[i]:model.offsets[i + 1]] - model.init[model.offsets[i]:model.offsets[i + 1]]).abs()
                 > 1e-6 * (1 + model.init[model.offsets[i]:model.offsets[i + 1]].abs())).sum())
        total += 1.0 if (op < 0 and m == 0 and bank.op[left] < 0) else (nsym + m) * math.log2(unique + m)
    return total


def projection_rss(F, Y, W):
    """Weighted RSS of the best per-target fit a_k + F b_k (F: n x d)."""
    sw = W.clamp_min(0).sqrt()
    A = torch.cat([torch.ones(F.shape[0], 1, dtype=F.dtype, device=F.device), F], 1)
    total = 0.0
    for k in range(Y.shape[1]):
        Aw, yw = sw[:, k:k + 1] * A, sw[:, k] * Y[:, k]
        Q, _ = torch.linalg.qr(Aw)
        total += float((yw - Q @ (Q.T @ yw)).square().sum())
    return total


def start_vector(model, term_params):
    """Joint start for a model: each term's screened constants, else its symbolic values."""
    theta = model.init.clone()
    for i, param in enumerate(term_params or []):
        if param is not None:
            theta[model.offsets[i]:model.offsets[i + 1]] = param[1]
    return theta
