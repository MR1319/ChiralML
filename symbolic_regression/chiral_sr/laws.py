"""The symbolic laws: their terms, refitting them with or without chemistry offsets, and reading a fitted law.

A law is a logistic regression on a few symbolic terms T_k built from the 15 inputs:

    logit P(strong) = b0 + sum_k beta_k * z_k  [+ u_chemistry]          z_k = T_k standardised on the training rows

Fitted as SyMANTIC fits its own models: the summed log-loss plus 0.5 * RIDGE * sum beta_k^2, intercept unpenalised,
by damped Newton steps (the engine's fit_glm). With offsets, one u per chemistry of the training rows is fitted
jointly, with penalty 0.5 * penalty * sum u^2: the posterior mode of a random-intercept logistic regression with
u ~ N(0, 1/penalty). A chemistry absent from the training rows gets u = 0.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .data import short
from .symantic.fast.classification import fit_glm

RIDGE = 0.1         # on the standardised term coefficients (SyMANTIC's default)
PENALTY = 0.3       # on the chemistry offsets: prior SD 1 / sqrt(0.3) = 1.83 logit units
C = 'Concentration_mg_mL'

# Three laws, each the lowest-loss 5-term model of a SyMANTIC search (logistic loss, strict units, depth 2, seed 7):
#   B32  + - x /                      on all 773 rows
#   b37  + - x /                      on the 717 initial rows (the same search as B32, without the 56)
#   u29  + - x / sqrt x^2 x^3         on the 717 initial rows
LAWS = {
    'B32': dict(rows='773', operators='binary', terms=[
        '((SC_C_atoms-BB_C_atoms)*(Concentration_mg_mL*SC_n_branch_points))',
        '(Solvent_NumHAcceptors*(SC_n_units*Backbone_fastsolv))',
        '((Concentration_mg_mL*Backbone_fastsolv)*(Concentration_mg_mL/BB_conjugated_units))',
        '((SC_C_atoms-BB_C_atoms)*(SC_n_units*BB_C_atoms))',
        '((Concentration_mg_mL*MW_Mn)*(BB_C_atoms/BB_hetero_atoms))']),
    'b37': dict(rows='717', operators='binary', terms=[
        '((SC_C_atoms-BB_C_atoms)*(Concentration_mg_mL*SC_n_branch_points))',
        '((Solvent_NumHAcceptors-Monomer_fastsolv)*(SC_n_units*Monomer_fastsolv))',
        '((Solvent_NumHAcceptors+Monomer_fastsolv)*(Concentration_mg_mL*MW_Mn))',
        '((Concentration_mg_mL*SC_C_atoms)*(Concentration_mg_mL*SC_n_branch_points))',
        '((SC_C_atoms-BB_C_atoms)*(SC_n_units*BB_C_atoms))']),
    'u29': dict(rows='717', operators='units', terms=[
        '((SC_C_atoms-BB_C_atoms)*(Concentration_mg_mL*SC_n_branch_points))',
        '((SC_C_atoms-BB_hetero_atoms)*(Solvent_NumHAcceptors*Backbone_fastsolv))',
        '((Concentration_mg_mL*SC_C_atoms))**(3)',
        '((Concentration_mg_mL)**(3)*(Solvent_MolLogP+Backbone_fastsolv))',
        '((Solvent_MolLogP)**(3)-(Monomer_fastsolv)**(3))']),
}
POSITIVE = {'Temperature', 'Concentration_mg_mL', 'MW_Mn', 'BB_C_atoms', 'BB_hetero_atoms', 'BB_conjugated_units'}


def term_values(terms, X):
    """n x k matrix: every term evaluated on every row of X (the 15 inputs)."""
    syms = {c: sp.Symbol(c, positive=c in POSITIVE) for c in X.columns}
    cols = [X[c].to_numpy(float) for c in X.columns]
    out = []
    for t in terms:
        f = sp.lambdify(list(syms.values()), sp.sympify(t, locals=syms), 'numpy')
        out.append(np.broadcast_to(np.asarray(f(*cols), float), len(X)))
    F = np.column_stack(out) if out else np.zeros((len(X), 0))
    if not np.isfinite(F).all():
        raise ValueError('a term is not finite on some rows')
    return F


@dataclass
class Fit:
    terms: list
    intercept: float
    effects: np.ndarray                             # logit change per training SD of each term
    mean: np.ndarray
    sd: np.ndarray
    offsets: dict = field(default_factory=dict)     # chemistry code -> u (empty for a law without offsets)

    def contributions(self, F):
        """n x k: each term's contribution to the logit (0 = the average training row)."""
        return (F - self.mean) / self.sd * self.effects

    def offset(self, chem):
        return np.array([self.offsets.get(c, 0.0) for c in chem]) if self.offsets else np.zeros(len(chem))

    def logit(self, F, chem=None):
        eta = self.intercept + self.contributions(F).sum(1)
        return eta + self.offset(chem) if self.offsets else eta

    def probability(self, F, chem=None):
        return 1 / (1 + np.exp(-self.logit(F, chem)))

    def equation(self, digits=4):
        """The fitted law in the raw units of the inputs (offsets not shown)."""
        raw = self.effects / self.sd
        b = self.intercept - (self.effects * self.mean / self.sd).sum()
        parts = [f'{w:+.{digits}g}·{short(t)}' for w, t in zip(raw, self.terms)]
        return 'logit P = ' + ' '.join(parts) + f' {b:+.{digits}g}' + (' + u_chemistry' if self.offsets else '')


def fit(F, y, train, terms, chem=None, penalty=PENALTY):
    """Fit the law on the rows `train`; with `chem` (one code per row), also one offset per chemistry in `train`."""
    mean, sd = F[train].mean(0), F[train].std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (F[train] - mean) / sd
    if chem is None:
        levels, G = np.array([], dtype=int), np.zeros((len(train), 0))
    else:
        levels = np.unique(chem[train])
        G = (chem[train][:, None] == levels[None, :]).astype(float)
    ridge = torch.as_tensor(np.r_[np.full(F.shape[1], RIDGE), np.full(len(levels), penalty)], dtype=torch.float64)
    theta = fit_glm(torch.as_tensor(np.c_[Z, G], dtype=torch.float64)[None],
                    torch.as_tensor(y[train], dtype=torch.float64)[:, None], True,
                    torch.ones(len(train), dtype=torch.float64), ridge)[0][0][:, 0].numpy()
    k = F.shape[1]
    return Fit(list(terms), float(theta[0]), theta[1:1 + k], mean, sd, dict(zip(levels.tolist(), theta[1 + k:])))


def law_plane(fitted, F, chem=None):
    """(concentration part, pair part) per row: the terms containing c, and the intercept + the other terms
    (+ the offset). Their sum is the logit, so P = 0.5 on the line x + y = 0. The pair part is fixed for a
    polymer-solvent pair at one temperature: changing c moves a row sideways."""
    con = fitted.contributions(F)
    has_c = np.array([C in t for t in fitted.terms])
    pair = fitted.intercept + con[:, ~has_c].sum(1)
    if fitted.offsets:
        pair = pair + fitted.offset(chem)
    return con[:, has_c].sum(1), pair


def choose_penalty(F, y, train, terms, chem, groups, grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0), shuffles=3):
    """The offset penalty with the lowest held-out log-loss in grouped 5-fold CV inside `train` (groups: the
    polymer || solvent pairs). Returns (best penalty, {penalty: mean held-out log-loss})."""
    loss = {}
    for lam in grid:
        ll = []
        for seed in range(shuffles):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            for a, b in cv.split(train, y[train], groups[train]):
                eta = fit(F, y, train[a], terms, chem, lam).logit(F[train[b]], chem[train[b]])
                ll.append(np.mean(np.logaddexp(0, eta) - y[train[b]] * eta))
        loss[lam] = float(np.mean(ll))
    return min(loss, key=loss.get), loss
