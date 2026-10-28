"""Physical-unit algebra for dimensional feature construction.

A unit is an exponent vector over base unit symbols, e.g. 'L/T**2' -> (1, -2)
over (L, T). Rational exponents (from pow(1/2)) are stored as float64 and
hashed after rounding. ``relational_units`` declare equivalences such as
('u1*u2', 'u3'); every unit is mapped to a canonical representative of its
equivalence class by eliminating the pivot coordinates of those relations.
The map is linear, so canonical primitives yield canonical derived units.

Dimensionless is the zero vector; the symbol '1' (as used by SyMANTIC's
``dimensionality`` lists) and plain numbers denote it.
"""
import re

import numpy as np
import sympy as sp

_HASH_SCALE = 2**20
_DIMENSIONLESS = sp.Symbol("1")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _expression(unit):
    """Parse a unit given as text or SymPy without SymPy's reserved names (S, N, E, I...)."""
    if unit is None:
        return sp.Integer(1)
    if isinstance(unit, sp.Basic):
        return unit
    text = str(unit).strip()
    if text in ("", "1"):
        return sp.Integer(1)
    names = {name: sp.Symbol(name) for name in _IDENTIFIER.findall(text)}
    return sp.sympify(text, locals=names)


def unit_keys(units):
    """One int64 key per row; equal keys <=> equal (canonical) units."""
    units = np.atleast_2d(np.asarray(units, dtype=np.float64))
    if units.shape[1] == 0:
        return np.zeros(units.shape[0], dtype=np.int64)
    scaled = np.round(units * _HASH_SCALE).astype(np.int64)
    key = np.zeros(units.shape[0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(scaled.shape[1]):
            key = key * np.uint64(1000003) + scaled[:, j].astype(np.uint64)
    return key.view(np.int64)


class UnitSystem:
    """Base symbols, canonical primitive units, and the target unit."""

    def __init__(self, dimensionality, relational_units=None, output_dim=None):
        exprs = [_expression(d) for d in dimensionality]
        pairs = [(_expression(a), _expression(b)) for a, b in (relational_units or [])]
        target = None if output_dim is None else _expression(output_dim)
        pool = exprs + [e for pair in pairs for e in pair] + ([target] if target is not None else [])
        found = set().union(*(e.free_symbols for e in pool)) - {_DIMENSIONLESS}
        self.symbols = sorted(found, key=str)
        self.index = {s: i for i, s in enumerate(self.symbols)}
        relations = [self._vector(a) - self._vector(b) for a, b in pairs]
        self._pivots = self._reduce(relations)
        self.primitive = np.array([self.canonical(self._vector(e)) for e in exprs]).reshape(len(exprs), len(self.symbols))
        self.target = None if target is None else self.canonical(self._vector(target))

    @property
    def size(self):
        return len(self.symbols)

    def _vector(self, expr):
        vec = np.zeros(len(self.symbols))
        expr = sp.expand_power_base(sp.powdenest(expr, force=True), force=True)
        for base, exponent in expr.as_powers_dict().items():
            if base == _DIMENSIONLESS or base.is_number:
                continue
            if not isinstance(base, sp.Symbol):
                raise ValueError(f"Cannot interpret unit factor {base!r}; use products of powers of symbols")
            vec[self.index[base]] += float(exponent)
        return vec

    def _reduce(self, relations):
        if not relations:
            return []
        reduced, pivots = sp.Matrix(np.array(relations)).applyfunc(sp.nsimplify).rref()
        return [(p, np.array(reduced.row(i), dtype=np.float64).ravel()) for i, p in enumerate(pivots)]

    def canonical(self, vec):
        vec = np.array(vec, dtype=np.float64)
        for pivot, row in self._pivots:
            vec = vec - vec[pivot] * row
        return vec

    def format(self, vec):
        """Human-readable unit, e.g. 'u1**2*u2**(-1)'."""
        expr = sp.Integer(1)
        for symbol, exponent in zip(self.symbols, vec):
            if abs(exponent) > 1e-12:
                expr *= symbol ** sp.nsimplify(exponent, rational=True)
        return str(expr)
