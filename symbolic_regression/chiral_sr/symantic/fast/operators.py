"""Operator registry: numerical maps, printed forms, unit rules, and codes.

Printed forms follow the original SyMANTIC names so equations stay readable
and SymPy-parsable. Deliberate changes: ``log`` is computed with log10, so it
is printed as ``log(x, 10)``; ``/2pi`` and ``*2pi`` print as ``x/(2*pi)`` and
``x*(2*pi)`` (the original ``x/2pi`` does not parse).

Unit rules for unary operators (units are exponent vectors, see units.py):
the child's units are ``factor * parent_units``; operators in
``REQUIRES_DIMENSIONLESS`` only apply to dimensionless parents when units are
strict. Transcendental functions always produce dimensionless output.
"""
from fractions import Fraction
import math
import re

import torch

BINARY = ("+", "-", "*", "/")
# 'hinge' / 'hinge(q)': relu(x - knot), with the knot at quantile q (default 0.5)
# of the parent feature's TRAINING values -- MARS-style threshold terms. The
# knot is never stored: wherever a feature is evaluated it is recomputed from
# the parent's training values in the bank, so new data use the training knot.
TRANSCENDENTAL = ("exp", "ln", "log", "sin", "sinh", "cos", "cosh", "tanh", "exp(-1)")
SHIFTS = ("+1", "-1")
_TWO_PI = 2 * math.pi

_UNARY = {
    "exp": (torch.exp, "(exp({}))"),
    "ln": (torch.log, "(ln({}))"),
    "log": (torch.log10, "(log({}, 10))"),
    "sin": (torch.sin, "(sin({}))"),
    "sinh": (torch.sinh, "(sinh({}))"),
    "cos": (torch.cos, "(cos({}))"),
    "cosh": (torch.cosh, "(cosh({}))"),
    "tanh": (torch.tanh, "(tanh({}))"),
    "^-1": (torch.reciprocal, "(({})**-1)"),
    "exp(-1)": (lambda x: torch.exp(-x), "(exp(-{}))"),
    "+1": (lambda x: x + 1, "({}+1)"),
    "-1": (lambda x: x - 1, "({}-1)"),
    "/2": (lambda x: x / 2, "({}/2)"),
    "/2pi": (lambda x: x / _TWO_PI, "({}/(2*pi))"),
    "*2pi": (lambda x: x * _TWO_PI, "({}*(2*pi))"),
}


class OperatorSet:
    """Splits user operators into binary/unary and assigns stable codes.

    The code of an operator is its position in the user's list, which is also
    its bit in each feature's operator set (used by the complexity measure).
    """

    def __init__(self, operators, strict_units=True):
        if not operators:
            raise ValueError("Provide at least one operator")
        self.names = list(operators)
        self.binary = {}
        self.unary = {}
        self.unit_factor = {}
        self.requires_dimensionless = {}
        self.hinge_q = {}
        for code, op in enumerate(self.names):
            if op in BINARY:
                self.binary[op] = code
                continue
            if op == "hinge" or op.startswith("hinge("):
                q = 0.5 if op == "hinge" else float(Fraction(re.findall(r"\(([^)]*)\)", op)[0]))
                if not 0.0 < q < 1.0:
                    raise ValueError(f"Hinge quantile must be in (0, 1), got {op!r}")
                self.hinge_q[code] = q
                self.unary[code] = (None, None)  # evaluated by apply/format with an explicit knot
                factor = 1.0
            elif op in _UNARY:
                fn, fmt = _UNARY[op]
                self.unary[code] = (fn, fmt.format)
                factor = {"^-1": -1.0}.get(op, 0.0 if op in TRANSCENDENTAL else 1.0)
            elif op.startswith("pow"):
                found = re.findall(r"\(([^)]*)\)", op)
                if len(found) != 1:
                    raise ValueError(f"Cannot parse power operator {op!r}; use e.g. 'pow(2)' or 'pow(1/2)'")
                exponent = float(Fraction(found[0]))
                text = found[0]
                self.unary[code] = (lambda x, e=exponent: torch.pow(x, e),
                                    lambda s, t=text: f"({s})**({t})")
                factor = exponent
            else:
                raise ValueError(f"Unsupported operator {op!r}")
            self.unit_factor[code] = factor
            self.requires_dimensionless[code] = strict_units and (op in TRANSCENDENTAL or op in SHIFTS)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Operators must be unique")

    def __len__(self):
        return len(self.names)

    def code(self, op):
        return self.binary[op]

    def has(self, op):
        return op in self.binary

    def is_hinge(self, code):
        return code in self.hinge_q

    def knot(self, code, train_values):
        """Hinge knot(s): quantile of each training row (rows, n) -> (rows, 1), or (n,) -> scalar."""
        return _row_quantile(train_values, self.hinge_q[code])

    def apply(self, code, left, right=None, knot=None):
        """Evaluate one operator on feature rows (feature-major tensors).

        For a hinge, ``knot`` must be given when ``left`` is not the training data
        (prediction); without it the knot is taken from ``left`` itself."""
        if right is None:
            if code in self.hinge_q:
                k = self.knot(code, left) if knot is None else knot
                return torch.relu(left - k)
            return self.unary[code][0](left)
        op = self.names[code]
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        return left / right

    def format(self, code, left, right=None, knot=None):
        if right is None:
            if code in self.hinge_q:
                shift = "" if knot is None or float(knot) == 0.0 else f"{-float(knot):+.6g}"
                return f"(relu({left}{shift}))"
            return self.unary[code][1](left)
        return f"({left}{self.names[code]}{right})"


def _row_quantile(x, q):
    """Linear-interpolated quantile along the last axis (numpy's default), via sort."""
    s = torch.sort(x, dim=-1).values
    pos = q * (s.shape[-1] - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.shape[-1] - 1)
    out = s[..., lo] + (pos - lo) * (s[..., hi] - s[..., lo])
    return out[..., None] if x.dim() > 1 else out
