"""Re-running the SyMANTIC searches that produced the laws (bundled engine in chiral_sr/symantic).

Settings of every search here: logistic loss, at most 5 terms, depth 2 (terms are built from the 15 inputs by up
to two levels of operators), strict units (+ and - only join quantities of the same unit), Pareto selection
'utopia', seed 7. The reported law is the lowest-loss model with 5 terms on the Pareto front.
"""
import pandas as pd

OPERATORS = {'binary': ['+', '-', '*', '/'],
             'units': ['+', '-', '*', '/', 'pow(1/2)', 'pow(2)', 'pow(3)']}


def search(d, rows, operators='binary', n_term=5, max_level=2, seed=7, verbose=False):
    """Pareto front of a search on the rows `rows`: one line per model (terms, complexity, training log-loss,
    printed equation)."""
    from .symantic.fast import SymanticClassifier
    clf = SymanticClassifier(operators=OPERATORS[operators], n_term=n_term, max_level=max_level,
                             criterion='logistic', selection='utopia', random_state=seed, disp=verbose,
                             dimensionality=d.units, strict_units=True)
    clf.fit(d.X.iloc[rows].reset_index(drop=True), d.y[rows])
    return pd.DataFrame([dict(n_terms=len(m['terms']), complexity=m['complexity'], log_loss=m['log_loss'],
                              equation=m['equation']) for m in clf.models_])


def best(front, k=5):
    """The lowest-loss model with at most k terms."""
    f = front[front.n_terms <= k]
    return f.loc[f.log_loss.idxmin()]


def terms(equation):
    """The term strings of a printed equation 'logit P(1) = c1*(t1) + c2*(t2) + ... + b'."""
    rhs, parts, depth, cur, i = equation.split('=', 1)[1], [], 0, '', 0
    while i < len(rhs):
        ch = rhs[i]
        depth += (ch == '(') - (ch == ')')
        if depth == 0 and rhs.startswith(' + ', i):
            parts.append(cur)
            cur, i = '', i + 3
            continue
        cur += ch
        i += 1
    parts.append(cur)
    return [p.strip().split('*', 1)[1] for p in parts if '*' in p]           # the last part is the intercept
