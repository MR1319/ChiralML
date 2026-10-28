"""Evaluation protocols and scores, shared by the laws and the random forest.

  authors' folds     Figure4.ipynb's protocol: 20 repeats of StratifiedGroupKFold(5, shuffle=True,
                     random_state=1000 + repeat), grouped by polymer || solvent; AP of the pooled out-of-fold scores
                     of each repeat. About 85% of test rows have their chemistry in the training rows.
  new polymers       leave one chemistry out (59 folds): every test polymer is unseen, molecular-weight variants
                     included; AP of the pooled scores.
  717 -> 56          train on the 717 initial rows, test on the 56 additional rows (the order the data came in).

A model is passed as predict(train, test, seed) -> scores of the test rows (higher = more likely strong); the seed
is used by the random forest only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

N_REPEATS, N_SPLITS = 20, 5
RF_PARAMS = dict(n_estimators=1100, max_depth=12, min_samples_leaf=2, min_samples_split=2, max_features=0.5,
                 class_weight='balanced', bootstrap=True, n_jobs=-1)       # Figure4.ipynb's CLASSIFIER_PARAMS


def authors_folds(d, rows, repeat):
    """The 5 (train, test) index pairs of one repeat, exactly as Figure4.ipynb builds them."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + repeat)
    return [(rows[a], rows[b]) for a, b in cv.split(np.zeros((len(rows), 1)), d.y[rows], d.groups[rows])]


def authors_cv(d, rows, predict, repeats=N_REPEATS):
    """{repeat: out-of-fold scores (NaN outside rows)}; fold f of repeat r uses seed (1000 + r) * 100 + f."""
    out = {}
    for r in range(repeats):
        s = np.full(len(d.y), np.nan)
        for f, (tr, te) in enumerate(authors_folds(d, rows, r), start=1):
            s[te] = predict(tr, te, (1000 + r) * 100 + f)
        out[r] = s
    return out


def leave_one_chemistry_out(d, rows, predict):
    """Pooled scores with each chemistry held out once (seed 1000 + its index)."""
    s = np.full(len(d.y), np.nan)
    for i, c in enumerate(np.unique(d.chem[rows])):
        tr, te = rows[d.chem[rows] != c], rows[d.chem[rows] == c]
        s[te] = predict(tr, te, 1000 + i)
    return s


def random_forest(d, X=None):
    """The authors' random forest as predict(train, test, seed); X defaults to their 41 features (NaN kept, as in
    Figure4.ipynb: scikit-learn's forest handles it)."""
    Z = (d.X41 if X is None else X).to_numpy(float)

    def predict(tr, te, seed):
        model = RandomForestClassifier(**RF_PARAMS, random_state=seed).fit(Z[tr], d.y[tr])
        return model.predict_proba(Z[te])[:, 1]
    return predict


def rf_seed_average(d, train, test, X=None, seeds=20):
    """Mean probability of the random forest over seeds 0..19 (for the single 717 -> 56 split)."""
    predict = random_forest(d, X)
    return np.mean([predict(train, test, s) for s in range(seeds)], axis=0)


def polymer_prior(d, train, test, k=2.0):
    """Each test row scored by its polymer's share of strong rows in training (smoothed towards the overall
    share with k pseudo-rows): what knowing the polymer alone gives."""
    base = d.y[train].mean()
    poly = d.meta['polymer'].to_numpy()
    g = pd.DataFrame({'p': poly[train], 'y': d.y[train]}).groupby('p')['y'].agg(['sum', 'size'])
    rate = (g['sum'] + k * base) / (g['size'] + k)
    return pd.Series(poly[test]).map(rate).fillna(base).to_numpy()


# ----------------------------------------------------------------------------- scores and intervals

def ap(d, scores, rows, weight=None):
    return average_precision_score(d.y[rows], scores[rows], sample_weight=weight)


def auc(d, scores, rows, weight=None):
    return roc_auc_score(d.y[rows], scores[rows], sample_weight=weight)


def ap_per_repeat(d, cv_scores, rows):
    return np.array([ap(d, s, rows) for s in cv_scores.values()])


def cluster_weights(clusters, n_boot=2000, seed=0):
    """n_boot x len(clusters) row weights: resample whole clusters (e.g. chemistries) with replacement."""
    codes, uniq = pd.factorize(clusters)
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
                     for _ in range(n_boot)]).astype(float)


def paired_delta_cv(d, a, b, rows, n_boot=1000, seed=0):
    """AP(a) - AP(b) on the authors' folds: mean over repeats, repeats won by a, and a 95% CI from resampling
    chemistries (the same resample in every repeat, deltas averaged over repeats)."""
    yr = d.y[rows]
    W = cluster_weights(d.chem[rows], n_boot, seed)
    deltas = [ap(d, a[r], rows) - ap(d, b[r], rows) for r in a]
    boot = np.full((n_boot, len(a)), np.nan)
    for j, r in enumerate(a):
        for i, w in enumerate(W):
            if 0 < (w * yr).sum() < w.sum():
                boot[i, j] = ap(d, a[r], rows, w) - ap(d, b[r], rows, w)
    lo, hi = np.nanpercentile(np.nanmean(boot, axis=1), [2.5, 97.5])
    return dict(delta=float(np.mean(deltas)), ci=(float(lo), float(hi)), wins=int(np.sum(np.array(deltas) > 0)))


def bootstrap(d, scores, rows, clusters, metric=ap, n_boot=2000, seed=0):
    """{name: 2.5 / 97.5 percentiles of the metric} and the bootstrap draws, resampling `clusters` (one label per
    row of `rows`, e.g. chemistries or test pairs); resamples without both classes are skipped. `scores`: {name:
    scores over all rows}. Every name gets the same resamples, so differences can be read pairwise."""
    yr = d.y[rows]
    W = cluster_weights(clusters, n_boot, seed)
    W = W[[(w * yr).sum() > 0 and (w * (1 - yr)).sum() > 0 for w in W]]
    draws = {m: np.array([metric(d, s, rows, w) for w in W]) for m, s in scores.items()}
    return {m: np.percentile(v, [2.5, 97.5]) for m, v in draws.items()}, draws


def holdout_table(d, probability, rows, clusters, n_boot=2000, seed=0):
    """Each model on held-out rows: AP and AUC with 95% intervals from resampling `clusters`, the calls at
    P = 0.5 and the accuracy. Returns (table, AP draws); the draws share their resamples across models."""
    ci_ap, draws = bootstrap(d, probability, rows, clusters, ap, n_boot, seed)
    ci_auc, _ = bootstrap(d, probability, rows, clusters, auc, n_boot, seed)
    yt = d.y[rows]
    out = {}
    for m, p in probability.items():
        called = p[rows] > 0.5
        out[m] = {'AP': round(ap(d, p, rows), 3), 'AP 95% CI': f'[{ci_ap[m][0]:.3f}, {ci_ap[m][1]:.3f}]',
                  'AUC': round(auc(d, p, rows), 3), 'AUC 95% CI': f'[{ci_auc[m][0]:.3f}, {ci_auc[m][1]:.3f}]',
                  'strong rows caught (P > 0.5)': f'{(called & (yt == 1)).sum()} of {yt.sum()}',
                  'not-strong rows called strong': f'{(called & (yt == 0)).sum()} of {(yt == 0).sum()}',
                  'accuracy': round((called == (yt == 1)).mean(), 3)}
    return pd.DataFrame(out).T, draws


def paired_difference(d, probability, draws, rows, a, b):
    """AP(a) - AP(b) on the rows and its 95% interval from the shared bootstrap draws."""
    lo, hi = np.percentile(draws[a] - draws[b], [2.5, 97.5])
    return ap(d, probability[a], rows) - ap(d, probability[b], rows), (lo, hi)
