"""Figures for the notebook: the law plane and confusion matrices."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

STRONG, NOT_STRONG = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SHARE = LinearSegmentedColormap.from_list('share', ['#f1f0ec', INK2])


def limits(*parts, pad=0.4):
    """Common (xlo, xhi, ylo, yhi) for several (x, y) law-plane parts."""
    xs = np.concatenate([p[0] for p in parts])
    ys = np.concatenate([p[1] for p in parts])
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


def law_plane(ax, x, y_part, label, wrong=None, lims=None, xlabel='concentration part (logit)',
              ylabel='pair part (logit)'):
    """Rows in the law plane, coloured by the measured label. Solid line: P = 0.5 (x + y = 0); dashed: P = 0.25
    and 0.75. With `wrong` (boolean per row), the rows on the wrong side of P = 0.5 are ringed."""
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')
    order = np.random.default_rng(0).permutation(len(x))
    ax.scatter(x[order], y_part[order], s=22, c=np.where(label[order] == 1, STRONG, NOT_STRONG), alpha=0.75,
               edgecolors='white', linewidths=0.5, zorder=3)
    if wrong is not None:
        ax.scatter(x[wrong], y_part[wrong], s=75, facecolors='none', edgecolors=INK, linewidths=0.9, zorder=4)
    lo, hi, ylo, yhi = lims or limits((x, y_part))
    xs = np.array([lo, hi])
    for off in (0.0, np.log(3), -np.log(3)):
        ax.plot(xs, off - xs, color=INK2 if off == 0 else MUTED, ls='-' if off == 0 else '--',
                lw=1.2 if off == 0 else 0.9, zorder=2)
    ax.set(xlim=(lo, hi), ylim=(ylo, yhi), xlabel=xlabel, ylabel=ylabel)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], ls='', marker='o', mfc=STRONG, mec='white', label=f'strong, n = {int(label.sum())}'),
               Line2D([], [], ls='', marker='o', mfc=NOT_STRONG, mec='white',
                      label=f'not strong, n = {int((1 - label).sum())}')]
    if wrong is not None:
        handles.append(Line2D([], [], ls='', marker='o', ms=9, mfc='none', mec=INK, label='wrong side of P = 0.5'))
    ax.legend(handles=handles, loc='lower left', frameon=False)


def predictions(axes, names, label, probabilities, groups):
    """One panel per model, side by side: each row's predicted P(strong) on x, the rows listed on the shared y axis
    (`names`, top to bottom), coloured by the measured label. Dashed line: P = 0.5; thin lines separate `groups`
    (e.g. polymers)."""
    ys = np.arange(len(names))[::-1]
    axes[0].figure.set_facecolor('white')
    for ax, (model, p) in zip(axes, probabilities.items()):
        ax.set_facecolor('white')
        ax.axvline(0.5, color=MUTED, ls='--', lw=0.9, zorder=1)
        for k in np.where(groups[1:] != groups[:-1])[0]:
            ax.axhline(ys[k] - 0.5, color=GRID, lw=0.8, zorder=0)
        ax.scatter(p, ys, s=36, c=np.where(label == 1, STRONG, NOT_STRONG), edgecolors='white', linewidths=0.5,
                   zorder=3)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.7, len(names) - 0.3), xlabel='predicted P(strong)')
        ax.set_title(model, loc='left', fontsize=10)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.grid(axis='y', visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(ys, names, fontsize=7.5)
    axes[0].figure.legend(handles=[Line2D([], [], ls='', marker='o', mfc=STRONG, mec='white',
                                          label=f'measured strong, n = {int(label.sum())}'),
                                   Line2D([], [], ls='', marker='o', mfc=NOT_STRONG, mec='white',
                                          label=f'measured not strong, n = {int((1 - label).sum())}')],
                          loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)


def counts(label, logit):
    """[[TP, FN], [FP, TN]] at P = 0.5 (logit 0)."""
    p = logit > 0
    return np.array([[(p & (label == 1)).sum(), (~p & (label == 1)).sum()],
                     [(p & (label == 0)).sum(), (~p & (label == 0)).sum()]], float)


def confusion(ax, M, title='', base=None, decimals=0):
    """A 2 x 2 confusion matrix M = [[TP, FN], [FP, TN]] shaded by the share of the measured class (row); with
    `base`, the change from that matrix is shown in brackets."""
    share = M / M.sum(1, keepdims=True)
    ax.imshow(share, cmap=SHARE, vmin=0, vmax=1)
    for (i, k), v in np.ndenumerate(M):
        txt = f'{v:.{decimals}f}\n{share[i, k]:.0%}'
        if base is not None:
            txt += f'   ({v - base[i, k]:+.{decimals}f})'
        ax.text(k, i, txt, ha='center', va='center', fontsize=10, color='white' if share[i, k] > 0.55 else INK)
    ax.set_xticks([0, 1], ['predicted strong', 'predicted not strong'])
    ax.set_yticks([0, 1], [f'strong\n(n = {M[0].sum():.0f})', f'not strong\n(n = {M[1].sum():.0f})'])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    tp, fn, fp, tn = M.ravel()
    ax.set_title(f'{title}\n{fn + fp:.{decimals}f} wrong · recall {tp / (tp + fn):.0%} · precision '
                 f'{tp / (tp + fp):.0%}', loc='left', fontsize=9.5)
