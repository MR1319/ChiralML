# Symbolic regression for chiral assembly

This folder adds a symbolic-regression analysis of the experimental dataset in this repository. It asks whether
one short, unit-consistent equation can predict strong chiral assembly (|g| ≥ 10⁻³), and how it compares with the
random forest of `data_analysis/notebooks/Figure4.ipynb`.

The equations come from SyMANTIC with a logistic loss.
SyMANTIC builds candidate terms from 15 physically meaningful inputs and selects the best combination of at most 5
terms. Units are enforced: + and − only join quantities of the same unit.

## Contents

| Path | What it is |
|---|---|
| `symbolic_regression.ipynb` | The analysis, step by step |
| `chiral_sr/data.py` | Reads `data_analysis/data/analysis_data.xlsx` as Figure4.ipynb does: same label, same polymer ‖ solvent groups, same 41 random-forest features. Adds the 15 inputs with their units and a chemistry key (one repeat unit, molecular-weight variants together; 59 chemistries) |
| `chiral_sr/laws.py` | The laws (B32, b37, u29) and their refit, with or without one offset per chemistry; the law plane; choosing the offset penalty by inner cross-validation |
| `chiral_sr/protocol.py` | Figure4.ipynb's 20 × grouped 5-fold protocol, leave one chemistry out, the 717 → 56 split, the random forest with Figure4.ipynb's settings and seeds, bootstrap intervals |
| `chiral_sr/search.py` | Re-runs the SyMANTIC searches that produced the laws |
| `chiral_sr/plots.py` | Law-plane and confusion-matrix figures |
| `chiral_sr/symantic/` | Bundled copy of the SyMANTIC engine used for this analysis, so no separate install is needed |

## Setup

Start from the repository's analysis environment (Python 3.12) and add the engine's three dependencies:

```bash
pip install -r symbolic_regression/requirements.txt
```

Open `symbolic_regression/symbolic_regression.ipynb` and run the cells in order. The notebook finds
`data_analysis/data/analysis_data.xlsx` the same way the figure notebooks do. No data is copied into this folder.

## The three laws

| Law | Found on | Operators | Role in the analysis |
|---|---|---|---|
| **B32** | all 773 rows | + − × ÷ | The equation for the whole dataset |
| **b37** | the 717 initial rows | + − × ÷ | B32's search without the 56 additional rows: an honest test on those 56 |
| **u29** | the 717 initial rows | + − × ÷ √ x² x³ | The same, with powers allowed |

The structure of B32 (which terms it uses) was chosen on all 773 rows. In every cross-validation of B32, the
coefficients are refit on each training fold, but the structure is fixed. Those scores are therefore optimistic,
in the same way as scoring any final model on data that informed its design. The 56 additional rows are not a test
for B32, because its structure saw them. On that split, b37 and u29 are the honest models.

## Chemistry offsets

The laws can be fitted with one intercept shift per chemistry (a random-intercept logistic regression). Each
chemistry's shift is penalised towards 0, so chemistries with few rows move little. A chemistry absent from the
training rows gets no shift. The offsets help when the test polymers are known and cannot help for new polymers;
the notebook shows both cases.
