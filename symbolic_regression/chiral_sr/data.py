"""The analysis data of this repository, arranged for the symbolic-regression notebook.

Reads data_analysis/data/analysis_data.xlsx exactly as data_analysis/notebooks/Figure4.ipynb does (same label,
same polymer || solvent groups, same 41 random-forest features) and adds what the equations need: the 15 inputs
with their physical units and a chemistry key for the offsets.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N = '_Norm_MolWt'

# The 15 inputs of the symbolic regression and their SI units (SyMANTIC's dimensional analysis). Counts are per
# repeat unit / (MolWt_monomer / 1000), i.e. mol of that entity per kg of polymer; logP and the FASTSOLV
# solubilities are logarithms, so dimensionless.
INPUTS = {
    'Temperature': 'K',
    'Concentration_mg_mL': 'kg/m**3',
    'MW_Mn': 'kg/mol',
    'SC_C_atoms' + N: 'mol/kg',
    'SC_O_atoms' + N: 'mol/kg',
    'SC_n_units' + N: 'mol/kg',
    'SC_n_branch_points' + N: 'mol/kg',
    'BB_C_atoms' + N: 'mol/kg',
    'BB_hetero_atoms' + N: 'mol/kg',
    'BB_conjugated_units' + N: 'mol/kg',
    'Solvent_MolLogP': '1',
    'Solvent_NumHDonors': '1',
    'Solvent_NumHAcceptors': '1',
    'Monomer_fastsolv': '1',
    'Backbone_fastsolv': '1',
}

DESCRIPTIONS = {
    'Temperature': 'condition: temperature',
    'Concentration_mg_mL': 'condition: polymer concentration',
    'MW_Mn': 'chain length: number-average molecular weight',
    'SC_C_atoms': 'side chain: carbon atoms',
    'SC_O_atoms': 'side chain: oxygen atoms (glycol / alkoxy)',
    'SC_n_units': 'side chain: side chains (grafting density)',
    'SC_n_branch_points': 'side chain: branch points',
    'BB_C_atoms': 'backbone: carbon atoms',
    'BB_hetero_atoms': 'backbone: heteroatoms',
    'BB_conjugated_units': 'backbone: conjugated units',
    'Solvent_MolLogP': 'solvent: hydrophobicity (logP)',
    'Solvent_NumHDonors': 'solvent: H-bond donors',
    'Solvent_NumHAcceptors': 'solvent: H-bond acceptors',
    'Monomer_fastsolv': 'repeat-unit solubility in the solvent (FASTSOLV)',
    'Backbone_fastsolv': 'backbone solubility in the solvent (FASTSOLV)',
}

# The unit the values are stored in. The search uses only the dimension (INPUTS), so e.g. Mn in g/mol and kg/mol
# are the same quantity to it.
VALUE_UNITS = {'Temperature': 'K', 'Concentration_mg_mL': 'mg/mL', 'MW_Mn': 'g/mol', 'Solvent_MolLogP': '-',
               'Solvent_NumHDonors': '-', 'Solvent_NumHAcceptors': '-', 'Monomer_fastsolv': 'log10(mol/L)',
               'Backbone_fastsolv': 'log10(mol/L)'}

# Short names for printing equations.
SHORT = {'Concentration_mg_mL': 'c', 'Temperature': 'T', 'MW_Mn': 'Mn', 'SC_C_atoms': 'SC_C', 'SC_O_atoms': 'SC_O',
         'SC_n_units': 'SC_units', 'SC_n_branch_points': 'SC_branch', 'BB_C_atoms': 'BB_C',
         'BB_hetero_atoms': 'BB_hetero', 'BB_conjugated_units': 'BB_conj', 'Solvent_MolLogP': 'logP',
         'Solvent_NumHDonors': 'HBD', 'Solvent_NumHAcceptors': 'HBA', 'Monomer_fastsolv': 'F_mono',
         'Backbone_fastsolv': 'F_bb'}

# Figure4.ipynb's correlation pruning (77 -> 41 features for the random forest).
DROP_FEATURES = [
    'BB_conjugated_units_Norm_MolWt', 'R_ring_6',
    'Monomer_NumHDonors_Norm_MolWt', 'Monomer_NumHAcceptors_Norm_MolWt',
    'SC_atoms_before_plus_after_CO_Norm_MolWt', 'SideChain_Eccentricity',
    'Backbone_Eccentricity', 'Monomer_Eccentricity', 'R_atom_all',
    'SC_bonds_CC_or_CO_Norm_MolWt', 'BB_C_atoms_Norm_MolWt',
    'Backbone_InertialShapeFactor', 'Solvent_Eccentricity', 'Solvent_TPSA',
    'SideChain_TPSA', 'R_ring_3', 'R_ring_1',
    'SC_atoms_after_branching_CO_Norm_MolWt', 'Monomer_InertialShapeFactor',
    'SC_total_atoms_CO_Norm_MolWt', 'dG_solv_cosmo',
    'Monomer_NumSpiroAtoms_Norm_MolWt',
    'Monomer_NumSaturatedRings_Norm_MolWt',
    'Monomer_NumBridgeheadAtoms_Norm_MolWt', 'MW_Mw',
    'SC_C_atoms_Norm_MolWt', 'BB_CeqC_bonds_Norm_MolWt',
    'BB_CC_rotatable_bonds_Norm_MolWt', 'SC_total_branches_Norm_MolWt',
    'SC_atoms_before_branching_CO_Norm_MolWt', 'Backbone_TPSA',
    'BBminusSC_MolLogP', 'SideChain_InertialShapeFactor',
    'BB_rings_total_Norm_MolWt', 'Monomer_NumAromaticRings_Norm_MolWt',
    'SideChain_MolLogP',
]
G_THRESHOLD = 1e-3                      # strong chiral assembly: |g| >= 1e-3


def find_repository_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / 'data_analysis' / 'data' / 'analysis_data.xlsx').exists():
            return candidate
    raise FileNotFoundError('Could not locate data_analysis/data/analysis_data.xlsx.')


def short(text):
    """An equation or term with the short input names."""
    for name in sorted(SHORT, key=len, reverse=True):
        text = text.replace(name, SHORT[name])
    return text.replace('**', '^').replace('*', '·')


@dataclass
class Dataset:
    X: pd.DataFrame         # the 15 inputs (names without the _Norm_MolWt suffix), one row per experiment
    X41: pd.DataFrame       # the 41 random-forest features of Figure4.ipynb
    meta: pd.DataFrame      # labels and experimental metadata
    y: np.ndarray           # 1 = strong chiral assembly (|g| >= 1e-3)
    groups: np.ndarray      # 'polymer || solvent', the cross-validation groups of Figure4.ipynb
    chem: np.ndarray        # chemistry code: one repeat unit, molecular-weight variants together (59 levels)
    rows: dict              # '773' (initial then additional, as Figure4.ipynb), '717' (initial), '56' (additional)

    @property
    def units(self):
        """Units of the columns of X, in their order (for SyMANTIC's dimensional analysis)."""
        return list(INPUTS.values())


def chemistry(features, meta):
    """Rows of one chemistry share every polymer-level descriptor except the molecular-weight ones (Mn, Mw, PDI),
    so e.g. N2200 at 3.6k ... 130k is one chemistry. Polymer-level = constant within a polymer name."""
    X = features.drop(columns='experiment_id')
    cols = [c for c in X.columns if c not in ('Temperature', 'Concentration_mg_mL', 'MW_Mn', 'MW_Mw', 'PDI')
            and (X[c].groupby(meta['polymer']).nunique(dropna=False) <= 1).all()]
    key = X[cols].astype(float).round(8).fillna(-1e300).astype(str).agg('|'.join, axis=1)
    return pd.factorize(key)[0]


def load(root=None):
    path = Path(root or find_repository_root()) / 'data_analysis' / 'data' / 'analysis_data.xlsx'
    features = pd.read_excel(path, sheet_name='Exp_ML_features')
    meta = pd.read_excel(path, sheet_name='Exp_ML_metadata')
    if not features['experiment_id'].equals(meta['experiment_id']):
        raise ValueError('Exp_ML_features and Exp_ML_metadata are not aligned by experiment_id.')
    norm = lambda s: s.astype('string').str.replace(r'\s+', ' ', regex=True).str.lower()
    groups = (norm(meta['polymer']) + ' || ' + norm(meta['solvent'])).to_numpy(str)
    y = (meta['abs_g_factor'] >= G_THRESHOLD).astype(int).to_numpy()
    X = features[list(INPUTS)].astype(float)
    X.columns = [c.removesuffix(N) for c in X.columns]
    X41 = features.drop(columns=['experiment_id', *DROP_FEATURES])
    initial = np.where(meta['dataset_split'].eq('initial'))[0]
    additional = np.where(meta['dataset_split'].eq('additional'))[0]
    return Dataset(X=X, X41=X41, meta=meta, y=y, groups=groups, chem=chemistry(features, meta),
                   rows={'773': np.r_[initial, additional], '717': initial, '56': additional})
