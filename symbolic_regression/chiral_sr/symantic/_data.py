"""Normalize pandas input before passing it to numerical backends."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


def prepare_dataframe(df, *, drop_constant=True):
    """Return finite float64 features, target, and retained feature positions.

    The first column is always the target. Positions are relative to the
    remaining input columns, so physical units stay aligned after filtering.
    Numeric extension dtypes are supported without mutating the caller's data.
    Set drop_constant=False to retain physical constants in dimensional models.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df) < 2 or df.shape[1] < 2:
        raise ValueError("Provide at least two rows, a target, and one feature")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique")

    def is_real_numeric(dtype):
        return is_numeric_dtype(dtype) and not is_bool_dtype(dtype) and not is_complex_dtype(dtype)

    if not is_real_numeric(df.dtypes.iloc[0]):
        raise ValueError("The first column (target) must contain real numeric data")
    target = df.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    positions = [i for i, dtype in enumerate(df.dtypes.iloc[1:]) if is_real_numeric(dtype)]
    features = df.iloc[:, [i + 1 for i in positions]].copy()
    values = features.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if not np.isfinite(target).all() or not np.isfinite(values).all():
        raise ValueError("Target and numeric features must contain finite values; fill or remove missing values first")
    if np.ptp(target) == 0:
        raise ValueError("The target must have nonzero variance")
    keep = np.ptp(values, axis=0) != 0 if drop_constant else np.ones(values.shape[1], dtype=bool)
    positions = [position for position, retained in zip(positions, keep) if retained]
    names = [name if isinstance(name, str) else f"x{position + 1}"
             for name, position in zip(features.columns[keep], positions)]
    if len(names) != len(set(names)):
        raise ValueError("Column names must remain unique when converted to strings")
    if not names:
        raise ValueError("Provide at least one numeric feature" + (" with nonzero variance" if drop_constant else ""))
    features = pd.DataFrame(values[:, keep], index=df.index, columns=names)
    return features, target, positions
