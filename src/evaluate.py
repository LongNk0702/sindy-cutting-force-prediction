from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

from .config import DECIMAL_PRECISION, FIGURE_DIR, FiGURE_STYLE

#Basic metrics
def regression_metrics(
        y_true: np.ndarray,y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute overall (variance-weighted) multi-output regression metrics.

    Args:
        y_true: shape (n_samples, n_outputs)
        y_pred: shape (n_samples, n_outputs)

    Returns:
        dict with keys: R2, RMSE, MAE
    """
    _check_shapes(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"R2": float(r2), "RMSE": rmse, "MAE": mae}

def metrics_per_axis(
    y_true: np.ndarray, y_pred: np.ndarray, axis_names: Iterable[str] = ("Fx", "Fy", "Fz")
) -> pd.DataFrame:
     """
    Compute metrics per axis (column) and return as a DataFrame.

    Args:
        y_true: (n_samples, 3) for Fx, Fy, Fz (or any #outputs)
        y_pred: (n_samples, 3)
        axis_names: iterable of axis labels

    Returns:
        DataFrame with rows = axis_names and columns = R2, RMSE, MAE
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = []
    for i, name in enumerate(axis_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        results.append({"axis": name, "R2": float(r2), "RMSE": rmse, "MAE": mae})

    df = pd.DataFrame(results).set_index("axis")
    return df.round(DECIMAL_PRECISION)

#cross-validation evaluation
def cross_validate_sindy(
        build_model_fn: Callable[[], object],
        X: np.ndarray,
        Y: np.ndarray,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int=42,
        axis_names: Iterable[str] = ("Fx", "Fy", "Fz"),
)-> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Perform K-fold CV for SINDy models.

    You must pass `build_model_fn` that returns a *fresh* SINDy model per fold.
    The model must implement .fit(X, t=None, x_dot=Y) and .predict(X).

    Args:
        build_model_fn: callable -> new SINDy model (unfitted)
        X: (n_samples, n_features)
        Y: (n_samples, n_outputs)
        n_splits: K in KFold
        shuffle: shuffle before split
        random_state: rng seed for KFold
        axis_names: labels for outputs

    Returns:
        fold_table: DataFrame of per-fold metrics (R2, RMSE, MAE)
        mean_metrics: dict of mean metrics across folds
        per_axis_mean: DataFrame of mean per-axis metrics across folds
    """
    _check_shapes_xy(X, Y)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_rows: List[Dict[str, float]] = []
    per_axis_list: List[pd.DataFrame] = []

    for fold_idx, (tr, te) in enumerate(kf.split(X), start=1):
        model = build_model_fn()
        model.fit(X[tr], t=None, x_dot=Y[tr])

        Yp = model.predict(X[te])

        m_overall = regression_metrics(Y[te], Yp)
        m_axis = metrics_per_axis(Y[te], Yp, axis_names=axis_names)

        fold_row = {"fold": fold_idx, **m_overall}
        fold_rows.append(fold_row)
        per_axis_list.append(m_axis.assign(fold=fold_idx))

    fold_table = pd.DataFrame(fold_rows).set_index("fold")
    mean_metrics = fold_table.mean(axis=0).to_dict()

    # mean per-axis metrics across folds
    per_axis_df = pd.concat(per_axis_list)  # index: axis, columns: metrics + fold
    per_axis_mean = (
        per_axis_df.reset_index()
        .groupby("axis")[["R2", "RMSE", "MAE"]]
        .mean()
        .round(DECIMAL_PRECISION)
    )

    return fold_table.round(DECIMAL_PRECISION), _round_dict(mean_metrics), per_axis_mean

#plots
def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_index: int,
    axis_name: str,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create a parity plot (y_true vs y_pred) for a single axis.

    Args:
        y_true: (n_samples, n_outputs)
        y_pred: (n_samples, n_outputs)
        axis_index: column index to plot (e.g., 0 for Fx)
        axis_name: label to show on the plot
        savepath: where to save (defaults to FIGURE_DIR / f"parity_{axis_name}.png")
        show: whether to plt.show()

    Returns:
        Path to the saved figure.
    """
    _check_shapes(y_true, y_pred)

    # Style
    plt.figure(figsize=FIGURE_STYLE["figsize"], dpi=FIGURE_STYLE["dpi"])
    plt.scatter(y_true[:, axis_index], y_pred[:, axis_index], s=22, alpha=0.9)

    # reference diagonal
    min_v = min(np.min(y_true[:, axis_index]), np.min(y_pred[:, axis_index]))
    max_v = max(np.max(y_true[:, axis_index]), np.max(y_pred[:, axis_index]))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.2)

    plt.xlabel(f"Measured {axis_name} (N)")
    plt.ylabel(f"Predicted {axis_name} (N)")
    plt.title(f"Parity Plot – {axis_name}")
    plt.grid(alpha=FIGURE_STYLE.get("grid_alpha", 0.3))

    # path
    out = savepath or (FIGURE_DIR / f"parity_{axis_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    if show:
        plt.show()
    plt.close()
    return out


def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_index: int,
    axis_name: str,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Plot residuals (y_true - y_pred) vs y_true for a single axis.

    Args:
        y_true: (n_samples, n_outputs)
        y_pred: (n_samples, n_outputs)
        axis_index: which column to plot
        axis_name: label
        savepath: custom path to save
        show: display the plot

    Returns:
        Path to saved figure.
    """
    _check_shapes(y_true, y_pred)

    res = y_true[:, axis_index] - y_pred[:, axis_index]

    plt.figure(figsize=FIGURE_STYLE["figsize"], dpi=FIGURE_STYLE["dpi"])
    plt.scatter(y_true[:, axis_index], res, s=22, alpha=0.9)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel(f"Measured {axis_name} (N)")
    plt.ylabel("Residual (N)")
    plt.title(f"Residual Plot – {axis_name}")
    plt.grid(alpha=FIGURE_STYLE.get("grid_alpha", 0.3))

    out = savepath or (FIGURE_DIR / f"residual_{axis_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    if show:
        plt.show()
    plt.close()
    return out

#reporting
def export_metrics_csv(
    overall_metrics: Dict[str, float],
    per_axis_metrics: pd.DataFrame,
    fold_table: Optional[pd.DataFrame] = None,
    out_dir: Path = FIGURE_DIR,
    filename_prefix: str = "sindy_eval",
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Save metrics to CSV files:
    - overall_metrics → <prefix>_overall.csv
    - per_axis_metrics → <prefix>_per_axis.csv
    - fold_table → <prefix>_folds.csv (optional)

    Returns:
        paths (overall_path, per_axis_path, folds_path_or_None)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([overall_metrics]).round(DECIMAL_PRECISION)
    per_axis_df = per_axis_metrics.round(DECIMAL_PRECISION)

    p_overall = out_dir / f"{filename_prefix}_overall.csv"
    p_axis = out_dir / f"{filename_prefix}_per_axis.csv"
    overall_df.to_csv(p_overall, index=False)
    per_axis_df.to_csv(p_axis)

    p_folds = None
    if fold_table is not None:
        p_folds = out_dir / f"{filename_prefix}_folds.csv"
        fold_table.round(DECIMAL_PRECISION).to_csv(p_folds)

    return p_overall, p_axis, p_folds

#internal helpers
def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays (n_samples, n_outputs).")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )


def _check_shapes_xy(X: np.ndarray, Y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features).")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n_samples, n_outputs).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample size mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]} rows."
        )


def _round_dict(d: Dict[str, float], ndigits: int = DECIMAL_PRECISION) -> Dict[str, float]:
    return {k: (round(v, ndigits) if isinstance(v, (int, float)) else v) for k, v in d.items()}

#self-test
if __name__ == "__main__":
    # Minimal sanity check with random data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(27, 4))
    Y = rng.normal(size=(27, 3))

    def dummy_builder():
        # Minimal stand-in SINDy-like model with sklearn API
        from sklearn.linear_model import LinearRegression

        class _Wrapper:
            def __init__(self):
                self.m = LinearRegression()

            def fit(self, X, t=None, x_dot=None):
                self.m.fit(X, x_dot)

            def predict(self, X):
                return self.m.predict(X)

        return _Wrapper()

    folds, mean_m, axis_m = cross_validate_sindy(dummy_builder, X, Y, n_splits=5)
    print("Folds:\n", folds)
    print("Mean metrics:", mean_m)
    print("Per-axis mean:\n", axis_m)