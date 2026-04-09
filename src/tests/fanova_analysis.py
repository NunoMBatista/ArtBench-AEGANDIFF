"""
fANOVA hyperparameter importance analysis from one or more W&B sweeps.

Usage
-----
Single sweep (label defaults to sweep id):
    python src/fanova_analysis.py https://wandb.ai/entity/project/sweeps/abc123

Text file with multiple sweeps (one per line: <url_or_id>  <label>):
    python src/fanova_analysis.py sweeps.txt

    File format — first token is the sweep ref, rest of the line is the label:
        https://wandb.ai/.../sweeps/abc123   CGAN
        https://wandb.ai/.../sweeps/def456   Diffusion
        # lines starting with # are ignored

Options
-------
    --metric     W&B metric to analyse (default: fid)
    --goal       minimize | maximize (default: minimize)
    --entity     W&B entity  [overrides URL]
    --project    W&B project [overrides URL]
    --method     fanova | permutation | mdi  (default: fanova)
    --top        show only top-N parameters (default: all)
    --min-runs   minimum finished runs required (default: 5)

Output
------
    outputs/fanova_<timestamp>/<label>.png  for each sweep

Dependencies
------------
    pip install wandb optuna scikit-learn matplotlib pandas numpy
"""

import argparse
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="fANOVA hyperparameter importance from W&B sweep(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="Sweep URL/ID, or path to a text file with one '<url> <label>' per line",
    )
    p.add_argument("--metric", default="fid", help="Metric to analyse (default: fid)")
    p.add_argument("--goal", default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--method", default="fanova", choices=["fanova", "permutation", "mdi"])
    p.add_argument("--top", type=int, default=None, help="Show top-N params (default: all)")
    p.add_argument("--min-runs", type=int, default=5)
    return p


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def load_sweep_list(input_arg: str) -> list[tuple[str, str]]:
    """
    Return list of (sweep_ref, label).
    If input_arg is an existing file, parse it line by line.
    Otherwise treat it as a single sweep ref with the sweep id as label.
    """
    p = Path(input_arg)
    if p.exists() and p.is_file():
        entries = []
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split(None, 1)
            ref = tokens[0]
            label = tokens[1].strip().strip('"\'') if len(tokens) > 1 else _sweep_id_from_ref(ref)
            entries.append((ref, label))
        if not entries:
            sys.exit(f"ERROR: No valid entries found in {p}")
        return entries

    # Single sweep ref passed directly
    label = _sweep_id_from_ref(input_arg)
    return [(input_arg, label)]


def _sweep_id_from_ref(ref: str) -> str:
    """Best-effort label: last path component of URL or bare id."""
    m = re.search(r"sweeps/([^/?#]+)", ref)
    if m:
        return m.group(1)
    return ref.rstrip("/").split("/")[-1]


def parse_sweep_ref(
    sweep_ref: str,
    default_entity: Optional[str] = None,
    default_project: Optional[str] = None,
) -> tuple[str, str, str]:
    """Return (entity, project, sweep_id). Exits on failure."""
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?#]+)", sweep_ref)
    if m:
        entity, project, sweep_id = m.group(1), m.group(2), m.group(3)
    else:
        parts = sweep_ref.strip("/").split("/")
        if len(parts) == 3:
            entity, project, sweep_id = parts
        elif len(parts) == 2:
            entity, project, sweep_id = default_entity, parts[0], parts[1]
        elif len(parts) == 1:
            entity, project, sweep_id = default_entity, default_project, parts[0]
        else:
            sys.exit(f"ERROR: Cannot parse sweep reference: {sweep_ref!r}")

    entity  = entity  or default_entity
    project = project or default_project

    if not entity or not project:
        sys.exit(
            "ERROR: Could not determine entity/project.\n"
            "  Use a full W&B URL, or pass --entity / --project."
        )
    return entity, project, sweep_id


# ---------------------------------------------------------------------------
# W&B data fetch
# ---------------------------------------------------------------------------

def fetch_runs(entity: str, project: str, sweep_id: str, metric: str) -> pd.DataFrame:
    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}"
    print(f"  Fetching {sweep_path} ...")

    try:
        sweep = api.sweep(sweep_path)
    except Exception as exc:
        sys.exit(f"ERROR: Could not fetch sweep — {exc}")

    rows, skipped = [], 0
    for run in sweep.runs:
        if run.state != "finished":
            skipped += 1
            continue
        value = run.summary.get(metric)
        if value is None:
            skipped += 1
            continue
        row = dict(run.config)
        row["__metric__"] = float(value)
        rows.append(row)

    print(f"    {len(rows)} finished runs with '{metric}'  ({skipped} skipped)")

    if not rows:
        sys.exit(f"ERROR: No finished runs found with metric '{metric}'")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, metric_col: str = "__metric__"):
    y = df[metric_col].copy()
    X = df.drop(columns=[metric_col]).copy()

    # Drop columns whose values are unhashable (nested dicts/lists from W&B)
    unhashable = [
        c for c in X.columns
        if X[c].apply(lambda v: isinstance(v, (dict, list))).any()
    ]
    if unhashable:
        print(f"    Dropping nested columns: {unhashable}")
        X = X.drop(columns=unhashable)

    # Drop constant columns
    constant = X.columns[X.nunique() <= 1].tolist()
    if constant:
        print(f"    Dropping constant columns: {constant}")
        X = X.drop(columns=constant)

    # Encode bools and strings
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        elif X[col].dtype == object or X[col].dtype.name == "category":
            X[col] = pd.Categorical(X[col]).codes

    # Drop non-numeric leftovers
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        print(f"    Dropping non-numeric columns: {non_num}")
        X = X.drop(columns=non_num)

    X = X.dropna(axis=1)
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


# ---------------------------------------------------------------------------
# Importance methods
# ---------------------------------------------------------------------------

def importance_fanova(X: pd.DataFrame, y: pd.Series, goal: str) -> pd.Series:
    try:
        import optuna
        from optuna.importance import FanovaImportanceEvaluator, get_param_importances
        from optuna.distributions import CategoricalDistribution
        from optuna.trial import FrozenTrial, TrialState
        from datetime import datetime as dt
    except ImportError:
        sys.exit("ERROR: pip install optuna  (required for fANOVA)")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize" if goal == "minimize" else "maximize")

    def _to_native(v):
        return v.item() if hasattr(v, "item") else v

    distributions = {
        col: CategoricalDistribution(tuple(_to_native(v) for v in X[col].unique()))
        for col in X.columns
    }
    now = dt.now()
    trials = []
    for i, (idx, row) in enumerate(X.iterrows()):
        params = {k: v.item() if hasattr(v, "item") else v for k, v in row.items()}
        trials.append(FrozenTrial(
            number=i, trial_id=i,
            state=TrialState.COMPLETE,
            value=float(y.loc[idx]),
            values=None,
            params=params,
            distributions=distributions,
            user_attrs={}, system_attrs={},
            intermediate_values={},
            datetime_start=now, datetime_complete=now,
        ))
    study.add_trials(trials)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=42))
    return pd.Series(imp, name="importance")


def importance_permutation(X: pd.DataFrame, y: pd.Series, goal: str) -> pd.Series:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance as pi
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    r = pi(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns, name="importance").clip(lower=0)


def importance_mdi(X: pd.DataFrame, y: pd.Series, goal: str) -> pd.Series:
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns, name="importance")


METHODS = {"fanova": importance_fanova, "permutation": importance_permutation, "mdi": importance_mdi}

METHOD_LABELS = {
    "fanova": "fANOVA",
    "permutation": "Permutation importance",
    "mdi": "Mean Decrease Impurity",
}


# ---------------------------------------------------------------------------
# Plotting — bar chart only, compact
# ---------------------------------------------------------------------------

def plot_importances(
    importances: pd.Series,
    label: str,
    method: str,
    out_path: Path,
    top: Optional[int] = None,
):
    importances = importances.sort_values(ascending=False)
    if top is not None:
        importances = importances.head(top)

    n = len(importances)
    fig, ax = plt.subplots(figsize=(max(3, n * 0.9 + 1), 3.5))

    colors = plt.cm.viridis_r(np.linspace(0.15, 0.85, n))
    bars = ax.bar(importances.index, importances.values, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, importances.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + importances.values.max() * 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(f"Importance ({METHOD_LABELS[method]})", fontsize=8)
    ax.set_ylim(0, importances.values.max() * 1.18)
    ax.tick_params(axis="x", rotation=15, labelsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    sweeps = load_sweep_list(args.input)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"fanova_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    for sweep_ref, label in sweeps:
        print(f"[{label}]")
        entity, project, sweep_id = parse_sweep_ref(sweep_ref, args.entity, args.project)

        df_raw = fetch_runs(entity, project, sweep_id, args.metric)
        if len(df_raw) < args.min_runs:
            print(f"    SKIP: only {len(df_raw)} runs (need {args.min_runs})\n")
            continue

        X, y = prepare_data(df_raw)
        print(f"    Hyperparameters: {list(X.columns)}")
        print(f"    Runs: {len(X)}")

        if X.shape[1] == 0:
            print("    SKIP: no varying hyperparameters after filtering\n")
            continue

        print(f"    Computing importance via '{args.method}' ...")
        importances = METHODS[args.method](X, y, args.goal)

        total = importances.sum()
        if total > 0:
            importances = importances / total

        print("    Importances (normalised):")
        for param, val in importances.sort_values(ascending=False).items():
            print(f"      {param:25s}  {val:.4f}  {'█' * int(val * 30)}")

        # Sanitise label for filename
        safe_label = re.sub(r"[^\w\-]", "_", label).strip("_") or sweep_id
        out_path = out_dir / f"{safe_label}.png"

        plot_importances(
            importances=importances,
            label=label,
            method=args.method,
            out_path=out_path,
            top=args.top,
        )
        print()


if __name__ == "__main__":
    main()
