#!/usr/bin/env python3
"""Scaffold-aware statistics and figures for the structure–dynamics campaign."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "structure_dynamics_publication_v1"
FIG = OUT / "figures"
IDENTIFIERS = {"molecule_id", "smiles", "scaffold", "training_index", "cyp_index",
               "cyp_target", "mean_lyapunov", "std_lyapunov", "minimum_lyapunov",
               "maximum_lyapunov", "positive_repeat_fraction", "positive_block_fraction",
               "short_dynamic_score"}


def _bh(pvalues):
    values = np.asarray(pvalues, dtype=float); order = np.argsort(values); adjusted = np.empty(len(values))
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        index = order[rank_index]; rank = rank_index + 1
        running = min(running, values[index] * len(values) / rank); adjusted[index] = running
    return adjusted


def _correlations(frame, features):
    rows = []
    for scope, group in [("all", frame), *frame.groupby("cyp_target")]:
        for feature in features:
            values = group[[feature, "mean_lyapunov"]].dropna()
            if values[feature].nunique() < 2:
                rho, p = np.nan, 1.0
            else:
                rho, p = spearmanr(values[feature], values.mean_lyapunov)
            rows.append({"scope": scope, "feature": feature, "n": len(values),
                         "spearman_rho": rho, "p_value": p})
    result = pd.DataFrame(rows)
    result["q_value_bh"] = result.groupby("scope").p_value.transform(_bh)
    return result.sort_values(["scope", "q_value_bh", "spearman_rho"], ascending=[True, True, False])


def _scaffold_bootstrap(frame, features, repeats=2000):
    rng = np.random.default_rng(260830); scaffolds = frame.scaffold.unique(); rows = []
    scaffold_rows = {scaffold: np.flatnonzero(frame.scaffold.to_numpy() == scaffold)
                     for scaffold in scaffolds}
    y_all = frame.mean_lyapunov.to_numpy(dtype=float)
    for feature in features:
        x_all = frame[feature].to_numpy(dtype=float)
        estimates = []
        for _ in range(repeats):
            sampled = rng.choice(scaffolds, len(scaffolds), replace=True)
            indices = np.concatenate([scaffold_rows[scaffold] for scaffold in sampled])
            x, y = x_all[indices], y_all[indices]
            finite = np.isfinite(x) & np.isfinite(y); x, y = x[finite], y[finite]
            estimates.append(np.nan if np.unique(x).size < 2 else spearmanr(x, y).statistic)
        observed = spearmanr(frame[feature], frame.mean_lyapunov, nan_policy="omit").statistic
        rows.append({"feature": feature, "spearman_rho": observed,
                     "cluster_bootstrap_low": np.nanquantile(estimates, .025),
                     "cluster_bootstrap_high": np.nanquantile(estimates, .975)})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)


def _grouped_models(frame, features):
    x = frame[features]; y = frame.mean_lyapunov.to_numpy(); groups = frame.scaffold.to_numpy()
    splits = min(5, len(np.unique(groups)))
    models = {
        "elastic_net": make_pipeline(SimpleImputer(), StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=.25, max_iter=20000)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=500, min_samples_leaf=4,
                                                                           max_features=.75, random_state=260830, n_jobs=-1)),
    }
    rows, importance_rows = [], []
    for model_name, model in models.items():
        predictions = np.full(len(frame), np.nan)
        for fold, (train, test) in enumerate(GroupKFold(splits).split(x, y, groups), 1):
            model.fit(x.iloc[train], y[train]); predictions[test] = model.predict(x.iloc[test])
            if model_name == "extra_trees":
                imp = permutation_importance(model, x.iloc[test], y[test], n_repeats=20,
                                             random_state=260830 + fold, scoring="neg_mean_absolute_error")
                for feature, mean, std in zip(features, imp.importances_mean, imp.importances_std):
                    importance_rows.append({"fold": fold, "feature": feature,
                                            "importance": mean, "importance_std": std})
        rows.append({"model": model_name, "scaffold_grouped_folds": splits,
                     "mae": mean_absolute_error(y, predictions), "r2": r2_score(y, predictions),
                     "spearman_rho": spearmanr(y, predictions).statistic})
        pd.DataFrame({"molecule_id": frame.molecule_id, "cyp_target": frame.cyp_target,
                      "model": model_name, "observed_lyapunov": y,
                      "cross_validated_prediction": predictions}).to_csv(
                          OUT / f"{model_name}_cross_validated_predictions.csv", index=False)
    importance = pd.DataFrame(importance_rows).groupby("feature").agg(
        mean_importance=("importance", "mean"), std_importance=("importance", "std")
    ).reset_index().sort_values("mean_importance", ascending=False)
    return pd.DataFrame(rows), importance


def main():
    warnings.filterwarnings("ignore", message="An input array is constant")
    FIG.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(OUT / "structure_dynamics_population.csv")
    interventions = pd.read_csv(OUT / "causal_interventions.csv")
    features = [column for column in frame.select_dtypes(include="number").columns
                if column not in IDENTIFIERS and frame[column].nunique(dropna=True) > 1]
    correlations = _correlations(frame, features); correlations.to_csv(OUT / "descriptor_correlations.csv", index=False)
    bootstrap = _scaffold_bootstrap(frame, features); bootstrap.to_csv(OUT / "scaffold_bootstrap_correlations.csv", index=False)
    models, importance = _grouped_models(frame, features)
    models.to_csv(OUT / "scaffold_grouped_model_performance.csv", index=False)
    importance.to_csv(OUT / "descriptor_permutation_importance.csv", index=False)
    baselines = interventions[interventions.intervention == "baseline"].set_index(["molecule_id", "cyp_target"]).mean_lyapunov
    interventions["baseline_lyapunov"] = [baselines.loc[(m, c)] for m, c in zip(interventions.molecule_id, interventions.cyp_target)]
    interventions["lyapunov_change"] = interventions.mean_lyapunov - interventions.baseline_lyapunov
    interventions.to_csv(OUT / "causal_interventions_with_effects.csv", index=False)

    plt.style.use("dark_background"); cyan, magenta, lime = "#00e5ff", "#ff1493", "#a6ff00"
    top = bootstrap.head(15).sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(9, 7)); xerr = np.vstack((top.spearman_rho-top.cluster_bootstrap_low,
                                                             top.cluster_bootstrap_high-top.spearman_rho))
    ax.errorbar(top.spearman_rho, range(len(top)), xerr=xerr, fmt="o", color=cyan, ecolor="#777777")
    ax.set_yticks(range(len(top)), top.feature.str.replace("_", " ")); ax.axvline(0, color="white", alpha=.4)
    ax.set(xlabel="Spearman correlation with largest Lyapunov exponent",
           title="Structural associations with Graph-CA instability\n95% scaffold-cluster bootstrap intervals")
    fig.tight_layout(); fig.savefig(FIG / "01_structure_correlations.png", dpi=220); plt.close(fig)

    top_imp = importance.head(15).sort_values("mean_importance")
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top_imp.feature.str.replace("_", " "), top_imp.mean_importance,
                                                   xerr=top_imp.std_importance, color=magenta, alpha=.85)
    ax.set(xlabel="Held-out MAE increase after permutation", title="Scaffold-held-out descriptor importance")
    fig.tight_layout(); fig.savefig(FIG / "02_scaffold_permutation_importance.png", dpi=220); plt.close(fig)

    effects = interventions[interventions.intervention != "baseline"].copy()
    strongest = effects.reindex(effects.lyapunov_change.abs().sort_values(ascending=False).index).head(30)
    labels = strongest.molecule_id.str.replace("OCNT-", "") + " | " + strongest.intervention + " | " + strongest.target
    fig, ax = plt.subplots(figsize=(11, 10)); colours = np.where(strongest.lyapunov_change > 0, magenta, cyan)
    ax.barh(range(len(strongest)), strongest.lyapunov_change, color=colours); ax.set_yticks(range(len(strongest)), labels)
    ax.axvline(0, color="white", alpha=.5); ax.invert_yaxis()
    ax.set(xlabel="Change in largest Lyapunov exponent", title="Strongest causal structural interventions")
    fig.tight_layout(); fig.savefig(FIG / "03_causal_intervention_effects.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, ((molecule, cyp), group) in zip(axes, effects.groupby(["molecule_id", "cyp_target"])):
        order = group.groupby("intervention").lyapunov_change.apply(lambda x: np.mean(np.abs(x))).sort_values().index
        data = [group[group.intervention == kind].lyapunov_change for kind in order]
        ax.boxplot(data, tick_labels=[x.replace("_", " ") for x in order], vert=False,
                   patch_artist=True, boxprops={"facecolor": lime, "alpha": .65})
        ax.axvline(0, color="white", alpha=.5); ax.set_title(f"{molecule} / {cyp}")
    axes[0].set_xlabel("Change in largest Lyapunov exponent"); axes[1].set_xlabel("Change in largest Lyapunov exponent")
    fig.suptitle("Intervention families for the two strange-attractor cases")
    fig.tight_layout(); fig.savefig(FIG / "04_intervention_families.png", dpi=220); plt.close(fig)

    strongest_assoc = bootstrap.iloc[0]
    strongest_intervention = strongest.iloc[0]
    report = f"""# Structure–dynamics publication campaign

## Design

The frozen Kuramoto–Sakaguchi Graph-CA was evaluated on {len(frame)} held-out molecule–CYP cases, balanced across the four CYP endpoints and stratified across the earlier dynamical screen. Each largest Lyapunov exponent used a {json.loads((OUT / 'campaign_metadata.json').read_text())['burn_in']}-generation burn-in, repeated perturbations, circular phase distance, and repeated Benettin renormalisation. The chemical analysis contains 2D descriptors, graph topology, bond composition, and reproducible ETKDG 3D conformer descriptors.

Scaffold-cluster bootstrap intervals preserve dependence among molecules sharing a Bemis–Murcko scaffold. Predictive tests use scaffold-grouped cross-validation. The intervention experiment freezes every learned parameter and changes individual message-passing bonds, bond identities, ring status, or chemically meaningful atom-feature groups in trajectories 7 and 8.

## Principal results

- Strongest univariate structural association: **{strongest_assoc.feature.replace('_', ' ')}**, Spearman rho {strongest_assoc.spearman_rho:.3f}, scaffold-bootstrap 95% interval [{strongest_assoc.cluster_bootstrap_low:.3f}, {strongest_assoc.cluster_bootstrap_high:.3f}].
- Extra Trees scaffold-held-out performance: R² {models.set_index('model').loc['extra_trees', 'r2']:.3f}, Spearman rho {models.set_index('model').loc['extra_trees', 'spearman_rho']:.3f}.
- Largest intervention effect: **{strongest_intervention.molecule_id}**, {strongest_intervention.intervention.replace('_', ' ')} at {strongest_intervention.target}, changing the largest exponent by {strongest_intervention.lyapunov_change:+.5f} per generation.

## Interpretation boundary

Descriptor associations identify structural correlates of the learned dynamics. The frozen-model interventions provide direct computational evidence that particular graph connections and encoded chemical features control the measured instability. The 3D descriptors are correlates of molecular constitution because the Graph-CA receives atom and bond features rather than Cartesian coordinates.

## Reproducible outputs

- `structure_dynamics_population.csv`: cohort descriptors and repeated Lyapunov estimates.
- `descriptor_correlations.csv` and `scaffold_bootstrap_correlations.csv`: univariate statistics.
- `scaffold_grouped_model_performance.csv`: scaffold-held-out multivariate tests.
- `causal_interventions_with_effects.csv`: every frozen-model intervention and effect size.
- `figures/`: publication-resolution figures.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__": main()
