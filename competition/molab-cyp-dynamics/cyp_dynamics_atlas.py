# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.5",
#     "marimo>=0.15",
#     "pandas>=2.2",
#     "rdkit>=2025.3",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw

    return Chem, Draw, alt, mo, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # CYP Dynamics Atlas

        **An interactive investigation of learned molecular graph dynamics across CYP1A2, CYP2C9, CYP2D6, and CYP3A4 inhibition.**

        This notebook asks whether learned graph-state dynamics reveal CYP-selective chemistry beyond conventional molecular structure. The trajectories shown here are computational states produced by graph cellular automata. They are distinct from physical molecular-dynamics trajectories.

        _Built for the 2026 marimo and OpenADMET cheminformatics notebook competition._
        """
    )
    return


@app.cell
def _(pd):
    raw_base = "https://raw.githubusercontent.com/acnash/Strange-Matter-Engine/main/competition/molab-cyp-dynamics/data"
    cohort = pd.read_csv(f"{raw_base}/cyp_dynamics_cohort.csv")
    regime_summary = pd.read_csv(f"{raw_base}/dynamical_regime_summary.csv")
    descriptor_correlations = pd.read_csv(f"{raw_base}/descriptor_correlations.csv")
    interventions = pd.read_csv(f"{raw_base}/causal_interventions.csv")
    attractors = pd.read_csv(f"{raw_base}/attractor_screen.csv")
    return attractors, cohort, descriptor_correlations, interventions, regime_summary


@app.cell
def _(mo):
    isoform = mo.ui.dropdown(
        options=["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"],
        value="CYP1A2",
        label="Choose a CYP isoform",
    )
    colour_metric = mo.ui.dropdown(
        options={
            "Dynamical instability": "mean_lyapunov",
            "Molecular weight": "molecular_weight",
            "Lipophilicity (logP)": "logp",
            "Aromatic atom fraction": "aromatic_atom_fraction",
            "Graph connectivity": "algebraic_connectivity",
        },
        value="mean_lyapunov",
        label="Colour the atlas by",
    )
    mo.hstack([isoform, colour_metric], justify="start", gap=2)
    return colour_metric, isoform


@app.cell
def _(cohort, isoform, mo):
    isoform_cohort = cohort[cohort["cyp_target"] == isoform.value].copy()
    molecule_options = {
        f"{row.molecule_id} · pIC50 {row.measured_pic50:.2f}": row.molecule_id
        for row in isoform_cohort.itertuples()
    }
    molecule = mo.ui.dropdown(
        options=molecule_options,
        value=next(iter(molecule_options.values())),
        label="Inspect a molecule",
        searchable=True,
    )
    molecule
    return isoform_cohort, molecule


@app.cell
def _(cohort, isoform, mo):
    total_molecules = cohort["molecule_id"].nunique()
    isoform_cases = int((cohort["cyp_target"] == isoform.value).sum())
    measured_cases = int(
        cohort.loc[cohort["cyp_target"] == isoform.value, "measured_pic50"].notna().sum()
    )
    mo.hstack(
        [
            mo.stat(value=f"{total_molecules:,}", label="Unique molecules"),
            mo.stat(value=f"{isoform_cases:,}", label=f"{isoform.value} trajectories"),
            mo.stat(value=f"{measured_cases:,}", label="Matched measurements"),
            mo.stat(value="1,000", label="Lyapunov burn-in generations"),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(alt, colour_metric, isoform, isoform_cohort, mo):
    atlas_chart = (
        alt.Chart(isoform_cohort)
        .mark_circle(size=105, opacity=0.78, stroke="#ffffff", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "measured_pic50:Q",
                title=f"Measured {isoform.value} direct-inhibition pIC50",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "mean_lyapunov:Q",
                title="Largest Lyapunov exponent per generation",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                f"{colour_metric.value}:Q",
                title=colour_metric.selected_key,
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("molecule_id:N", title="Molecule"),
                alt.Tooltip("measured_pic50:Q", title="pIC50", format=".2f"),
                alt.Tooltip("mean_lyapunov:Q", title="Lyapunov", format=".4f"),
                alt.Tooltip("molecular_weight:Q", title="MW", format=".1f"),
                alt.Tooltip("logp:Q", title="logP", format=".2f"),
            ],
        )
        .properties(height=420, title=f"{isoform.value}: assay potency versus learned instability")
        .interactive()
    )
    mo.ui.altair_chart(atlas_chart)
    return


@app.cell
def _(Chem, Draw, cohort, molecule, mo):
    selected_record = cohort.loc[cohort["molecule_id"] == molecule.value].iloc[0]
    selected_mol = Chem.MolFromSmiles(selected_record["smiles"])
    selected_svg = Draw.MolsToGridImage(
        [selected_mol],
        legends=[selected_record["molecule_id"]],
        molsPerRow=1,
        subImgSize=(460, 330),
        useSVG=True,
    )
    selected_facts = mo.md(
        f"""
        ### {selected_record['molecule_id']} · {selected_record['cyp_target']}

        | Measurement | Value |
        |:--|--:|
        | Direct-inhibition pIC50 | **{selected_record['measured_pic50']:.2f}** |
        | Largest Lyapunov exponent | **{selected_record['mean_lyapunov']:.5f}** |
        | Molecular weight | {selected_record['molecular_weight']:.1f} |
        | logP | {selected_record['logp']:.2f} |
        | Topological polar surface area | {selected_record['tpsa']:.1f} Å² |
        | Algebraic connectivity | {selected_record['algebraic_connectivity']:.4f} |

        **SMILES:** `{selected_record['smiles']}`
        """
    )
    mo.hstack([mo.Html(selected_svg), selected_facts], widths=[1, 1.25], gap=2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Does the CYP label leave a signature in the dynamics?

        Each transition rule was evaluated across molecule–CYP cases. The adjusted analysis controls for the molecule-level dynamical summary, asking whether the target-conditioned state retains separable CYP geometry. The chart below compares adjusted silhouette scores. Higher values indicate stronger separation among the four isoforms.
        """
    )
    return


@app.cell
def _(alt, mo, regime_summary):
    regime_order = regime_summary.sort_values("adjusted_silhouette", ascending=False)
    regime_chart = (
        alt.Chart(regime_order)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("adjusted_silhouette:Q", title="Adjusted CYP silhouette score"),
            y=alt.Y("rule:N", sort="-x", title=None),
            color=alt.Color(
                "adjusted_silhouette:Q",
                scale=alt.Scale(scheme="teals"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("rule:N", title="Transition rule"),
                alt.Tooltip("adjusted_silhouette:Q", format=".3f"),
                alt.Tooltip("adjusted_permutation_p:Q", title="Permutation p", format=".3g"),
                alt.Tooltip("adjusted_n_molecules:Q", title="Molecules"),
            ],
        )
        .properties(height=330)
    )
    mo.ui.altair_chart(regime_chart)
    return


@app.cell
def _(descriptor_correlations, mo):
    overall_correlations = descriptor_correlations[
        descriptor_correlations["scope"] == "overall"
    ].copy()
    strongest_feature = overall_correlations.iloc[
        overall_correlations["spearman_rho"].abs().argmax()
    ]
    mo.callout(
        mo.md(
            f"""
            ### Structural clue

            The strongest univariate structural association with learned instability is **{strongest_feature['feature'].replace('_', ' ')}**, with Spearman ρ = **{strongest_feature['spearman_rho']:.3f}** across the held-out cohort.

            This is an association within the learned system. Scaffold-grouped validation and frozen-model interventions provide the stronger tests shown in the next build.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(attractors, mo):
    attractor_view = attractors[
        [
            "molecule_id",
            "cyp_target",
            "transition_rule",
            "classification",
            "correlation_dimension",
            "spectral_entropy",
            "dominant_period",
        ]
    ].rename(
        columns={
            "molecule_id": "Molecule",
            "cyp_target": "CYP",
            "transition_rule": "Rule",
            "classification": "Long-horizon classification",
            "correlation_dimension": "Correlation dimension",
            "spectral_entropy": "Spectral entropy",
            "dominant_period": "Dominant period",
        }
    )
    mo.md("## Long-horizon candidates")
    mo.ui.table(attractor_view, selection=None, pagination=False)
    return


@app.cell
def _(interventions, mo):
    strongest_intervention = interventions.iloc[
        interventions["lyapunov_change"].abs().argmax()
    ]
    mo.callout(
        mo.md(
            f"""
            ## A controlled computational intervention

            The largest frozen-model intervention in the current campaign occurs for **{strongest_intervention['molecule_id']}** at **{strongest_intervention['cyp_target']}**. Changing **{strongest_intervention['intervention']}** at `{strongest_intervention['target']}` shifts the largest Lyapunov exponent by **{strongest_intervention['lyapunov_change']:+.5f} per generation**.

            Every learned parameter stays fixed during these interventions. The edit isolates how an encoded bond or atom-feature contribution controls the resulting graph-state instability.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            ## Current answer

            The learned graph dynamics contain reproducible structure associated with CYP conditioning, molecular connectivity, and chemically meaningful computational interventions. The present evidence supports a relationship between molecular constitution and the stability of the learned trajectories. The next stage will test whether this dynamical information improves within-scaffold explanation of assay variation and will expose atom-level trajectories through linked interactive views.

            ### Interpretation boundary

            These trajectories describe learned information propagation over molecular graphs. They do not simulate atomic motion, conformational sampling, binding kinetics, or enzyme structure.
            """
        ),
        kind="neutral",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Data provenance and AI disclosure

        The assay measurements originate from the released OpenADMET CYP Inhibition Blind Challenge training data. The dynamical tables were generated by the Strange Matter Engine analysis pipeline from held-out molecule–CYP cases. Source files, preparation code, statistical outputs, and this notebook are version-controlled in the public GitHub repository.

        OpenAI Codex assisted with notebook engineering, interface implementation, testing, and editorial refinement under human scientific direction. Scientific claims are tied to reproducible outputs and explicitly stated interpretation boundaries.
        """
    )
    return


if __name__ == "__main__":
    app.run()
