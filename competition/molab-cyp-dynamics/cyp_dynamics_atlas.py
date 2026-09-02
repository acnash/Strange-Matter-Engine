# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.5",
#     "marimo>=0.15",
#     "pandas>=2.2",
#     "plotly>=6.0",
#     "rdkit>=2025.3",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="Molecular Signal Observatory")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from rdkit import Chem
    from rdkit.Chem import Draw

    return Chem, Draw, alt, go, mo, pd


@app.cell
def _(mo):
    hero = mo.Html(
        """
        <style>
        :root {
          --void: #050711;
          --panel: rgba(10, 18, 35, 0.88);
          --cyan: #22f5ff;
          --magenta: #ff38c7;
          --violet: #9678ff;
          --ink: #e9fbff;
          --muted: #91a9bd;
        }
        body, .marimo, .marimo-main {
          background:
            radial-gradient(circle at 18% 0%, rgba(34,245,255,.10), transparent 34rem),
            radial-gradient(circle at 88% 18%, rgba(255,56,199,.09), transparent 30rem),
            var(--void) !important;
          color: var(--ink) !important;
        }
        .marimo-main { max-width: 1440px !important; }
        h1, h2, h3 { color: var(--ink) !important; letter-spacing: .02em; }
        h2 { border-bottom: 1px solid rgba(34,245,255,.3); padding-bottom: .45rem; }
        p, li, td, th, label { color: var(--ink); }
        code { color: var(--cyan) !important; background: rgba(34,245,255,.08) !important; }
        a { color: var(--cyan) !important; }
        .hero-shell {
          position: relative; overflow: hidden; padding: 3rem 3.2rem; margin: .5rem 0 2rem;
          border: 1px solid rgba(34,245,255,.42); border-radius: 20px;
          background: linear-gradient(125deg, rgba(8,18,38,.98), rgba(20,7,31,.92));
          box-shadow: 0 0 60px rgba(34,245,255,.10), inset 0 0 50px rgba(150,120,255,.06);
        }
        .hero-shell:after {
          content: ""; position: absolute; inset: 0; pointer-events: none;
          background: repeating-linear-gradient(0deg, transparent 0 4px, rgba(255,255,255,.018) 5px);
        }
        .eyebrow { color: var(--cyan); font: 700 .75rem/1.2 monospace; letter-spacing: .22em; }
        .hero-title { margin: .55rem 0 .4rem; font-size: clamp(2.4rem, 6vw, 5.8rem); line-height: .92;
          background: linear-gradient(90deg, #f6ffff 8%, var(--cyan) 52%, var(--magenta));
          -webkit-background-clip: text; color: transparent; text-transform: uppercase; }
        .hero-copy { max-width: 850px; color: var(--muted); font-size: 1.08rem; }
        .signal-card { border: 1px solid rgba(150,120,255,.28); border-radius: 16px;
          background: var(--panel); padding: 1rem 1.2rem; box-shadow: inset 0 0 30px rgba(34,245,255,.025); }
        </style>
        <section class="hero-shell">
          <div class="eyebrow">OPENADMET × STRANGE MATTER ENGINE // LIVE GRAPH SIGNAL</div>
          <h1 class="hero-title">Molecular Signal Observatory</h1>
          <p class="hero-copy">Follow learned information as it propagates through molecular graphs, descends through generations, and settles into dynamical regimes conditioned by CYP identity.</p>
        </section>
        """
    )
    question = mo.md(
        r"""
        ## The question

        **When the same learned system reads molecular structure in a CYP-specific context, what kinds of information flow appear, and which structures produce persistent, periodic, or strange-attractor-like behaviour?**

        The trajectories are computational graph states. Colour represents learned activity on molecular atoms, and the descending axis represents successive generations. They do not describe atomic motion.
        """
    )
    mo.vstack([hero, question])
    return


@app.cell
def _(pd):
    raw_base = "https://raw.githubusercontent.com/acnash/Strange-Matter-Engine/main/competition/molab-cyp-dynamics/data"
    cohort = pd.read_csv(f"{raw_base}/cyp_dynamics_cohort.csv")
    regime_summary = pd.read_csv(f"{raw_base}/dynamical_regime_summary.csv")
    descriptor_correlations = pd.read_csv(f"{raw_base}/descriptor_correlations.csv")
    descriptor_correlations["scope"] = descriptor_correlations["scope"].replace(
        {"all": "overall"}
    )
    interventions = pd.read_csv(f"{raw_base}/causal_interventions.csv")
    attractors = pd.read_csv(f"{raw_base}/attractor_screen.csv")
    cascades = pd.read_csv(f"{raw_base}/molecular_cascades.csv.gz")
    phase_trajectories = pd.read_csv(f"{raw_base}/phase_trajectories.csv.gz")
    return (
        attractors,
        cascades,
        cohort,
        descriptor_correlations,
        interventions,
        phase_trajectories,
        regime_summary,
    )


@app.cell
def _(attractors, cascades, mo):
    available_cases = attractors[attractors["case_id"].isin(cascades["case_id"].unique())]
    dynamic_options = {
        f"{row.molecule_id} · {row.cyp_target} · {row.transition_rule.replace('_', ' ')}": row.case_id
        for row in available_cases.itertuples()
    }
    dynamic_case = mo.ui.dropdown(
        options=dynamic_options,
        value=next(iter(dynamic_options)),
        label="Select a learned dynamical system",
        searchable=True,
    )
    generation = mo.ui.slider(
        start=0,
        stop=1000,
        step=50,
        value=1000,
        label="Reveal generations",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## Enter the signal"),
            mo.hstack([dynamic_case, generation], widths=[2, 3], gap=2),
        ]
    )
    return available_cases, dynamic_case, generation


@app.cell
def _(available_cases, dynamic_case, generation, mo):
    dynamic_record = available_cases.loc[
        available_cases["case_id"] == dynamic_case.value
    ].iloc[0]
    mo.Html(
        f"""
        <div class="signal-card">
          <div class="eyebrow">ACTIVE TRANSMISSION // GENERATION {generation.value:04d}</div>
          <h3>{dynamic_record['molecule_id']} × {dynamic_record['cyp_target']}</h3>
          <p><b>{dynamic_record['transition_rule'].replace('_', ' ').title()}</b> · {dynamic_record['classification']}</p>
          <p>Late motion {dynamic_record['late_motion']:.3g} · correlation dimension {dynamic_record['correlation_dimension']:.2f} · spectral entropy {dynamic_record['spectral_entropy']:.2f}</p>
        </div>
        """
    )
    return dynamic_record


@app.cell
def _(cascades, dynamic_case, generation, go, mo):
    cascade_view = cascades[
        (cascades["case_id"] == dynamic_case.value)
        & (cascades["generation"] <= generation.value)
    ]
    newest_generation = cascade_view["generation"].max()
    newest = cascade_view[cascade_view["generation"] == newest_generation]
    cascade_figure = go.Figure()
    cascade_figure.add_trace(
        go.Scatter3d(
            x=cascade_view["x"], y=cascade_view["z"], z=cascade_view["y"],
            mode="markers",
            marker=dict(
                size=2.7,
                color=cascade_view["activity"],
                colorscale=[[0, "#22f5ff"], [0.5, "#9678ff"], [1, "#ff38c7"]],
                opacity=0.48,
                colorbar=dict(title="Learned<br>activity", thickness=12),
            ),
            customdata=cascade_view[["generation", "atom", "element"]],
            hovertemplate="generation %{customdata[0]}<br>atom %{customdata[1]} · %{customdata[2]}<br>activity %{marker.color:.2f}<extra></extra>",
            name="signal history",
        )
    )
    cascade_figure.add_trace(
        go.Scatter3d(
            x=newest["x"], y=newest["z"], z=newest["y"],
            mode="markers",
            marker=dict(size=6, color="#fff7a8", line=dict(color="#22f5ff", width=1)),
            hoverinfo="skip",
            name=f"generation {int(newest_generation)}",
        )
    )
    cascade_figure.update_layout(
        height=680,
        paper_bgcolor="#050711",
        plot_bgcolor="#050711",
        font=dict(color="#dffcff", family="monospace"),
        margin=dict(l=0, r=0, t=55, b=0),
        title="MOLECULAR INFORMATION CASCADE",
        legend=dict(bgcolor="rgba(5,7,17,.7)"),
        scene=dict(
            bgcolor="#050711",
            xaxis=dict(title="molecular x", gridcolor="#163044", zerolinecolor="#22f5ff"),
            yaxis=dict(title="molecular z", gridcolor="#163044", zerolinecolor="#22f5ff"),
            zaxis=dict(title="generation descent", gridcolor="#281b45", autorange="reversed"),
            camera=dict(eye=dict(x=1.45, y=1.55, z=0.72)),
            aspectmode="data",
        ),
    )
    mo.ui.plotly(cascade_figure, config={"displaylogo": False})
    return


@app.cell
def _(dynamic_case, dynamic_record, go, mo, phase_trajectories):
    phase_view = phase_trajectories[
        phase_trajectories["case_id"] == dynamic_case.value
    ].copy()
    phase_figure = go.Figure(
        go.Scatter3d(
            x=phase_view["pc1"], y=phase_view["pc2"], z=phase_view["pc3"],
            mode="lines+markers",
            line=dict(color=phase_view["generation"], colorscale="Turbo", width=4),
            marker=dict(size=1.8, color=phase_view["generation"], colorscale="Turbo", opacity=.68),
            customdata=phase_view[["generation"]],
            hovertemplate="generation %{customdata[0]}<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<br>PC3 %{z:.3f}<extra></extra>",
        )
    )
    phase_figure.update_layout(
        height=590,
        paper_bgcolor="#050711", plot_bgcolor="#050711",
        font=dict(color="#dffcff", family="monospace"),
        margin=dict(l=0, r=0, t=55, b=0),
        title="POST-BURN-IN PHASE PORTRAIT · GENERATIONS 1,000–5,000",
        scene=dict(
            bgcolor="#050711",
            xaxis=dict(title="dynamical PC1", gridcolor="#163044"),
            yaxis=dict(title="dynamical PC2", gridcolor="#163044"),
            zaxis=dict(title="dynamical PC3", gridcolor="#281b45"),
            camera=dict(eye=dict(x=1.55, y=-1.55, z=.9)),
        ),
    )
    interpretation = mo.md(
        f"""
        ### Reading the orbit

        This projection compresses the multichannel graph state into its first three dynamical components. The selected orbit has a measured correlation dimension of **{dynamic_record['correlation_dimension']:.2f}**, recurrence determinism of **{dynamic_record['recurrence_determinism']:.2f}**, and a dominant period of **{dynamic_record['dominant_period']:.1f} generations**. Rotate and zoom the orbit to inspect whether it collapses, cycles, or continues to explore structured state space.
        """
    )
    mo.vstack([mo.ui.plotly(phase_figure, config={"displaylogo": False}), interpretation])
    return


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
        value="Dynamical instability",
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
        value=next(iter(molecule_options)),
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
def _(alt, mo, pd, regime_summary):
    regime_order = regime_summary.copy()
    for column in (
        "adjusted_silhouette",
        "adjusted_permutation_p",
        "adjusted_n_molecules",
    ):
        regime_order[column] = pd.to_numeric(regime_order[column], errors="coerce")
    regime_order = regime_order.dropna(subset=["adjusted_silhouette"]).sort_values(
        "adjusted_silhouette", ascending=False
    )
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
    strongest_feature = overall_correlations.loc[
        overall_correlations["spearman_rho"].abs().idxmax()
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
