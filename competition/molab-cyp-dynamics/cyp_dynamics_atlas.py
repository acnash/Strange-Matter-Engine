# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.5",
#     "marimo>=0.15",
#     "pandas>=2.2",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # CYP Dynamics Atlas

        **An interactive investigation of learned molecular graph dynamics across CYP1A2, CYP2C9, CYP2D6, and CYP3A4 inhibition.**

        This notebook asks whether learned graph-state dynamics reveal CYP-selective chemistry beyond conventional molecular structure. The trajectories shown here are computational states produced by graph cellular automata. They are distinct from physical molecular-dynamics trajectories.

        _Prototype workspace for the 2026 marimo and OpenADMET cheminformatics notebook competition._
        """
    )
    return


@app.cell
def _(mo):
    isoform = mo.ui.dropdown(
        options=["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"],
        value="CYP1A2",
        label="Choose a CYP isoform",
    )
    isoform
    return (isoform,)


@app.cell
def _(isoform, mo):
    mo.callout(
        mo.md(
            f"""
            ### Current lens: {isoform.value}

            The next build will connect this control to the released training measurements, structural neighbourhoods, dynamical-regime atlas, atom-level trajectories, and frozen-model intervention results.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Planned interactive investigation

        1. **Chemical context:** inspect structures, assay measurements, scaffolds, and nearest neighbours.
        2. **Dynamics theatre:** animate atom-level state propagation and linked time-series views.
        3. **CYP atlas:** explore how learned dynamical regimes distribute across the four isoforms.
        4. **Intervention laboratory:** trace how chemically meaningful graph edits alter the dynamics.
        5. **Evidence:** quantify what dynamics add under scaffold-grouped validation and uncertainty analysis.

        ### Reproducibility and disclosure

        The final notebook will document data provenance, computational boundaries, package versions, cached artifacts, and AI assistance. Every displayed scientific result will be reproducible from released data or explicitly identified precomputed artifacts.
        """
    )
    return


if __name__ == "__main__":
    app.run()
