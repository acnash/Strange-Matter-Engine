"""Hidden scientific instrumentation for the Strange Matter Engine teaching notebooks."""

import base64
import io
import json
import uuid
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "development" / "cyp_graph_smoke_test.csv"

NEON = {
    "cyan": "#27E1FF",
    "magenta": "#FF3CAC",
    "lime": "#F9F871",
    "orange": "#FF9F43",
    "violet": "#7A5CFA",
    "white": "#DCE6F2",
    "grey": "#65758B",
    "black": "#070914",
    "panel": "#11152A",
}

ELEMENT_COLOURS = {
    "C": NEON["white"],
    "N": NEON["cyan"],
    "O": NEON["magenta"],
    "S": NEON["lime"],
    "F": "#43F6A7",
    "Cl": "#43F6A7",
    "Br": NEON["orange"],
    "P": NEON["lime"],
}

plt.rcParams.update(
    {
        "figure.facecolor": NEON["black"],
        "axes.facecolor": NEON["black"],
        "savefig.facecolor": NEON["black"],
        "text.color": NEON["white"],
        "axes.labelcolor": NEON["white"],
        "axes.edgecolor": NEON["cyan"],
        "xtick.color": NEON["white"],
        "ytick.color": NEON["white"],
        "font.size": 11,
    }
)


def neon_css():
    return HTML(
        """
        <style>
        :root { --cyber-cyan:#27E1FF; --cyber-pink:#FF3CAC; --cyber-yellow:#F9F871; }
        .jp-CodeCell .jp-Cell-inputWrapper, div.code_cell div.input { display:none !important; }
        .jp-RenderedHTMLCommon h1, .text_cell_render h1 { color:#087F99 !important; letter-spacing:0.02em; border-bottom:1px solid #B9C5D6; padding-bottom:0.3em; }
        .jp-RenderedHTMLCommon h2, .text_cell_render h2 { color:#C21875 !important; }
        .jp-RenderedHTMLCommon h3, .text_cell_render h3 { color:#6950C5 !important; }
        .jp-RenderedHTMLCommon code, .text_cell_render code { color:#FF9F43 !important; background:#11152A !important; }
        .jp-RenderedHTMLCommon blockquote, .text_cell_render blockquote { border-left:3px solid #27E1FF; background:#EEF3F9; padding:0.5rem 1rem; }
        .sme-card { background:#11152A; border:1px solid #27304D; border-left:3px solid #27E1FF; border-radius:4px; padding:14px; margin:10px 0; }
        </style>
        """
    )


def load_data():
    return pd.read_csv(DATA_PATH)


def get_record(molecule_id):
    frame = load_data()
    row = frame.loc[frame["molecule_id"] == molecule_id]
    if row.empty:
        raise KeyError(molecule_id)
    return row.iloc[0]


def get_molecule(molecule_id):
    row = get_record(molecule_id)
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    if mol is None:
        raise ValueError(f"SMILES could not be parsed: {molecule_id}")
    AllChem.Compute2DCoords(mol)
    return row, mol


def atom_feature_sets(mol):
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    donors, acceptors = set(), set()
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donors.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())
    aromatic_ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_ring_atoms.update(ring)
    return donors, acceptors, aromatic_ring_atoms


def atom_properties(mol):
    donors, acceptors, aromatic_ring_atoms = atom_feature_sets(mol)
    rows = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        rows.append(
            {
                "atom": idx,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(),
                "hybridisation": str(atom.GetHybridization()),
                "heavy_atom_degree": atom.GetDegree(),
                "attached_hydrogens": atom.GetTotalNumHs(),
                "donor": idx in donors,
                "acceptor": idx in acceptors,
                "in_ring": atom.IsInRing(),
                "in_aromatic_ring": idx in aromatic_ring_atoms,
                "mass_Da": atom.GetMass(),
                "neighbours": ", ".join(str(n.GetIdx()) for n in atom.GetNeighbors()),
            }
        )
    return pd.DataFrame(rows)


def bond_properties(mol):
    rows = []
    for bond in mol.GetBonds():
        rows.append(
            {
                "edge": bond.GetIdx(),
                "atom_i": bond.GetBeginAtomIdx(),
                "atom_j": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()),
                "bond_order": bond.GetBondTypeAsDouble(),
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "in_ring": bond.IsInRing(),
                "stereo": str(bond.GetStereo()),
            }
        )
    return pd.DataFrame(rows)


def _coordinates(mol):
    conformer = mol.GetConformer()
    coords = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    span = np.ptp(coords, axis=0)
    span[span == 0] = 1
    return (coords - coords.mean(axis=0)) / span.max()


def _rdkit_depiction(mol, selected_atom=None, selected_bond=None, width=1000, height=700):
    import io

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.backgroundColour = (7 / 255, 9 / 255, 20 / 255, 1.0)
    options.atomNoteColour = (39 / 255, 225 / 255, 1.0, 1.0)
    options.annotationFontScale = 0.85
    options.bondLineWidth = 2.2
    options.additionalAtomLabelPadding = 0.08
    options.updateAtomPalette(
        {
            6: (220 / 255, 230 / 255, 242 / 255),
            7: (39 / 255, 225 / 255, 1.0),
            8: (1.0, 60 / 255, 172 / 255),
            9: (67 / 255, 246 / 255, 167 / 255),
            15: (249 / 255, 248 / 255, 113 / 255),
            16: (249 / 255, 248 / 255, 113 / 255),
            17: (67 / 255, 246 / 255, 167 / 255),
            35: (1.0, 159 / 255, 67 / 255),
        }
    )
    if selected_atom is not None:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=[selected_atom],
            highlightAtomColors={selected_atom: (1.0, 60 / 255, 172 / 255)},
            highlightAtomRadii={selected_atom: 0.38},
        )
    elif selected_bond is not None:
        drawer.DrawMolecule(
            mol,
            [],
            [selected_bond],
            {},
            {selected_bond: (39 / 255, 225 / 255, 1.0)},
            {},
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def plot_molecule(mol, selected_atom=None, selected_bond=None, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(_rdkit_depiction(mol, selected_atom=selected_atom, selected_bond=selected_bond))
    ax.set_title(title or "Atom-numbered molecular structure", color=NEON["cyan"], pad=15, weight="bold")
    ax.set_axis_off()
    return ax


def plot_graph(mol, selected_atom=None, title="Molecular graph $G=(V,E)$", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            aromatic=bond.GetIsAromatic(),
            order=bond.GetBondTypeAsDouble(),
        )
    coords = _coordinates(mol)
    pos = {i: coords[i] for i in range(mol.GetNumAtoms())}
    node_colours = [ELEMENT_COLOURS.get(graph.nodes[i]["element"], NEON["white"]) for i in graph.nodes]
    edge_colours = [NEON["violet"] if graph.edges[e]["aromatic"] else NEON["cyan"] for e in graph.edges]
    widths = [1.5 + graph.edges[e]["order"] for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colours, width=widths, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colours, node_size=650, edgecolors=NEON["black"], linewidths=2, ax=ax)
    if selected_atom is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[selected_atom], node_color=[node_colours[selected_atom]], node_size=900, edgecolors=NEON["magenta"], linewidths=4, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={i: f"{graph.nodes[i]['element']}\n{i}" for i in graph.nodes}, font_color=NEON["black"], font_weight="bold", ax=ax)
    ax.set_title(title, color=NEON["magenta"], pad=15, weight="bold")
    ax.set_axis_off()
    ax.margins(0.18)
    return graph, ax


def molecule_summary(row, mol):
    return pd.DataFrame(
        {
            "quantity": ["Molecule", "CYP context", "Experimental pIC50", "Atoms |V|", "Bonds |E|", "Molecular weight", "Formal charge", "Rings", "Aromatic rings", "Rotatable bonds"],
            "value": [
                row["molecule_id"],
                row["cyp_target"],
                f"{row['experimental_pic50']:.3f}",
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                f"{Descriptors.MolWt(mol):.2f} Da",
                Chem.GetFormalCharge(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                Lipinski.NumRotatableBonds(mol),
            ],
        }
    )


ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Other"]
HYBRIDISATIONS = ["SP", "SP2", "SP3", "OTHER"]


def encode_atom(mol, atom_index):
    properties = atom_properties(mol).set_index("atom").loc[atom_index]
    element = properties["element"] if properties["element"] in ELEMENTS[:-1] else "Other"
    hybrid = properties["hybridisation"] if properties["hybridisation"] in HYBRIDISATIONS[:-1] else "OTHER"
    names = [f"element_{e}" for e in ELEMENTS]
    values = [1.0 if element == e else 0.0 for e in ELEMENTS]
    names += ["formal_charge", "aromatic"]
    values += [float(properties["formal_charge"]), float(properties["aromatic"])]
    names += [f"hybrid_{h.lower()}" for h in HYBRIDISATIONS]
    values += [1.0 if hybrid == h else 0.0 for h in HYBRIDISATIONS]
    names += ["heavy_atom_degree", "attached_hydrogens", "donor", "acceptor", "in_ring", "in_aromatic_ring"]
    values += [
        float(properties["heavy_atom_degree"]),
        float(properties["attached_hydrogens"]),
        float(properties["donor"]),
        float(properties["acceptor"]),
        float(properties["in_ring"]),
        float(properties["in_aromatic_ring"]),
    ]
    return names, np.array(values, dtype=float)


def encoding_matrix(mol):
    vectors = [encode_atom(mol, i)[1] for i in range(mol.GetNumAtoms())]
    return encode_atom(mol, 0)[0], np.vstack(vectors)


def plot_encoding(mol, selected_atom=0, ax=None):
    names, matrix = encoding_matrix(mol)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(4, mol.GetNumAtoms() * 0.42)))
    cmap = LinearSegmentedColormap.from_list("neon", [NEON["black"], NEON["violet"], NEON["cyan"], NEON["lime"]])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(mol.GetNumAtoms()), [f"Atom {i}" for i in range(mol.GetNumAtoms())])
    ax.get_yticklabels()[selected_atom].set_color(NEON["lime"])
    ax.get_yticklabels()[selected_atom].set_weight("bold")
    ax.set_title("Initial chemical-state matrix C", color=NEON["cyan"], weight="bold", pad=12)
    ax.set_xlabel("Feature channel")
    ax.set_ylabel("Cell / atom")
    ax.figure.colorbar(image, ax=ax, pad=0.01, label="encoded value")
    return names, matrix, ax


def _selector(description="Molecule"):
    data = load_data()
    return widgets.Dropdown(options=list(data["molecule_id"]), description=description, layout=widgets.Layout(width="350px"))


def gallery_explorer():
    records = {}
    for molecule_id in load_data()["molecule_id"]:
        row, mol = get_molecule(molecule_id)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        plot_molecule(mol, title=f"{row['molecule_id']} | {row['cyp_target']}", ax=axes[0])
        plot_graph(mol, ax=axes[1])
        plt.tight_layout()
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        summary = molecule_summary(row, mol)
        records[molecule_id] = {
            "image": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            "summary": summary.to_dict(orient="records"),
        }

    control_id = f"sme-gallery-{uuid.uuid4().hex}"
    options = "".join(f'<option value="{key}">{key}</option>' for key in records)
    payload = json.dumps(records).replace("</", "<\\/")
    return HTML(
        f"""
        <div id="{control_id}" class="sme-gallery">
          <label for="{control_id}-selector">Molecule</label>
          <select id="{control_id}-selector">{options}</select>
          <img alt="Selected molecule and graph" />
          <table><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody></tbody></table>
        </div>
        <style>
          #{control_id} {{ color:#182235; font-family:system-ui,sans-serif; }}
          #{control_id} label {{ font-weight:700; margin-right:0.7rem; }}
          #{control_id} select {{ min-width:230px; padding:0.35rem 0.5rem; border:1px solid #087F99; border-radius:4px; background:white; color:#182235; }}
          #{control_id} img {{ display:block; width:100%; max-width:1400px; margin:0.8rem 0; background:#070914; }}
          #{control_id} table {{ border-collapse:collapse; width:min(680px,100%); color:#182235; background:white; }}
          #{control_id} th {{ background:#E8F3F7; color:#087F99; text-align:left; }}
          #{control_id} th, #{control_id} td {{ border:1px solid #C8D3E0; padding:0.42rem 0.65rem; }}
        </style>
        <script>
        (() => {{
          const root = document.getElementById({json.dumps(control_id)});
          const records = {payload};
          const selector = root.querySelector('select');
          const image = root.querySelector('img');
          const body = root.querySelector('tbody');
          function render() {{
            const record = records[selector.value];
            image.src = 'data:image/png;base64,' + record.image;
            body.replaceChildren(...record.summary.map(item => {{
              const row = document.createElement('tr');
              const quantity = document.createElement('td');
              const value = document.createElement('td');
              quantity.textContent = item.quantity;
              value.textContent = item.value;
              row.append(quantity, value);
              return row;
            }}));
          }}
          selector.addEventListener('change', render);
          render();
        }})();
        </script>
        """
    )


def atom_explorer(mode="properties"):
    selector = _selector()
    atom_selector = widgets.IntSlider(value=0, min=0, max=9, step=1, description="Atom 0", readout=False, continuous_update=False)
    output = widgets.Output()

    def update_atom_limit(*_):
        _, mol = get_molecule(selector.value)
        atom_selector.max = mol.GetNumAtoms() - 1
        atom_selector.value = min(atom_selector.value, atom_selector.max)
        refresh()

    def refresh(*_):
        atom_selector.description = f"Atom {atom_selector.value}"
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            idx = min(atom_selector.value, mol.GetNumAtoms() - 1)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            plot_molecule(mol, selected_atom=idx, title=f"Selected cell: atom {idx}", ax=axes[0])
            plot_graph(mol, selected_atom=idx, title=f"Neighbourhood N({idx})", ax=axes[1])
            plt.show()
            props = atom_properties(mol).set_index("atom").loc[[idx]].T.reset_index()
            props.columns = ["property", "value"]
            if mode == "graph":
                props = props.loc[props["property"].isin(["heavy_atom_degree", "neighbours"])]
                props["property"] = props["property"].replace(
                    {"heavy_atom_degree": "degree dᵢ", "neighbours": "neighbour set N(i)"}
                )
            display(props.style.hide(axis="index").set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))
            if mode == "encoding":
                names, vector = encode_atom(mol, idx)
                encoded = pd.DataFrame({"channel": names, "value": vector})
                display(encoded.style.hide(axis="index").set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))
                fig, ax = plt.subplots(figsize=(14, max(4, mol.GetNumAtoms() * 0.42)))
                plot_encoding(mol, selected_atom=idx, ax=ax)
                plt.tight_layout()
                plt.show()

    selector.observe(update_atom_limit, names="value")
    atom_selector.observe(refresh, names="value")
    update_atom_limit()
    return widgets.VBox([widgets.HBox([selector, atom_selector]), output])


def bond_explorer():
    selector = _selector()
    bond_selector = widgets.IntSlider(value=0, min=0, max=8, step=1, description="Bond 0", readout=False, continuous_update=False)
    output = widgets.Output()

    def update_limit(*_):
        _, mol = get_molecule(selector.value)
        bond_selector.max = max(0, mol.GetNumBonds() - 1)
        bond_selector.value = min(bond_selector.value, bond_selector.max)
        refresh()

    def refresh(*_):
        bond_selector.description = f"Bond {bond_selector.value}"
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            idx = min(bond_selector.value, mol.GetNumBonds() - 1)
            bond = mol.GetBondWithIdx(idx)
            fig, ax = plt.subplots(figsize=(8, 5.5))
            plot_molecule(mol, selected_bond=idx, title=f"Edge {idx}: atom {bond.GetBeginAtomIdx()} ↔ atom {bond.GetEndAtomIdx()}", ax=ax)
            plt.show()
            table = bond_properties(mol).set_index("edge").loc[[idx]].T.reset_index().rename(columns={"index": "property", idx: "value"})
            display(table.style.hide(axis="index").set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    selector.observe(update_limit, names="value")
    bond_selector.observe(refresh, names="value")
    update_limit()
    return widgets.VBox([widgets.HBox([selector, bond_selector]), output])


ELECTRONEGATIVITY = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96}


def diffuse(mol, alpha=0.35, steps=12):
    states = np.zeros((steps + 1, mol.GetNumAtoms()), dtype=float)
    states[0] = [ELECTRONEGATIVITY.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()]
    for t in range(steps):
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
            neighbour_mean = states[t, neighbours].mean() if neighbours else states[t, i]
            states[t + 1, i] = (1 - alpha) * states[t, i] + alpha * neighbour_mean
    return states


def ca_explorer():
    selector = _selector()
    alpha = widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.05, description="α = 0.35", readout=False, continuous_update=False)
    steps = widgets.IntSlider(value=12, min=1, max=40, step=1, description="T = 12", readout=False, continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        alpha.description = f"α = {alpha.value:.2f}"
        steps.description = f"T = {steps.value}"
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            states = diffuse(mol, alpha.value, steps.value)
            cmap = LinearSegmentedColormap.from_list("state", [NEON["black"], NEON["violet"], NEON["cyan"], NEON["lime"]])
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            im = axes[0].imshow(states.T, aspect="auto", cmap=cmap, interpolation="nearest")
            axes[0].set_title("Molecular spacetime", color=NEON["cyan"], weight="bold")
            axes[0].set_xlabel("Generation $t$")
            axes[0].set_ylabel("Atom / cell $i$")
            fig.colorbar(im, ax=axes[0], label="state value")
            step_distance = np.linalg.norm(np.diff(states, axis=0), axis=1)
            axes[1].plot(range(1, len(step_distance) + 1), step_distance, color=NEON["magenta"], lw=3)
            axes[1].scatter(range(1, len(step_distance) + 1), step_distance, color=NEON["lime"], s=25)
            axes[1].set_title("Step-to-step distance", color=NEON["magenta"], weight="bold")
            axes[1].set_xlabel("Generation $t$")
            axes[1].set_ylabel(r"$\|X^{(t)}-X^{(t-1)}\|_2$")
            axes[1].grid(color=NEON["grey"], alpha=0.2)
            plt.tight_layout()
            plt.show()
            print(f"Final step distance: {step_distance[-1]:.6f}")

    for control in (selector, alpha, steps):
        control.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([selector, alpha, steps]), output])


def static_overview(molecule_id="DEV_CYP1A2_001"):
    row, mol = get_molecule(molecule_id)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_molecule(mol, title=f"{row['molecule_id']} | atom-numbered structure", ax=axes[0])
    plot_graph(mol, title="The same molecule as $G=(V,E)$", ax=axes[1])
    plt.tight_layout()
    plt.show()


def neighbour_mean_matrix(mol):
    """Row-normalised adjacency used by the transparent teaching rules."""
    n_atoms = mol.GetNumAtoms()
    matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        neighbours = [neighbour.GetIdx() for neighbour in atom.GetNeighbors()]
        if neighbours:
            matrix[i, neighbours] = 1.0 / len(neighbours)
        else:
            matrix[i, i] = 1.0
    return matrix


def diffuse_from_initial(mol, initial_state, alpha=0.35, steps=24):
    transition = (1.0 - alpha) * np.eye(mol.GetNumAtoms()) + alpha * neighbour_mean_matrix(mol)
    states = np.zeros((steps + 1, mol.GetNumAtoms()), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for t in range(steps):
        states[t + 1] = transition @ states[t]
    return states


def trajectory_summary(states, tolerance=1e-3, persistence=3):
    step_distance = np.linalg.norm(np.diff(states, axis=0), axis=1)
    convergence_time = np.nan
    for start in range(max(0, len(step_distance) - persistence + 1)):
        if np.all(step_distance[start : start + persistence] < tolerance):
            convergence_time = start + 1
            break
    centred = states - states.mean(axis=0, keepdims=True)
    lag_pairs = []
    for atom_series in centred.T:
        denominator = np.dot(atom_series, atom_series)
        if denominator > 1e-12:
            lag_pairs.append(np.dot(atom_series[:-1], atom_series[1:]) / denominator)
    spectrum = np.abs(np.fft.rfft(centred, axis=0)) ** 2
    nonzero_power = spectrum[1:].sum()
    dominant_fraction = float(spectrum[1:].max() / nonzero_power) if nonzero_power > 1e-12 else 0.0
    observed_convergence = float(convergence_time) if np.isfinite(convergence_time) else float(len(states))
    return {
        "final_step_distance": float(step_distance[-1]),
        "convergence_time": observed_convergence,
        "mean_atom_variance": float(np.var(states, axis=1).mean()),
        "mean_temporal_variance": float(np.var(states, axis=0).mean()),
        "oscillation_amplitude": float(np.ptp(states, axis=0).mean()),
        "mean_lag1_autocorrelation": float(np.mean(lag_pairs)) if lag_pairs else 0.0,
        "dominant_frequency_power_fraction": dominant_fraction,
        "transient_length": float(convergence_time) if np.isfinite(convergence_time) else float(len(states) - 1),
    }


FINGERPRINT_NAMES = [
    "final_step_distance",
    "convergence_time",
    "mean_atom_variance",
    "mean_temporal_variance",
    "oscillation_amplitude",
    "mean_lag1_autocorrelation",
    "dominant_frequency_power_fraction",
    "transient_length",
]


def dynamical_fingerprint(mol, alpha=0.35, steps=24, tolerance=1e-3):
    states = diffuse(mol, alpha=alpha, steps=steps)
    summary = trajectory_summary(states, tolerance=tolerance)
    return states, np.array([summary[name] for name in FINGERPRINT_NAMES]), summary


def fingerprint_explorer():
    selector = _selector()
    alpha = widgets.FloatSlider(value=0.35, min=0.05, max=0.95, step=0.05, description="alpha 0.35", readout=False, continuous_update=False)
    steps = widgets.IntSlider(value=24, min=8, max=60, step=4, description="T 24", readout=False, continuous_update=False)
    tolerance = widgets.FloatLogSlider(value=1e-3, base=10, min=-5, max=-1, step=0.5, description="tolerance", continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        alpha.description = f"alpha {alpha.value:.2f}"
        steps.description = f"T {steps.value}"
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            states, fingerprint, summary = dynamical_fingerprint(mol, alpha.value, steps.value, tolerance.value)
            distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            cmap = LinearSegmentedColormap.from_list("fingerprint", [NEON["black"], NEON["violet"], NEON["cyan"], NEON["lime"]])
            image = axes[0].imshow(states.T, aspect="auto", cmap=cmap, interpolation="nearest")
            axes[0].set(title="Complete molecular trajectory", xlabel="Generation t", ylabel="Atom i")
            fig.colorbar(image, ax=axes[0], label="state")
            axes[1].semilogy(range(1, len(distances) + 1), np.maximum(distances, 1e-12), color=NEON["magenta"], lw=3)
            axes[1].axhline(tolerance.value, color=NEON["lime"], ls="--", label="tolerance")
            axes[1].set(title="Convergence measurement", xlabel="Generation t", ylabel="step distance")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            table = pd.DataFrame({"fingerprint component": FINGERPRINT_NAMES, "value": fingerprint})
            display(table.style.hide(axis="index").format({"value": "{:.6g}"}).set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    for control in (selector, alpha, steps, tolerance):
        control.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([selector, alpha, steps, tolerance]), output])


def perturbation_analysis(mol, atom_index=0, epsilon=1e-3, alpha=0.35, steps=32):
    initial = np.array([ELECTRONEGATIVITY.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()], dtype=float)
    perturbed = initial.copy()
    perturbed[atom_index] += epsilon
    reference_states = diffuse_from_initial(mol, initial, alpha=alpha, steps=steps)
    perturbed_states = diffuse_from_initial(mol, perturbed, alpha=alpha, steps=steps)
    separation = np.linalg.norm(perturbed_states - reference_states, axis=1)
    times = np.arange(1, steps + 1)
    finite_time_rate = np.log(np.maximum(separation[1:], 1e-15) / epsilon) / times
    return reference_states, perturbed_states, separation, finite_time_rate


def perturbation_explorer():
    selector = _selector()
    atom_index = widgets.IntSlider(value=0, min=0, max=9, step=1, description="atom 0", readout=False, continuous_update=False)
    epsilon = widgets.FloatLogSlider(value=1e-3, base=10, min=-6, max=-1, step=0.5, description="epsilon", continuous_update=False)
    alpha = widgets.FloatSlider(value=0.35, min=0.05, max=0.95, step=0.05, description="alpha 0.35", readout=False, continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        row, mol = get_molecule(selector.value)
        atom_index.max = mol.GetNumAtoms() - 1
        atom_index.value = min(atom_index.value, atom_index.max)
        atom_index.description = f"atom {atom_index.value}"
        alpha.description = f"alpha {alpha.value:.2f}"
        reference, perturbed, separation, rates = perturbation_analysis(mol, atom_index.value, epsilon.value, alpha.value)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        axes[0].plot(reference[:, atom_index.value], color=NEON["cyan"], lw=3, label="reference")
        axes[0].plot(perturbed[:, atom_index.value], color=NEON["magenta"], lw=2, ls="--", label="perturbed")
        axes[0].set(title=f"Atom {atom_index.value}: two nearby trajectories", xlabel="Generation t", ylabel="state")
        axes[0].legend()
        axes[1].semilogy(np.maximum(separation, 1e-15), color=NEON["lime"], lw=3)
        axes[1].set(title="Global trajectory separation", xlabel="Generation t", ylabel="distance delta(t)")
        plt.tight_layout()
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        result = pd.DataFrame(
            {
                "quantity": ["initial perturbation", "final separation", "largest finite-time rate", "mean finite-time rate"],
                "value": [epsilon.value, separation[-1], np.max(rates), np.mean(rates)],
            }
        )
        result_html = result.style.hide(axis="index").format({"value": "{:.6g}"}).set_properties(
            **{"background-color": NEON["panel"], "color": NEON["white"]}
        ).to_html()
        output.outputs = (
            {
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(image_buffer.getvalue()).decode("ascii")},
                "metadata": {},
            },
            {
                "output_type": "display_data",
                "data": {"text/html": result_html, "text/plain": result.to_string(index=False)},
                "metadata": {},
            },
        )

    for control in (selector, atom_index, epsilon, alpha):
        control.observe(refresh, names="value")
    controls = (selector, atom_index, epsilon, alpha)
    display(*controls, output)
    refresh()
    return (*controls, output)


def parameterised_trajectory(mol, theta_self=0.6, theta_neighbour=0.4, theta_bias=0.0, steps=24):
    neighbour_matrix = neighbour_mean_matrix(mol)
    states = np.zeros((steps + 1, mol.GetNumAtoms()), dtype=float)
    electronegativity = np.array([ELECTRONEGATIVITY.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()])
    states[0] = (electronegativity - electronegativity.mean()) / (electronegativity.std() + 1e-8)
    for t in range(steps):
        states[t + 1] = np.tanh(theta_self * states[t] + theta_neighbour * (neighbour_matrix @ states[t]) + theta_bias)
    return states


def parameterised_rule_explorer():
    selector = _selector()
    theta_self = widgets.FloatSlider(value=0.6, min=-1.5, max=1.5, step=0.1, description="self 0.6", readout=False, continuous_update=False)
    theta_neighbour = widgets.FloatSlider(value=0.4, min=-1.5, max=1.5, step=0.1, description="neighbour 0.4", readout=False, continuous_update=False)
    theta_bias = widgets.FloatSlider(value=0.0, min=-0.75, max=0.75, step=0.05, description="bias 0.0", readout=False, continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        theta_self.description = f"self {theta_self.value:.1f}"
        theta_neighbour.description = f"neighbour {theta_neighbour.value:.1f}"
        theta_bias.description = f"bias {theta_bias.value:.2f}"
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            states = parameterised_trajectory(mol, theta_self.value, theta_neighbour.value, theta_bias.value)
            distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.3))
            cmap = LinearSegmentedColormap.from_list("learned", [NEON["black"], NEON["violet"], NEON["cyan"], NEON["lime"]])
            axes[0].imshow(states.T, aspect="auto", cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
            axes[0].set(title="Parameterised shared-rule trajectory", xlabel="Generation t", ylabel="Atom i")
            axes[1].plot(distances, color=NEON["magenta"], lw=3)
            axes[1].set(title="Movement under the chosen parameters", xlabel="Generation t", ylabel="step distance")
            plt.tight_layout()
            plt.show()

    for control in (selector, theta_self, theta_neighbour, theta_bias):
        control.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([selector, theta_self, theta_neighbour, theta_bias]), output])


def bptt_gradients(mol, target, theta_self=0.6, theta_neighbour=0.4, theta_bias=0.0, steps=16):
    neighbour_matrix = neighbour_mean_matrix(mol)
    states = parameterised_trajectory(mol, theta_self, theta_neighbour, theta_bias, steps)
    prediction = float(states[-1].mean())
    loss = 0.5 * (prediction - target) ** 2
    adjoint = np.full(mol.GetNumAtoms(), (prediction - target) / mol.GetNumAtoms())
    gradient = np.zeros(3, dtype=float)
    adjoint_norms = [np.linalg.norm(adjoint)]
    transition_linear = theta_self * np.eye(mol.GetNumAtoms()) + theta_neighbour * neighbour_matrix
    for t in range(steps - 1, -1, -1):
        local = 1.0 - states[t + 1] ** 2
        preactivation_gradient = adjoint * local
        gradient[0] += np.dot(preactivation_gradient, states[t])
        gradient[1] += np.dot(preactivation_gradient, neighbour_matrix @ states[t])
        gradient[2] += preactivation_gradient.sum()
        adjoint = transition_linear.T @ preactivation_gradient
        adjoint_norms.append(np.linalg.norm(adjoint))
    return states, prediction, loss, gradient, np.array(adjoint_norms[::-1])


def bptt_explorer():
    selector = _selector()
    theta_self = widgets.FloatSlider(value=0.6, min=-1.2, max=1.2, step=0.1, description="self 0.6", readout=False, continuous_update=False)
    theta_neighbour = widgets.FloatSlider(value=0.4, min=-1.2, max=1.2, step=0.1, description="neighbour 0.4", readout=False, continuous_update=False)
    steps = widgets.IntSlider(value=16, min=2, max=40, step=2, description="T 16", readout=False, continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            all_targets = load_data()["experimental_pic50"]
            target = float((row["experimental_pic50"] - all_targets.mean()) / all_targets.std(ddof=0))
            states, prediction, loss, gradient, adjoint_norms = bptt_gradients(mol, target, theta_self.value, theta_neighbour.value, 0.0, steps.value)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
            axes[0].plot(states.mean(axis=1), color=NEON["cyan"], lw=3, label="mean state")
            axes[0].axhline(target, color=NEON["magenta"], ls="--", label="scaled target")
            axes[0].set(title="Forward pass", xlabel="Generation t", ylabel="molecular mean")
            axes[0].legend()
            axes[1].semilogy(np.maximum(adjoint_norms, 1e-15), color=NEON["lime"], lw=3)
            axes[1].set(title="Backward gradient signal", xlabel="Generation t", ylabel="adjoint norm")
            plt.tight_layout()
            plt.show()
            table = pd.DataFrame({"quantity": ["prediction", "scaled target", "loss", "dL/d self", "dL/d neighbour", "dL/d bias"], "value": [prediction, target, loss, *gradient]})
            display(table.style.hide(axis="index").format({"value": "{:.6g}"}).set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    for control in (selector, theta_self, theta_neighbour, steps):
        control.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([selector, theta_self, theta_neighbour, steps]), output])


def teaching_feature_table(alpha=0.35, steps=24, tolerance=1e-3):
    rows = []
    for _, record in load_data().iterrows():
        mol = Chem.MolFromSmiles(record["canonical_smiles"])
        AllChem.Compute2DCoords(mol)
        _, fingerprint, _ = dynamical_fingerprint(mol, alpha, steps, tolerance)
        rows.append([record["molecule_id"], record["cyp_target"], record["experimental_pic50"], *fingerprint])
    return pd.DataFrame(rows, columns=["molecule_id", "cyp_target", "experimental_pic50", *FINGERPRINT_NAMES])


def ridge_fit(x, y, penalty=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardised = (x - mean) / scale
    y_mean = y.mean()
    coefficients = np.linalg.solve(standardised.T @ standardised + penalty * np.eye(standardised.shape[1]), standardised.T @ (y - y_mean))
    predictions = y_mean + standardised @ coefficients
    return predictions, coefficients, y_mean, mean, scale


def ridge_explorer():
    penalty = widgets.FloatLogSlider(value=1.0, base=10, min=-3, max=3, step=0.25, description="lambda", continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        with output:
            clear_output(wait=True)
            table = teaching_feature_table()
            x = table[FINGERPRINT_NAMES].to_numpy()
            y = table["experimental_pic50"].to_numpy()
            prediction, coefficients, _, _, _ = ridge_fit(x, y, penalty.value)
            residuals = y - prediction
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
            colours = [NEON["cyan"], NEON["magenta"], NEON["lime"], NEON["orange"]] * 3
            axes[0].scatter(y, prediction, c=colours[: len(y)], s=90, edgecolor=NEON["white"])
            limits = [min(y.min(), prediction.min()) - 0.2, max(y.max(), prediction.max()) + 0.2]
            axes[0].plot(limits, limits, color=NEON["white"], ls="--")
            axes[0].set(title="Teaching-set fit", xlabel="experimental pIC50", ylabel="predicted pIC50", xlim=limits, ylim=limits)
            axes[1].barh(FINGERPRINT_NAMES, coefficients, color=NEON["violet"])
            axes[1].set(title="Regularised coefficients", xlabel="coefficient beta")
            plt.tight_layout()
            plt.show()
            metrics = pd.DataFrame({"quantity": ["lambda", "training RMSE", "coefficient L2 norm"], "value": [penalty.value, np.sqrt(np.mean(residuals ** 2)), np.linalg.norm(coefficients)]})
            display(metrics.style.hide(axis="index").format({"value": "{:.6g}"}).set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    penalty.observe(refresh, names="value")
    refresh()
    return widgets.VBox([penalty, output])


BASELINE_NAMES = ["molecular_weight", "logP", "TPSA", "rings", "rotatable_bonds"]


def baseline_features(mol):
    return np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRings(mol), Lipinski.NumRotatableBonds(mol)], dtype=float)


def validation_table(penalty=1.0, scheme="leave-one-CYP-out"):
    data = teaching_feature_table()
    dynamic_x = data[FINGERPRINT_NAMES].to_numpy()
    baseline_x = []
    for molecule_id in data["molecule_id"]:
        _, mol = get_molecule(molecule_id)
        baseline_x.append(baseline_features(mol))
    baseline_x = np.vstack(baseline_x)
    y = data["experimental_pic50"].to_numpy()
    if scheme == "leave-one-molecule-out":
        folds = [np.array([i]) for i in range(len(data))]
    else:
        folds = [np.where(data["cyp_target"].to_numpy() == target)[0] for target in data["cyp_target"].unique()]

    def cross_validated_predictions(x):
        predictions = np.zeros_like(y)
        for test_indices in folds:
            train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
            _, coefficients, intercept, mean, scale = ridge_fit(x[train_indices], y[train_indices], penalty)
            predictions[test_indices] = intercept + ((x[test_indices] - mean) / scale) @ coefficients
        return predictions

    baseline_prediction = cross_validated_predictions(baseline_x)
    dynamic_prediction = cross_validated_predictions(dynamic_x)
    rows = []
    for name, prediction in [("descriptor baseline", baseline_prediction), ("dynamical fingerprint", dynamic_prediction)]:
        residual = y - prediction
        rows.append({"model": name, "RMSE": np.sqrt(np.mean(residual ** 2)), "MAE": np.mean(np.abs(residual)), "predictions": prediction})
    return data, rows


def validation_explorer():
    scheme = widgets.Dropdown(options=["leave-one-CYP-out", "leave-one-molecule-out"], value="leave-one-CYP-out", description="split")
    penalty = widgets.FloatLogSlider(value=1.0, base=10, min=-3, max=3, step=0.25, description="lambda", continuous_update=False)
    output = widgets.Output()

    def refresh(*_):
        with output:
            clear_output(wait=True)
            data, results = validation_table(penalty.value, scheme.value)
            y = data["experimental_pic50"].to_numpy()
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
            for ax, result, colour in zip(axes, results, [NEON["cyan"], NEON["magenta"]]):
                prediction = result["predictions"]
                limits = [min(y.min(), prediction.min()) - 0.3, max(y.max(), prediction.max()) + 0.3]
                ax.scatter(y, prediction, color=colour, s=90, edgecolor=NEON["white"])
                ax.plot(limits, limits, color=NEON["white"], ls="--")
                ax.set(title=result["model"], xlabel="experimental pIC50", ylabel="held-out prediction", xlim=limits, ylim=limits)
            plt.tight_layout()
            plt.show()
            metrics = pd.DataFrame([{key: value for key, value in result.items() if key != "predictions"} for result in results])
            display(metrics.style.hide(axis="index").format({"RMSE": "{:.4f}", "MAE": "{:.4f}"}).set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    for control in (scheme, penalty):
        control.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([scheme, penalty]), output])
