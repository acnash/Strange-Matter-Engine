"""Hidden scientific instrumentation for the Strange Matter Engine teaching notebooks."""

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors, Lipinski, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "development" / "cyp_graph_smoke_test.csv"

NEON = {
    "cyan": "#00F5FF",
    "magenta": "#FF2BD6",
    "lime": "#B6FF00",
    "orange": "#FF8A00",
    "violet": "#8A5CFF",
    "white": "#F4FBFF",
    "grey": "#718096",
    "black": "#05070D",
    "panel": "#0B1020",
}

ELEMENT_COLOURS = {
    "C": NEON["cyan"],
    "N": NEON["violet"],
    "O": NEON["magenta"],
    "S": NEON["lime"],
    "F": NEON["orange"],
    "Cl": "#49FFB5",
    "Br": "#FF4D6D",
    "P": "#FFD166",
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
        :root { --neon-cyan:#00F5FF; --neon-magenta:#FF2BD6; --neon-lime:#B6FF00; }
        .jp-Notebook, .notebook-container { background:#05070D !important; }
        .jp-Cell-inputWrapper, div.input { display:none !important; }
        .jp-RenderedHTMLCommon, .text_cell_render { color:#EAFBFF !important; }
        .jp-RenderedHTMLCommon h1, .text_cell_render h1 { color:#00F5FF !important; text-shadow:0 0 12px #00F5FF; }
        .jp-RenderedHTMLCommon h2, .text_cell_render h2 { color:#FF2BD6 !important; }
        .jp-RenderedHTMLCommon h3, .text_cell_render h3 { color:#B6FF00 !important; }
        .jp-RenderedHTMLCommon code, .text_cell_render code { color:#FF8A00 !important; background:#0B1020 !important; }
        .jp-RenderedHTMLCommon blockquote, .text_cell_render blockquote { border-left:4px solid #00F5FF; color:#EAFBFF; background:#0B1020; padding:0.5rem 1rem; }
        .widget-label, .widget-readout, .jupyter-widgets label { color:#EAFBFF !important; }
        .sme-card { background:#0B1020; border:1px solid #00F5FF; border-radius:10px; padding:14px; margin:10px 0; box-shadow:0 0 10px #00F5FF33; }
        table { color:#EAFBFF !important; }
        th { color:#00F5FF !important; }
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


def plot_molecule(mol, selected_atom=None, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    coords = _coordinates(mol)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        xy = coords[[i, j]]
        colour = NEON["magenta"] if bond.GetIsAromatic() else NEON["grey"]
        width = 3.0 if bond.GetBondTypeAsDouble() > 1 else 2.0
        ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=width, alpha=0.85, zorder=1)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        colour = ELEMENT_COLOURS.get(atom.GetSymbol(), NEON["white"])
        edge = NEON["lime"] if i == selected_atom else NEON["black"]
        size = 900 if i == selected_atom else 600
        ax.scatter(*coords[i], s=size, c=colour, edgecolors=edge, linewidths=4, zorder=3)
        ax.text(*coords[i], f"{atom.GetSymbol()}\n{i}", ha="center", va="center", color=NEON["black"], weight="bold", zorder=4)
    ax.set_title(title or "Atom-numbered molecular structure", color=NEON["cyan"], pad=15, weight="bold")
    ax.set_axis_off()
    ax.margins(0.18)
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
    edge_colours = [NEON["magenta"] if graph.edges[e]["aromatic"] else NEON["cyan"] for e in graph.edges]
    widths = [1.5 + graph.edges[e]["order"] for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colours, width=widths, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colours, node_size=650, edgecolors=NEON["black"], linewidths=2, ax=ax)
    if selected_atom is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[selected_atom], node_color=[node_colours[selected_atom]], node_size=900, edgecolors=NEON["lime"], linewidths=4, ax=ax)
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
    ax.set_title("Initial chemical-state matrix $C$", color=NEON["cyan"], weight="bold", pad=12)
    ax.set_xlabel("Feature channel")
    ax.set_ylabel("Cell / atom")
    ax.figure.colorbar(image, ax=ax, pad=0.01, label="encoded value")
    return names, matrix, ax


def _selector(description="Molecule"):
    data = load_data()
    return widgets.Dropdown(options=list(data["molecule_id"]), description=description, layout=widgets.Layout(width="350px"))


def gallery_explorer():
    selector = _selector()
    output = widgets.Output()

    def refresh(*_):
        with output:
            clear_output(wait=True)
            row, mol = get_molecule(selector.value)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            plot_molecule(mol, title=f"{row['molecule_id']} | {row['cyp_target']}", ax=axes[0])
            plot_graph(mol, ax=axes[1])
            plt.show()
            display(molecule_summary(row, mol).style.hide(axis="index").set_properties(**{"background-color": NEON["panel"], "color": NEON["white"]}))

    selector.observe(refresh, names="value")
    refresh()
    return widgets.VBox([selector, output])


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
            plot_graph(mol, selected_atom=idx, title=f"Neighbourhood $N({idx})$", ax=axes[1])
            plt.show()
            props = atom_properties(mol).set_index("atom").loc[[idx]].T.reset_index()
            props.columns = ["property", "value"]
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
            plot_molecule(mol, title=f"Edge {idx}: atom {bond.GetBeginAtomIdx()} ↔ atom {bond.GetEndAtomIdx()}", ax=ax)
            coords = _coordinates(mol)
            xy = coords[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            ax.plot(xy[:, 0], xy[:, 1], color=NEON["lime"], lw=8, alpha=0.75, zorder=2)
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
