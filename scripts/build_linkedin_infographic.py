#!/usr/bin/env python3
"""Build the detailed LinkedIn Strange Matter Engine infographic."""

from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures/molecular_spacetime_linkedin_infographic.png"
INK, PANEL, CYAN, MAGENTA, LIME, ORANGE, WHITE, MUTED, GRID = (
    "#070914", "#10162A", "#27E1FF", "#FF3CAC", "#B6FF3B", "#FF9F1C",
    "#E7F2FF", "#94A6C3", "#33415F",
)
TARGETS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGET_COLOURS = dict(zip(TARGETS, (CYAN, MAGENTA, LIME, ORANGE)))
RULE_NAMES = {
    "activator_inhibitor": "Activator-inhibitor", "conservative_graph_flux": "Conservative flux",
    "coupled_map": "Coupled map", "damped_symplectic": "Damped symplectic",
    "delayed_memory": "Delayed memory", "fitzhugh_nagumo": "FitzHugh-Nagumo",
    "gated_residual": "Gated residual", "gray_scott": "Gray-Scott",
    "inertial_reaction_diffusion": "Inertial reaction-diffusion",
    "kuramoto_sakaguchi": "Kuramoto-Sakaguchi",
}
LIGAND_ID = "OCNT-2328784"
LIGAND_SMILES = "CNC(=O)OC1=CC=CC2=CC=CC=C21"


def panel(fig, rect):
    fig.patches.append(FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3], boxstyle="round,pad=0.007,rounding_size=.016",
        transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=1.15, zorder=-5))


def txt(fig, x, y, value, size, colour=WHITE, weight="normal", ha="left", va="top"):
    fig.text(x, y, value, fontsize=size, color=colour, fontweight=weight,
             ha=ha, va=va, family="DejaVu Sans")


def ligand_image():
    mol = Chem.MolFromSmiles(LIGAND_SMILES)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(720, 380)
    opts = drawer.drawOptions(); opts.clearBackground = False
    opts.setBackgroundColour((7/255, 9/255, 20/255, 1))
    opts.setSymbolColour((231/255, 242/255, 1, 1))
    opts.setAtomPalette({
        6: (231/255, 242/255, 1), 7: (39/255, 225/255, 1),
        8: (1, 60/255, 172/255), 16: (182/255, 1, 59/255),
    })
    opts.addAtomIndices = True; opts.bondLineWidth = 2.3; opts.baseFontSize = .62
    drawer.DrawMolecule(mol); drawer.FinishDrawing()
    return np.asarray(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA"))


def load_rmse():
    rows=[]
    for path in sorted((ROOT/"results").glob("production_*_challenge_aligned_v5/study_summary.json")):
        data=json.loads(path.read_text())
        rows.append((data["rule"],float(data["final_metrics"]["restored_validation_rmse"])))
    return sorted(rows,key=lambda x:x[1])


def rmse_plot(fig):
    rmse=load_rmse(); ax=fig.add_axes([.145,.402,.290,.170],facecolor=PANEL)
    rules=[RULE_NAMES[r] for r,_ in rmse][::-1]; vals=[v for _,v in rmse][::-1]
    y=np.arange(len(rules)); colours=[LIME if v==min(vals) else CYAN for v in vals]
    ax.hlines(y,.86,vals,color=GRID,lw=2); ax.scatter(vals,y,c=colours,s=42,zorder=3)
    ax.set_xlim(.86,.96); ax.set_yticks(y,rules,fontsize=7.1,color=WHITE)
    ax.set_xticks([.87,.89,.91,.93,.95]); ax.tick_params(axis="x",colors=MUTED,labelsize=7)
    ax.grid(axis="x",color=GRID,alpha=.42); ax.spines[:].set_visible(False)
    return min(vals)


def dynamics_plot(fig):
    path=ROOT/"results/production_coupled_map_challenge_aligned_v5/runs/final_model/validation_dynamics.csv"
    d=pd.read_csv(path); ax=fig.add_axes([.555,.431,.350,.144],facecolor=PANEL)
    for target in TARGETS:
        q=d[d.cyp_target==target]
        ax.scatter(q.recurrence_ratio,q.late_motion,s=6,alpha=.48,c=TARGET_COLOURS[target],label=target)
    ax.set_xlabel(""); ax.set_ylabel("late motion",color=MUTED,fontsize=7)
    ax.tick_params(colors=MUTED,labelsize=6.7); ax.grid(color=GRID,alpha=.35); ax.spines[:].set_visible(False)
    ax.legend(loc="upper right",frameon=False,labelcolor=WHITE,fontsize=5.5,ncol=2)


def trajectory_examples(fig):
    ax=fig.add_axes([.355,.158,.255,.094],facecolor=PANEL)
    t=np.linspace(0,1,100)
    ax.plot(t,.72*np.exp(-5*t)+.10,color=CYAN,lw=2,label="point attractor")
    ax.plot(t,.49+.14*np.sin(10*np.pi*t),color=MAGENTA,lw=1.7,label="periodic")
    ax.plot(t,.23+.11*np.sin(22*t+5*np.sin(6*t)),color=ORANGE,lw=1.3,label="complex candidate")
    ax.set(xlim=(0,1),ylim=(0,1)); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.legend(frameon=False,labelcolor=WHITE,fontsize=5.4,ncol=1,loc="upper right")


def main():
    plt.rcParams.update({"figure.facecolor": INK,"savefig.facecolor": INK,"axes.facecolor": PANEL,
                         "font.family":"DejaVu Sans","mathtext.fontset":"dejavusans"})
    fig=plt.figure(figsize=(10.8,13.5),dpi=100); fig.subplots_adjust(0,0,1,1)

    txt(fig,.055,.966,"MOLECULAR SPACE-TIME",31,CYAN,"bold")
    txt(fig,.055,.925,"Predicting pIC50 while probing emergent dynamics",17,WHITE,"bold")
    txt(fig,.055,.895,"OpenADMET CYP inhibition challenge  |  graph cellular automata",10.5,MUTED)
    fig.lines.append(plt.Line2D([.055,.945],[.872,.872],transform=fig.transFigure,color=MAGENTA,lw=2.7))

    # Real molecule and chemical encoding.
    panel(fig,[.050,.675,.895,.170])
    mol_ax=fig.add_axes([.063,.695,.255,.122]); mol_ax.imshow(ligand_image()); mol_ax.axis("off")
    txt(fig,.070,.817,f"REAL CHALLENGE LIGAND  {LIGAND_ID}",8.5,CYAN,"bold")
    txt(fig,.335,.817,"WHAT EACH ATOM KNOWS AT GENERATION 0",10.5,MAGENTA,"bold")
    txt(fig,.335,.785,"IDENTITY",8,LIME,"bold"); txt(fig,.408,.785,"element  •  atomic number  •  mass  •  radii",8.6,WHITE)
    txt(fig,.335,.758,"ELECTRONS",8,LIME,"bold"); txt(fig,.408,.758,"formal charge  •  electronegativity  •  polarizability",8.6,WHITE)
    txt(fig,.335,.731,"CHEMISTRY",8,LIME,"bold"); txt(fig,.408,.731,"aromaticity  •  valence  •  hybridisation  •  donor/acceptor",8.6,WHITE)
    txt(fig,.335,.704,"CONTEXT",8,LIME,"bold"); txt(fig,.408,.704,"ring geometry  •  local neighbours  •  CYP target",8.6,WHITE)
    txt(fig,.070,.686,"Bonds define who exchanges information; bond type and conjugation modulate the message.",8.2,MUTED)

    # Prediction and target-conditioned dynamics.
    panel(fig,[.050,.355,.425,.290]); panel(fig,[.505,.355,.440,.290])
    txt(fig,.070,.622,"pIC50 PREDICTION",12.5,CYAN,"bold")
    txt(fig,.070,.596,"10 transition rules  |  grouped validation  |  lower is better",8.1,MUTED)
    best=rmse_plot(fig)
    txt(fig,.070,.380,f"BEST VALIDATION RMSE  {best:.3f} pIC50 units",10,LIME,"bold")

    txt(fig,.525,.622,"TARGET-CONDITIONED DYNAMICS",12.5,MAGENTA,"bold")
    txt(fig,.525,.596,"Do different CYP targets occupy different regions of dynamical space?",8.1,MUTED)
    dynamics_plot(fig)
    txt(fig,.525,.408,"RECURRENCE RATIO",7.5,CYAN,"bold")
    txt(fig,.645,.408,"how closely a later state returns to an earlier state",7.1,WHITE)
    txt(fig,.525,.386,"LATE MOTION",7.5,CYAN,"bold")
    txt(fig,.625,.386,"how much the state still changes near the trajectory end",7.1,WHITE)
    txt(fig,.525,.365,"CYP silhouette 0.667  |  scaffold-grouped accuracy 0.999",8.5,LIME,"bold")

    # Mathematics and trajectory fingerprint.
    panel(fig,[.050,.135,.895,.190])
    txt(fig,.070,.301,"A LOCAL RULE REPEATED THROUGH MOLECULAR SPACE-TIME",11.5,ORANGE,"bold")
    txt(fig,.070,.271,r"$m_i^{(t)}=\sum_{j\in\mathcal{N}(i)}g(e_{ij})\,\psi(h_i^{(t)},h_j^{(t)})$",13,WHITE)
    txt(fig,.070,.238,r"$h_i^{(t+1)}=F_\theta\!\left(h_i^{(t)},m_i^{(t)},c_{\mathrm{CYP}}\right)$",13,CYAN)
    txt(fig,.070,.205,"GRAY-SCOTT EXAMPLE",7.3,ORANGE,"bold")
    txt(fig,.070,.184,r"$\Delta u=D_uL_Gu-uv^2+f(1-u)$",8.7,WHITE)
    txt(fig,.070,.160,r"$\Delta v=D_vL_Gv+uv^2-(f+k)v$",8.7,WHITE)
    txt(fig,.070,.143,r"$L_G$ carries reaction-diffusion signals along bonds.",6.1,MUTED)

    txt(fig,.355,.271,"TRAJECTORY FINGERPRINT",8.5,MAGENTA,"bold")
    trajectory_examples(fig)
    txt(fig,.355,.145,"convergence  •  amplitude  •  recurrence  •  entropy  •  spectrum",7.2,MUTED)

    txt(fig,.635,.271,"ONE MODEL, TWO OUTPUTS",8.5,LIME,"bold")
    txt(fig,.635,.240,r"$\widehat{\mathrm{pIC50}}=b+w^\mathsf{T}z$",13,WHITE)
    txt(fig,.635,.207,"1  Predict experimental inhibition",8.3,CYAN,"bold")
    txt(fig,.635,.181,"2  Map the internal dynamical regime",8.3,MAGENTA,"bold")
    txt(fig,.635,.151,"z = molecule-level summary of the full atom-by-time trajectory",7.1,MUTED)

    txt(fig,.055,.105,"THE DYNAMICAL SEARCH",9.5,MAGENTA,"bold")
    txt(fig,.055,.079,"point attractors   •   periodicity   •   strange-attractor candidates   •   chaotic-behaviour candidates",10.2,WHITE,"bold")
    txt(fig,.055,.045,"Finite-time signatures guide longer, perturbation-tested trajectories - they are candidates, not final claims.",7.8,MUTED)
    txt(fig,.945,.045,"STRANGE MATTER ENGINE",8,CYAN,"bold",ha="right")

    OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=200); plt.close(fig)
    print(OUT)


if __name__=="__main__": main()
