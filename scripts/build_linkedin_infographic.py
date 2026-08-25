#!/usr/bin/env python3
"""Build a LinkedIn-ready Strange Matter Engine scientific infographic."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "molecular_spacetime_linkedin_infographic.png"
INK, PANEL, PANEL2, CYAN, MAGENTA, LIME, ORANGE, WHITE, MUTED, GRID = (
    "#070914", "#10162A", "#151D36", "#27E1FF", "#FF3CAC", "#B6FF3B",
    "#FF9F1C", "#E7F2FF", "#94A6C3", "#33415F",
)
TARGETS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGET_COLOURS = dict(zip(TARGETS, (CYAN, MAGENTA, LIME, ORANGE)))
RULE_NAMES = {
    "activator_inhibitor": "Activator-inhibitor",
    "conservative_graph_flux": "Conservative flux",
    "coupled_map": "Coupled map",
    "damped_symplectic": "Damped symplectic",
    "delayed_memory": "Delayed memory",
    "fitzhugh_nagumo": "FitzHugh-Nagumo",
    "gated_residual": "Gated residual",
    "gray_scott": "Gray-Scott",
    "inertial_reaction_diffusion": "Inertial reaction-diffusion",
    "kuramoto_sakaguchi": "Kuramoto-Sakaguchi",
}


def panel(fig, rect, radius=.018):
    patch = FancyBboxPatch((rect[0], rect[1]), rect[2], rect[3],
                           boxstyle=f"round,pad=0.008,rounding_size={radius}",
                           transform=fig.transFigure, facecolor=PANEL,
                           edgecolor=GRID, linewidth=1.2, zorder=-5)
    fig.patches.append(patch)


def text(fig, x, y, value, size, colour=WHITE, weight="normal", ha="left", va="top"):
    fig.text(x, y, value, fontsize=size, color=colour, fontweight=weight,
             ha=ha, va=va, family="DejaVu Sans")


def molecule_icon(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    points = np.array([[.12,.55],[.30,.74],[.51,.61],[.70,.76],[.87,.55],[.66,.35],[.42,.34]])
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,2),(0,6)]
    for i,j in edges:
        ax.plot([points[i,0],points[j,0]],[points[i,1],points[j,1]],color=CYAN,lw=3,alpha=.72)
    node_colours = [CYAN, WHITE, MAGENTA, WHITE, LIME, ORANGE, WHITE]
    for (x,y),c in zip(points,node_colours):
        ax.add_patch(Circle((x,y),.055,facecolor=c,edgecolor=INK,lw=1.5))
    for k,(x,y) in enumerate(points):
        ax.text(x,y,str(k+1),ha="center",va="center",fontsize=7,color=INK,fontweight="bold")


def load_rmse():
    rows=[]
    for path in sorted((ROOT/"results").glob("production_*_challenge_aligned_v5/study_summary.json")):
        data=json.loads(path.read_text())
        rows.append((data["rule"],float(data["final_metrics"]["restored_validation_rmse"])))
    return sorted(rows,key=lambda x:x[1])


def main():
    plt.rcParams.update({"figure.facecolor": INK, "savefig.facecolor": INK,
                         "axes.facecolor": PANEL, "font.family": "DejaVu Sans"})
    fig=plt.figure(figsize=(10.8,13.5),dpi=100)
    fig.subplots_adjust(0,0,1,1)

    text(fig,.06,.955,"MOLECULAR SPACE-TIME",35,CYAN,"bold")
    text(fig,.06,.910,"Predicting pIC50 while probing emergent dynamics",20,WHITE,"bold")
    text(fig,.06,.877,"OpenADMET CYP inhibition challenge  |  graph cellular automata",12,MUTED)
    fig.lines.append(plt.Line2D([.06,.94],[.852,.852],transform=fig.transFigure,color=MAGENTA,lw=3))

    panel(fig,[.055,.685,.89,.135])
    icon=fig.add_axes([.075,.704,.18,.095]); molecule_icon(icon)
    text(fig,.285,.795,"THE IDEA",11,MAGENTA,"bold")
    text(fig,.285,.766,"Atoms become cells. Bonds define the neighbourhood.",15,WHITE,"bold")
    text(fig,.285,.733,"Local information exchange unfolds across repeated generations -\ncreating a molecular space-time trajectory.",12,MUTED)
    text(fig,.285,.692,"Fundamental machine learning techniques connect those trajectories to pIC50.",11,LIME,"bold")

    panel(fig,[.055,.345,.43,.305]); panel(fig,[.515,.345,.43,.305])
    text(fig,.075,.625,"pIC50 PREDICTION",14,CYAN,"bold")
    text(fig,.075,.600,"10 rules | grouped validation | lower is better",9,MUTED)
    rmse=load_rmse()
    ax=fig.add_axes([.145,.408,.295,.172],facecolor=PANEL)
    rules=[RULE_NAMES[r] for r,_ in rmse][::-1]; vals=[v for _,v in rmse][::-1]
    y=np.arange(len(rules)); colours=[LIME if abs(v-min(vals))<1e-8 else CYAN for v in vals]
    ax.hlines(y,.86,vals,color=GRID,lw=2); ax.scatter(vals,y,c=colours,s=45,zorder=3)
    ax.set_xlim(.86,.96); ax.set_yticks(y,rules,fontsize=7.4,color=WHITE)
    ax.set_xticks([.87,.89,.91,.93,.95]); ax.tick_params(axis="x",colors=MUTED,labelsize=8)
    ax.grid(axis="x",color=GRID,alpha=.45); ax.spines[:].set_visible(False)
    text(fig,.075,.380,f"BEST VALIDATION RMSE  {min(vals):.3f} pIC50 units",11,LIME,"bold")

    text(fig,.535,.625,"TARGET-CONDITIONED DYNAMICS",14,MAGENTA,"bold")
    text(fig,.535,.600,"Coupled-map example | CYP clusters",10,MUTED)
    path=ROOT/"results/production_coupled_map_challenge_aligned_v5/runs/final_model/validation_dynamics.csv"
    d=pd.read_csv(path)
    ax2=fig.add_axes([.555,.430,.35,.142],facecolor=PANEL)
    for target in TARGETS:
        q=d[d.cyp_target==target]
        ax2.scatter(q.recurrence_ratio,q.late_motion,s=7,alpha=.45,c=TARGET_COLOURS[target],label=target)
    ax2.set_xlabel("recurrence ratio",color=MUTED,fontsize=8); ax2.set_ylabel("late motion",color=MUTED,fontsize=8)
    ax2.tick_params(colors=MUTED,labelsize=7); ax2.grid(color=GRID,alpha=.35); ax2.spines[:].set_visible(False)
    ax2.legend(loc="best",frameon=False,labelcolor=WHITE,fontsize=6,ncol=2)
    text(fig,.535,.401,"CYP silhouette  0.667",10,LIME,"bold")
    text(fig,.535,.378,"Scaffold-grouped accuracy  0.999",9,WHITE,"bold")
    text(fig,.535,.357,"Distinct regimes persist beyond related chemical frameworks.",8.5,MUTED)

    panel(fig,[.055,.145,.89,.165])
    text(fig,.075,.285,"ONE MODEL. TWO SCIENTIFIC QUESTIONS.",14,ORANGE,"bold")
    text(fig,.075,.250,"1",24,CYAN,"bold"); text(fig,.108,.253,"Can molecular space-time predict experimental pIC50?",12,WHITE,"bold")
    text(fig,.075,.213,"2",24,MAGENTA,"bold"); text(fig,.108,.216,"What dynamical regimes emerge inside the evolving molecular graph?",12,WHITE,"bold")
    text(fig,.075,.170,"SIGNALS",9,MUTED,"bold")
    text(fig,.145,.170,"recurrence  •  late motion  •  spectral entropy  •  perturbation response",10,CYAN,"bold")

    text(fig,.06,.115,"THE DYNAMICAL SEARCH",11,MAGENTA,"bold")
    text(fig,.06,.087,"point attractors   •   periodicity   •   strange-attractor candidates   •   chaotic-behaviour candidates",12,WHITE,"bold")
    text(fig,.06,.052,"Finite-time signatures are candidates for longer, perturbation-tested trajectories - not final claims.",9,MUTED)
    text(fig,.94,.052,"STRANGE MATTER ENGINE",9,CYAN,"bold",ha="right")

    OUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT,dpi=200)
    plt.close(fig)
    print(OUT)


if __name__=="__main__": main()
