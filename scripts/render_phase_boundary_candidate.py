"""Two-panel publication candidate; schematic geometry and recorded seam tests."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures/phase_boundary_candidate'
OUT.mkdir(parents=True, exist_ok=True)
records = json.loads((ROOT / 'results/strange_attractor_audit_20260929/results.json').read_text())
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
fig, axes = plt.subplots(1, 2, figsize=(174/25.4, 91/25.4),
    gridspec_kw={'width_ratios': [1, 1.25]})
fig.subplots_adjust(left=.035, right=.98, top=.85, bottom=.19, wspace=.43)
a, b = axes
a.set_title('A) Phase boundary', loc='left', fontsize=10, weight='bold', pad=19)
a.set_aspect('equal', adjustable='datalim'); a.axis('off')
theta = np.linspace(-np.pi, np.pi, 601)
a.plot(np.cos(theta), np.sin(theta), color='#555555', lw=1)
a.plot([-1.32, -.68], [0, 0], color='black', ls=':', lw=.8)
a.scatter([-1], [0], s=17, facecolor='white', edgecolor='black', zorder=5)
angle = .16  # Exaggerated spacing for the schematic only; no measured data changed.
for sign, color in [(1, '#00A8C6'), (-1, '#B135A8')]:
    px, py = -np.cos(angle), sign*np.sin(angle)
    a.scatter([px], [py], s=45, color=color, edgecolor='black', linewidth=.5, zorder=6)
    label = r'$h=1-\varepsilon$' if sign == 1 else r'$h=-1+\varepsilon$'
    a.annotate(label, xy=(px, py), xytext=(-1.40, sign*.68), ha='left', va='center',
        arrowprops={'arrowstyle':'-', 'color':'black', 'lw':.7,
                    'connectionstyle':'angle3,angleA=0,angleB=90'})
a.text(-1.40, 0, r'$-1\equiv 1$', ha='right', va='center', fontsize=9)
a.text(1.12, 0, r'$h=0$', va='center', fontsize=9)
a.set_xlim(-2.05, 1.55); a.set_ylim(-1.15, 1.15)
a.text(.5, -.16, r'$h=\theta/\pi$', transform=a.transAxes, ha='center', fontsize=9)
b.set_title('B) One-update separation', loc='left', fontsize=10, weight='bold', pad=19)
styles = {'OCNT-0494110': ('#0072B2', 'o', 'CYP2C9'),
          'OCNT-2328784': ('#D55E00', 's', 'CYP1A2')}
for rec in sorted(records, key=lambda r:r['molecule_id']):
    color, marker, cyp = styles[rec['molecule_id']]
    tests = sorted(rec['seam_test'], key=lambda v:v['initial_circular_distance'])
    b.plot([v['initial_circular_distance'] for v in tests],
           [v['next_circular_distance'] for v in tests],
           color=color, marker=marker, markersize=4, lw=1.15,
           label=f"{rec['molecule_id']} · {cyp}")
b.set_xscale('log'); b.set_xlim(6e-11, 7e-4); b.set_ylim(0, .0072)
b.set_xticks([1e-10, 1e-8, 1e-6, 1e-4])
b.set_yticks([0, .002, .004, .006])
fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-3,-3))
b.yaxis.set_major_formatter(fmt)
b.set_xlabel('Initial separation (dimensionless)', labelpad=7)
b.set_ylabel('Separation after one update\n(dimensionless)', labelpad=7)
b.spines[['top','right']].set_visible(False)
b.grid(axis='y', color='#DDDDDD', linewidth=.5)
b.legend(loc='lower left', frameon=False, fontsize=8, handlelength=1.7)
fig.savefig(OUT/'phase_boundary_candidate.png', dpi=600, facecolor='white')
fig.savefig(OUT/'phase_boundary_candidate_preview.png', dpi=180, facecolor='white')
plt.close(fig)
print(OUT/'phase_boundary_candidate.png')
