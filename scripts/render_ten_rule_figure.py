"""Reproduce Figure 2 from archived states and one frozen-model rollout.

Run with a Python environment containing numpy, matplotlib, torch and pandas.
--data-root points to the checkout containing the original result archives.
No model fitting, prediction scoring or model promotion is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RULES = [
    ('gated_residual', 'Gated residual'),
    ('inertial_reaction_diffusion', 'Inertial reaction-diffusion'),
    ('fitzhugh_nagumo', 'FitzHugh-Nagumo'),
    ('kuramoto_sakaguchi', 'Kuramoto-Sakaguchi'),
    ('delayed_memory', 'Delayed memory'),
    ('gray_scott', 'Gray-Scott'),
    ('coupled_map', 'Coupled map'),
    ('activator_inhibitor', 'Activator-inhibitor'),
    ('damped_symplectic', 'Damped symplectic'),
    ('conservative_graph_flux', 'Conservative graph flux'),
]
SELECTED = {'fitzhugh_nagumo', 'delayed_memory', 'gray_scott',
            'damped_symplectic', 'conservative_graph_flux'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--linkedin', action='store_true', help='Restyle saved figure data without running calculations')
    parser.add_argument('--saved-data', action='store_true', help='Render the publication figure from its saved arrays only')
    args = parser.parse_args()
    reuse = args.linkedin or args.saved_data
    root, dest = args.data_root.resolve(), args.output_root.resolve()
    archive = root / 'results/long_horizon_attractor_campaign_v1/base_trajectories'
    run = root / 'results/figure2_ten_rules_20261017'
    run.mkdir(parents=True, exist_ok=True)
    generated = run / 'conservative_graph_flux'
    checkpoint = root / 'results/production_conservative_graph_flux_enhanced_v3/runs/final_model/model.pt'
    ck_hash = digest(checkpoint) if not reuse else None
    if not reuse and not (generated / 'trajectory_generation_complete.json').exists():
        env = {k:v for k,v in os.environ.items() if not k.startswith('SME_')}
        env.update({
            'SME_CHECKPOINT': str(checkpoint),
            'SME_GRAPH_CACHE': str(root / 'tmp/strange_matter_graph_ca_graphs.pkl'),
            'SME_SCREENING_CSV': str(checkpoint.parent / 'validation_dynamics.csv'),
            'SME_EXTENDED_OUTPUT': str(generated),
            'SME_EXTENDED_GENERATIONS': '256', 'SME_EXTENDED_BURN_IN': '128',
            'SME_EXTENDED_CANDIDATES': '4', 'SME_EXTENDED_SAVE_NODE_TRAJECTORIES': '1',
            'SME_EXTENDED_TRAJECTORY_ONLY': '1',
            'SME_RUN_NAME': 'figure2_ten_rules_20261017/setup',
        })
        subprocess.run([sys.executable, str(root / 'scripts/run_graph_ca_visual_prototype.py'),
                        'extended-dynamics'], env=env, cwd=root, check=True)
    if not reuse:
        assert digest(checkpoint) == ck_hash, 'Frozen checkpoint changed'
    states_dir = dest / 'publication/figure_data/figure2_ten_rules'
    figures = dest / 'publication/figures'
    pdfs = dest / 'output/pdf'
    for p in (states_dir, figures, pdfs):
        p.mkdir(parents=True, exist_ok=True)
    panels, records = [], []
    for index, (rule, title) in enumerate(RULES):
        folder = generated if rule == 'conservative_graph_flux' else archive / rule
        source = folder / 'node_trajectories/case_001.npz'
        if reuse:
            source = states_dir / f'{index+1:02d}_{rule}.npz'
        with np.load(source, allow_pickle=False) as data:
            states = data['trajectory'][:257].copy()
            molecule = str(data['molecule_id'])
            cyp = str(data['cyp_target'])
            training_index = int(data['training_index'])
        assert states.ndim == 3 and len(states) == 257 and np.isfinite(states).all()
        assert cyp == 'CYP1A2'
        if rule == 'conservative_graph_flux':
            conservation_error = float(np.abs(states.sum(1)-states[0].sum(0)).max())
            assert conservation_error < 1e-4, 'Unexpected loss of graph-flux conservation'
        # Common colour limits must represent the data without clipping.
        if rule != 'conservative_graph_flux':
            assert np.min(states[:,:,0]) >= -1.000001 and np.max(states[:,:,0]) <= 1.000001
        saved = states_dir / f'{index+1:02d}_{rule}.npz'
        if reuse:
            panels.append((states, molecule, rule, title))
            continue
        np.savez_compressed(saved, trajectory=states, molecule_id=molecule, cyp_target=cyp,
                            training_index=training_index, rule=rule, generations=np.arange(257))
        with np.load(saved) as checked:
            assert np.array_equal(checked['trajectory'], states)
        metadata = json.loads((folder / 'trajectory_generation_complete.json').read_text())
        records.append(dict(panel=chr(65+index), rule=rule, molecule_id=molecule,
                            cyp_context=cyp, shape=list(states.shape),
                            source=str(source.relative_to(root)), source_sha256=digest(source),
                            checkpoint=metadata['checkpoint'],
                            data_file=saved.name, data_sha256=digest(saved),
                            bp_submission_family=rule in SELECTED,
                            colour_limits=([-float(np.ceil(np.abs(states[:,:,0]).max())),
                                            float(np.ceil(np.abs(states[:,:,0]).max()))]
                                           if rule == 'conservative_graph_flux' else [-1,1]),
                            conservation_error=(conservation_error
                                                if rule == 'conservative_graph_flux' else None),
                            channel_1_min=float(states[:,:,0].min()),
                            channel_1_max=float(states[:,:,0].max())))
        panels.append((states, molecule, rule, title))
    plt.rcParams.update({'font.family':'Arial', 'font.size':9, 'axes.labelsize':9,
                         'xtick.labelsize':9, 'ytick.labelsize':9, 'pdf.fonttype':42,
                         'text.color':'#000000', 'axes.edgecolor':'#000000',
                         'axes.labelcolor':'#000000', 'axes.titlecolor':'#000000',
                         'xtick.color':'#000000', 'ytick.color':'#000000'})
    if args.linkedin:
        plt.rcParams.update({'figure.facecolor':'#08182D', 'axes.facecolor':'#08182D',
                             'text.color':'#C9DDF0', 'axes.labelcolor':'#80DFFF',
                             'xtick.color':'#C9DDF0', 'ytick.color':'#C9DDF0',
                             'axes.edgecolor':'#32CFE8', 'axes.titlecolor':'#00E5FF'})
    print('Validated and saved ten trajectory arrays; rendering.', flush=True)
    fig, axes = plt.subplots(5, 2, figsize=(174/25.4, 200/25.4))
    fig.subplots_adjust(left=.087, right=.987, bottom=.14, top=.94, hspace=.83, wspace=.24)
    for i,(states,molecule,rule,title) in enumerate(panels):
        row,col = i%5,i//5
        ax = axes[row,col]
        count = states.shape[1]
        cyclic = rule == 'kuramoto_sakaguchi'
        limit = float(np.ceil(np.abs(states[:,:,0]).max())) if rule == 'conservative_graph_flux' else 1
        im = ax.imshow(states[:,:,0], cmap='twilight_shifted' if cyclic else 'RdBu_r',
                       vmin=-limit,vmax=limit,origin='upper',aspect='auto',interpolation='nearest',
                       extent=(.5,count+.5,256.5,-.5),rasterized=True)
        ax.set_title(f'{chr(65+i)})  {title}', loc='left', fontsize=10, fontweight='bold', pad=20)
        ax.text(0,1.055,f'{molecule}  |  CYP1A2', transform=ax.transAxes,fontsize=8.5,va='bottom')
        ax.set_xticks(sorted(set([1,(count+1)//2,count])))
        ax.set_yticks([0,128,256])
        if col == 1:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=2.5,pad=2,width=.6)
        if row == 4:
            ax.set_xlabel('Atom index',labelpad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        if rule == 'conservative_graph_flux':
            flux_im, flux_limit = im, limit
        elif cyclic:
            cyclic_im = im
        else:
            linear_im = im
    fig.text(.013,.55,'Generation',rotation=90,va='center',fontsize=10)
    for bounds, im, title, limit in [([.09,.045,.245,.011],linear_im,'State (A-C, E-I)',1),
                                     ([.415,.045,.245,.011],cyclic_im,'Phase / π (D)',1),
                                     ([.74,.045,.245,.011],flux_im,'State (J)',flux_limit)]:
        cb = fig.colorbar(im,cax=fig.add_axes(bounds),orientation='horizontal',ticks=[-limit,0,limit])
        cb.ax.set_title(title,fontsize=9,pad=4)
        cb.ax.tick_params(labelsize=9,length=2,pad=2)
        cb.outline.set_linewidth(.6)
    png = figures / '02_ten_rule_1d_cellular_evolution.png'
    if args.linkedin:
        png = dest / 'output/social/ten_rule_molecular_space_time_linkedin.png'
        png.parent.mkdir(parents=True, exist_ok=True)
    print('Saving figure PNG.', flush=True)
    fig.savefig(png,dpi=600,facecolor='#08182D' if args.linkedin else 'white')
    plt.close(fig)
    if reuse:
        print(json.dumps(dict(png=str(png),source='Saved publication arrays; values and colour scales unchanged'),indent=2))
        return
    manifest = dict(generations=[0,256],channel=1,normalisation='none',
                    interpolation='nearest',figure_mm=[174,200],png_dpi=600,
                    font_pt=dict(title=10,axis_and_ticks=9,molecule_id=8.5),
                    panel_order='A-E down left column; F-J down right column',
                    conservative_checkpoint_sha256=ck_hash,panels=records)
    (states_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(png=str(png),data=str(states_dir)),indent=2))

if __name__ == '__main__':
    main()
