"""Reproduce Appendix E from recorded states, without training or scoring.

First run: --data-root ORIGINAL_CHECKOUT. Subsequent runs: --saved-data.
The compact archives retain the plotted channel, atom order, bonds and conformer.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[1]
RULES=[('gated_residual','Gated residual'),
 ('inertial_reaction_diffusion','Inertial reaction-diffusion'),
 ('fitzhugh_nagumo','FitzHugh-Nagumo'),
 ('kuramoto_sakaguchi','Kuramoto-Sakaguchi'),
 ('delayed_memory','Delayed memory'),('gray_scott','Gray-Scott'),
 ('coupled_map','Coupled map'),('activator_inhibitor','Activator-inhibitor'),
 ('damped_symplectic','Damped symplectic'),
 ('conservative_graph_flux','Conservative graph flux')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--saved-data',action='store_true')
    args=parser.parse_args()
    data_dir=ROOT/'publication/figure_data/appendix_e_gallery'
    figures=ROOT/'publication/figures/appendix_e_gallery'
    previews=ROOT/'output/figures/appendix_e_gallery'
    for folder in (data_dir,figures,previews): folder.mkdir(parents=True,exist_ok=True)
    if not args.saved_data:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        if args.data_root is None: parser.error('--data-root is required on first run')
        train=args.data_root/'data/openadmet-cyp-challenge-2026/cyp-challenge-TRAIN_inhibition.csv'
        with train.open() as handle:
            smiles={r['Molecule_Name']:r['SMILES'] for r in csv.DictReader(handle)}
    manifest=[]
    plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':'black',
        'axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black'})
    cyclic=LinearSegmentedColormap.from_list('phase',
        ['#00cbd6','#4364d8','#a83bca','#eb278b','#f5b34a','#95d54d','#00cbd6'])
    linear=LinearSegmentedColormap.from_list('state',
        ['#224993','#26bfd0','#f5f1df','#ed60a6','#862b93'])
    for page,(rule,title) in enumerate(RULES,1):
        cases=[]
        if not args.saved_data:
            source_dir=args.data_root/'results/long_horizon_attractor_campaign_v1/base_trajectories'/rule
            if rule=='conservative_graph_flux':
                source_dir=args.data_root/'results/figure2_ten_rules_20261017/conservative_graph_flux'
            metadata=json.loads((source_dir/'trajectory_generation_complete.json').read_text())
            checkpoint=Path(metadata['checkpoint'])
            if not checkpoint.is_absolute(): checkpoint=args.data_root/checkpoint
            log=checkpoint.parent/'console.log'
            # Verify these are gradient-trained fixed-rule production checkpoints.
            assert 'mean_raw_gradient_norm' in log.read_text(encoding='utf-8'),log
        for row in range(3):
            saved=data_dir/f'{page:02d}_{rule}_{row+1}.npz'
            if not args.saved_data:
                source=source_dir/'node_trajectories'/f'case_{row+1:03d}.npz'
                with np.load(source,allow_pickle=False) as a:
                    values=a['trajectory'][:1001,:,0].copy()
                    molecule,target=str(a['molecule_id']),str(a['cyp_target'])
                mol=Chem.MolFromSmiles(smiles[molecule]); embedded=Chem.AddHs(mol)
                seed=260830+row
                params=AllChem.ETKDGv3(); params.randomSeed=seed
                assert AllChem.EmbedMolecule(embedded,params)==0,molecule
                AllChem.MMFFOptimizeMolecule(embedded,maxIters=500)
                embedded=Chem.RemoveHs(embedded)
                xyz=embedded.GetConformer().GetPositions(); xyz-=xyz.mean(axis=0)
                bonds=np.array([(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()])
                assert len(xyz)==values.shape[1] and np.isfinite(values).all()
                provenance={'source':str(source.relative_to(args.data_root)),
                    'source_sha256':sha(source),'checkpoint':str(checkpoint),
                    'training_log_sha256':sha(log),'training':'BP-Graph-CA (fixed rule; backpropagation)',
                    'rule':rule,'molecule':molecule,'cyp_context':target,'channel':1,
                    'conformer_seed':seed,'display_rotations':3,'generations':len(values)-1}
                np.savez_compressed(saved,values=values,xyz=xyz,bonds=bonds,
                    molecule=molecule,target=target,provenance=json.dumps(provenance))
            with np.load(saved,allow_pickle=False) as a:
                values=a['values']; xyz=a['xyz']; bonds=a['bonds']
                molecule,target=str(a['molecule']),str(a['target'])
                provenance=json.loads(str(a['provenance']))
            manifest.append(dict(page=page,row=row+1,file=saved.name,sha256=sha(saved),**provenance))
            cases.append((values,xyz,bonds,molecule,target))
        phase=rule=='kuramoto_sakaguchi'
        limit=1 if rule!='conservative_graph_flux' else max(1,float(np.ceil(max(np.max(np.abs(c[0])) for c in cases))))
        cmap=cyclic if phase else linear; norm=Normalize(-limit,limit)
        fig,axes=plt.subplots(3,2,figsize=(8.27,11.69))
        fig.subplots_adjust(left=.10,right=.92,top=.875,bottom=.16,hspace=.44,wspace=.28)
        fig.text(.10,.971,f'E{page}. {title}',fontsize=15,weight='bold')
        fig.text(.10,.944,'Training: BP-Graph-CA (fixed rule; backpropagation)',fontsize=11)
        fig.text(.29,.914,'Flat atom-generation view',ha='center',fontsize=12)
        fig.text(.75,.914,'Rotating molecular trace',ha='center',fontsize=12)
        for row,(values,xyz,bonds,molecule,target) in enumerate(cases):
            end=len(values)-1
            ax=axes[row,0]
            ax.imshow(values,aspect='auto',interpolation='nearest',origin='upper',
                extent=(.5,len(xyz)+.5,end+.5,-.5),cmap=cmap,norm=norm)
            ax.set_title(f'{chr(65+row*2)}) {molecule} | {target}',loc='left',fontsize=10.5,pad=8)
            ax.set_xlabel('Atom index'); ax.set_ylabel('Generation')
            ax.set_xticks(sorted(set([1,len(xyz)//2,len(xyz)])))
            ax.set_yticks([0,end//2,end]); ax.tick_params(labelsize=10)
            ax=axes[row,1]; generations=np.arange(len(values))
            angle=generations*2*np.pi*3/end
            x=np.cos(angle)[:,None]*xyz[None,:,0]+np.sin(angle)[:,None]*xyz[None,:,2]
            z=-np.sin(angle)[:,None]*xyz[None,:,0]+np.cos(angle)[:,None]*xyz[None,:,2]
            xp=x+.22*z; xp/=max(np.max(np.abs(xp)),1e-10)
            # Molecular vertical offsets are bounded to 2.5% of the displayed time range.
            yp=generations[:,None]-xyz[None,:,1]/max(np.max(np.abs(xyz[:,1])),1e-10)*end*.025
            for atom in range(len(xyz)):
                points=np.stack([xp[:,atom],yp[:,atom]],axis=-1)
                ax.add_collection(LineCollection(np.stack([points[:-1],points[1:]],axis=1),
                    colors=cmap(norm(values[:-1,atom])),linewidths=.85,alpha=.8,rasterized=True))
            for generation in (0,end//2,end):
                for a,b in bonds:
                    p=np.array([xp[generation,a],yp[generation,a]])
                    q=np.array([xp[generation,b],yp[generation,b]])
                    f=np.linspace(0,1,13); pts=p+f[:,None]*(q-p)
                    col=(1-f[:-1,None])*cmap(norm(values[generation,a]))+f[:-1,None]*cmap(norm(values[generation,b]))
                    ax.add_collection(LineCollection(np.stack([pts[:-1],pts[1:]],axis=1),colors=col,linewidths=1.3))
                ax.scatter(xp[generation],yp[generation],s=13,c=values[generation],cmap=cmap,norm=norm,
                    edgecolors='#333333',linewidths=.25,zorder=5)
            ax.set_xlim(-1.12,1.12); ax.set_ylim(end*1.04,-end*.04)
            ax.set_title(f'{chr(66+row*2)}) Same trajectory',loc='left',fontsize=10.5,pad=8)
            ax.set_yticks([0,end//2,end]); ax.set_xticks([]); ax.tick_params(labelsize=10)
            ax.spines[['top','right','bottom']].set_visible(False)
        cax=fig.add_axes([.28,.103,.44,.014])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',
            ticks=[-limit,0,limit],label='First phase channel / π' if phase else 'First state channel (dimensionless)')
        filename=f'E{page:02d}_{rule}.png'
        fig.savefig(figures/filename,dpi=400,facecolor='white')
        fig.savefig(previews/filename,dpi=105,facecolor='white')
        plt.close(fig)
        print(f'Completed E{page}: {title}',flush=True)
    (data_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
