"""Frozen-state numerical audit, with full-state plot archives and no training."""
import os, sys, json, math, hashlib, types, time, faulthandler
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/strange_attractor_audit_20260929'
CHECK = ROOT / 'results/production_kuramoto_sakaguchi_multiscale_ensemble_v2/runs/final_model/model.pt'

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))

def audit(**kw):
    model, data = kw['model'].double(), kw['data']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for p in model.parameters(): p.requires_grad_(False)
    cp = torch.load(CHECK, map_location='cpu', weights_only=False)
    hp = cp['hyperparameters']
    a,b,c,scale = [float(hp[k]) for k in ('dyn_a','dyn_b','dyn_c','update_scale')]
    wrap = lambda x: torch.atan2(torch.sin(math.pi*x),torch.cos(math.pi*x))/math.pi
    diff = lambda x,y: wrap(x-y)
    summaries=[]
    for idx,cyp in [(4309,0),(86,1)]:
        rec=data['train'][idx]; device=next(model.parameters()).device
        x=torch.as_tensor(rec['x'],dtype=torch.float64,device=device)
        x=model._chemical_features(x)
        src=torch.as_tensor(rec['src'],device=device); dst=torch.as_tensor(rec['dst'],device=device)
        edge=torch.as_tensor(rec['edge'],dtype=torch.float64,device=device)
        gate=torch.sigmoid(model.bond_gate(edge)/float(hp['bond_temperature']))
        n=len(x); hdim=model.bias.numel(); dim=n*hdim
        deg=torch.bincount(dst,minlength=n).clamp_min(1).to(x.dtype)[:,None]
        context=torch.zeros(4,dtype=x.dtype,device=device); context[cyp]=1
        const=model.chem(x)+model.context(context)+model.bias
        wself=model.self_layer.weight.T; wn=model.neighbour.weight.T; wf=model.frequency_drive.weight.T
        bond=model.bond(edge); lag=math.pi*(c-.5)
        def step(h):
            agg=torch.zeros_like(h)
            agg.index_add_(-2,dst,gate*(h[...,src,:]@wn)+bond)
            reaction=torch.tanh(h@wself+agg/deg+const)
            freq=torch.tanh(model.frequency_drive(reaction))
            coupling=torch.zeros_like(h)
            coupling.index_add_(-2,dst,gate*torch.sin(math.pi*(h[...,src,:]-h[...,dst,:])-lag))
            return wrap(h+scale/math.pi*(a*freq+b*coupling/deg))
        def tangent(h,v):
            agg=torch.zeros_like(h); agg.index_add_(0,dst,gate*(h[src]@wn)+bond)
            r=torch.tanh(h@wself+agg/deg+const)
            f=torch.tanh(model.frequency_drive(r))
            da=torch.zeros_like(v); da.index_add_(1,dst,gate*(v[:,src]@wn))
            dr=(v@wself+da/deg)*(1-r*r)
            df=(dr@wf)*(1-f*f)
            dc=torch.zeros_like(v)
            dc.index_add_(1,dst,gate*torch.cos(math.pi*(h[src]-h[dst])-lag)*math.pi*(v[:,src]-v[:,dst]))
            return v+scale/math.pi*(a*df+b*dc/deg)
        h=model.initial_node_state(rec)
        original_error=float(torch.max(torch.abs(step(h)-model.kuramoto_step(rec,cyp,h))))
        assert original_error<1e-12, original_error
        torch.manual_seed(20260929+cyp)
        v=torch.randn((1,n,hdim),device=device,dtype=x.dtype)
        _,ad=torch.autograd.functional.jvp(step,h,v[0])
        derivative_error=float((tangent(h,v)[0]-ad).abs().max())
        assert derivative_error<1e-10,derivative_error
        if os.environ.get('AUDIT_FULL_SPECTRUM') == '1':
            archive=np.load(OUT/f"{rec['name']}_{cyp}_trajectories.npz")
            h=torch.as_tensor(archive['states'][-1],device=device)
            torch.manual_seed(862+cyp)
            q,_=torch.linalg.qr(torch.randn(dim,dim,device=device,dtype=x.dtype))
            full=[]
            for t in range(1,11001):
                q=tangent(h,q.T.reshape(dim,n,hdim)).reshape(dim,dim).T
                h=step(h)
                if t%10==0:
                    q,r=torch.linalg.qr(q)
                    if t>1000: full.append(torch.log(torch.diag(r).abs()).cpu().numpy()/10)
                if t%1000==0:
                    write('full_spectrum_progress.json',dict(molecule=rec['name'],generation=t,total=11000))
            full=np.stack(full); spectrum=np.sort(full.mean(0))[::-1]
            cumulative=np.cumsum(spectrum); nonnegative=np.flatnonzero(cumulative>=0)
            j=int(nonnegative[-1]+1) if len(nonnegative) else 0
            ky=float(j+cumulative[j-1]/abs(spectrum[j])) if 0<j<dim else float(j)
            result=dict(molecule_id=rec['name'],state_dimension=dim,measured_generations=10000,
                tangent_alignment_generations=1000,spectrum=spectrum.tolist(),
                positive_exponents=int((spectrum>0).sum()),sum_exponents=float(spectrum.sum()),
                kaplan_yorke_estimate=ky,
                caveat='Piecewise derivative spectrum; Kaplan-Yorke is a heuristic, not a measured fractal dimension.')
            write(f"{rec['name']}_{cyp}_full_spectrum.json",result)
            np.savez_compressed(OUT/f"{rec['name']}_{cyp}_full_spectrum.npz",local_exponents=full)
            continue
        for _ in range(5000): h=step(h)
        # The learned frequency also depends on raw wrapped coordinates.
        seam=[]
        for eps in (1e-4,1e-6,1e-8,1e-10):
            left=h.clone(); right=h.clone(); left[0,0]=1-eps; right[0,0]=-1+eps
            seam.append(dict(epsilon=eps,initial_circular_distance=float(torch.linalg.vector_norm(diff(left,right))),
                             next_circular_distance=float(torch.linalg.vector_norm(diff(step(left),step(right))))))
        q,_=torch.linalg.qr(torch.randn(dim,8,device=device,dtype=x.dtype))
        eps=torch.tensor([1e-5,1e-7,1e-9],device=device,dtype=x.dtype)
        direction=torch.randn_like(h); direction/=torch.linalg.vector_norm(direction)
        companions=wrap(h[None]+eps[:,None,None]*direction)
        displaced=[]
        for radius in (0.1,0.5,1.,2.):
            direction=torch.randn_like(h); direction/=torch.linalg.vector_norm(direction)
            displaced.append(wrap(h+radius*direction))
        batch=torch.cat([h[None],companions,torch.stack(displaced)])
        traj=[h.cpu().numpy()]; basins=[]; logs=[]; finite=[]; crossings=0
        started=time.time()
        for t in range(1,20001):
            v=tangent(batch[0],q.T.reshape(8,n,hdim))
            q=v.reshape(8,dim).T
            previous=batch[0].clone(); batch=step(batch)
            crossings+=int(((batch[0]-previous).abs()>1).sum())
            traj.append(batch[0].cpu().numpy())
            if t%10==0:
                q,r=torch.linalg.qr(q)
                logs.append(torch.log(torch.diag(r).abs()).cpu().numpy()/10)
                delta=diff(batch[1:4],batch[0]); norm=torch.linalg.vector_norm(delta.flatten(1),dim=1)
                finite.append((torch.log(norm/eps)/10).cpu().numpy())
                batch[1:4]=wrap(batch[0]+eps[:,None,None]*delta/norm[:,None,None])
                basins.append(batch[4:].cpu().numpy())
            if t%2000==0:
                write('progress.json',dict(status='running',molecule=rec['name'],generations=t,total=20000,
                    elapsed_seconds=time.time()-started,mean_tangent_exponents=np.mean(logs,axis=0).tolist()))
        trajectory=np.stack(traj); basin=np.stack(basins); local=np.stack(logs); fd=np.stack(finite)
        name=f"{rec['name']}_{cyp}"
        np.savez_compressed(OUT/f'{name}_trajectories.npz',states=trajectory,basin_states=basin,
            tangent_local_exponents=local,finite_difference_local_exponents=fd,epsilon=eps.cpu().numpy(),
            src=rec['src'],dst=rec['dst'],atom_features=rec['x'],edge_features=rec['edge'],
            smiles=rec['canonical_smiles'],burn_in=5000,generations=20000,basin_stride=10)
        # Full circular embedding; compare late independent windows to calibrate basin distances.
        embed=lambda z: np.concatenate([np.sin(math.pi*z),np.cos(math.pi*z)],axis=-1).reshape(len(z),-1)
        z=embed(trajectory[::10]); rng=np.random.default_rng(731+cyp)
        proj=rng.normal(size=(z.shape[1],32)); proj/=np.linalg.norm(proj,axis=0)
        projected=np.einsum('ij,jk->ik',z,proj,optimize=False)
        distance=lambda u,v: float(np.mean(np.abs(np.sort(u,axis=0)-np.sort(v,axis=0))))
        refdist=distance(projected[1000:1500],projected[1500:2000])
        basin_results=[]
        for k,radius in enumerate((.1,.5,1.,2.)):
            bz=np.einsum('ij,jk->ik',embed(basin[:,k]),proj,optimize=False)
            late=distance(bz[1500:2000],projected[1500:2000])
            basin_results.append(dict(radius=radius,late_distance=late,reference_window_distance=refdist,
                                      late_to_reference_window_ratio=late/refdist))
        # No low-dimensional projection is treated as proof of fractality.
        write('progress.json',dict(status='geometry_analysis',molecule=rec['name']))
        centered=z-z.mean(0)
        torch.manual_seed(991+cyp)
        _,_,basis=torch.pca_lowrank(torch.as_tensor(centered,device=device),q=24,center=False,niter=6)
        vt=basis.T.cpu().numpy()
        dimension=[]; curves={}
        for width in (6,12,24,z.shape[1]):
            zz=np.einsum('ij,jk->ik',centered,vt[:width].T,optimize=False) if width<z.shape[1] else z
            for theiler in (10,50,100):
                ii=rng.integers(0,len(z),100000); jj=rng.integers(0,len(z),100000)
                ok=np.abs(ii-jj)>theiler; ii=ii[ok]; jj=jj[ok]
                ds=np.concatenate([np.linalg.norm(zz[ii[k:k+2000]]-zz[jj[k:k+2000]],axis=1) for k in range(0,len(ii),2000)])
                radii=np.quantile(ds,np.linspace(.01,.25,25)); corr=np.array([(ds<=r).mean() for r in radii])
                slopes=np.diff(np.log(corr))/np.diff(np.log(radii))
                lx=np.log(radii); ly=np.log(corr); lx-=lx.mean(); ly-=ly.mean()
                dimension.append(dict(embedding_width=width,theiler_generations=theiler*10,
                    fitted_slope=float(np.sum(lx*ly)/np.sum(lx*lx)),
                    local_slope_min=float(slopes.min()),local_slope_max=float(slopes.max()),
                    log10_radius_span=float(np.log10(radii[-1]/radii[0]))))
                curves[f'{width}_{theiler}_radius']=radii; curves[f'{width}_{theiler}_correlation']=corr
        np.savez_compressed(OUT/f'{name}_geometry.npz',**curves)
        windows=local.reshape(4,500,8).mean(1)
        summary=dict(molecule_id=rec['name'],cyp_index=cyp,state_dimension=dim,
            one_step_agreement=original_error,jvp_agreement=derivative_error,
            tangent_exponents=local.mean(0).tolist(),tangent_5000_generation_windows=windows.tolist(),
            finite_difference_exponents=fd.mean(0).tolist(),branch_crossings=crossings,seam_test=seam,
            maximum_absolute_state=float(np.abs(trajectory).max()),basin=basin_results,dimension=dimension,
            measured_generations=20000,burn_in=5000,elapsed_seconds=time.time()-started)
        summaries.append(summary); write('results.json',summaries)
    if os.environ.get('AUDIT_FULL_SPECTRUM') == '1':
        write('full_spectrum_progress.json',dict(status='complete',cases=2))
    else:
        write('progress.json',dict(status='complete',cases=len(summaries)))

if __name__=='__main__':
    faulthandler.dump_traceback_later(60,repeat=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for key in list(os.environ):
        if key.startswith('SME_'): del os.environ[key]
    os.environ.update(SME_DEVICE='cuda',SME_CHECKPOINT=str(CHECK),SME_EXTENDED_OUTPUT=str(OUT),
                      SME_SCREENING_CSV=str(ROOT/'results/long_horizon_attractor_campaign_v1/spectrum_setup/selected_candidates.csv'),
                      SME_RUN_NAME='strange_attractor_audit_20260929/setup')
    write('full_spectrum_provenance.json' if os.environ.get('AUDIT_FULL_SPECTRUM')=='1' else 'provenance.json',dict(checkpoint=str(CHECK),checkpoint_sha256=hashlib.sha256(CHECK.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training=False,
        precision='float64',burn_in=5000,measured_generations=20000))
    stub=types.ModuleType('run_extended_coupled_map_dynamics'); stub.run_extended_analysis=audit
    sys.modules['run_extended_coupled_map_dynamics']=stub
    import run_graph_ca_visual_prototype as runner
    runner.train(extended_dynamics=True)
