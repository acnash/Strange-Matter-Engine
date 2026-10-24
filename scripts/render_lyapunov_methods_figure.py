"""Render the methods figure from compact, archived diagnostics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'publication/figure_data/lyapunov_methods'
OUT=ROOT/'publication/figures'
cyan,pink,purple='#008fba','#ce328b','#7053b6'
plt.rcParams.update({'font.family':'Arial','font.size':18,'text.color':'black',
 'axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black'})
a=np.load(DATA/'direct.npz');b=np.load(DATA/'renormalised.npz')
fig=plt.figure(figsize=(15,7.5))
schemes=[];plots=[]
for x,title in zip([.075,.415,.755],['A) Direct perturbation','B) Repeated renormalisation','C) Lyapunov spectrum']):
    fig.text(x,.94,title,fontsize=18,weight='bold')
    s=fig.add_axes([x,.57,.225,.30]);s.set(xlim=(0,1),ylim=(0,1));s.axis('off');schemes.append(s)
    p=fig.add_axes([x,.12,.225,.33]);plots.append(p)
    p.spines[['top','right']].set_visible(False);p.grid(axis='y',color='#e6e6e6',lw=.6)
s=schemes[0];t=np.linspace(.05,.95,180);ref=.28+.05*np.sin(t*7)
other=ref+.035+.42*((t-.05)/.9)**1.8
s.plot(t,ref,color=cyan,lw=2.7);s.plot(t,other,color=pink,lw=2.7)
s.text(.10,.08,'Reference',fontsize=18,color='black')
s.text(.36,.81,'Companion',fontsize=18,color='black')
s.annotate('',xy=(.89,other[-12]),xytext=(.89,ref[-12]),arrowprops={'arrowstyle':'<->','color':'black'})
s.text(.92,.48,'d',fontsize=18)
s=schemes[1];eps=.20;s.axhline(eps,xmin=.03,xmax=.97,color='#888888',ls=':',lw=1)
for i,end in enumerate([.62,.79,.66]):
    start=.07+i*.30;stop=start+.24;tt=np.linspace(start,stop,60)
    s.plot(tt,eps+(end-eps)*((tt-start)/(stop-start))**1.5,color=pink,lw=2.7)
    s.annotate('',xy=(stop,eps),xytext=(stop,end),arrowprops={'arrowstyle':'->','color':'black','linestyle':'--'})
    s.text((start+stop)/2,.025,r'$\tau$',ha='center',fontsize=18)
s.text(.005,eps,r'$\varepsilon$',ha='right',va='center',fontsize=18)
s=schemes[2]
s.set_position([.72,.51,.27,.38])
s.set(xlim=(0,3.5),ylim=(0,2.5));s.set_aspect('equal')
radius=.32
start=np.array([.63,1.82]); evolved=np.array([2.65,1.82])
reset=np.array([.63,.55]); qr=np.array([2.65,.55])
transform=np.array([[1.45,.85],[.30,.80]])
v1,v2=transform[:,0]*radius,transform[:,1]*radius
q1=v1/np.linalg.norm(v1); proj=q1*np.dot(q1,v2)
residual=v2-proj;q2=residual/np.linalg.norm(residual)
def vector(origin,v,color,alpha=1):
    s.annotate('',xy=origin+v,xytext=origin,arrowprops={'arrowstyle':'->','color':color,'lw':2.4,'alpha':alpha,'shrinkA':0,'shrinkB':0})
theta=np.linspace(0,2*np.pi,120)
circle=radius*np.stack([np.cos(theta),np.sin(theta)])
for origin,vecs,shape in [(start,[np.array([radius,0]),np.array([0,radius])],circle),
                         (evolved,[v1,v2],transform@circle),
                         (reset,[radius*q1,radius*q2],circle)]:
    s.add_patch(Polygon(shape.T+origin,closed=True,facecolor='#edf4f7',edgecolor='#bac6cd',lw=1))
    for v,col in zip(vecs,[cyan,pink]):vector(origin,v,col)
    s.scatter(*origin,color='black',s=12,zorder=5)
vector(qr,v1,cyan)
vector(qr,v2,pink,alpha=.25)
s.plot([qr[0]+v2[0],qr[0]+proj[0]],[qr[1]+v2[1],qr[1]+proj[1]],color=pink,lw=2.6)
s.plot([qr[0],qr[0]+proj[0]],[qr[1],qr[1]+proj[1]],color='#555555',ls=':',lw=1)
corner=qr+proj
square=np.array([corner-.07*q1,corner-.07*q1+.07*q2,corner+.07*q2])
s.plot(square[:,0],square[:,1],color='#555555',lw=.9)
s.text(qr[0]+.22,qr[1]-.20,r'$r_{11}$',fontsize=16,color='black')
s.text(qr[0]+.42,qr[1]+.18,r'$r_{22}$',fontsize=16,color='black')
for origin,label in [(start,'Start'),(evolved,'Evolve'),(qr,'Measure (QR)'),(reset,'Reset')]:
    s.text(origin[0],origin[1]-.52,label,ha='center',fontsize=15)
for aa,bb in [((1.12,1.82),(2.02,1.82)),((3.25,1.47),(3.25,1.02)),((2.12,.55),(1.12,.55)),((.15,.80),(.15,1.50))]:
    s.annotate('',xy=bb,xytext=aa,arrowprops={'arrowstyle':'->','color':'black','lw':1.2})
p=plots[0]
for curve in a['separation']:p.semilogy(np.arange(len(curve)),curve,color=cyan,alpha=.5,lw=1)
p.set(xlabel='Generation',ylabel='Full-state separation',xlim=(0,5000),xticks=[0,2500,5000])
p=plots[1];time=(np.arange(b['cumulative_exponents'].shape[1])+1)*int(b['interval'])
for curve in b['cumulative_exponents']:p.plot(time,curve,color=pink,alpha=.5,lw=1)
p.axhline(0,color='#777777',lw=.8)
p.set(xlabel='Generations after burn-in',ylabel=r'Running $\lambda_1$ (generation$^{-1}$)',xlim=(0,4000),xticks=[0,2000,4000])
p.yaxis.labelpad=5
p=plots[2];finals=np.stack([np.load(DATA/f'spectrum_{i}.npz')['cumulative_spectra'][-1] for i in [1,2]])
for values in finals:p.plot(np.arange(1,9),values,'o-',color=purple,alpha=.8,markersize=4,lw=1.2)
p.axhline(0,color='#777777',lw=.8)
p.set(xlabel='Exponent index',ylabel=r'$\lambda_i$ (generation$^{-1}$)',xticks=range(1,9),ylim=(-.005,finals.max()*1.18))
OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/'04_lyapunov_methods.png',dpi=500,facecolor='white')
preview=ROOT/'output/figures/lyapunov_methods';preview.mkdir(parents=True,exist_ok=True)
fig.savefig(preview/'preview.png',dpi=125,facecolor='white')
