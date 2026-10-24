"""Render the manuscript's predictive evaluation workflow from reported scores."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'publication/figures/05_predictive_evaluation_workflow.png'
PREVIEW = ROOT / 'output/figures/predictive_evaluation_workflow/preview.png'
PREVIEW.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'text.color': 'black'})
fig = plt.figure(figsize=(18, 11.3), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,18), ylim=(0,11.3)); ax.axis('off')
colours = ['#007E98', '#7752A5', '#C37622']
fills = ['#F6FBFC', '#FAF8FC', '#FFFBF5']
xs = [2.65, 7.7, 12.75]; widths = [4.62, 4.62, 4.85]

def text(x,y,s,size=12,weight='normal',**kw):
    ax.text(x,y+3,s,fontsize=size,weight=weight,va='top',linespacing=1.22,**kw)

def box(x,y,w,h,c,fill):
    ax.add_patch(FancyBboxPatch((x,y+3),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',
                              facecolor=fill,edgecolor=c,linewidth=1.2))

box(2.65,7.57,9.67,.45,colours[0],fills[0])
box(12.75,7.57,4.85,.45,colours[2],fills[2])
text(7.485,7.91,'FITTING DATA',14,'bold',ha='center')
text(15.175,7.91,'RESERVED HOLDOUT',14,'bold',ha='center')
text(.4,7.27,'Method',14,'bold')
for x,title,sub in zip(xs,['1. Screen candidates','2. Confirm and select','3. Freeze and evaluate'],
                      ['Two scaffold folds','Five scaffold folds × two random seeds','Fixed final prediction systems']):
    text(x,7.27,title,14,'bold'); text(x,6.94,sub,12)

rows = [
('A) BP-Graph-CA','Backpropagation',
 'Screen ten transition-rule families for each\nof the four CYP targets.\nRetain three configurations per target.',
 'Train each finalist ten times. Use held-fold\npredictions to select models and fit their\ncombination. Build all four CYP prediction\nsystems.',
 'Four-target MA-ST-RAE: 0.748490.'),
('B) ES-Graph-CA','Evolutionary strategy',
 'Screen 32 optimiser configurations for CYP3A4.\nBest ST-RAE: 0.638768.\nMatched BP reference: 0.626897.',
 'Confirm the five best configurations.\nSelected CYP3A4 ST-RAE:\n0.659505 ± 0.043028.',
 'The experiment ended after confirmation.\nReserved-holdout evaluation was not performed.'),
('C) GA-Graph-CA','Genetic configuration\nsearch',
 'Screen four successive populations of 24\nmodel configurations for CYP3A4.\nBackpropagation trains the weights.',
 'Confirm four designs and the existing BP\nreference configuration.\nSelected CYP3A4 ST-RAE:\n0.596981 ± 0.042987.',
 'Four-target MA-ST-RAE: 0.743723.\nCombine the new CYP3A4 model with the\nexisting BP models for the other three\nCYP targets.'),
('D) GP-Graph-CA','Genetic transition-rule\nsearch',
 'Screen four successive populations of 24\ntransition rules for CYP3A4.\nBackpropagation trains the weights.',
 'Confirm five candidate rules.\nSelected CYP3A4 ST-RAE:\n0.598075 ± 0.036679.',
 'Four-target MA-ST-RAE: 0.745824.\nCombine the new CYP3A4 model with the\nexisting BP models for the other three\nCYP targets.')]
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
def wrap_cell(value, width):
    limit = width * fig.dpi
    prop = FontProperties(family='Arial', size=18)
    lines, line = [], ''
    for word in value.split():
        trial = (line+' '+word).strip()
        if line and renderer.get_text_width_height_descent(trial, prop, False)[0] > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    return '\n'.join(lines+[line])
for i,(name,desc,*cells) in enumerate(rows):
    top=6.48-i*2.22
    text(.25,top-.12,name,16,'bold'); text(.25,top-.50,desc,15)
    for j,cell in enumerate(cells):
        box(xs[j],top-2.02,widths[j],2.02,colours[j],fills[j])
        cell = wrap_cell(cell, widths[j]-.30)
        text(xs[j]+.14,top-.16,cell,18)
    for j in range(2):
        if i==1 and j==1: continue
        ax.add_patch(FancyArrowPatch((xs[j]+widths[j]+.055,top+2),
                                    (xs[j+1]-.055,top+2),arrowstyle='-|>',
                                    mutation_scale=13,color='#555555',linewidth=1.2))
fig.savefig(OUT,dpi=400,facecolor='white')
fig.savefig(PREVIEW,dpi=120,facecolor='white')
plt.close(fig)
print(OUT)
