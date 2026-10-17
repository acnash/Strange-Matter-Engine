"""Prepare the locally edited manuscript for vector-quality browser printing."""
from pathlib import Path
import sys, re, importlib.util
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tmp/pdfs/render_dependencies'))
import markdown
from latex2mathml.converter import convert
source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT/'publication/publication.md'
text = source.read_text(encoding='utf-8')
equations = []
def equation(m):
    equations.append(convert(m.group(1), display='block'))
    return '\n\nEQUATIONPLACEHOLDER'+str(len(equations)-1)+'END\n\n'
text = re.sub(r'```math\s*\n(.*?)\n```', equation, text, flags=re.S)
body = markdown.markdown(text, extensions=['tables','fenced_code','sane_lists'])
for i, math in enumerate(equations):
    body = body.replace('<p>EQUATIONPLACEHOLDER'+str(i)+'END</p>', '<div class="equation">'+math+'</div>')
images = []
def image(m):
    value = m.group(1)
    path = Path(value) if re.match(r'^[A-Za-z]:/',value) else source.parent/value
    assert path.exists(), path
    images.append(str(path))
    return 'src="'+path.resolve().as_uri()+'"'
body = re.sub(r'src="([^"]+)"', image, body)
spec=importlib.util.spec_from_file_location('style',ROOT/'scripts/render_publication_pdf.py')
style=importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
css=style.CSS+'''
html {font-family: "Times New Roman", "Segoe UI", serif; color:#17202a;}
.equation {margin:4mm 0; break-inside:avoid; text-align:center;}
math {font-family:"Cambria Math"; font-size:10pt;}
figure {margin:4mm 0; break-inside:avoid;}
figure img {width:100%; max-height:160mm; object-fit:contain;}
figcaption {font-size:9pt; line-height:1.35;}
figure.figure-two {break-before:page; break-after:page; margin:0;}
figure.figure-two p {margin:0;}
figure.figure-two img {width:174mm; height:200mm; max-height:none;}
figure.figure-two figcaption {margin-top:2mm; font-size:9pt; line-height:1.25;}
.main-text > p {margin:0; text-indent:1.5em;}
.main-text > p.caption {text-indent:0; margin:3mm 0 2mm; break-after:avoid;}
.main-text figure p, .main-text figcaption {text-indent:0;}
'''
body=re.sub(r'(<p><img.*?</p>)\s*<p>(<strong>Figure \d+\..*?)</p>',r'<figure>\1<figcaption>\2</figcaption></figure>',body,flags=re.S)
body=re.sub(r'<figure>(?=<p><img[^>]+02_ten_rule_1d_cellular_evolution)', '<figure class="figure-two">', body)
# Apply paragraph typography only to Sections 1-5. Front matter, appendix,
# references, figures and table captions keep their existing presentation.
start=body.index('<h2>1. Introduction</h2>')
end=body.index('<h2>Author Contributions</h2>', start)
main=body[start:end]
main=re.sub(r'<p>(?=<strong>Table \d+\.)', '<p class="caption">', main)
body=body[:start]+'<div class="main-text">'+main+'</div>'+body[end:]
dest=ROOT/'tmp/pdfs/current_manuscript_print.html'
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Molecular Space-Time</title><style>'+css+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
print(f'HTML: {dest}; equations: {len(equations)}; figures: {len(images)}')
