"""Print the current Figure 2 and its manuscript caption at journal page size."""
from pathlib import Path
import re

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

root = Path(__file__).resolve().parents[1]
source = root / 'publication/publication.md'
caption = next(line for line in source.read_text(encoding='utf-8').splitlines()
               if line.startswith('**Figure 2.'))
caption = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', caption)
font_dir = Path('C:/Windows/Fonts')
regular, bold = font_dir/'arial.ttf', font_dir/'arialbd.ttf'
if not regular.exists():
    font_dir = Path(reportlab.__file__).parent/'fonts'
    regular, bold = font_dir/'Vera.ttf', font_dir/'VeraBd.ttf'
pdfmetrics.registerFont(TTFont('Caption',str(regular)))
pdfmetrics.registerFont(TTFont('CaptionBold',str(bold)))
pdfmetrics.registerFontFamily('Caption',normal='Caption',bold='CaptionBold')
output = root / 'output/pdf/figure2_ten_rules_A4.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(output), pagesize=A4)
c.setTitle('Figure 2: ten molecular graph cellular automaton transition rules')
c.drawImage(str(root / 'publication/figures/02_ten_rule_1d_cellular_evolution.png'),
            18*mm, 79*mm, width=174*mm, height=200*mm)
style = ParagraphStyle('caption',fontName='Caption',fontSize=9,leading=11.2)
p = Paragraph(caption, style)
w,h = p.wrap(174*mm,60*mm)
assert h < 60*mm, 'Caption exceeds space on A4 page'
p.drawOn(c,18*mm,77*mm-h)
c.showPage()
c.save()
print(output)
