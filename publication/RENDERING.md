# Publication editing and rendering

Edit `publication/publication.md` on the `Complex-Systems` branch. Its image links are repository-relative. Figure 2 uses `figures/02_five_rule_1d_cellular_evolution_all_text_plus2pt.png`; its original plot pixels are retained with larger typography.

The current print-quality PDF is `output/pdf/molecule_space_time_publication.pdf`. The same rendering is also saved as `output/pdf/molecular_space_time_latest_high_quality.pdf`.

## Portable high-quality renderer

Install Python packages `markdown` and `latex2mathml`, and the Node package `playwright` in your local environment. Install Playwright Chromium with `npx playwright install chromium`, or set `PDF_BROWSER_PATH` to your Chrome or Edge executable.

From the repository root, run:

```sh
python scripts/render_current_manuscript.py
node scripts/print_current_manuscript.cjs
```

The renderer verifies that all figures loaded and creates `output/pdf/molecular_space_time_latest_high_quality.pdf`. Display equations are converted to MathML, and text remains selectable. Review the PDF visually after any changes.

The PowerShell scripts `scripts/resize_figure2_labels.ps1` and `scripts/figure2_all_text_larger.ps1` preserve the Figure 2 heatmap pixels while changing the surrounding typography. They require Windows, System.Drawing and Arial. Their source image is retained in `publication/figures/02_five_rule_1d_cellular_evolution_before_label_edit.png`.
