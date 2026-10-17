# Publication editing and rendering

Edit `publication/publication.md` on the `Complex-Systems` branch. Its image links are repository-relative. Figure 2 uses `figures/02_ten_rule_1d_cellular_evolution.png`, a 600-dpi image designed for 174 mm by 200 mm printing on a dedicated A4 page, with 10-point headings and 8.5–9-point labels. The first five panels preserve the original channel-1 data for generations 0–256.

The previous complete-manuscript PDFs are `output/pdf/molecule_space_time_publication.pdf` and `output/pdf/molecular_space_time_latest_high_quality.pdf`. Regenerate them after the current manuscript edits. The new Figure 2 print proof is `output/pdf/figure2_ten_rules_A4.pdf`.

## Portable high-quality renderer

The renderer sets Sections 1-5 with a 1.5-em first-line paragraph indent and no added space between paragraphs. Headings, figure and table captions, front matter, Appendix A and references retain their existing spacing. Markdown paragraph separators remain in the source to preserve paragraph structure; the PDF renderer controls their visual spacing.

Install Python packages `markdown` and `latex2mathml`, and the Node package `playwright` in your local environment. Install Playwright Chromium with `npx playwright install chromium`, or set `PDF_BROWSER_PATH` to your Chrome or Edge executable.

From the repository root, run:

```sh
python scripts/render_current_manuscript.py
node scripts/print_current_manuscript.cjs
```

The renderer verifies that all figures loaded and creates `output/pdf/molecular_space_time_latest_high_quality.pdf`. Display equations are converted to MathML, and text remains selectable. Review the PDF visually after any changes.

For the low-resolution preview, run `node scripts/print_current_manuscript.cjs --preview` after preparing the HTML. It keeps text selectable and reduces figure widths to at most 800 pixels, creating `output/pdf/molecular_space_time_low_resolution_preview.pdf`. This is the current complete-manuscript preview with the revised paragraph groupings. Use a full checkout on the other machine so the manuscript's linked `results/` figures and data are available.

## Ten-rule Figure 2

`scripts/render_ten_rule_figure.py --data-root PATH` reads the original archived trajectories, generates the conservative graph-flux trajectory using a frozen checkpoint if required, saves the plotted full-channel arrays in `publication/figure_data/figure2_ten_rules`, and renders the PNG. It requires NumPy, Matplotlib, PyTorch and pandas in the existing scientific environment. On Windows, activate the Conda environment so its DLL directories are available. The data root must contain the original checkpoints, graph cache and long-horizon trajectory archive. The manifest records source paths, hashes, molecule identifiers, channel counts and colour ranges. No training is performed. The conservative graph-flux panel uses its own colour limits because its values exceed the other state panels' range.

`python scripts/print_figure2_a4.py` uses ReportLab to place the PNG and the current manuscript legend on one A4 page. The full-manuscript renderer also gives Figure 2 a dedicated page.

The PowerShell scripts `scripts/resize_figure2_labels.ps1` and `scripts/figure2_all_text_larger.ps1` reproduce the superseded five-panel version. Its source images remain available for comparison.
