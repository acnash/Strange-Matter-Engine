"""Verify and inventory the manuscript release without running scientific models.

Run with Python and pypdf. The report records exact local file hashes and checks
that the supporting files have been added to Git's index before publication.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / 'publication/publication.md'
PDF = ROOT / 'output/pdf/SUBMISSION_publication_high_resolution_20261024.pdf'
REPORT = ROOT / 'publication/release_20261024_manifest.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def repo_path(value):
    value = str(value).replace('\\', '/')
    marker = 'Strange Matter Engine/'
    return ROOT / (value.split(marker, 1)[1] if marker in value else value)

def main():
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    paths = {MANUSCRIPT, PDF}
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', MANUSCRIPT.read_text(encoding='utf-8'))
    images += re.findall(r'<img[^>]+src="([^"]+)"', MANUSCRIPT.read_text(encoding='utf-8'))
    paths.update((MANUSCRIPT.parent / value).resolve() for value in images)
    for folder in ['publication/figure_data', 'results/strange_attractor_audit_20260929',
                   'results/structure_dynamics_publication_v1']:
        paths.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.log')
    paths.update(ROOT / name for name in tracked if name.startswith('scripts/') and name.endswith('.py'))
    paths.add(ROOT / 'results/long_horizon_attractor_campaign_v1/validation_dynamics_population_summary.csv')
    paths.add(ROOT / 'data/openadmet-cyp-challenge-2026/cyp-challenge-TRAIN_inhibition.csv')
    gallery = json.loads((ROOT / 'publication/figure_data/appendix_e_gallery/manifest.json').read_text())
    for row in gallery:
        compact = ROOT / 'publication/figure_data/appendix_e_gallery' / row['file']
        assert digest(compact) == row['sha256'], compact
        original = repo_path(row['source'])
        assert digest(original) == row['source_sha256'], original
        paths.update([original, repo_path(row['checkpoint'])])
    sources = json.loads((ROOT / 'publication/figure_data/lyapunov_methods/sources.json').read_text())
    for compact, original in [('direct.npz', sources['direct']),
                              ('renormalised.npz', sources['renormalised']),
                              ('spectrum_1.npz', sources['spectrum'][0]),
                              ('spectrum_2.npz', sources['spectrum'][1])]:
        original = repo_path(original)
        assert digest(original) == digest(ROOT / 'publication/figure_data/lyapunov_methods' / compact)
        paths.add(original)
    audit = json.loads((ROOT / 'results/strange_attractor_audit_20260929/provenance.json').read_text())
    checkpoint = repo_path(audit['checkpoint'])
    assert digest(checkpoint) == audit['checkpoint_sha256'], checkpoint
    paths.add(checkpoint)
    missing = [str(p) for p in paths if not p.exists()]
    assert not missing, missing
    untracked = [p.relative_to(ROOT).as_posix() for p in paths
                 if p.relative_to(ROOT).as_posix() not in tracked]
    assert not untracked, untracked

    reader = PdfReader(PDF)
    fonts = {}
    large_images = []
    def inspect(resources):
        for ref in resources.get('/Font', {}).values():
            font = ref.get_object()
            base = str(font.get('/BaseFont'))
            descendants = font.get('/DescendantFonts', [font])
            embedded = all(any(key in f.get_object().get('/FontDescriptor', {}).get_object()
                               for key in ['/FontFile', '/FontFile2', '/FontFile3'])
                           for f in descendants)
            fonts[base] = {'subtype': str(font.get('/Subtype')), 'embedded': embedded}
        for ref in resources.get('/XObject', {}).values():
            obj = ref.get_object()
            if obj.get('/Subtype') == '/Form':
                inspect(obj.get('/Resources', {}))
            elif obj.get('/Subtype') == '/Image':
                w, h = obj.get('/Width', 0), obj.get('/Height', 0)
                if w*h > 34_000_000:
                    large_images.append([w, h])
    for page in reader.pages:
        inspect(page['/Resources'])
    assert all(f['embedded'] and f['subtype'] != '/Type3' for f in fonts.values()), fonts
    root = reader.trailer['/Root']
    assert not reader.is_encrypted
    assert '/JavaScript' not in root.get('/Names', {})
    links = sum(1 for p in reader.pages for a in p.get('/Annots', [])
                if a.get_object().get('/Subtype') == '/Link')
    report = {
        'date': '2026-10-24',
        'scope': 'File completeness, provenance hashes and PDF technical checks; no training or scientific reruns.',
        'manuscript_sha256': digest(MANUSCRIPT),
        'pdf': {'path': PDF.relative_to(ROOT).as_posix(), 'sha256': digest(PDF),
                'bytes': PDF.stat().st_size, 'pages': len(reader.pages), 'fonts': fonts,
                'text_characters': sum(len(p.extract_text() or '') for p in reader.pages),
                'tagged': '/StructTreeRoot' in root, 'link_annotations': links,
                'images_over_34_megapixels': large_images},
        'figure_references_checked': len(images),
        'gallery_provenance_records_checked': len(gallery),
        'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                  for p in sorted(paths)]}
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
    print(f'Indexed supporting files checked: {len(paths)}')

if __name__ == '__main__':
    main()
