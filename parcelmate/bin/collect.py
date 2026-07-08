import os
import sys
import csv
import glob
import html
import argparse

from parcelmate.cfg import get_cfg
from parcelmate.constants import PLOT_DIR, KNOCKOUT_NAME, LOSS_NAME


def _stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def _find_plots(output_dir):
    """Return {category: [png_path, ...]} for one run, grouped by plot subdirectory."""
    plot_root = os.path.join(output_dir, PLOT_DIR)
    groups = {}
    for png in sorted(glob.glob(os.path.join(plot_root, '**', '*.png'), recursive=True)):
        category = os.path.relpath(os.path.dirname(png), plot_root)
        groups.setdefault(category, []).append(png)
    return groups


def _params_str(params):
    return ', '.join('%s=%s' % (k.split('.')[-1], v) for k, v in params.items())


# Column order and headers for the knockout loss table.
_LOSS_COLUMNS = ['condition', 'kind', 'domain', 'loss', 'perplexity', 'n_tokens']
# Row background per condition kind, matching the knockout plot palette.
_KIND_BG = {'healthy': '#eaf0f8', 'knockout': '#f8ecec', 'baseline': '#f4f4f4'}


def _load_loss_summary(output_dir):
    """Return the rows of a run's knockout ``loss_summary.csv`` (or None)."""
    path = os.path.join(output_dir, KNOCKOUT_NAME, '%s_summary.csv' % LOSS_NAME)
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows or None


def _fmt(column, value):
    """Format a loss-table cell for display."""
    if value is None or value == '':
        return ''
    if column in ('loss', 'perplexity'):
        try:
            return '%.3f' % float(value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _loss_summary_md(rows):
    lines = ['\n**knockout loss summary**\n',
             '| %s |' % ' | '.join(_LOSS_COLUMNS),
             '|%s|' % '|'.join(['---'] * len(_LOSS_COLUMNS))]
    for r in rows:
        lines.append('| %s |' % ' | '.join(_fmt(c, r.get(c)) for c in _LOSS_COLUMNS))
    return '\n'.join(lines)


def _loss_summary_html(rows):
    parts = ['<div class="cat"><h3>knockout loss summary</h3>',
             '<table class="loss"><thead><tr>%s</tr></thead><tbody>'
             % ''.join('<th>%s</th>' % html.escape(c) for c in _LOSS_COLUMNS)]
    for r in rows:
        bg = _KIND_BG.get(r.get('kind', ''), '#ffffff')
        cells = ''.join('<td>%s</td>' % html.escape(_fmt(c, r.get(c))) for c in _LOSS_COLUMNS)
        parts.append('<tr style="background:%s">%s</tr>' % (bg, cells))
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def collect(manifest_path, outdir):
    manifest = get_cfg(manifest_path)
    runs = manifest.get('runs', [])

    md_lines = ['# Sweep: %s\n' % manifest.get('name', ''), '%d runs\n' % len(runs)]
    html_parts = [
        '<!doctype html><meta charset="utf-8"><title>Sweep: %s</title>' % html.escape(str(manifest.get('name', ''))),
        '<style>body{font-family:sans-serif;margin:2rem;}'
        '.run{border-top:2px solid #ccc;padding-top:1rem;margin-top:1rem;}'
        '.cat{margin:1rem 0;}img{max-width:320px;height:auto;margin:4px;border:1px solid #ddd;'
        'vertical-align:top;}h2{color:#333;}code{background:#f4f4f4;padding:2px 4px;}'
        'table.loss{border-collapse:collapse;font-size:0.9em;}'
        'table.loss th,table.loss td{border:1px solid #ddd;padding:3px 8px;text-align:right;}'
        'table.loss th:nth-child(-n+3),table.loss td:nth-child(-n+3){text-align:left;}</style>',
        '<h1>Sweep: %s</h1>' % html.escape(str(manifest.get('name', ''))),
    ]

    for run in runs:
        params = run.get('params', {})
        output_dir = run['output_dir']
        groups = _find_plots(output_dir)
        n_plots = sum(len(v) for v in groups.values())
        status = '%d plots' % n_plots if n_plots else 'no outputs yet'

        md_lines.append('\n## %s' % run['tag'])
        md_lines.append('- params: `%s`' % _params_str(params))
        md_lines.append('- output_dir: `%s` (%s)' % (output_dir, status))

        html_parts.append('<div class="run"><h2>%s</h2>' % html.escape(run['tag']))
        html_parts.append('<p><b>params:</b> <code>%s</code><br><b>output:</b> <code>%s</code> (%s)</p>'
                          % (html.escape(_params_str(params)), html.escape(output_dir), status))

        loss_rows = _load_loss_summary(output_dir)
        if loss_rows:
            md_lines.append(_loss_summary_md(loss_rows))
            html_parts.append(_loss_summary_html(loss_rows))

        for category in sorted(groups):
            md_lines.append('\n**%s**\n' % category)
            html_parts.append('<div class="cat"><h3>%s</h3>' % html.escape(category))
            for png in groups[category]:
                rel = os.path.relpath(png, outdir)
                md_lines.append('![%s](%s)' % (os.path.basename(png), rel))
                html_parts.append('<img src="%s" title="%s">' % (html.escape(rel), html.escape(os.path.basename(png))))
            html_parts.append('</div>')
        html_parts.append('</div>')

    md_path = os.path.join(outdir, 'index.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    html_path = os.path.join(outdir, 'dashboard.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_parts) + '\n')

    _stderr('Wrote %s\nWrote %s\n' % (md_path, html_path))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'Collect plots from all runs in a sweep into an HTML dashboard and a markdown index.'
    )
    argparser.add_argument('manifest_path', help='Path to a sweep manifest.yaml (produced by sweep.py).')
    argparser.add_argument('-o', '--outdir', default=None,
                           help='Where to write index.md and dashboard.html. Default: manifest directory.')
    args = argparser.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.manifest_path))
    os.makedirs(outdir, exist_ok=True)
    collect(args.manifest_path, outdir)
