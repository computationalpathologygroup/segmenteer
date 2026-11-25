import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from plot_evaluation import DATASETS, METADATA, RESULTS, TOP, load_benchmark, slide_key


DATASET = 'twi'
METRICS = ('dice', 'under', 'over', 'he_dice', 'he_failure', 'non_he_dice', 'non_he_failure')
PERCENT = {'under', 'over', 'he_failure', 'non_he_failure'}
LOWER_IS_BETTER = PERCENT


def file_hash(path):
    with path.open('rb') as stream:
        return sha256(stream.read()).hexdigest()


def interval(values):
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def estimate(value, bounds, n, **extra):
    return {'mean': float(value), 'lower': float(bounds[0]), 'upper': float(bounds[1]), 'n': int(n), **extra}


def wilson(failed, n):
    if not n:
        raise ValueError('Empty failure-rate cohort')
    p = failed / n
    z = NormalDist().inv_cdf(0.975)
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    radius = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return estimate(p, (max(0., center-radius), min(1., center+radius)), n, failures=int(failed))


def bootstrap(values, stains, n_resamples, rng):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires nonempty finite observations')
    groups = [np.flatnonzero(stains == stain) for stain in sorted(set(stains))]
    draws = np.empty((n_resamples, values.shape[1]))
    for start in range(0, n_resamples, 256):
        end = min(start+256, n_resamples)
        total = np.zeros((end-start, values.shape[1]))
        for ids in groups:
            count = len(ids)
            if count == 1:
                total += values[ids[0]]
            else:
                indices = rng.integers(0, count, size=(end-start, count))
                total += values[ids][indices].sum(axis=1)
        draws[start:end] = total/len(values)
    return draws


def dice_bootstrap(dice, stains, he, n_resamples, rng):
    he_draws = bootstrap(dice[he], stains[he], n_resamples, rng)
    non_he_draws = bootstrap(dice[~he], stains[~he], n_resamples, rng)
    overall = (he_draws*he.sum() + non_he_draws*(~he).sum())/len(he)
    return overall, he_draws, non_he_draws


def area_values(frame, item, ids, column):
    s = pd.to_numeric(frame[column], errors='coerce')
    if not np.isfinite(s).all() or s.lt(0).any():
        raise ValueError(f"{item['run']}: invalid {column} on a successful evaluation")
    if column == 'under_segmentation_rate' and s.gt(1+1e-9).any():
        raise ValueError(f"{item['run']}: under-segmentation exceeds 1")
    if len(set(ids)-set(s.index)) != item['assigned']:
        raise ValueError(f"{item['run']}: missing area observations not explained by assigned failures")
    return s.reindex(ids).to_numpy(float)


def compute(n_resamples, seed):
    meta, chosen, _, public, _ = load_benchmark(DATASET)
    if len(chosen) != TOP:
        raise ValueError(f'Expected {TOP} eligible distinct methods, found {len(chosen)}')
    if len({item['family'] for item in chosen}) != TOP:
        raise ValueError('Duplicate method family')
    ids = list(meta.index)
    if meta.Staining.isna().any() or meta.Staining.astype(str).str.strip().eq('').any():
        raise ValueError('Missing staining metadata')
    stains = meta.Staining.astype(str).str.strip().to_numpy()
    he = np.isin(np.char.upper(stains.astype(str)), ['HE', 'H&E', 'H & E'])
    if not he.any() or he.all():
        raise ValueError('Both H&E and non-H&E cohorts are required')
    dice = np.array([[item['scores'][sid] for item in chosen] for sid in ids], dtype=float)
    folder, workbook = DATASETS[DATASET]
    source = RESULTS / folder / 'per_slide.csv'
    df = pd.read_csv(source, dtype={'method': str, 'slide_id': str}, low_memory=False)
    df['sid'] = df.slide_id.map(slide_key)
    rng = np.random.default_rng(seed)
    overall, he_draws, non_he_draws = dice_bootstrap(dice, stains, he, n_resamples, rng)
    rows = []
    for j, (item, reference) in enumerate(zip(chosen, public)):
        frame = df.loc[df.method.eq(item['run']) & df.status.eq('ok') & df.sid.isin(ids)].set_index('sid')
        metrics = {}
        for field, mask, draws in [('dice', np.ones(len(ids), dtype=bool), overall), ('he_dice', he, he_draws), ('non_he_dice', ~he, non_he_draws)]:
            metrics[field] = estimate(dice[mask,j].mean(), interval(draws[:,j]), mask.sum())
        for field, mask in [('he_failure', he), ('non_he_failure', ~he)]:
            metrics[field] = wilson(int((dice[mask,j] < 0.5).sum()), int(mask.sum()))
        for field, column in [('under', 'under_segmentation_rate'), ('over', 'over_segmentation_rate')]:
            values = area_values(frame, item, ids, column)
            valid = np.isfinite(values)
            draws = bootstrap(values[valid], stains[valid], n_resamples, rng)[:,0]
            metrics[field] = estimate(values[valid].mean(), interval(draws), valid.sum())
        for field in ('dice', 'under', 'over'):
            if not np.isclose(metrics[field]['mean'], reference[field], rtol=0, atol=1e-12):
                raise ValueError(f"{item['run']}: {field} differs from plotting summary")
        rows.append({'rank': j+1, 'method': item['display']['method'], 'family': item['family'], 'run': item['run'], 'mpp': item['display']['mpp'], 'configuration': item['display']['configuration'], 'assigned_dice_zero': item['assigned'], 'metrics': metrics})
    comparisons = []
    for j in range(1, len(rows)):
        difference = overall[:,0]-overall[:,j]
        comparisons.append({'reference_method': rows[0]['method'], 'reference_run': rows[0]['run'], 'comparison_method': rows[j]['method'], 'comparison_run': rows[j]['run'], **estimate((dice[:,0]-dice[:,j]).mean(), interval(difference), len(ids))})
    return {'dataset': DATASET, 'generated_at': datetime.now(timezone.utc).isoformat(), 'n': len(ids), 'n_he': int(he.sum()), 'n_non_he': int((~he).sum()), 'stain_counts': {str(k): int(v) for k,v in pd.Series(stains).value_counts().sort_index().items()}, 'bootstrap_resamples': n_resamples, 'seed': seed, 'confidence_level': 0.95, 'bootstrap_method': 'Percentile slide bootstrap within each exact staining category, holding observed category counts fixed. Dice draws are shared across methods. Area metrics use each method’s valid geometry cohort.', 'failure_threshold_dice': 0.5, 'failure_interval': 'Wilson score', 'metric_units': 'All numeric metrics are fractions of one; over-segmentation may exceed one. LaTeX displays area and failure rates as percentages.', 'selection': 'Best complete configuration per method family by overall mean Dice using plot_evaluation.load_benchmark; top ten distinct families.', 'comparison_policy': 'Exploratory paired percentile intervals against the observed top-ranked method, unadjusted for multiplicity or configuration selection.', 'input_sha256': {'per_slide.csv': file_hash(source), 'summary.csv': file_hash(source.with_name('summary.csv')), 'metadata': file_hash(METADATA / workbook), 'plot_evaluation.py': file_hash(Path(__file__).with_name('plot_evaluation.py'))}, 'methods': rows, 'paired_comparisons': comparisons}


def latex_escape(value):
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(table.get(c,c) for c in str(value))


def latex_table(data):
    rows = data['methods']
    best = {field: (min if field in LOWER_IS_BETTER else max)(r['metrics'][field]['mean'] for r in rows) for field in METRICS}
    caption = (
        f"Overall and stain-stratified tissue-background segmentation performance of the {len(rows)} highest-ranking eligible methods across {data['n']} whole-slide images ({data['n_he']} H\\&E and {data['n_non_he']} non-H\\&E). "
        r'The best configuration per method was selected by overall mean Dice. Under (FN/GT) and Over (FP/GT) are the false-negative and false-positive areas as percentages of ground-truth tissue area. '
        f"The 95\\% confidence intervals for mean Dice, FN/GT and FP/GT were calculated using {data['bootstrap_resamples']:,} stain-stratified bootstrap resamples. "
        r'Wilson score intervals are reported for severe failure percentages (Dice $<0.5$). '
    )
    if any(r['assigned_dice_zero'] for r in rows):
        caption += 'Documented missing predictions were assigned Dice zero and counted as failures. Area-based metrics use only valid prediction geometries.'
    else:
        caption = caption.rstrip()
    lines = [r'\begin{table}[H]', r'\centering', r'\caption{' + caption + '}', r'\label{tab:twi_top10}', r'\begingroup', r'\fontsize{9}{11}\selectfont', r'\setlength{\tabcolsep}{1.35pt}', r'\renewcommand{\arraystretch}{1.18}', r'\newcommand{\twimethod}[1]{\raisebox{3.5pt}[0pt][0pt]{\mbox{#1}}}', r'\newcommand{\twival}[2]{\shortstack[c]{#1\\[0.6pt]{\fontsize{8}{9.3}\selectfont\color{black!77}#2}}}', r'\begin{tabular*}{0.99\linewidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}m{1.44in} c c c @{\hspace{11pt}} c c @{\hspace{11pt}} c c @{}}', r'\toprule', r'\multirow{2}{*}[-1pt]{\textbf{Method}} & \multicolumn{3}{c}{\textbf{Overall}} & \multicolumn{2}{c}{\textbf{H\&E}} & \multicolumn{2}{c}{\textbf{Non-H\&E}} \\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}', r'& \textbf{Dice} & \textbf{Under (\%)} & \textbf{Over (\%)} & \textbf{Dice} & \textbf{$<0.5$ (\%)} & \textbf{Dice} & \textbf{$<0.5$ (\%)} \\', r'\midrule']
    for i, row in enumerate(rows):
        cells = [r'\twimethod{' + latex_escape(row['method']) + '}']
        for field in METRICS:
            metric = row['metrics'][field]
            scale, decimals = (100,1) if field in PERCENT else (1,3)
            value, lo, hi = (f'{metric[k]*scale:.{decimals}f}' for k in ('mean','lower','upper'))
            if np.isclose(metric['mean'], best[field], rtol=0, atol=1e-12):
                value = r'\textbf{' + value + '}'
            cells.append(r'\twival{' + value + '}{' + lo + '--' + hi + '}')
        lines.append(' & '.join(cells) + (r' \\[3.2pt]' if i+1 < len(rows) else r' \\'))
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', r'\endgroup', r'\end{table}']) + '\n'


def write_outputs(data, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data['methods']:
        flat = {k:v for k,v in row.items() if k != 'metrics'}
        for field, stats in row['metrics'].items():
            flat.update({f'{field}_{k}':v for k,v in stats.items()})
        rows.append(flat)
    pd.DataFrame(rows).to_csv(out / 'top10_statistics.csv', index=False)
    pd.DataFrame(data['paired_comparisons']).to_csv(out / 'paired_bootstrap_comparisons.csv', index=False)
    (out / 'top10_statistics.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out / 'table1_twi_compact_9pt.tex').write_text(latex_table(data), encoding='utf-8')
    for row in data['methods']:
        stats = row['metrics']['dice']
        print(f"{row['rank']:2d} {row['method']:28s} {stats['mean']:.3f} ({stats['lower']:.3f}, {stats['upper']:.3f})")
    print(out.resolve())


def main():
    parser = argparse.ArgumentParser(description='Compute TWI top-10 statistics and manuscript table')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=RESULTS / DATASETS[DATASET][0] / 'statistics')
    args = parser.parse_args()
    if args.resamples < 1000 or args.seed < 0:
        parser.error('Use at least 1000 resamples and a nonnegative seed')
    write_outputs(compute(args.resamples, args.seed), args.output_dir)


if __name__ == '__main__':
    main()
