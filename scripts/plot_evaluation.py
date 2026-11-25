from pathlib import Path
from datetime import datetime, timezone
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
METADATA = ROOT / 'metadata'
PUBLIC_JSON = ROOT / 'docs' / 'assets' / 'benchmark-results.json'
DATASETS = {
    'twi': ('fast_evaluation_benchmark-tissue-masks', 'Tissue_with_Issues_dataset_overview.xlsx'),
    'cpg_rumc': ('fast_evaluation_benchmark-tissue-masks_cpg', 'datateam_crowdsource_data_overview.xlsx'),
}
TOP = 10
MIN_COVERAGE = {'twi': 1.0, 'cpg_rumc': 1.0}
PLOT_DATASETS = ('twi',)
EXCLUDE = {'twi': {'trident_cpg'}, 'cpg_rumc': set()}
REFERENCE_EXCLUSIONS = {'twi': set(), 'cpg_rumc': {'TBS_S01_SLIDE_056', 'TBS_S01_SLIDE_063'}}
CONFIRMED_FAILURES = {'twi': {('rtlucassen_slidesegmenter', 'slide_Helicobacter-Pylori_2')}, 'cpg_rumc': set()}
NAMES = {
    'canny_foreground': ('Canny foreground', ''),
    'atlaspatch_sam2': ('AtlasPatch-SAM2', 'SAM2.1-Hiera-T'),
    'rtlucassen_slidesegmenter': ('SlideSegmenter', ''),
    'trident_pathprofiler': ('TRIDENT PathProfiler', 'U-Net'),
    'entropy_masker': ('Entropy masker', ''),
    'trident_hest': ('TRIDENT HEST', 'DeepLabV3-ResNet50'),
    'fesi': ('FESI', ''),
    'bigpicture': ('BigPicture', ''),
    'cx_tissue_seg': ('CxTissueSeg', ''),
    'nki_all_stains': ('NKI All Stains', 'MONAI U-Net'),
    'trident_grandqc': ('TRIDENT GrandQC', 'U-Net++'),
    'trident_rtlucassen': ('TRIDENT SlideSegmenter', ''),
    'trident_cpg': ('TRIDENT CPG', ''),
    'grandqc_patchwise': ('GrandQC patchwise', 'U-Net++'),
    'improved_fesi': ('Improved FESI', ''),
    'fastsam_x': ('FastSAM-X', 'YOLOv8-seg'),
}
CLASSICAL = {'canny_foreground', 'entropy_masker', 'fesi', 'improved_fesi', 'hsv-threshold', 'hsv-threshold-v2', 'histomicstk_tissue_saliency', 'li_tissue', 'morphological_disk3', 'otsu_tissue', 'watershed_tissue_mindist200um', 'yen_tissue'}
MPP_IN_NAME = {'entropy_masker', 'fesi', 'improved_fesi', 'fastsam_x', 'grandqc_patchwise'}
GROUPS = ('Organ', 'Data source', 'Staining')
DICE_COLORS = ('#D67E82', '#F0D78D', '#EEEFE9', '#B5DEE0', '#73C6E2', '#168EB3')
INK = '#252A2E'
MAGENTA = '#cf2778'
BLUE = '#73c6e2'
STAIN_NAMES = {'HE': 'H&E', 'PDL1': 'PD-L1', 'KI67': 'Ki-67', 'P16/KI67': 'p16/Ki-67'}
IHC_STAINS = {'ER','PGR','HER2','KI67','PDL1','BCL2','BCL6','CD10','MUM1','MYC','CMYC','P16','P16/KI67','CD68','CD34','SOX10','S100','P63','CKAE1/AE3','P53','PIN4','ERG','PSA','GR','AR','PTEN','SMA','CD31','CD3','CD20','SYP'}
METHOD_LABELS = {'Canny foreground':'Canny\nforeground', 'AtlasPatch-SAM2':'AtlasPatch-\nSAM2', 'TRIDENT PathProfiler':'TRIDENT\nPathProfiler', 'TRIDENT HEST':'TRIDENT\nHEST', 'NKI All Stains':'NKI All\nStains', 'TRIDENT SlideSegmenter':'TRIDENT\nSlideSegmenter', 'TRIDENT GrandQC':'TRIDENT\nGrandQC'}


def slide_key(value):
    name = Path(str(value).replace('\\', '/')).name.strip()
    return re.sub(r'\.(ome\.tiff|xml|geojson|svs|tif|tiff|ndpi|mrxs|scn|qptiff)$', '', name, flags=re.I).casefold()


def column_key(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().casefold()).strip('_')


def load_metadata(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, engine='openpyxl')
    rows = {}
    for sheet, frame in sorted(sheets.items(), key=lambda item: (item[0].strip().casefold() != 'general info', item[0])):
        frame.columns = [column_key(c) for c in frame.columns]
        if 'slide_id' not in frame.columns:
            continue
        if 'general info' in {s.strip().casefold() for s in sheets} and sheet.strip().casefold() != 'general info':
            continue
        for row in frame.to_dict('records'):
            key = slide_key(row['slide_id'])
            if not key:
                continue
            values = {'Organ': row.get('organ',''), 'Data source': row.get('data_source',''), 'Staining': row.get('staining', row.get('stain',''))}
            record = rows.setdefault(key, {})
            for group, value in values.items():
                value = str(value).strip()
                if value and record.get(group) and record[group] != value:
                    raise ValueError(f'Conflicting metadata in {path.name}: {group} for {key}')
                if value:
                    record[group] = value
    if not rows:
        raise ValueError(f'No slide_id column or slide records in {path}')
    return pd.DataFrame(list(rows.values()), index=list(rows)).reindex(columns=GROUPS).rename_axis('sid')


def identity(run):
    prefix = run.split('__', 1)[0].casefold()
    family = re.sub(r'_mpp\d+(?:\.\d+)?$', '', prefix)
    if re.fullmatch(r'grandqc_patchwise_mpp[0-9.]+_jpeg[0-9]+_tile[0-9]+', family):
        family = 'grandqc_patchwise'
    if family == 'fastsam_fastsam-x':
        family = 'fastsam_x'
    resolution = re.search(r'(?:^|_)mpp=?([0-9]+(?:\.[0-9]+)?)(?=_|$)', run, re.I)
    mpp = float(resolution.group(1)) if resolution else None
    name, arch = NAMES.get(family, (family.replace('_', ' ').replace('-', ' ').title(), ''))
    base_name = name
    if family in MPP_IN_NAME and mpp is not None:
        name += f' ({mpp:g} MPP)'
    hole = re.search(r'(?:^|_)apply-hole-filling=([01])(?=_|$)', run)
    config = ('Hole filling ' + ('on' if hole.group(1) == '1' else 'off')) if hole else ''
    return family, {'method': name, 'base_name': base_name, 'architecture': arch, 'mpp': mpp, 'configuration': config}


def finite_mean(frame, field, positive=False):
    if field not in frame:
        return None
    v = pd.to_numeric(frame[field], errors='coerce')
    v = v[np.isfinite(v) & (v.gt(0) if positive else v.ge(0))]
    return float(v.mean()) if len(v) else None


def load_benchmark(key):
    folder, workbook = DATASETS[key]
    source = RESULTS / folder
    meta = load_metadata(METADATA / workbook)
    cohort = set(meta.index)
    df = pd.read_csv(source / 'per_slide.csv', low_memory=False, dtype={'slide_id': str, 'method': str})
    required = {'method', 'slide_id', 'status', 'dice'}
    if not required.issubset(df.columns):
        raise ValueError(f'{source}: missing columns {sorted(required - set(df.columns))}')
    if df[['method','slide_id']].isna().any().any():
        raise ValueError(f'{source}: missing method or slide ID')
    if df.method.str.contains(r'[/\\]', regex=True).any():
        raise ValueError(f'{source}: method identifiers must be run names, not paths')
    df['sid'] = df.slide_id.map(slide_key)
    df['family'] = df.method.map(lambda run: identity(run)[0])
    if df.duplicated(['method','sid']).any():
        raise ValueError(f'{source}: duplicate configuration/slide rows')
    summary = pd.read_csv(source / 'summary.csv', low_memory=False)
    if 'n_gt_available' in summary:
        counts = set(pd.to_numeric(summary.n_gt_available, errors='raise').dropna().astype(int))
        if counts != {len(cohort)}:
            raise ValueError(f'{key}: workbook has {len(cohort)} slide IDs, evaluation reports {sorted(counts)} ground truths; reconcile cohort first')
    unknown = set(df.sid) - cohort
    if unknown:
        raise ValueError(f'{key}: {len(unknown)} evaluated slide IDs do not match metadata; reconcile slide IDs first')
    df['dice'] = pd.to_numeric(df.dice, errors='coerce')
    valid = df.status.eq('ok') & np.isfinite(df.dice) & df.dice.between(0, 1)
    if (df.status.eq('ok') & ~valid).any():
        raise ValueError(f'{source}: successful rows with invalid Dice')
    excluded = {slide_key(sid) for sid in REFERENCE_EXCLUSIONS[key]}
    if excluded - cohort:
        raise ValueError(f'{key}: reference exclusions not found in metadata')
    if df.loc[valid, 'sid'].isin(excluded).any():
        raise ValueError(f'{key}: a previously excluded reference has a successful evaluation; review REFERENCE_EXCLUSIONS')
    cohort -= excluded
    meta = meta.loc[sorted(cohort)]
    if excluded:
        print(f'{key}: {len(cohort)} evaluable slides; {len(excluded)} invalid reference annotations excluded')
    candidates = []
    incomplete = []
    for run, frame in df.groupby('method', sort=True):
        family, display = identity(run)
        scores = frame.loc[valid.loc[frame.index] & frame.sid.isin(cohort), ['sid', 'dice']].set_index('sid').dice.to_dict()
        actual = len(scores)
        assigned = 0
        for failure_family, failure_slide in CONFIRMED_FAILURES[key]:
            sid = slide_key(failure_slide)
            if family == failure_family and sid in cohort and sid not in scores:
                if len(df.loc[df.family.eq(family), 'method'].unique()) != 1:
                    raise ValueError(f'{key}: confirmed failure must identify a unique configuration for {family}')
                scores[sid] = 0.0
                assigned += 1
        item = {'run': run, 'family': family, 'display': display, 'scores': scores, 'actual': actual, 'assigned': assigned}
        if family in EXCLUDE[key]:
            continue
        if set(scores) == cohort:
            candidates.append(item)
        elif key == 'cpg_rumc':
            incomplete.append(item)
    if key == 'cpg_rumc' and 'method' in summary:
        for run in sorted(set(summary.method.dropna().astype(str)) - set(df.method)):
            if '/' in run or '\\' in run:
                raise ValueError(f'{source}: summary method identifiers must be run names, not paths')
            family, display = identity(run)
            if family not in EXCLUDE[key]:
                incomplete.append({'run': run, 'family': family, 'display': display, 'scores': {}, 'actual': 0, 'assigned': 0})
    if not candidates:
        pending = website_rows(df, valid, [], cohort, len(cohort), incomplete=incomplete)
        print(f'{key}: no configurations meet {MIN_COVERAGE[key]:.0%} coverage; no ranked leaderboard entries')
        return meta, [], {}, [], pending
    ranking_cohort = set.intersection(*(set(item['scores']) for item in candidates))
    if not ranking_cohort:
        raise ValueError(f'{key}: no common ranking cohort among eligible configurations')
    meta_ranked = meta.loc[sorted(ranking_cohort)]
    for item in candidates:
        item['scores'] = {sid: item['scores'][sid] for sid in sorted(ranking_cohort)}
        item['dice'] = float(np.mean(list(item['scores'].values())))
    candidates.sort(key=lambda item: (-item['dice'], item['run']))
    chosen = []
    seen = set()
    for item in candidates:
        if item['family'] in seen:
            continue
        chosen.append(item)
        seen.add(item['family'])
        if len(chosen) == TOP:
            break
    if len({item['display']['method'] for item in chosen}) != len(chosen):
        raise ValueError(f'{key}: duplicate display labels; add distinct names to NAMES')
    timing = {}
    timing_columns = {'timing_status','algorithm_time_s','total_processing_time_s'}
    for item in chosen:
        frame = df.loc[df.method.eq(item['run']) & valid & df.sid.isin(ranking_cohort)].copy()
        if timing_columns.issubset(frame.columns):
            for field in ('algorithm_time_s','total_processing_time_s'):
                frame[field] = pd.to_numeric(frame[field], errors='coerce')
            frame = frame.loc[frame.timing_status.eq('ok') & np.isfinite(frame.algorithm_time_s) & np.isfinite(frame.total_processing_time_s) & frame.algorithm_time_s.gt(0) & frame.total_processing_time_s.gt(0)]
        else:
            frame = frame.iloc[:0]
        timing[item['run']] = frame.set_index('sid')
    paired = set.intersection(*(set(frame.index) for frame in timing.values()))
    if not paired:
        print(f'{key}: no common valid timing cohort; Pareto and timing values unavailable')
    for run, frame in timing.items():
        timing[run] = frame.loc[sorted(paired)]
    public = []
    for rank, item in enumerate(chosen, 1):
        frame = df.loc[df.method.eq(item['run']) & valid & df.sid.isin(ranking_cohort)]
        timed = timing[item['run']]
        public.append({**item['display'], 'rank': rank, 'dice': item['dice'], 'over': finite_mean(frame, 'over_segmentation_rate'), 'under': finite_mean(frame, 'under_segmentation_rate'), 'seconds': finite_mean(timed, 'total_processing_time_s', True), 'seconds_per_mp': finite_mean(timed, 'seconds_per_native_megapixel', True), 'evaluated': item['actual'], 'eligible': len(cohort), 'failure_zero': item['assigned'], 'ranked_n': len(ranking_cohort), 'paired_n': len(paired), 'paired_dice': finite_mean(timed, 'dice')})
    print(f'{key}: {len(chosen)} selected methods, {len(ranking_cohort)}/{len(cohort)} common ranking slides, {len(paired)} paired timings')
    return meta_ranked, chosen, timing, public, website_rows(df, valid, candidates, ranking_cohort, len(cohort), public, chosen, incomplete)


def website_rows(df, valid, candidates, ranking_cohort, cohort_size, public=(), chosen=(), incomplete=()):
    top = {item['run']: row for item, row in zip(chosen, public)}
    timing = {}
    for item in candidates:
        frame = df.loc[df.method.eq(item['run']) & valid & df.sid.isin(ranking_cohort)].copy()
        if {'timing_status', 'algorithm_time_s', 'total_processing_time_s'}.issubset(frame.columns):
            for field in ('algorithm_time_s', 'total_processing_time_s'):
                frame[field] = pd.to_numeric(frame[field], errors='coerce')
            frame = frame.loc[frame.timing_status.eq('ok') & np.isfinite(frame.algorithm_time_s) & np.isfinite(frame.total_processing_time_s) & frame.algorithm_time_s.gt(0) & frame.total_processing_time_s.gt(0)]
            if not frame.empty:
                timing[item['run']] = frame.set_index('sid')
    paired = set.intersection(*(set(frame.index) for frame in timing.values())) if timing else set()
    rows = []
    for index, item in enumerate([*candidates, *incomplete], 1):
        rank = index if index <= len(candidates) else None
        run = item['run']
        frame = df.loc[df.method.eq(run) & valid & df.sid.isin(ranking_cohort)]
        timed = timing[run].loc[sorted(paired)] if run in timing else frame.iloc[:0]
        display = item['display']
        row = {'_run': run, 'method': display['base_name'], 'architecture': display['architecture'], 'mpp': display['mpp'], 'configuration': display['configuration'], 'rank': rank}
        if rank is None:
            row['status'] = 'in_progress'
        else:
            row.update(status='complete', dice=item['dice'], over=finite_mean(frame, 'over_segmentation_rate'), under=finite_mean(frame, 'under_segmentation_rate'), seconds=finite_mean(timed, 'total_processing_time_s', True), failure_zero=item['assigned'], ranked_n=cohort_size)
            if run in top:
                row.update(pareto_rank=top[run]['rank'], pareto_seconds=top[run]['seconds'], pareto_dice=top[run]['paired_dice'], pareto_n=top[run]['paired_n'])
        rows.append(row)
    return rows


VERIFIED_DEVICE_ALIASES = frozenset({
    'grandqc_patchwise_mpp10_jpeg80_tile512__batch-size=4_checkpoint-path=None_confidence-threshold=0.5_device=cuda0_jpeg-quality=80_min-area=0_min__cfg=631e835ca1cd',
    'grandqc_patchwise_mpp10_jpeg80_tile512__batch-size=4_checkpoint-path=None_confidence-threshold=0.5_device=cuda_jpeg-quality=80_min-area=0_min__cfg=83f282381f4a',
})


def device_key(run):
    readable = run.rsplit('__cfg=', 1)[0]
    prefix, sep, params = readable.partition('__')
    if not sep:
        return None, None
    parts = params.split('_')
    devices = [p for p in parts if p.startswith('device=')]
    if len(devices) != 1 or (len(run.encode('utf-8')) >= 160 and run not in VERIFIED_DEVICE_ALIASES):
        return None, None
    return (prefix, tuple(p for p in parts if not p.startswith('device='))), devices[0]


def export_website(benchmarks):
    datasets = {key: [dict(row) for row in value[4]] for key, value in benchmarks.items()}
    left, right = datasets['twi'], datasets['cpg_rumc']
    by_run = {row['_run']: row for row in left}
    pairs = [(row, None) for row in left]
    indices = {row['_run']: i for i, row in enumerate(left)}
    unmatched = []
    for row in right:
        if row['_run'] in by_run:
            i = indices[row['_run']]
            pairs[i] = (pairs[i][0], row)
        else:
            unmatched.append(row)
    for row in unmatched:
        key, device = device_key(row['_run'])
        matches = [i for i, (a, b) in enumerate(pairs) if a is not None and b is None and key is not None and device_key(a['_run'])[0] == key and device_key(a['_run'])[1] != device]
        left_count = sum(device_key(a['_run'])[0] == key for a in left) if key is not None else 0
        right_count = sum(device_key(b['_run'])[0] == key for b in right) if key is not None else 0
        if len(matches) == left_count == right_count == 1:
            i = matches[0]
            pairs[i] = (pairs[i][0], row)
        else:
            pairs.append((None, row))
    pairs = [(a, b) for a, b in pairs if any(row is not None and row['rank'] is not None for row in (a, b))]
    datasets = {'twi': [a for a, _ in pairs if a is not None], 'cpg_rumc': [b for _, b in pairs if b is not None]}
    identities = {}
    for i, (a, b) in enumerate(pairs, 1):
        row = a if a is not None else b
        key = (row['method'], row['mpp'], row['architecture'])
        identities.setdefault(key, []).append((i, a, b))
    for group in identities.values():
        seen = {}
        variants = []
        for i, a, b in group:
            row = a if a is not None else b
            options = dict(re.findall(r'(?:^|_)([a-z][a-z0-9-]*)=([+-]?[0-9]+(?:\.[0-9]+)?|True|False)(?=_|$)', row['_run'], flags=re.I))
            options = {k: v for k, v in options.items() if k not in {'mpp', 'apply-hole-filling', 'device', 'cfg'} and not any(x in k for x in ('path', 'checkpoint', 'weight', 'file', 'dir'))}
            variants.append((i, a, b, row['configuration'], options))
        keys = set.intersection(*(set(v[4]) for v in variants))
        differing = sorted(k for k in keys if len({v[4][k] for v in variants}) > 1)
        for i, a, b, config, options in variants:
            config = ' · '.join(filter(None, [config, ', '.join(f'{k}={options[k]}' for k in differing)]))
            seen.setdefault(config, []).append((i, a, b))
        for config, members in seen.items():
            for variant, (i, a, b) in enumerate(members, 1):
                label = ' · '.join(filter(None, [config, f'Configuration {variant}' if len(members) > 1 else '']))
                for row in (a, b):
                    if row is not None:
                        row['id'] = i
                        row['configuration'] = label
                        row.pop('_run')
    payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'cohorts': {key: len(value[0]) for key, value in benchmarks.items()}, 'datasets': datasets}
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = PUBLIC_JSON.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(PUBLIC_JSON)
    print(PUBLIC_JSON)


def load_all():
    return {key: load_benchmark(key) for key in DATASETS}


def save(fig, folder, name, dpi=240):
    fig.savefig(folder / (name + '.pdf'), facecolor='white')
    fig.savefig(folder / (name + '.png'), facecolor='white', dpi=dpi)
    print(folder / (name + '.pdf'))


def wrapped(name):
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    return re.sub(r' \(([0-9.]+ MPP)\)$', r'\n(\1)', name)


def plot_heatmap(meta, chosen, group, folder):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib.collections import PolyCollection
    if not chosen:
        return
    values = meta[group].replace('', np.nan)
    if values.notna().sum() == 0:
        print(f'{folder.name}: no {group} metadata; skipping heatmap')
        return
    values = values.fillna('Unknown')
    totals = values.value_counts().to_dict()
    if group == 'Staining':
        labels = sorted(totals, key=lambda s: (0 if s == 'HE' else 1 if s in IHC_STAINS else 2, -totals[s], s.casefold()))
    else:
        labels = sorted(totals, key=lambda s: (-totals[s], s.casefold()))
    classical = [item for item in chosen if item['family'] in CLASSICAL]
    deep = [item for item in chosen if item['family'] not in CLASSICAL]
    order = classical + deep
    n = len(labels)
    cmap = LinearSegmentedColormap.from_list('segmenteer_dice', DICE_COLORS, N=512)
    norm = Normalize(vmin=0, vmax=1, clip=True)
    tile_w, tile_h, x_step, y_step = .90, .73, 1.00, .83
    fig_w = 6.75
    bottom, top = 1.86, .12
    fig_h = bottom + top + .168*n
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.font_manager import FontProperties
    row_font = FontProperties(family='DejaVu Sans', size=9.5)
    row_labels = [f'{STAIN_NAMES.get(label,label) if group == "Staining" else label} (N={totals[label]})' for label in labels]
    left = max(1.9, max(renderer.get_text_width_height_descent(label,row_font,False)[0] for label in row_labels)/fig.dpi+.22)
    ax = fig.add_axes([left/fig_w,bottom/fig_h,(fig_w-left-.12)/fig_w,(fig_h-bottom-top)/fig_h])
    ax.set_axis_off()
    xs = [j*x_step + (.52 if classical and deep and j >= len(classical) else 0) for j in range(len(order))]
    ys = [i*y_step for i in range(n)]
    last_x, last_y = xs[-1]+tile_w, ys[-1]+tile_h
    ax.set_xlim(-.05,last_x+.06)
    ax.set_ylim(last_y+.07,-.03)
    for i, label in enumerate(labels):
        ids = values.index[values.eq(label)]
        for j, item in enumerate(order):
            score = float(np.mean([item['scores'][sid] for sid in ids]))
            x, y = xs[j], ys[i]
            ax.add_patch(Rectangle((x,y),tile_w,tile_h,facecolor=cmap(norm(score)),edgecolor='white',linewidth=.62))
    for label,y in zip(labels,ys):
        name = STAIN_NAMES.get(label,label) if group == 'Staining' else label
        ax.text(-.18,y+tile_h/2,f'{name} (N={totals[label]})',ha='right',va='center',fontsize=9.5,color=INK,clip_on=False)
    texts = [ax.text(x+tile_w/2,last_y+.24,wrapped(item['display']['method']),ha='right',va='top',rotation=60,rotation_mode='anchor',multialignment='right',fontsize=9.5,color=INK,clip_on=False) for item,x in zip(order,xs)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lowest = min(t.get_window_extent(renderer).y0 for t in texts)
    bracket_y = ax.transData.inverted().transform((0,lowest))[1]+.43
    for a,b,label in ((0,len(classical)-1,'Classical methods'),(len(classical),len(order)-1,'Deep learning')):
        if b < a:
            continue
        x1,x2=xs[a]-.05,xs[b]+tile_w+.05
        ax.plot([x1,x1,x2,x2],[bracket_y-.16,bracket_y,bracket_y,bracket_y-.16],color='#73777B',lw=.78,clip_on=False)
        ax.text((x1+x2)/2,bracket_y+.33,label,ha='center',va='top',fontsize=10.5,fontweight='semibold',color='#384044',clip_on=False)
    canvas_w,canvas_h=fig.canvas.get_width_height()
    pt=fig.dpi/72
    left,right=fig.dpi*.20,fig.dpi*1.25
    y=fig.dpi*.78
    h=7*pt/2
    tofig=fig.transFigure.inverted().transform
    quads,colors=[],[]
    for k in range(120):
        a=left+k*(right-left)/120
        b=left+(k+1)*(right-left)/120
        quads.append(tofig(np.array(((a,y+h),(b,y+h),(b,y-h),(a,y-h)))))
        colors.append(cmap(1-(k+.5)/120))
    fig.add_artist(PolyCollection(quads,facecolors=colors,edgecolors='none',linewidths=0,transform=fig.transFigure,zorder=12,clip_on=False))
    fig.add_artist(Polygon(tofig(np.array(((left,y+h),(right,y+h),(right,y-h),(left,y-h)))),closed=True,facecolor='none',edgecolor='#5D6670',linewidth=.55,transform=fig.transFigure,zorder=13,clip_on=False))
    for xx,yy,text,ha,va,size,weight in ((left,y+h+10*pt,'Mean Dice','left','bottom',11,'semibold'),(left,y-h-10*pt,'1.00','left','top',10,'normal'),(right,y-h-10*pt,'0.00','right','top',10,'normal')):
        fx,fy=tofig((xx,yy))
        fig.text(fx,fy,text,ha=ha,va=va,fontsize=size,fontweight=weight,color=INK,clip_on=False,zorder=15)
    name={'Organ':'organ_dice_top10','Data source':'data_source_dice_top10','Staining':'stain_generalization'}[group]
    save(fig,folder,name,250 if group=='Staining' else 200)
    plt.close(fig)


def plot_pareto(chosen, timing, public, folder):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    if not chosen or not public[0]['paired_n']:
        return
    agg = pd.DataFrame([{'name': row['method'], 'dice': row['paired_dice'], 'time': row['seconds'], 'device': ','.join(sorted(set(timing[item['run']].get('device', pd.Series(dtype=str)).dropna().astype(str))))} for item,row in zip(chosen,public)]).sort_values('time').reset_index(drop=True)
    x,y=agg.time.to_numpy(),agg.dice.to_numpy()
    agg['frontier']=[not np.any((x<=xi)&(y>=yi)&((x<xi)|(y>yi))) for xi,yi in zip(x,y)]
    def setup(ax):
        ax.set_facecolor('white')
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        for side in ('left','bottom'):
            ax.spines[side].set_linewidth(.6)
            ax.spines[side].set_color('#30373c')
        ax.tick_params(labelsize=10,length=3,color='#30373c',pad=5)
        ax.tick_params(which='minor',length=1.8,color='#8B969D')
    def draw(ax,small):
        setup(ax)
        ax.set_xscale('log')
        ax.set_xlim(min(x)*.46,max(x)*1.85)
        ax.set_ylim(max(0,min(.65,float(min(y))-.05)),min(1.02,max(.97,float(max(y))+.055)))
        ax.grid(axis='y',linewidth=.45,color='#E4EAEE')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        front=agg[agg.frontier]
        ax.plot(front.time,front.dice,color=MAGENTA,lw=1.1,zorder=3)
        for r in agg.itertuples():
            ax.scatter(r.time,r.dice,s=38 if small else 47,marker='s' if 'cuda' in r.device.casefold() else 'o',color=MAGENTA if r.frontier else BLUE,edgecolor='#9c235e' if r.frontier else '#168eb3',linewidth=.6,zorder=4)
        ax.set_ylabel('Mean Dice',fontsize=11)
        ax.set_xlabel('Mean processing time per WSI (s)',fontsize=10.5 if small else 11)
        from matplotlib.transforms import Bbox
        from matplotlib.path import Path as MplPath
        fig=ax.figure
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        bounds=ax.get_window_extent(renderer)
        pt=fig.dpi/72
        radius=np.sqrt(38 if small else 47)*pt/2+1.5
        points=ax.transData.transform(np.column_stack([x,y]))
        markers=[Bbox.from_bounds(px-radius,py-radius,2*radius,2*radius) for px,py in points]
        frontier_path=MplPath(ax.transData.transform(front[['time','dice']].to_numpy())) if len(front)>1 else None
        candidates=[]
        rows=list(agg.itertuples())
        for r,(px,py) in zip(rows,points):
            text=wrapped(r.name)
            t=ax.annotate(text,(r.time,r.dice),xytext=(0,0),textcoords='offset points',ha='left',va='bottom',fontsize=8.5 if small else 10,color=MAGENTA if r.frontier else '#263238',zorder=5)
            box=t.get_window_extent(renderer)
            w,h=box.width,box.height
            t.remove()
            options=[]
            for gap in (5,7,9,12,16,22,30,40):
                for sx,sy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
                    for shift in ((0,-3,3,-6,6) if sy==0 or sx==0 else (0,)):
                        ox=sx*gap+(shift if sx==0 else 0)
                        oy=sy*gap+(shift if sy==0 else 0)
                        ha='left' if sx>0 else 'right' if sx<0 else 'center'
                        va='bottom' if sy>0 else 'top' if sy<0 else 'center'
                        lx=px+ox*pt-(w if ha=='right' else w/2 if ha=='center' else 0)
                        ly=py+oy*pt-(h if va=='top' else h/2 if va=='center' else 0)
                        box=Bbox.from_bounds(lx,ly,w,h).padded(1.5)
                        if not (bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)) or any(box.overlaps(m) for m in markers):
                            continue
                        if frontier_path is not None and frontier_path.intersects_bbox(box,filled=False):
                            continue
                        qx,qy=np.clip(px,box.x0,box.x1),np.clip(py,box.y0,box.y1)
                        distance=np.hypot(qx-px,qy-py)/pt
                        cost=distance+(.5 if sx==0 else 0)+.12*abs(shift)
                        options.append((cost,box,ox,oy,ha,va,distance))
            candidates.append(sorted(options,key=lambda o:o[0]))
        placement={}
        def assign(remaining):
            if not remaining:
                return True
            available={i:[o for o in candidates[i] if not any(o[1].overlaps(p[1]) for p in placement.values())] for i in remaining}
            i=min(remaining,key=lambda j:len(available[j]))
            for option in available[i]:
                placement[i]=option
                if assign([j for j in remaining if j!=i]):
                    return True
                placement.pop(i)
            return False
        for limit in (8,12,18,30,50,70):
            all_candidates=candidates
            candidates=[[o for o in opts if o[6]<=limit] for opts in all_candidates]
            if assign(list(range(len(rows)))):
                break
            candidates=all_candidates
        else:
            raise ValueError(f'{folder.name}: no non-overlapping Pareto label placement; enlarge the figure')
        for i,r in enumerate(rows):
            _,_,ox,oy,ha,va,distance=placement[i]
            text=wrapped(r.name)
            kwargs={} if distance<=10 else {'arrowprops':dict(arrowstyle='-',color='#73777B',lw=.55,shrinkA=2,shrinkB=4)}
            ax.annotate(text,(r.time,r.dice),xytext=(ox,oy),textcoords='offset points',ha=ha,va=va,fontsize=8.5 if small else 10,color=MAGENTA if r.frontier else '#263238',zorder=5,**kwargs)
    fig,ax=plt.subplots(figsize=(7.0,4.65))
    fig.subplots_adjust(left=.12,right=.965,bottom=.15,top=.95)
    draw(ax,False)
    save(fig,folder,'pareto_top10_single')
    plt.close(fig)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.0,3.90),gridspec_kw={'width_ratios':[2.1,1.0]})
    fig.subplots_adjust(left=.078,right=.98,bottom=.155,top=.93,wspace=.55)
    draw(ax,True)
    setup(bx)
    bx.set_xscale('log')
    ordered=agg.sort_values('dice',ascending=False)
    by_name={item['display']['method']:timing[item['run']] for item in chosen}
    names=list(ordered.name)
    for row,r in enumerate(ordered.itertuples()):
        vals=by_name[r.name].total_processing_time_s.to_numpy(float)
        p10,p25,p50,p75,p90=np.percentile(vals,[10,25,50,75,90])
        c=MAGENTA if r.frontier else BLUE
        pos=len(names)-1-row
        bx.plot([p10,p90],[pos,pos],color=c,linewidth=.8,zorder=2)
        bx.plot([p25,p75],[pos,pos],color=c,linewidth=3.3,solid_capstyle='butt',zorder=3)
        bx.scatter([p50],[pos],s=16,marker='o',facecolor='white',edgecolor='#30373c',linewidth=.7,zorder=4)
    bx.set_yticks(range(len(names)),names[::-1],fontsize=9)
    bx.tick_params(axis='y',length=0,pad=5)
    bx.set_ylim(-.85,len(names)-.15)
    bx.set_xlabel('Processing time (s)',fontsize=10.5)
    bx.xaxis.set_major_locator(LogLocator(base=10,subs=(1,)))
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.grid(axis='x',linewidth=.45,color='#E4EAEE')
    ax.annotate('a',xy=(0,1),xycoords='axes fraction',xytext=(-5,3),textcoords='offset points',fontsize=10,fontweight='semibold',ha='right',va='bottom',color=INK,annotation_clip=False)
    bx.annotate('b',xy=(0,1),xycoords='axes fraction',xytext=(-5,3),textcoords='offset points',fontsize=10,fontweight='semibold',ha='right',va='bottom',color=INK,annotation_clip=False)
    save(fig,folder,'pareto_top10_two_panel')
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
    benchmarks=load_all()
    for key in PLOT_DATASETS:
        meta,chosen,timing,public,_=benchmarks[key]
        if not chosen:
            continue
        folder=RESULTS / DATASETS[key][0] / 'figures'
        folder.mkdir(parents=True,exist_ok=True)
        for group in GROUPS:
            plot_heatmap(meta,chosen,group,folder)
        plot_pareto(chosen,timing,public,folder)
    export_website(benchmarks)


if __name__ == '__main__':
    main()
