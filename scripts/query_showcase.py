#!/usr/bin/env python3
"""Export benchmark leaderboards and a metadata-selected slide showcase."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


# Category keys (not values) follow benchmark_data_annotation.xlsx.
SHOWCASE_FIELDS = {'source': 'data_source', 'center': 'source_site',
                   'organ': 'organ', 'stain': 'staining'}
UNAVAILABLE_METADATA = {'', 'unknown', 'n/a', 'nan', 'not available',
                        'none', 'null', '-'}

IMAGE_EXTENSIONS = {'.svs','.ndpi','.mrxs','.scn','.tif','.tiff','.qptiff',
                    '.bif','.vms','.vmu','.png','.jpg','.jpeg','.webp'}
SEED = 'segmenteer-tissues-with-issues-matched-300-v1'


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def norm(value):
    return ''.join(ch for ch in unicodedata.normalize('NFKD', str(value)).casefold()
                   if not unicodedata.combining(ch)).replace('ł','l')


def slide_stem(value):
    return Path(str(value).strip()).stem


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def locate_metadata(args):
    """Locate the supplied workbook or CSV metadata."""
    if args.annotations:
        if not args.annotations.is_file():
            raise FileNotFoundError(f'Annotation metadata not found: {args.annotations}')
        return args.annotations
    options = [args.metrics.parent / 'benchmark_data_annotation.xlsx',
               args.metrics.parent / 'benchmark_data_annotation.csv',
               args.slides.parent / 'benchmark_data_annotation.xlsx',
               Path.cwd() / 'benchmark_data_annotation.xlsx',
               Path(__file__).parent / 'benchmark_data_annotation.xlsx']
    return next((p for p in options if p.is_file()), None)


def valid_metadata_value(value):
    return norm(value).strip() not in UNAVAILABLE_METADATA


def showcase_metadata(args):
    path = locate_metadata(args)
    if path is None:
        raise FileNotFoundError('No metadata workbook found. Pass --annotations benchmark_data_annotation.xlsx')
    metadata = load_stratification_metadata(path)
    if not metadata:
        raise ValueError(f'No slide metadata found in {path}')
    return metadata, str(path)


def list_showcase_values(metadata, category):
    """List source/center/organ/stain values actually present in the supplied metadata."""
    from collections import Counter
    categories = SHOWCASE_FIELDS if category == 'all' else {category: SHOWCASE_FIELDS[category]}
    for key, field in categories.items():
        values = Counter(str(row.get(field, '')).strip() for row in metadata.values()
                         if valid_metadata_value(row.get(field, '')))
        print(f'{key.upper()} ({len(values)} values)')
        for value, count in sorted(values.items(), key=lambda pair: (-pair[1], pair[0].casefold())):
            print(f'  {value} ({count})')


def filter_showcase_cases(metadata, category, value, parser):
    """Filter metadata before evaluating model predictions (not after ranking)."""
    if category == 'all':
        return metadata
    field = SHOWCASE_FIELDS[category]
    target = norm(value).strip()
    found = {sid: row for sid, row in metadata.items()
             if norm(row.get(field, '')).strip() == target}
    if not found:
        available = sorted({str(row.get(field, '')).strip() for row in metadata.values()
                            if valid_metadata_value(row.get(field, ''))}, key=str.casefold)
        parser.error(f'No slides in {category}={value!r}. Values: '+', '.join(available))
    return found


def load_metrics(metrics_path):
    by_run = defaultdict(dict)
    run_labels = {}
    total = 0
    for r in csv_rows(metrics_path):
        total += 1
        if str(r.get('evaluation_error','')).strip() or r.get('evaluation_mode','').casefold() != 'supervised':
            continue
        slide = Path(str(r.get('filename','')).strip()).name
        run = (r.get('run_id') or r.get('method_name') or '').strip()
        vals = {name: number(r.get(col)) for name,col in
                [('dice','dice'),('nsd','nsd'),('under','under_segmentation_rate'),('over','over_segmentation_rate')]}
        if vals['dice'] is None:
            vals['dice'] = number(r.get('dice_score'))
        if not run or not slide or any(v is None for v in vals.values()):
            continue
        if not 0 <= vals['dice'] <= 1 or not 0 <= vals['nsd'] <= 1:
            continue
        # Deliberately evaluate specific run configurations independently.
        previous = by_run[run].get(slide)
        if previous is not None and previous != vals:
            raise ValueError(f'Conflicting records for run {run}, slide {slide}')
        by_run[run][slide] = vals
        run_labels.setdefault(run, (r.get('formal_name') or '').strip())
    if not by_run:
        raise ValueError(f'No valid supervised per-slide scores in {metrics_path}')
    return by_run, run_labels, total


def display_label(run_id, formal_name):
    """Resolve architecture-aware display labels without changing run IDs."""
    # Prefer architecture-aware labels from saved evaluation records.
    formal_name = (formal_name or '').strip()
    if '/' in formal_name:
        return formal_name
    # Use the package identity resolver when available.
    try:
        from segmenteer.methods.identity import architecture_label
        label = architecture_label(run_id=run_id, formal_name=formal_name)
        if label and label != formal_name and label != run_id:
            name = label
        else:
            name = ''
    except ImportError:
        name = ''
    # Provide standalone architecture labels when Segmenteer is unavailable.
    if not name:
        mapped = [
            ('trident_pathprofiler','TRIDENT PathProfiler/U-Net'),
            ('trident_grandqc','TRIDENT GrandQC/U-Net++'),
            ('trident_hest','TRIDENT HEST/DeepLabV3-ResNet50'),
            ('trident_rtlucassen','TRIDENT RTLucassen/SlideSegmenter CNN'),
            ('atlaspatch_sam2','AtlasPatch/SAM2.1-Hiera-T'),
            ('grandqc_patchwise','GrandQC patchwise/U-Net++ (EfficientNet-B0)'),
            ('grandqc_','GrandQC/U-Net++ (EfficientNet-B0)'),
            ('nki_all_stains','NKI All-Stains/MONAI U-Net'),
            ('bigpicture','BigPicture/Resolution-agnostic CNN'),
            ('rtlucassen','RTLucassen/SlideSegmenter CNN'),
            ('fastsam_','FastSAM/YOLOv8-seg'),
            ('icgi_resenc','ICGI/nnU-Net v2 ResEnc (10 µm/px)'),
            ('icgi_nnunet','ICGI/nnU-Net v2 (10 µm/px)'),
            ('hest_deeplabv3','HEST/DeepLabV3-ResNet50'),
        ]
        slug = run_id.split('__', 1)[0].lower()
        name = next((label for prefix,label in mapped if slug.startswith(prefix)), '')
        if not name:
            # Fall back to the run identifier.
            name = slug.replace('_',' ').replace('-',' ').title()
    mpp = re.search(r'(?:^|_)mpp=(\d+(?:\.\d+)?)', run_id)
    if mpp and 'MPP' not in name.upper() and 'µm/px' not in name:
        name += f' ({mpp.group(1)} MPP)'
    return name


def load_available_timings(outputs, included, cohort, scope='available'):
    """Aggregate valid per-slide timings; use CSV only if JSON records are absent."""
    results = {}
    outputs = Path(outputs)
    cohort_stems = {slide_stem(slide) for slide in cohort}
    if not outputs.is_dir():
        return results

    def has_prediction(run, slide):
        base = outputs / run / 'predictions'
        return any((base / (slide + suffix)).is_file() for suffix in ('.xml', '.geojson'))

    csv_fallback = None
    for run in included:
        timing_dir = outputs / run / 'timings'
        samples = {}
        json_paths = list(timing_dir.glob('*.json')) if timing_dir.is_dir() else []
        for timing_file in json_paths:
            # Ignore incomplete timing records during concurrent exports.
            try:
                record = json.loads(timing_file.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, ValueError):
                continue
            if not isinstance(record, dict) or record.get('schema_version') != 3:
                continue
            if record.get('run_id') != run:
                continue
            slide = slide_stem(record.get('filename', ''))
            if not slide or slide != timing_file.stem:
                continue
            if scope == 'matched' and slide not in cohort_stems:
                continue
            seconds = number(record.get('total_processing_time_s'))
            if seconds is None or seconds < 0 or not has_prediction(run, slide):
                continue
            samples[slide] = seconds

        source = 'per-slide-json'
        if not json_paths:
            # Support experiments with legacy timing CSVs.
            if csv_fallback is None:
                fallback_file = outputs / 'benchmark_timings.csv'
                csv_fallback = list(csv_rows(fallback_file)) if fallback_file.is_file() else []
            source = 'benchmark-timings-csv'
            for record in csv_fallback:
                if (record.get('run_id') or '').strip() != run:
                    continue
                slide = slide_stem(record.get('filename', ''))
                if scope == 'matched' and slide not in cohort_stems:
                    continue
                seconds = number(record.get('total_processing_time_s'))
                if not slide or seconds is None or seconds < 0 or not has_prediction(run, slide):
                    continue
                if slide in samples and samples[slide] != seconds:
                    samples.pop(slide, None)
                    continue
                samples[slide] = seconds

        if samples:
            results[run] = {'mean': sum(samples.values()) / len(samples),
                            'n': len(samples), 'source': source, 'samples': samples}
    return results


def compute_leaderboard(by_run, labels, matched_slides, ranking_metric="dice", timings_root=None, timing_scope='available'):
    available = {run: rows for run, rows in by_run.items() if len(rows) >= matched_slides}
    if len(available) < 3:
        raise ValueError(f'Only {len(available)} configurations have >= {matched_slides} valid slides (need at least 3)')
    # Select configurations sharing the required number of evaluated slides.
    included, common = [], None
    for run, rows in sorted(available.items(), key=lambda r:(-len(r[1]),r[0])):
        overlap = set(rows) if common is None else common & rows.keys()
        if len(overlap) >= matched_slides:
            common = overlap
            included.append(run)
    if len(included) < 3:
        raise ValueError(f'Fewer than 3 configurations share {matched_slides} evaluated slides')
    cohort = sorted(common, key=lambda s:hashlib.sha256((SEED+'\0'+s).encode()).hexdigest())[:matched_slides]
    cohort_hash = hashlib.sha256('\n'.join(sorted(cohort)).encode()).hexdigest()
    ranked = []
    for run in included:
        values = {field:sum(by_run[run][slide][field] for slide in cohort)/matched_slides
                  for field in ('dice','nsd','under','over')}
        ranked.append(dict(run_id=run, label=display_label(run, labels.get(run,'')),
                           matched_slides=matched_slides, **values))
    timing = load_available_timings(timings_root, set(included), cohort, timing_scope) if timings_root else {}
    for model in ranked:
        stats = timing.get(model['run_id'], {})
        model['speed'] = stats.get('mean')
        model['timing_n'] = stats.get('n', 0)
        model['timing_source'] = stats.get('source')
    ranked.sort(key=lambda m:(-m[ranking_metric], m['run_id']))
    return ranked, cohort_hash, cohort, len(common), len(available)-len(included)


def prediction_for(root, run_id, sid):
    path = root / run_id / 'predictions' / f'{sid}.xml'
    return path if path.is_file() else None


def index_slides(slide_dir):
    found = {}
    if not slide_dir.is_dir():
        return found
    for path in slide_dir.rglob('*'):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not path.is_file() and not (path.suffix.lower() == '.mrxs' and path.is_dir()):
            continue
        key = path.name.split('.')[0] if '.ome.' in path.name.lower() else path.stem
        if key in found and path != found[key]:
            raise ValueError(f'Multiple slide files with ID {key}: {found[key]} and {path}')
        found[key] = path
    return found


def candidate_cases(cases, by_run, top3, outputs, difficulty, require_predictions, slides, metric):
    found = []
    per_model = {
        model['run_id']: {slide_id_for_grouping(filename): scores for filename, scores
                          in by_run[model['run_id']].items()}
        for model in top3
    }
    for sid, metadata in cases.items():
        if any(sid not in per_model[model['run_id']] for model in top3):
            continue
        vals = [per_model[model['run_id']][sid][metric] for model in top3]
        paths = [prediction_for(outputs, model['run_id'], sid) for model in top3]
        has_slide = sid in slides if slides is not None else None
        if require_predictions and (not all(paths) or not has_slide):
            continue
        worst, mean, spread = min(vals), sum(vals)/len(vals), max(vals)-min(vals)
        rank_value = -spread if difficulty == 'disagreement' else worst if difficulty == 'worst-model' else mean
        found.append(dict(slide_id=sid, organ=metadata.get('organ', ''), staining=metadata.get('staining', ''),
                          metric=metric, mean_metric=mean, worst_metric=worst, metric_spread=spread,
                          available_predictions=sum(bool(p) for p in paths),
                          has_slide=has_slide, _sort=rank_value))
    return sorted(found, key=lambda r:(r['_sort'], r['slide_id']))


def read_thumbnail(path, max_size):
    from PIL import Image
    if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:
        with Image.open(path) as im:
            native = im.size
            im.thumbnail((max_size,max_size), Image.Resampling.LANCZOS)
            return im.convert('RGB').copy(), native
    try:
        import openslide
        with openslide.OpenSlide(str(path)) as wsi:
            return wsi.get_thumbnail((max_size,max_size)).convert('RGB'), tuple(map(int,wsi.dimensions))
    except (ImportError, OSError, RuntimeError):
        if path.suffix.lower() not in {'.tif','.tiff','.qptiff'}:
            raise RuntimeError('WSI thumbnail requires OpenSlide (Python and native library)')
        import tifffile
        import numpy as np
        with tifffile.TiffFile(path) as tif:
            levels = tif.series[0].levels
            dimensions = [(int(x.pages[0].imagewidth),int(x.pages[0].imagelength)) for x in levels]
            candidates = [i for i,(w,h) in enumerate(dimensions) if max(w,h)>=max_size]
            idx = min(candidates,key=lambda i:dimensions[i][0]*dimensions[i][1]) if candidates else len(levels)-1
            w,h = dimensions[idx]
            if w*h>40_000_000:
                raise RuntimeError(f'No safe thumbnail pyramid level in {path.name}')
            image = levels[idx].asarray()
            if image.ndim==3 and image.shape[0] in (3,4) and image.shape[-1] not in (3,4):
                image=np.moveaxis(image,0,-1)
            im = Image.fromarray(image)
            im.thumbnail((max_size,max_size), Image.Resampling.LANCZOS)
            return im.convert('RGB'), dimensions[0]


def render_overlay(xml_path, native, thumb_size, destination, color="#FE6100"):
    from PIL import Image, ImageDraw
    root = ET.parse(xml_path).getroot()
    if root.tag != 'ASAP_Annotations':
        raise ValueError(f'Unsupported XML root in {xml_path}')
    rings = []
    for anno in root.findall('./Annotations/Annotation'):
        if anno.attrib.get('Type','').casefold() not in ('polygon','spline'):
            raise ValueError(f'Unsupported annotation type in {xml_path}')
        coords=sorted(anno.findall('./Coordinates/Coordinate'),key=lambda el:int(el.get('Order','0')))
        xy=[(number(el.get('X')),number(el.get('Y'))) for el in coords]
        if len(xy)<3 or any(x is None or y is None for x,y in xy):
            raise ValueError(f'Invalid polygon in {xml_path}')
        rings.append((anno.get('PartOfGroup','').casefold()=='hole',xy))
    factor=2
    w,h=thumb_size
    img=Image.new('L',(w*factor,h*factor),0)
    draw=ImageDraw.Draw(img)
    for is_hole in (False,True):
        for hole, xy in rings:
            if is_hole!=hole:continue
            draw.polygon([(x*w*factor/native[0],y*h*factor/native[1]) for x,y in xy], fill=0 if hole else 255)
    img=img.resize((w,h),Image.Resampling.LANCZOS)
    alpha=img.point(lambda x:round(x*.34))
    rgb=tuple(int(color[i:i+2],16) for i in (1,3,5))
    overlay=Image.new('RGBA',(w,h),(*rgb,0))
    overlay.putalpha(alpha)
    overlay.save(destination,'WEBP',lossless=True)


def write_json(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    tmp.replace(path)


def slide_id_for_grouping(value):
    """Preserve dotted TCGA/Prostate identifiers; remove only known file extensions."""
    identifier = str(value or '').strip().replace('\\', '/').rsplit('/', 1)[-1]
    for extension in ('.xml', '.svs', '.tiff', '.tif', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu'):
        if identifier.casefold().endswith(extension):
            return identifier[:-len(extension)]
    return identifier


def load_stratification_metadata(path):
    """Read both sheets of benchmark_data_annotation.xlsx using the standard library."""
    if path is None or not Path(path).is_file():
        return {}
    if Path(path).suffix.lower() != '.xlsx':
        meta = {}
        for row in csv_rows(path):
            normalized = {str(k).lower().strip().replace(' ', '_'): str(v or '').strip() for k, v in row.items() if k}
            identifier = slide_id_for_grouping(normalized.get('slide_id',''))
            if identifier: meta[identifier] = normalized
        return meta
    namespace = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    meta = {}
    with zipfile.ZipFile(path) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            entries = ET.fromstring(archive.read('xl/sharedStrings.xml')).findall(namespace+'si')
            shared = [''.join(n.text or '' for n in entry.iter(namespace+'t')) for entry in entries]
        sheetnames = sorted((n for n in archive.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',n)),
                            key=lambda name: int(re.search(r'sheet(\d+)',name).group(1)))
        for sheetname in sheetnames:
            root=ET.fromstring(archive.read(sheetname))
            rows=[]
            for item in root.findall('.//'+namespace+'sheetData/'+namespace+'row'):
                cells={}
                for cell in item.findall(namespace+'c'):
                    match=re.match(r'[A-Z]+',cell.get('r',''))
                    if not match: continue
                    val=cell.find(namespace+'v')
                    inline=cell.find(namespace+'is')
                    content=val.text if val is not None else ''.join(n.text or '' for n in inline.iter(namespace+'t')) if inline is not None else ''
                    if cell.get('t')=='s' and content is not None: content=shared[int(content)]
                    cells[match.group()]=str(content or '').strip()
                if cells: rows.append(cells)
            if not rows: continue
            titles={col: text.strip().casefold().replace(' ','_') for col,text in rows[0].items()}
            if 'slide_id' not in titles.values():continue
            for row in rows[1:]:
                data={name: row.get(col,'') for col,name in titles.items()}
                key=slide_id_for_grouping(data.get('slide_id',''))
                if not key: continue
                item=meta.setdefault(key,{})
                for fld in ('staining','organ','data_source','source_site','license'):
                    if data.get(fld) and not item.get(fld): item[fld]=data[fld]
    return meta


def distribution_summary(values):
    """Exact observed slide-level distribution. No inferential uncertainty claims."""
    values = sorted(values)
    if not values:
        return None
    def quantile(p):
        idx=(len(values)-1)*p
        i=int(idx)
        t=idx-i
        return values[i]*(1-t)+values[min(i+1,len(values)-1)]*t
    return dict(n=len(values), median=round(quantile(.5),6),
                q1=round(quantile(.25),6), q3=round(quantile(.75),6),
                min=round(values[0],6), max=round(values[-1],6),
                values=[round(v,6) for v in values])


def export_performance_breakdown(args, cohort, cohort_key, leaderboard, by_run):
    """Compute subgroup means using the overall matched cohort and methods."""
    file = locate_metadata(args)
    outfile = args.out_dir / 'benchmark-subgroups.json'
    if file is None:
        if outfile.is_file(): outfile.unlink()
        print('Subgroup leaderboard skipped: annotation metadata missing', file=sys.stderr)
        return
    metadata = load_stratification_metadata(file)
    fields = {'stain':'staining', 'organ':'organ', 'center':'source_site', 'source':'data_source'}
    grouped = {dimension:defaultdict(list) for dimension in fields}
    counts = {dimension:0 for dimension in fields}
    for filename in cohort:
        row = metadata.get(slide_id_for_grouping(filename), {})
        for dimension,field in fields.items():
            value = (row.get(field) or '').strip()
            if value and value.casefold() not in ('unknown','n/a','nan','not available','none','null','-'):
                grouped[dimension][value].append(filename)
                counts[dimension] += 1
    minimum=args.min_subgroup_slides
    timed = load_available_timings(args.outputs, {m['run_id'] for m in leaderboard}, cohort, 'matched')
    payload = dict(schema_version=1, cohort_key=cohort_key,matched_slides=len(cohort),
        min_group_slides=minimum,generated_at=datetime.now(timezone.utc).isoformat(),
        summary='Arithmetic means over per-slide scores, subgroup subset of overall matched cohort',
        groupings={})
    for dimension,groups in grouped.items():
        rows=[]
        for label,files in groups.items():
            if len(files)<minimum: continue
            scores={}
            for model in leaderboard:
                run=model['run_id']
                means={metric:round(sum(by_run[run][f][metric] for f in files)/len(files),6)
                       for metric in ('dice','nsd','under','over')}
                timing_samples=timed.get(run,{}).get('samples',{})
                measured=[timing_samples[slide_stem(f)] for f in files if slide_stem(f) in timing_samples]
                means['speed']=round(sum(measured)/len(measured),6) if measured else None
                means['timing_n']=len(measured)
                scores[run]=means
            rows.append(dict(label=label,n=len(files),scores=scores))
        rows.sort(key=lambda r:(-r['n'],r['label'].casefold()))
        payload['groupings'][dimension]=dict(annotated_slides=counts[dimension],
            missing_slides=len(cohort)-counts[dimension],groups=rows)
    write_json(outfile,payload)
    print('Published dynamic subgroup leaderboard: '+', '.join(
        f'{dimension}={len(info["groups"])}' for dimension,info in payload['groupings'].items()))



def clean_showcase_images(folder, keep=()):
    """Remove obsolete generated showcase images without touching other assets."""
    keep = set(keep)
    for path in Path(folder).glob('showcase*.webp'):
        if path.is_file() and path.name not in keep and (path.name == 'showcase.webp' or path.name.startswith('showcase-')):
            path.unlink()

def display_top_argument(value):
    """Parse a website-only row limit; ``all`` means no limit."""
    if value.casefold() == 'all':
        return None
    try:
        cap = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected a positive integer or "all"') from exc
    if cap <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer or "all"')
    return cap


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    p.add_argument('--version', action='version', version='query_showcase.py dynamic v9 (2026-10-28; script-controlled leaderboard limit)')
    p.add_argument('--metrics', type=Path, default=Path('outputs/tissue-segmentation/metrics.csv'))
    p.add_argument('--slides', type=Path, default=Path('/Volumes/PA_CPGARCHIVE/projects/tissue-segmentation/data/annotation/slides'))
    p.add_argument('--outputs', type=Path, default=Path('outputs/tissue-segmentation'))
    p.add_argument('--out-dir', type=Path, default=Path('docs/assets'))
    p.add_argument('--annotations', type=Path, help='full benchmark_data_annotation.xlsx or metadata CSV; autodetected if available')
    p.add_argument('--showcase-by', choices=['all','source','center','organ','stain'], default='all',
                   help='metadata category used to select eligible challenging cases')
    p.add_argument('--showcase-value', help='exact metadata category value (case/diacritic insensitive)')
    p.add_argument('--list-showcase-values', nargs='?', const='all', choices=['all','source','center','organ','stain'],
                   metavar='CATEGORY', help='print available values from annotation metadata and exit')
    p.add_argument('--matched-slides',type=int,default=300, help='common-slide cohort size for ranking the methods')
    p.add_argument('--top',type=int,default=3, help='number of top configurations to showcase')
    p.add_argument('--display-top', '--show-top', dest='display_top', type=display_top_argument,
                   default=None, metavar='N|all',
                   help='maximum methods shown in BOTH website leaderboards (default: all); distinct from --top for the showcase')
    p.add_argument('--min-subgroup-slides', type=int, default=5, help='minimum matched slides per subgroup')
    p.add_argument('--timing-scope',choices=['available','matched'],default='available',
                   help='average seconds/slide from all completed predictions (available), or only matched-cohort slides (matched)')
    p.add_argument('--metric','--rank-metric', dest='metric', choices=['dice','nsd'], default='dice', help='metric for selecting top methods and the most challenging case')
    p.add_argument('--difficulty',choices=['mean','worst-model','disagreement','mean-dice'],default='mean',help='case difficulty: lowest top-3 mean score, lowest single-model score, or largest disagreement')
    p.add_argument('--require-predictions',action='store_true',help='select only a case with all exact-run XMLs and source slide')
    p.add_argument('--slide-id',help='optional exact slide override within selected metadata filter')
    p.add_argument('--max-size',type=int,default=1200)
    p.add_argument('--overlay-color', default='#FFB000', help='HTML hex color used for generated prediction overlays')
    p.add_argument('--no-images',action='store_true',help='test ranking without thumbnail generation')
    args=p.parse_args()
    if not re.fullmatch(r'#[0-9a-fA-F]{6}',args.overlay_color):
        p.error('--overlay-color must be a six-digit hex color, e.g. #FFB000')
    args.overlay_color = args.overlay_color.upper()
    if args.difficulty == 'mean-dice':  # Retain compatibility with earlier CLI arguments.
        args.difficulty = 'mean'
    if args.matched_slides<1 or args.top<1 or args.max_size<64 or args.min_subgroup_slides<1:
        p.error('--matched-slides, --top and --min-subgroup-slides must be positive; --max-size >=64')
    if args.showcase_by == 'all' and args.showcase_value is not None:
        p.error('--showcase-value requires --showcase-by source, center, organ or stain')
    if args.showcase_by != 'all' and not args.showcase_value and args.list_showcase_values is None:
        p.error('--showcase-by requires --showcase-value (or run --list-showcase-values)')
    try:
        all_metadata, annotation_source = showcase_metadata(args)
    except (FileNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        p.error(str(exc))
    if args.list_showcase_values is not None:
        list_showcase_values(all_metadata, args.list_showcase_values)
        return 0
    cases = filter_showcase_cases(all_metadata, args.showcase_by, args.showcase_value, p)
    if not args.metrics.is_file():
        p.error(f'No metrics: {args.metrics}')
    if args.require_predictions and not args.slides.is_dir():
        print(f'WARNING: Slide folder missing: {args.slides}; leaderboard can still be refreshed',file=sys.stderr)
    try:
        by_run, labels, row_count = load_metrics(args.metrics)
        leaderboard, cohort_key, cohort, common_count, rejected = compute_leaderboard(
            by_run, labels, args.matched_slides, args.metric, args.outputs, args.timing_scope)
    except (FileNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        p.error(str(exc))
    if len(leaderboard)<args.top:
        p.error(f'Not enough matched methods for top {args.top}')
    top3=leaderboard[:args.top]
    # Publish quantitative results independently of showcase availability.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Remove obsolete exports that are no longer consumed by the site.
    for obsolete in ('benchmark-top-methods.json', 'showcase-difficulty-ranking.json',
                     'case-gdansk.json', 'gdansk-difficulty-ranking.json'):
        (args.out_dir / obsolete).unlink(missing_ok=True)
    leaderboard_payload = dict(schema_version=3,cohort_key=cohort_key,
        matched_slides=args.matched_slides,ranking_metric=args.metric,timing_scope=args.timing_scope,
        display_top=args.display_top,
        methods=[dict(method=m['label'],
                      method_name=m['label'].split('/', 1)[0].strip(),
                      architecture=m['label'].split('/', 1)[1].strip() if '/' in m['label'] else None,
                      **m, cohort_key=cohort_key)
                 for m in leaderboard],
        generated_at=datetime.now(timezone.utc).isoformat())
    write_json(args.out_dir/'benchmark-results.json', leaderboard_payload)
    export_performance_breakdown(args, cohort, cohort_key, leaderboard, by_run)
    print(f'Published {len(leaderboard)} matched methods to {(args.out_dir / "benchmark-results.json").resolve()}')
    print(f'Website leaderboard display: {"all methods" if args.display_top is None else f"top {args.display_top} methods"} (both leaderboards)')
    for model in leaderboard:
        if model['speed'] is not None:
            print(f"  Timing {model['label']}: {model['speed']:.2f} s/slide ({model['timing_n']} completed {'slide' if model['timing_n'] == 1 else 'slides'})")
    if not any(model['speed'] is not None for model in leaderboard):
        print(f'No usable per-prediction timing files found under {args.outputs}/<run-id>/timings/ yet')
    slide_index=index_slides(args.slides) if args.require_predictions or not args.no_images else None
    eligible=candidate_cases(cases, by_run,top3,args.outputs,args.difficulty,False,slide_index,args.metric)
    complete=candidate_cases(cases, by_run,top3,args.outputs,args.difficulty,True,slide_index,args.metric) if args.require_predictions else eligible
    if args.slide_id:
        complete=[c for c in complete if c['slide_id']==args.slide_id]
    showcase_filter = {'category': args.showcase_by, 'value': args.showcase_value}
    if not complete:
        lines=[f'No eligible case with {args.top} matched model scores'+(' and all predictions/slide' if args.require_predictions else ''),
               f'Showcase filter: {args.showcase_by}={args.showcase_value or "all"}',
               'Top methods: '+', '.join(m['label'] for m in top3),
               f'Candidate cases scored by all top methods: {len(eligible)}']
        if args.require_predictions and eligible:
            lines.append('Candidates (XMLs available out of top methods, original slide present):')
            lines.extend(f"  {c['slide_id']}: {c['available_predictions']}/{len(top3)} XMLs, slide={c['has_slide']}" for c in eligible[:10])
        # Clear outdated showcase artifacts when no case is available.
        write_json(args.out_dir/'case-showcase.json',dict(schema_version=3,
            status='unavailable', ranking_metric=args.metric, models=[],
            showcase_filter=showcase_filter, source_metadata=annotation_source,
            generated_at=datetime.now(timezone.utc).isoformat()))
        clean_showcase_images(args.out_dir)
        print('\n'.join(lines), file=sys.stderr)
        print('Leaderboard was updated. Showcase unavailable until all required source files exist.',file=sys.stderr)
        return 0
    row=complete[0]
    sid=row['slide_id']
    selected=cases[sid]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    # Stage fixed-name showcase assets before publishing them.
    thumbnail = None
    overlay_names = {}
    current_slide = slide_index.get(sid) if slide_index is not None else None
    written_images = set()
    if not args.no_images:
        if current_slide is None:
            print(f'WARNING: Original slide not found: {sid}', file=sys.stderr)
        else:
            try:
                with tempfile.TemporaryDirectory(prefix='.showcase-stage-', dir=args.out_dir) as temporary:
                    stage = Path(temporary)
                    thumb, native = read_thumbnail(current_slide, args.max_size)
                    thumbnail = 'showcase.webp'
                    thumb.save(stage / thumbnail, format='WEBP', quality=90)
                    written_images.add(thumbnail)
                    for idx, model in enumerate(top3, 1):
                        xml = prediction_for(args.outputs, model['run_id'], sid)
                        if xml:
                            image_name = f'showcase-model-{idx}.webp'
                            render_overlay(xml, native, thumb.size, stage / image_name, args.overlay_color)
                            overlay_names[model['run_id']] = image_name
                            written_images.add(image_name)
                    # Publish only after all requested images are rendered.
                    for image_name in sorted(written_images):
                        (stage / image_name).replace(args.out_dir / image_name)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                p.error(f'Could not render {sid}: {exc}. For scores only pass --no-images')
    models=[]
    for rank,model in enumerate(top3,1):
        run=model['run_id']
        xml=prediction_for(args.outputs,run,sid)
        models.append(dict(rank=rank,label=model['label'],run_id=run,
                           cohort_dice=round(model['dice'],6), cohort_nsd=round(model['nsd'],6),
                           metrics=next(scores for filename,scores in by_run[run].items() if slide_stem(filename)==sid),
                           prediction_exists=bool(xml),
                           overlay=overlay_names.get(run)))
    payload=dict(schema_version=3,status='ready',ranking_metric=args.metric,slide_id=sid,metadata=selected,
                 thumbnail=thumbnail,annotation_color=args.overlay_color,difficulty=args.difficulty,
                 difficulty_score=round(row['mean_metric'] if args.difficulty=='mean' else row['worst_metric'] if args.difficulty=='worst-model' else row['metric_spread'],6),
                 cohort=dict(slides=args.matched_slides, id=cohort_key, shared_slides=common_count),
                 models=models, showcase_filter=showcase_filter, source_metadata=annotation_source,
                 generated_at=datetime.now(timezone.utc).isoformat())
    write_json(args.out_dir/'case-showcase.json',payload)
    clean_showcase_images(args.out_dir, written_images)
    print('Top '+str(args.top)+' by '+args.metric.upper()+' (matched '+str(args.matched_slides)+' slides):')
    for model in top3:
        print(f"  {model['label']}: {model[args.metric]:.4f} [{model['run_id'][:65]}]")
    print(f"Selected showcase slide: {sid} ({selected.get('organ', '')}; {selected.get('staining', '')})")
    print(f"Difficulty: mean {args.metric.upper()}={row['mean_metric']:.4f}, lowest={row['worst_metric']:.4f}, spread={row['metric_spread']:.4f}")
    print(f'Case original: {current_slide or "not located"}')
    for model in models:
        print(f"  {model['label']}: case {args.metric.upper()}={model['metrics'][args.metric]:.4f}, XML={'yes' if model['prediction_exists'] else 'no'}")
    print(f'Updated: {(args.out_dir / "case-showcase.json").resolve()}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
