from __future__ import annotations
import csv
import hashlib
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from statistics import median
import json
import math
import os
from pathlib import Path
from threading import Event, Lock, Thread
import sqlite3
import sys
import time
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, shape
from shapely import make_valid
from shapely.validation import explain_validity
from shapely.ops import unary_union
from shapely.prepared import prep
try:
    import orjson
except ImportError:
    orjson = None
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
METADATA_DIR = REPO_ROOT / 'metadata'
METHODS_FILE = METADATA_DIR / 'methods_cpg.txt'
GROUND_TRUTH_DIR = METADATA_DIR / 'xmls_cpg'
ANNOTATIONS_CSV = None
RESULTS_DIR = REPO_ROOT / 'results' / 'fast_evaluation_benchmark-tissue-masks_cpg'
METADATA_FILE = METADATA_DIR / 'datateam_crowdsource_data_overview.xlsx'
OFFICIAL_METHODS_FILE = METHODS_FILE
GROUND_TRUTH_DIR_CPG = GROUND_TRUTH_DIR
RETRY_FAILED_ONLY = False
REPAIR_ON_FIRST_FAILURE = True
REPAIR_INVALID_GT_POLYGONS = True
REPAIR_OUTSIDE_GT_HOLES = True
MAX_HOLE_OUTSIDE_FRACTION = 0.02
REPAIR_POLICY_KEY = json.dumps({'repair_version': 1, 'invalid_gt': REPAIR_INVALID_GT_POLYGONS, 'outside_holes': REPAIR_OUTSIDE_GT_HOLES, 'max_hole_outside_fraction': MAX_HOLE_OUTSIDE_FRACTION}, sort_keys=True)
WORKERS = 8
GROUND_TRUTH_GROUPS = ('tissue',)
LIMIT_SLIDES = None
FORCE_RECOMPUTE = False
INVENTORY_ONLY = False
CSV_UPDATE_SECONDS = 30
PROGRESS_EVERY_SLIDES = 1
HEARTBEAT_SECONDS = 5.0
DISCOVERY_EVERY_SLIDES = 100
TRUST_PREVIOUS_XML_INVENTORY = False
VERIFY_CACHED_FILE_CHANGES = False
AUTO_FOLLOW_NEW_PREDICTIONS = False
REFRESH_PREDICTION_INVENTORY = True
LIVE_IDLE_POLL_SECONDS = 90
LIVE_EVALUATION_WAVE_SECONDS = 180
LIVE_MIN_PREDICTION_AGE_SECONDS = 5
INVENTORY_WORKERS = 6
SAVE_COMPLETENESS_PROVENANCE = True
INVENTORY_CACHE_FILENAME = 'prediction_inventory.json'
NEAR_EMPTY_PREDICTION_GT_AREA_FRACTION = 0.001
COLLECT_INFERENCE_TIMINGS = True
TIMINGS_ONLY = False
TIMING_SYNC_STATUSES = ('ok', 'error')
TIMING_BACKFILL_ALL_SLIDES = True
TIMING_READ_WORKERS = 12
TIMING_RETRY_UNAVAILABLE = True
TIMING_MISSING_RETRY_SECONDS = 600
REFRESH_TIMING_CACHE = False
CACHE_VERSION = 2
PREDICTION_EXTENSIONS = frozenset({'.geojson', '.xml'})
GT_EXTENSIONS = frozenset({'.geojson', '.json', '.xml'})
SCORE_FIELDS = ('method', 'method_path', 'slide_id', 'prediction_path', 'annotation_path', 'status', 'dice', 'over_segmentation_rate', 'under_segmentation_rate', 'pred_area_px2', 'gt_area_px2', 'intersection_area_px2', 'error', 'geometry_repaired', 'repair_details', 'prediction_empty', 'prediction_near_empty', 'prediction_content_status', 'prediction_area_ratio_to_gt', 'timing_status', 'timing_error', 'timing_path', 'timing_schema_version', 'total_processing_time_s', 'seconds_per_native_megapixel', 'initialization_time_s', 'algorithm_time_s', 'segmentation_call_time_s', 'image_loading_time_s', 'preprocessing_time_s', 'polygon_conversion_time_s', 'serialization_time_s', 'native_pixels', 'device', 'measurement_phase', 'timing_measured_at_utc')
SUMMARY_FIELDS = ('method', 'method_path', 'historical_total_unverified', 'n_gt_available', 'n_predictions', 'n_missing_predictions', 'n_evaluated', 'n_cached', 'n_failed', 'n_pending', 'n_repaired', 'n_empty_predictions', 'n_near_empty_predictions', 'n_nonempty_predictions', 'n_unassessed_predictions', 'n_zero_byte_prediction_files', 'empty_prediction_rate_pct', 'near_empty_prediction_rate_pct', 'n_timings_ok', 'n_timings_missing', 'n_timings_invalid', 'n_timings_unassessed', 'mean_total_processing_time_s', 'median_total_processing_time_s', 'mean_seconds_per_native_megapixel', 'mean_dice', 'mean_over_segmentation_rate', 'mean_under_segmentation_rate')
CACHE_KEY = json.dumps({'version': CACHE_VERSION, 'ground_truth_groups': sorted((group.strip().casefold() for group in GROUND_TRUTH_GROUPS))}, sort_keys=True)

def polygon_union(payload: dict):
    if not isinstance(payload, dict) or payload.get('type') != 'FeatureCollection' or (not isinstance(payload.get('features'), list)):
        raise ValueError('Expected GeoJSON FeatureCollection')
    geometries = []
    for i, feature in enumerate(payload['features']):
        try:
            geom = shape(feature['geometry'])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f'Invalid GeoJSON feature {i}: {exc}') from exc
        if geom.is_empty:
            continue
        if not geom.is_valid:
            raise ValueError(f'Invalid polygon geometry in feature {i}')
        if geom.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError(f'Non-polygon geometry in feature {i}: {geom.geom_type}')
        if not math.isfinite(geom.area) or geom.area <= 0:
            raise ValueError(f'Invalid polygon area in feature {i}')
        geometries.append(geom)
    if not geometries:
        return Polygon()
    result = unary_union(geometries) if len(geometries) > 1 else geometries[0]
    if not result.is_valid:
        raise ValueError('Invalid polygon union')
    return result

def asap_ring(node: ET.Element):
    name = node.attrib.get('Name', '<unnamed>')
    coords = node.findall('./Coordinates/Coordinate')
    if not coords:
        raise ValueError(f'ASAP annotation {name!r} has no coordinates')
    orders = [elem.attrib.get('Order') for elem in coords]
    if any((x is not None for x in orders)) and (not all((x is not None for x in orders))):
        raise ValueError(f'Partially specified coordinate order in {name!r}')
    if all((x is not None for x in orders)):
        order = [int(x) for x in orders]
        if len(set(order)) != len(order):
            raise ValueError(f'Duplicate coordinate orders in {name!r}')
        coords = [elem for _, elem in sorted(zip(order, coords), key=lambda item: item[0])]
    points = []
    for elem in coords:
        x, y = (float(elem.attrib['X']), float(elem.attrib['Y']))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f'Non-finite coordinate in {name!r}')
        if not points or points[-1] != (x, y):
            points.append((x, y))
    if len(points) > 1 and points[-1] == points[0]:
        points.pop()
    if len(set(points)) < 3:
        raise ValueError(f'Fewer than three vertices in {name!r}')
    return points + [points[0]]

def load_asap_geometry(path, tissue_groups: tuple[str, ...]):
    root = ET.parse(path).getroot()
    if root.tag != 'ASAP_Annotations':
        raise ValueError(f'Unexpected ASAP XML root: {root.tag}')
    wanted = {x.strip().casefold() for x in tissue_groups if x.strip()}
    tissue, names, pending_holes = ([], {}, [])
    for item in root.findall('./Annotations/Annotation'):
        name = item.attrib.get('Name', '')
        group = item.attrib.get('PartOfGroup', '').strip().casefold()
        kind = item.attrib.get('Type', '').strip().casefold()
        is_hole = group == 'hole'
        if group not in wanted and (not is_hole):
            if kind in {'polygon', 'spline', 'rectangle'}:
                raise ValueError(f'Unselected ASAP group {group!r} in {name!r}')
            continue
        if kind not in {'polygon', 'spline'}:
            raise ValueError(f'Unsupported ASAP type {kind!r} in {name!r}')
        geom = Polygon(asap_ring(item))
        if not geom.is_valid or geom.area <= 0:
            raise ValueError(f'Invalid ASAP polygon {name!r}')
        if is_hole:
            prefix = 'Hole for '
            parent = name.rsplit(' / ', 1)[0][len(prefix):] if name.startswith(prefix) else None
            pending_holes.append((name, parent, geom))
        else:
            names.setdefault(name, []).append(geom)
            tissue.append(geom)
    gt = unary_union(tissue) if tissue else Polygon()
    if pending_holes:
        holes = []
        for name, parent, hole in pending_holes:
            if parent is not None and parent not in names:
                raise ValueError(f'ASAP hole {name!r} has missing parent')
            if parent is not None and len(names[parent]) != 1:
                raise ValueError(f'ASAP hole {name!r} has ambiguous parent')
            if parent is not None and (not names[parent][0].covers(hole)):
                raise ValueError(f'ASAP hole {name!r} lies outside its parent')
            if not gt.covers(hole):
                raise ValueError(f'ASAP hole {name!r} lies outside tissue')
            holes.append(hole)
        gt = gt.difference(unary_union(holes))
        if not gt.is_valid:
            raise ValueError('Invalid ASAP tissue after hole subtraction')
    return gt

def _polygonal_components(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom] if geom.area > 0 else []
    if geom.geom_type in {'MultiPolygon', 'GeometryCollection'}:
        parts = []
        for part in geom.geoms:
            parts.extend(_polygonal_components(part))
        return parts
    return []

def _valid_polygon_area(geom, context, notes):
    if geom.is_valid and geom.area > 0:
        return geom
    if not REPAIR_INVALID_GT_POLYGONS:
        raise ValueError(f'{context}: invalid polygon (repairs disabled)')
    reason = explain_validity(geom)
    fixed = make_valid(geom)
    components = _polygonal_components(fixed)
    if not components:
        raise ValueError(f'{context}: make_valid returned no positive-area polygons ({reason})')
    area = unary_union(components) if len(components) > 1 else components[0]
    if area.is_empty or not area.is_valid or (not math.isfinite(area.area)) or (area.area <= 0):
        raise ValueError(f'{context}: make_valid did not produce valid polygonal area')
    notes.append(f'{context}: make_valid ({reason}); area {geom.area:.6g} -> {area.area:.6g}')
    return area

def _clip_hole_if_small_overhang(hole, allowed, name, where, notes):
    if allowed.covers(hole):
        return hole
    if not REPAIR_OUTSIDE_GT_HOLES:
        raise ValueError(f'ASAP hole {name!r} lies outside {where} (repair disabled)')
    clipped = hole.intersection(allowed)
    parts = _polygonal_components(clipped)
    if not parts:
        raise ValueError(f'ASAP hole {name!r} has no area inside {where}')
    clipped = unary_union(parts) if len(parts) > 1 else parts[0]
    outside_fraction = max(0.0, 1.0 - clipped.area / hole.area)
    if outside_fraction > MAX_HOLE_OUTSIDE_FRACTION + 1e-10:
        raise ValueError(f'ASAP hole {name!r} is {outside_fraction:.2%} outside {where}, exceeding the {MAX_HOLE_OUTSIDE_FRACTION:.2%} automatic-repair limit')
    if not clipped.is_valid or clipped.area <= 0:
        raise ValueError(f'ASAP hole {name!r} invalid after clipping to {where}')
    notes.append(f'{name}: clipped {outside_fraction:.3%} hole area outside {where}')
    return clipped

def load_asap_geometry_repaired(stream, tissue_groups: tuple[str, ...]):
    root = ET.parse(stream).getroot()
    if root.tag != 'ASAP_Annotations':
        raise ValueError(f'Unexpected ASAP XML root: {root.tag}')
    wanted = {x.strip().casefold() for x in tissue_groups if x.strip()}
    tissue, names, holes, notes = ([], {}, [], [])
    for item in root.findall('./Annotations/Annotation'):
        name = item.attrib.get('Name', '')
        group = item.attrib.get('PartOfGroup', '').strip().casefold()
        kind = item.attrib.get('Type', '').strip().casefold()
        is_hole = group == 'hole'
        if group not in wanted and (not is_hole):
            if kind in {'polygon', 'spline', 'rectangle'}:
                raise ValueError(f'Unselected ASAP group {group!r} in {name!r}')
            continue
        if kind not in {'polygon', 'spline'}:
            raise ValueError(f'Unsupported ASAP type {kind!r} in {name!r}')
        original = Polygon(asap_ring(item))
        geom = _valid_polygon_area(original, f'Annotation {name!r}', notes)
        if is_hole:
            prefix = 'Hole for '
            parent = name.rsplit(' / ', 1)[0][len(prefix):] if name.startswith(prefix) else None
            holes.append((name, parent, geom))
        else:
            names.setdefault(name, []).append(geom)
            tissue.append(geom)
    union = unary_union(tissue) if tissue else Polygon()
    union = _valid_polygon_area(union, 'GT tissue union', notes) if not union.is_empty and (not union.is_valid) else union
    resolved_holes = []
    for name, parent, hole in holes:
        if parent is not None:
            if parent not in names:
                raise ValueError(f'ASAP hole {name!r} has missing parent {parent!r}')
            if len(names[parent]) != 1:
                raise ValueError(f'ASAP hole {name!r} has ambiguous parent {parent!r}')
            hole = _clip_hole_if_small_overhang(hole, names[parent][0], name, 'its parent', notes)
        hole = _clip_hole_if_small_overhang(hole, union, name, 'tissue', notes)
        resolved_holes.append(hole)
    if resolved_holes:
        union = union.difference(unary_union(resolved_holes))
    if not union.is_valid:
        union = _valid_polygon_area(union, 'GT after hole subtraction', notes)
    if not notes:
        raise ValueError('No repairable GT geometry issue detected')
    if not union.is_valid or not math.isfinite(union.area):
        raise ValueError('Invalid ground truth after attempted in-memory repair')
    return (union, '; '.join(notes))

def load_repaired_gt_with_signature(path_str, tissue_groups):
    path = Path(path_str)
    if path.suffix.casefold() != '.xml':
        raise ValueError('In-memory GT repair supports ASAP XML annotations only')
    with path.open('rb') as stream:
        st = os.fstat(stream.fileno())
        geom, details = load_asap_geometry_repaired(stream, tissue_groups)
    return (geom, (st.st_size, st.st_mtime_ns), details)

def load_geometry_with_signature(path_str: str, tissue_groups: tuple[str, ...]):
    path = Path(path_str)
    with path.open('rb') as stream:
        st = os.fstat(stream.fileno())
        if st.st_size == 0:
            raise ValueError('Zero-byte XML/GeoJSON file')
        sig = (st.st_size, st.st_mtime_ns)
        if path.suffix.casefold() == '.xml':
            geom = load_asap_geometry(stream, tissue_groups)
        else:
            data = stream.read()
            payload = orjson.loads(data) if orjson is not None else json.loads(data)
            geom = polygon_union(payload)
    return (geom, sig)

def metrics(pred, gt, gt_prepared):
    pred_area, gt_area = (float(pred.area), float(gt.area))
    intersection = 0.0 if pred.is_empty or gt.is_empty or (not gt_prepared.intersects(pred)) else float(pred.intersection(gt).area)
    fp = max(0.0, pred_area - intersection)
    fn = max(0.0, gt_area - intersection)
    return {'dice': 2.0 * intersection / (pred_area + gt_area) if pred_area + gt_area else 1.0, 'over_segmentation_rate': fp / gt_area if gt_area else None, 'under_segmentation_rate': fn / gt_area if gt_area else None, 'pred_area_px2': pred_area, 'gt_area_px2': gt_area, 'intersection_area_px2': intersection}

def read_annotations(gt_dir: Path, csv_path: Path | None):
    if not gt_dir.is_dir():
        raise FileNotFoundError(f'Ground-truth XML folder does not exist: {gt_dir}')
    xml_by_stem = {}
    for path in gt_dir.iterdir():
        if path.suffix.casefold() != '.xml' or not path.is_file():
            continue
        key = path.stem.casefold()
        if key in xml_by_stem:
            raise ValueError(f'Duplicate ground-truth XML stem {path.stem!r} in {gt_dir}')
        xml_by_stem[key] = path
    print(f'[ANNOTATIONS] Found {len(xml_by_stem)} local XML files in {gt_dir}', flush=True)
    if not xml_by_stem:
        raise FileNotFoundError(f'No .xml annotations found directly inside {gt_dir}')
    mapping = {}
    used_stems = set()
    csv_rows = missing_rows = csv_matches = 0
    if csv_path is not None and csv_path.is_file():
        print(f'[ANNOTATIONS] Using optional slide-ID mapping: {csv_path}', flush=True)
        seen_csv = set()
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not {'slide_id', 'ground_truth_annotation'}.issubset(fields):
                raise ValueError('Annotation CSV must contain slide_id,ground_truth_annotation')
            for lineno, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f'Extra fields at annotation CSV row {lineno}')
                slide = (row.get('slide_id') or '').strip()
                original = (row.get('ground_truth_annotation') or '').strip()
                if not slide or not original:
                    raise ValueError(f'Missing slide ID or annotation at CSV row {lineno}')
                slide_key = slide.casefold()
                if slide_key in seen_csv:
                    raise ValueError(f'Duplicate slide_id {slide!r} in mapping CSV')
                seen_csv.add(slide_key)
                csv_rows += 1
                original_stem = Path(original).stem.casefold()
                xml_key = next((k for k in (slide_key, original_stem) if k in xml_by_stem), None)
                if xml_key is None:
                    missing_rows += 1
                    continue
                mapping[slide_key] = (slide, str(xml_by_stem[xml_key].resolve()))
                used_stems.add(xml_key)
                csv_matches += 1
                if csv_rows % DISCOVERY_EVERY_SLIDES == 0:
                    print(f'[ANNOTATIONS] CSV {csv_rows} rows: {csv_matches} local XML matches', flush=True)
        print(f'[ANNOTATIONS] CSV rows={csv_rows}, mapped to local XML={csv_matches}, without local XML={missing_rows}', flush=True)
    else:
        print('[ANNOTATIONS] Mapping CSV unavailable or disabled; matching XML filenames directly to slide IDs', flush=True)
    direct_matches = 0
    for stem, path in xml_by_stem.items():
        if stem in used_stems or stem in mapping:
            continue
        mapping[stem] = (path.stem, str(path.resolve()))
        direct_matches += 1
    print(f'[ANNOTATIONS] Eligible slide IDs={len(mapping)} ({csv_matches} CSV-mapped, {direct_matches} filename-matched)', flush=True)
    return (mapping, len(xml_by_stem), missing_rows)

def _load_inventory_cache(path: Path):
    if not path.is_file():
        return {'schema': 1, 'methods': {}}
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot read saved prediction inventory {path}: {exc}') from exc
    if doc.get('schema') != 1 or not isinstance(doc.get('methods'), dict):
        raise ValueError(f'Invalid inventory cache schema: {path}')
    return doc

def _scan_prediction_names(pred_dir: Path):
    xml = geojson = 0
    by_stem = {}
    with os.scandir(pred_dir) as entries:
        for item in entries:
            suffix = Path(item.name).suffix.casefold()
            if suffix not in PREDICTION_EXTENSIONS or not item.is_file():
                continue
            if suffix == '.xml':
                xml += 1
            else:
                geojson += 1
            stem = Path(item.name).stem.casefold()
            previous = by_stem.get(stem)
            if previous is not None:
                prev_suffix = Path(previous).suffix.casefold()
                if prev_suffix == suffix:
                    raise ValueError(f'Duplicate prediction stem {stem!r} in {pred_dir}')
                if suffix != '.xml':
                    continue
            by_stem[stem] = item.name
    return {'files': by_stem, 'n_xml': xml, 'n_geojson': geojson, 'scanned_at_utc': datetime.now(timezone.utc).isoformat(), 'prediction_directory': str(pred_dir), 'inventory_source': 'scanned'}

def ensure_output_isolated(out_dir: Path, method_dirs, annotation_csv: Path | None, annotation_paths=()):
    out_dir = out_dir.resolve()

    def contained(path: Path, parent: Path) -> bool:
        return path == parent or parent in path.parents
    for method in method_dirs:
        method = method.resolve()
        if contained(out_dir, method) or contained(method, out_dir):
            raise ValueError(f'Unsafe RESULTS_DIR {out_dir}: overlaps method directory {method}')
    gt_dir = GROUND_TRUTH_DIR.resolve()
    if contained(out_dir, gt_dir) or contained(gt_dir, out_dir):
        raise ValueError(f'Unsafe RESULTS_DIR {out_dir}: overlaps ground-truth directory {gt_dir}')
    protected = [METHODS_FILE.resolve(), Path(__file__).resolve(), *(Path(p).resolve() for p in annotation_paths)]
    if annotation_csv is not None:
        protected.append(annotation_csv.resolve())
    for input_path in protected:
        if contained(input_path, out_dir):
            raise ValueError(f'Unsafe RESULTS_DIR {out_dir}: contains input file {input_path}')

def export_csv(path: Path, fields, rows):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

def signature(path: str):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)

def connect_db(path: Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute("\n        CREATE TABLE IF NOT EXISTS evaluation (\n          method_path TEXT NOT NULL, slide_key TEXT NOT NULL,\n          method TEXT NOT NULL, slide_id TEXT NOT NULL,\n          prediction_path TEXT NOT NULL, annotation_path TEXT NOT NULL,\n          pred_size INTEGER NOT NULL, pred_mtime_ns INTEGER NOT NULL,\n          gt_size INTEGER NOT NULL, gt_mtime_ns INTEGER NOT NULL,\n          protocol INTEGER NOT NULL, protocol_key TEXT,\n          status TEXT NOT NULL,\n          dice REAL, over_segmentation_rate REAL, under_segmentation_rate REAL,\n          pred_area_px2 REAL, gt_area_px2 REAL, intersection_area_px2 REAL,\n          error TEXT NOT NULL, evaluated_at TEXT NOT NULL,\n          geometry_repaired INTEGER NOT NULL DEFAULT 0,\n          repair_details TEXT NOT NULL DEFAULT '',\n          repair_policy_key TEXT,\n          PRIMARY KEY(method_path, slide_key)\n        )\n    ")
    fields = {r[1] for r in conn.execute('PRAGMA table_info(evaluation)')}
    if 'protocol_key' not in fields:
        conn.execute('ALTER TABLE evaluation ADD COLUMN protocol_key TEXT')
    for col, ddl in (('geometry_repaired', 'INTEGER NOT NULL DEFAULT 0'), ('repair_details', "TEXT NOT NULL DEFAULT ''"), ('repair_policy_key', 'TEXT')):
        if col not in fields:
            conn.execute(f'ALTER TABLE evaluation ADD COLUMN {col} {ddl}')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS timing_metadata (\n          method_path TEXT NOT NULL, slide_key TEXT NOT NULL,\n          payload_json TEXT NOT NULL, collected_at_utc TEXT NOT NULL,\n          PRIMARY KEY(method_path, slide_key)\n        )\n    ')
    conn.commit()
    return conn

def cache_lookup(conn):
    return {(r['method_path'], r['slide_key']): dict(r) for r in conn.execute('SELECT * FROM evaluation')}

def cached_ok(old, entry):
    if old is None or FORCE_RECOMPUTE or old['status'] != 'ok' or (old['protocol_key'] != CACHE_KEY):
        return False
    if any((old[field] != entry[field] for field in ('prediction_path', 'annotation_path'))):
        return False
    if old.get('geometry_repaired') and old.get('repair_policy_key') != REPAIR_POLICY_KEY:
        return False
    if not VERIFY_CACHED_FILE_CHANGES:
        return True
    try:
        ps, pm = signature(entry['prediction_path'])
        gs, gm = signature(entry['annotation_path'])
    except OSError:
        return False
    return (ps, pm, gs, gm) == (old['pred_size'], old['pred_mtime_ns'], old['gt_size'], old['gt_mtime_ns'])

def evaluate_slide(task):
    _, annotation, entries, can_repair = task
    repair_details = ''
    gt_sig = (0, 0)
    try:
        gt, gt_sig = load_geometry_with_signature(annotation, GROUND_TRUTH_GROUPS)
    except Exception as strict_exc:
        if not can_repair or not isinstance(strict_exc, ValueError):
            msg = f'Ground truth: {type(strict_exc).__name__}: {strict_exc}'
            try:
                gt_sig = signature(annotation)
            except OSError:
                pass
            return [(entry, 'error', {}, msg, (0, 0), gt_sig, False, '') for entry in entries]
        try:
            gt, gt_sig, repair_details = load_repaired_gt_with_signature(annotation, GROUND_TRUTH_GROUPS)
        except Exception as repair_exc:
            msg = f'Ground truth: {type(strict_exc).__name__}: {strict_exc}; in-memory repair failed: {type(repair_exc).__name__}: {repair_exc}'
            try:
                gt_sig = signature(annotation)
            except OSError:
                pass
            return [(entry, 'error', {}, msg, (0, 0), gt_sig, False, '') for entry in entries]
    try:
        prepared = prep(gt)
    except Exception as exc:
        msg = f'Ground truth: prepared geometry failed: {type(exc).__name__}: {exc}'
        return [(entry, 'error', {}, msg, (0, 0), gt_sig, False, '') for entry in entries]
    completed = []
    for entry in entries:
        try:
            pred, pred_sig = load_geometry_with_signature(entry['prediction_path'], ('tissue',))
            completed.append((entry, 'ok', metrics(pred, gt, prepared), '', pred_sig, gt_sig, bool(repair_details), repair_details))
        except FileNotFoundError as exc:
            completed.append((entry, 'missing_prediction', {}, str(exc), (0, 0), gt_sig, False, ''))
        except Exception as exc:
            try:
                failed_sig = signature(entry['prediction_path'])
            except OSError:
                failed_sig = (0, 0)
            completed.append((entry, 'error', {}, f'Prediction: {type(exc).__name__}: {exc}', failed_sig, gt_sig, False, ''))
    return completed

def store_result(conn, entry, status, values, error, repaired=False, details=''):
    conn.execute('\n        INSERT OR REPLACE INTO evaluation (\n            method_path, slide_key, method, slide_id, prediction_path, annotation_path,\n            pred_size, pred_mtime_ns, gt_size, gt_mtime_ns, protocol, protocol_key,\n            status, dice, over_segmentation_rate, under_segmentation_rate,\n            pred_area_px2, gt_area_px2, intersection_area_px2, error, evaluated_at,\n            geometry_repaired, repair_details, repair_policy_key\n        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)\n    ', (entry['method_path'], entry['slide_id'].casefold(), entry['method'], entry['slide_id'], entry['prediction_path'], entry['annotation_path'], entry['pred_size'], entry['pred_mtime_ns'], entry['gt_size'], entry['gt_mtime_ns'], CACHE_VERSION, CACHE_KEY, status, values.get('dice'), values.get('over_segmentation_rate'), values.get('under_segmentation_rate'), values.get('pred_area_px2'), values.get('gt_area_px2'), values.get('intersection_area_px2'), error, datetime.now(timezone.utc).isoformat(), int(bool(repaired)), details if repaired else '', REPAIR_POLICY_KEY if repaired else ''))

def entry_key(entry):
    return (entry['method_path'], entry['slide_id'].casefold())

def prediction_content_flags(row):
    unknown = {'prediction_empty': '', 'prediction_near_empty': '', 'prediction_content_status': 'unknown', 'prediction_area_ratio_to_gt': ''}
    if row.get('status') != 'ok':
        if row.get('status') == 'missing_prediction':
            unknown['prediction_content_status'] = 'missing_prediction'
        elif row.get('status') == 'error' and str(row.get('error', '')).startswith('Prediction: ValueError: Zero-byte '):
            unknown['prediction_content_status'] = 'zero_byte_file'
        return unknown
    try:
        pred_area = float(row['pred_area_px2'])
        gt_area = float(row['gt_area_px2'])
    except (TypeError, ValueError, KeyError):
        return unknown
    if not math.isfinite(pred_area) or not math.isfinite(gt_area) or pred_area < 0 or (gt_area < 0):
        return unknown
    empty = pred_area == 0.0
    ratio = pred_area / gt_area if gt_area > 0 else None
    near_empty = not empty and ratio is not None and (0 < ratio <= NEAR_EMPTY_PREDICTION_GT_AREA_FRACTION)
    return {'prediction_empty': int(empty), 'prediction_near_empty': int(near_empty), 'prediction_content_status': 'empty' if empty else 'near_empty' if near_empty else 'nonempty', 'prediction_area_ratio_to_gt': ratio if ratio is not None else ''}

def empty_summary_counts(rows):
    labels = [prediction_content_flags(row)['prediction_content_status'] for row in rows]
    empty = labels.count('empty')
    near = labels.count('near_empty')
    zero_byte = labels.count('zero_byte_file')
    known = sum((label in {'empty', 'near_empty', 'nonempty'} for label in labels))
    return {'n_empty_predictions': empty, 'n_near_empty_predictions': near, 'n_nonempty_predictions': known - empty, 'n_unassessed_predictions': len(rows) - known, 'n_zero_byte_prediction_files': zero_byte, 'empty_prediction_rate_pct': 100.0 * empty / known if known else None, 'near_empty_prediction_rate_pct': 100.0 * near / known if known else None}
TIMING_NUMERIC_FIELDS = ('total_processing_time_s', 'seconds_per_native_megapixel', 'initialization_time_s', 'algorithm_time_s', 'segmentation_call_time_s', 'image_loading_time_s', 'preprocessing_time_s', 'polygon_conversion_time_s', 'serialization_time_s', 'native_pixels')
TIMING_EXPORT_FIELDS = ('timing_status', 'timing_error', 'timing_path', 'timing_schema_version', *TIMING_NUMERIC_FIELDS, 'device', 'measurement_phase', 'timing_measured_at_utc')

def parse_method_timing(entry):
    method_dir = Path(str(entry['method_path']))
    slide_id = str(entry['slide_id'])
    timing_path = method_dir / 'timings' / (slide_id + '.json')
    row = {name: '' for name in TIMING_EXPORT_FIELDS}
    row.update(timing_path=str(timing_path), timing_status='missing')
    try:
        with timing_path.open('rb') as f:
            payload = orjson.loads(f.read()) if orjson is not None else json.load(f)
    except FileNotFoundError:
        row['timing_error'] = 'timing JSON not present'
        return row
    except (OSError, ValueError, TypeError) as exc:
        row.update(timing_status='invalid', timing_error=f'{type(exc).__name__}: {exc}')
        return row
    try:
        if not isinstance(payload, dict):
            raise ValueError('timing JSON must be an object')
        if payload.get('run_id') != method_dir.name:
            raise ValueError('timing JSON run_id does not match method directory')
        source_filename = payload.get('filename')
        if not isinstance(source_filename, str) or Path(source_filename).stem.casefold() != slide_id.casefold():
            raise ValueError('timing JSON filename does not match evaluated slide_id')
        schema = payload.get('schema_version')
        if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
            raise ValueError('missing/invalid schema_version')
        row['timing_schema_version'] = schema
        for field in TIMING_NUMERIC_FIELDS:
            val = payload.get(field)
            if val is None:
                continue
            if isinstance(val, bool):
                raise ValueError(f'{field} is not numeric')
            value = float(val)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')
            row[field] = int(value) if field == 'native_pixels' and value.is_integer() else value
        if row['total_processing_time_s'] == '' or row['total_processing_time_s'] <= 0:
            raise ValueError('total_processing_time_s must be positive')
        for field in ('device', 'measurement_phase'):
            val = payload.get(field)
            if val is not None:
                row[field] = str(val).strip()
        row['timing_measured_at_utc'] = str(payload.get('measured_at_utc') or '')
        row.update(timing_status='ok', timing_error='')
    except (ValueError, TypeError, OverflowError) as exc:
        row.update(timing_status='invalid', timing_error=f'{type(exc).__name__}: {exc}')
        for field in TIMING_NUMERIC_FIELDS:
            row[field] = ''
    return row

class TimingCollector:

    def __init__(self, conn):
        self.conn = conn
        self.cache = {}
        self.checked_at = {}
        self.attempted = set()
        self.seen_ok = 0
        for record in conn.execute('SELECT method_path, slide_key, payload_json, collected_at_utc FROM timing_metadata'):
            key = (record['method_path'], record['slide_key'])
            try:
                self.cache[key] = json.loads(record['payload_json'])
                self.checked_at[key] = datetime.fromisoformat(record['collected_at_utc'])
            except (ValueError, TypeError):
                pass

    def enrich(self, rows):
        if not COLLECT_INFERENCE_TIMINGS:
            return
        queue = []
        allowed_statuses = {'ok', 'error', 'pending', 'missing_prediction'} if TIMINGS_ONLY and TIMING_BACKFILL_ALL_SLIDES else set(TIMING_SYNC_STATUSES)
        for row in rows:
            key = entry_key(row)
            old = self.cache.get(key)
            if row.get('status') not in allowed_statuses:
                continue
            if key in self.attempted:
                continue
            last_read = self.checked_at.get(key)
            retry_due = last_read is None or (datetime.now(timezone.utc) - last_read).total_seconds() >= TIMING_MISSING_RETRY_SECONDS
            if REFRESH_TIMING_CACHE or old is None or (TIMING_RETRY_UNAVAILABLE and old.get('timing_status') != 'ok' and retry_due):
                queue.append((key, {'method_path': row['method_path'], 'slide_id': row['slide_id']}))
                self.attempted.add(key)
        if queue:
            print(f'[TIMINGS] Reading {len(queue)} method/slide JSON records ({TIMING_READ_WORKERS} threads; valid cached records reused)', flush=True)
            with ThreadPoolExecutor(max_workers=TIMING_READ_WORKERS) as pool:
                futures = {pool.submit(parse_method_timing, entry): key for key, entry in queue}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {name: '' for name in TIMING_EXPORT_FIELDS}
                        record.update(timing_status='invalid', timing_error=f'{type(exc).__name__}: {exc}')
                    self.cache[key] = record
                    self.checked_at[key] = datetime.now(timezone.utc)
                    self.conn.execute('INSERT OR REPLACE INTO timing_metadata\n                        (method_path, slide_key, payload_json, collected_at_utc)\n                        VALUES (?, ?, ?, ?)', (key[0], key[1], json.dumps(record, ensure_ascii=False, allow_nan=False), datetime.now(timezone.utc).isoformat()))
            self.conn.commit()
            statuses = Counter((self.cache[key].get('timing_status') for key, _ in queue))
            print(f'[TIMINGS] New/updated {len(queue)}: ' + ', '.join((f'{name}={n}' for name, n in sorted(statuses.items()))), flush=True)
        for row in rows:
            record = self.cache.get(entry_key(row))
            row.update(record if record is not None else {name: '' for name in TIMING_EXPORT_FIELDS})
            if record is None:
                row['timing_status'] = 'not_checked'

def timing_summary_counts(records):
    statuses = Counter((str(r.get('timing_status') or 'not_checked') for r in records))
    good = [r for r in records if r.get('timing_status') == 'ok']

    def average(field):
        vals = [float(r[field]) for r in good if r.get(field) not in (None, '')]
        return sum(vals) / len(vals) if vals else None
    times = [float(r['total_processing_time_s']) for r in good]
    return {'n_timings_ok': statuses['ok'], 'n_timings_missing': statuses['missing'], 'n_timings_invalid': statuses['invalid'], 'n_timings_unassessed': len(records) - statuses['ok'] - statuses['missing'] - statuses['invalid'], 'mean_total_processing_time_s': average('total_processing_time_s'), 'median_total_processing_time_s': median(times) if times else None, 'mean_seconds_per_native_megapixel': average('seconds_per_native_megapixel')}

def write_reports(out_dir, method_dirs, result_map, cached_keys, gt_count, counts=None, annotations=None, timing_collector=None):
    rows = list(result_map.values())
    rows.sort(key=lambda r: (r['method'].casefold(), r['method_path'], r['slide_id'].casefold()))
    for row in rows:
        row.update(prediction_content_flags(row))
    if timing_collector is not None:
        timing_collector.enrich(rows)
    export_csv(out_dir / 'per_slide.csv', SCORE_FIELDS, rows)
    by_method = {str(path): [] for path in method_dirs}
    for row in rows:
        if row['method_path'] in by_method:
            by_method[row['method_path']].append(row)
    summaries = []
    for method in method_dirs:
        key = str(method)
        records = by_method[key]
        successes = [r for r in records if r['status'] == 'ok']

        def mean(field):
            vals = []
            for r in successes:
                if r.get(field) not in (None, ''):
                    val = float(r[field])
                    if math.isfinite(val):
                        vals.append(val)
            return sum(vals) / len(vals) if vals else None
        n_cached = sum((entry_key(r) in cached_keys and r['status'] == 'ok' for r in records))
        summaries.append({'method': method.name, 'method_path': key, 'historical_total_unverified': None, 'n_gt_available': gt_count, 'n_predictions': len(records), 'n_missing_predictions': gt_count - len(records) + sum((r['status'] == 'missing_prediction' for r in records)), 'n_evaluated': len(successes), 'n_cached': n_cached, 'n_failed': sum((r['status'] == 'error' for r in records)), 'n_pending': sum((r['status'] == 'pending' for r in records)), 'n_repaired': sum((r['status'] == 'ok' and str(r.get('geometry_repaired', 0)).lower() in ('1', 'true') for r in records)), **empty_summary_counts(records), **timing_summary_counts(records), 'mean_dice': mean('dice'), 'mean_over_segmentation_rate': mean('over_segmentation_rate'), 'mean_under_segmentation_rate': mean('under_segmentation_rate')})
    export_csv(out_dir / 'summary.csv', SUMMARY_FIELDS, summaries)
    if SAVE_COMPLETENESS_PROVENANCE:
        if counts is None or annotations is None:
            counts, annotations = load_completeness_context(out_dir, method_dirs)
        if counts is not None and annotations is not None:
            write_completeness_provenance(out_dir, method_dirs, counts, annotations, result_map, cached_keys)
    return summaries
COMPLETENESS_FIELDS = ('method', 'config_id', 'method_path', 'prediction_dir', 'expected_gt_slides', 'all_prediction_files', 'all_xml', 'all_geojson', 'gt_matched_predictions', 'gt_missing_predictions', 'n_successful', 'n_failed', 'n_missing_at_read', 'n_pending', 'n_repaired', 'n_reused_from_cache', 'n_empty_predictions', 'n_near_empty_predictions', 'n_nonempty_predictions', 'n_unassessed_predictions', 'n_zero_byte_prediction_files', 'empty_prediction_rate_pct', 'near_empty_prediction_rate_pct', 'prediction_coverage_pct', 'evaluation_coverage_pct', 'n_timings_ok', 'n_timings_missing', 'n_timings_invalid', 'n_timings_unassessed', 'mean_total_processing_time_s', 'median_total_processing_time_s', 'inventory_source', 'inventory_scanned_at_utc', 'report_generated_at_utc')
COMPLETENESS_SLIDE_FIELDS = ('method', 'config_id', 'method_path', 'slide_id', 'ground_truth_path', 'prediction_path', 'prediction_listed_in_inventory', 'status', 'cached_success', 'geometry_repaired', 'error', 'pred_area_px2', 'gt_area_px2', 'prediction_empty', 'prediction_near_empty', 'prediction_content_status', 'prediction_area_ratio_to_gt', 'timing_status', 'total_processing_time_s', 'seconds_per_native_megapixel', 'device', 'measurement_phase', 'timing_path', 'timing_error')
POTENTIALLY_EMPTY_FIELDS = ('method', 'config_id', 'method_path', 'slide_id', 'prediction_path', 'prediction_content_status', 'prediction_empty', 'prediction_near_empty', 'pred_area_px2', 'gt_area_px2', 'prediction_area_ratio_to_gt', 'dice')

def write_completeness_provenance(out_dir, method_dirs, counts, annotations, result_map, cached_keys):
    rows, slide_rows, potentially_empty_rows = ([], [], [])
    timestamp = datetime.now(timezone.utc).isoformat()
    for method in method_dirs:
        key = str(method)
        count = counts.get(key)
        if count is None:
            continue
        config_id = method.name.rsplit('__cfg=', 1)[-1] if '__cfg=' in method.name else ''
        ok = failed = missing_read = pending = repaired = cached = 0
        method_scored_rows = []
        for slide_key, (slide_id, gt_path) in sorted(annotations.items()):
            result = result_map.get((key, slide_key))
            if result is None:
                status, path, error, is_repaired, cached_success = ('missing_prediction', '', '', 0, 0)
            else:
                status = result.get('status') or 'pending'
                path = result.get('prediction_path', '')
                error = result.get('error', '')
                is_repaired = int(str(result.get('geometry_repaired', 0)).lower() in ('1', 'true'))
                cached_success = int((key, slide_key) in cached_keys and status == 'ok')
                if status == 'ok':
                    ok += 1
                    method_scored_rows.append(result)
                elif status == 'missing_prediction':
                    missing_read += 1
                elif status == 'pending':
                    pending += 1
                else:
                    failed += 1
                repaired += status == 'ok' and is_repaired
                cached += cached_success
            flags = prediction_content_flags(result) if result is not None else {'prediction_empty': '', 'prediction_near_empty': '', 'prediction_content_status': 'missing_prediction', 'prediction_area_ratio_to_gt': ''}
            if flags['prediction_content_status'] in {'empty', 'near_empty', 'zero_byte_file'}:
                potentially_empty_rows.append({'method': method.name, 'config_id': config_id, 'method_path': key, 'slide_id': slide_id, 'prediction_path': path, **flags, 'pred_area_px2': result.get('pred_area_px2', ''), 'gt_area_px2': result.get('gt_area_px2', ''), 'dice': result.get('dice', '')})
            slide_rows.append({'method': method.name, 'config_id': config_id, 'method_path': key, 'slide_id': slide_id, 'ground_truth_path': gt_path, 'prediction_path': path, 'prediction_listed_in_inventory': int(bool(path)), 'status': status, 'cached_success': cached_success, 'geometry_repaired': is_repaired, 'error': error, 'pred_area_px2': result.get('pred_area_px2', '') if result is not None else '', 'gt_area_px2': result.get('gt_area_px2', '') if result is not None else '', **flags, 'timing_status': result.get('timing_status', 'not_checked') if result else 'not_checked', 'total_processing_time_s': result.get('total_processing_time_s', '') if result else '', 'seconds_per_native_megapixel': result.get('seconds_per_native_megapixel', '') if result else '', 'device': result.get('device', '') if result else '', 'measurement_phase': result.get('measurement_phase', '') if result else '', 'timing_path': result.get('timing_path', '') if result else '', 'timing_error': result.get('timing_error', '') if result else ''})
        n_gt = len(annotations)
        rows.append({'method': method.name, 'config_id': config_id, 'method_path': key, 'prediction_dir': str(method / 'predictions'), 'expected_gt_slides': n_gt, 'all_prediction_files': count['all_predictions'], 'all_xml': count['all_xml'], 'all_geojson': count['all_geojson'], 'gt_matched_predictions': count['predictions'], 'gt_missing_predictions': count['missing_predictions'], 'n_successful': ok, 'n_failed': failed, 'n_missing_at_read': missing_read, 'n_pending': pending, 'n_repaired': repaired, **empty_summary_counts(method_scored_rows + [result_map[key, slide_key] for slide_key in annotations if (key, slide_key) in result_map and result_map[key, slide_key].get('status') != 'ok']), 'n_reused_from_cache': cached, **timing_summary_counts([result_map[key, slide_key] for slide_key in annotations if (key, slide_key) in result_map]), 'prediction_coverage_pct': 100 * count['predictions'] / n_gt if n_gt else None, 'evaluation_coverage_pct': 100 * ok / n_gt if n_gt else None, 'inventory_source': count['inventory_source'], 'inventory_scanned_at_utc': count['inventory_scanned_at_utc'], 'report_generated_at_utc': timestamp})
    export_csv(out_dir / 'method_completeness.csv', COMPLETENESS_FIELDS, rows)
    export_csv(out_dir / 'slide_completeness.csv', COMPLETENESS_SLIDE_FIELDS, slide_rows)
    export_csv(out_dir / 'potentially_empty_predictions.csv', POTENTIALLY_EMPTY_FIELDS, potentially_empty_rows)
    methods_digest = hashlib.sha256(METHODS_FILE.read_bytes()).hexdigest()
    gt_fingerprint = hashlib.sha256(json.dumps([(key, path) for key, (_slide, path) in sorted(annotations.items())], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    provenance = {'schema': 'segmenteer-completeness-v1', 'generated_at_utc': timestamp, 'methods_file': str(METHODS_FILE), 'methods_file_sha256': methods_digest, 'ground_truth_directory': str(GROUND_TRUTH_DIR), 'ground_truth_slide_ids': len(annotations), 'ground_truth_mapping_fingerprint_sha256': gt_fingerprint, 'inventory_note': 'Live mode: filenames automatically rescanned after each wave' if AUTO_FOLLOW_NEW_PREDICTIONS else 'One-shot mode: saved filename inventory; manual refresh optional', 'cache_file': str(out_dir / 'cache.sqlite3'), 'metric_protocol_key': CACHE_KEY, 'verify_cached_file_changes': VERIFY_CACHED_FILE_CHANGES, 'empty_detection': {'source': 'cached or freshly calculated exact vector pred_area_px2', 'exact_empty_area_px2': 0.0, 'near_empty_positive_area_max_gt_fraction': NEAR_EMPTY_PREDICTION_GT_AREA_FRACTION, 'rate_denominator': 'successful evaluations with known prediction area', 'unknown_policy': 'failed and pending cases have unknown mask content; zero-byte files are separately identified when read'}, 'timing_collection': {'input': '<method_path>/timings/<slide_id>.json', 'metric_source': 'Segmenteer original inference timing (not evaluator runtime)', 'source_files_read_only': True, 'fields': list(TIMING_EXPORT_FIELDS), 'timing_cache': 'SQLite timing_metadata table'}, 'method_completeness': rows}
    tmp = out_dir / 'completeness_provenance.json.tmp'
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=False)
    os.replace(tmp, out_dir / 'completeness_provenance.json')

def load_completeness_context(out_dir, method_dirs):
    cache_path = out_dir / INVENTORY_CACHE_FILENAME
    slide_path = out_dir / 'slide_completeness.csv'
    if not cache_path.is_file() or not slide_path.is_file():
        return (None, None)
    doc = _load_inventory_cache(cache_path)
    annotations = {}
    with slide_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            slide_id = row.get('slide_id') or ''
            if slide_id:
                annotations[slide_id.casefold()] = (slide_id, row.get('ground_truth_path', ''))
    counts = {}
    for method in method_dirs:
        cached = doc['methods'].get(str(method))
        if cached is None:
            continue
        available = sum((slide_key in cached['files'] for slide_key in annotations))
        counts[str(method)] = {'all_predictions': int(cached['n_xml']) + int(cached['n_geojson']), 'all_xml': int(cached['n_xml']), 'all_geojson': int(cached['n_geojson']), 'predictions': available, 'missing_predictions': len(annotations) - available, 'inventory_source': 'saved_inventory', 'inventory_scanned_at_utc': cached['scanned_at_utc']}
    return (counts, annotations)
'Official-only CPG Segmenteer evaluation; incremental, source inputs read-only.\n\nPlace beside the original `fast_evaluate_segmenteer_tissues_with_issues-5.py`\n(or `fast_evaluate_segmenteer_tissues_with_issues.py`) and `methods_cpg.txt`.\nRun: python3 fast_evaluate_segmenteer_tissues_cpg.py\n\nUses the ORIGINAL evaluation functions for ASAP XML geometry repair, Dice,\nover/under-segmentation, the SQLite cache, inference timings and CSV exports.\nChanges ONLY the input paths and output directory for the CPG dataset.\nNo unofficial list, alternate-machine fallback or 5-MPP deferral.\n\nResults are isolated under fast_evaluation_benchmark-tissue-masks_cpg/.\nNew method folders may be added to methods_cpg.txt later. Missing run folders,\nmissing slide predictions and partly completed runs are normal; rerunning\nprocesses new/changed inputs without discarding cached successful scores.\n'

@lru_cache(maxsize=1)
def active_run_ids() -> frozenset[str]:
    if not OFFICIAL_METHODS_FILE.is_file():
        raise FileNotFoundError(f'Required CPG run list not found: {OFFICIAL_METHODS_FILE}')
    return frozenset((Path(x.strip()).name.casefold() for x in OFFICIAL_METHODS_FILE.read_text(encoding='utf-8-sig').splitlines() if x.strip() and (not x.lstrip().startswith('#'))))

def is_excluded_run(run_name: str) -> bool:
    return Path(str(run_name)).name.casefold() not in active_run_ids()

def load_core():
    global SCORE_FIELDS
    if FORCE_RECOMPUTE or AUTO_FOLLOW_NEW_PREDICTIONS or RETRY_FAILED_ONLY or TIMINGS_ONLY:
        raise ValueError('Incremental evaluation requires FORCE_RECOMPUTE, AUTO_FOLLOW_NEW_PREDICTIONS, RETRY_FAILED_ONLY and TIMINGS_ONLY to be False')
    extras = ('prediction_source', 'source_method_path', 'source_run_id', 'source_config_match', 'official_prediction_available', 'unofficial_prediction_available', 'source_evaluated_at_utc')
    SCORE_FIELDS = tuple(dict.fromkeys((*SCORE_FIELDS, *extras)))
    return sys.modules[__name__]

def method_sources():
    if not OFFICIAL_METHODS_FILE.is_file():
        raise FileNotFoundError(f'Missing CPG methods list: {OFFICIAL_METHODS_FILE}')
    listed = {}
    for line_num, raw in enumerate(OFFICIAL_METHODS_FILE.read_text(encoding='utf-8-sig').splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith('#'):
            continue
        folder = Path(value).expanduser()
        if not folder.is_absolute():
            raise ValueError(f'{OFFICIAL_METHODS_FILE.name}:{line_num} must be an absolute path: {value}')
        if folder.name in listed:
            raise ValueError(f'Duplicate CPG method run directory: {folder.name}')
        listed[folder.name] = folder
    if not listed:
        raise ValueError(f'No method paths in {OFFICIAL_METHODS_FILE}')
    roots = {p.parent for p in listed.values()}
    if len(roots) != 1:
        raise ValueError('All CPG run directories must share one official benchmark root')
    root = next(iter(roots))
    if not root.is_dir():
        raise FileNotFoundError(f'CPG benchmark root not available: {root}')
    sources = {}
    for name, folder in sorted(listed.items()):
        if folder.is_dir():
            sources[name] = {'official': folder.resolve(), 'unofficial': None}
        else:
            print(f'[PENDING RUN] Method directory not yet created: {folder}', flush=True)
            sources[name] = {'official': None, 'unofficial': None}
    logical = {name: folder for name, folder in sorted(listed.items())}
    print(f"[METHODS] {len(logical)} configured official runs; {sum((s['official'] is not None for s in sources.values()))} existing folders", flush=True)
    extra = sorted((p.name for p in root.iterdir() if p.is_dir() and '__cfg=' in p.name and (p.name not in listed)))
    if extra:
        print(f'[UNLISTED] {len(extra)} other runs in {root}: append to methods_cpg.txt to evaluate')
        for name in extra[:15]:
            print(f'  {name}', flush=True)
    return (sources, logical)

def inventory(sources, gt):
    paths = sorted({p for choices in sources.values() for p in choices.values() if p is not None})

    def scan(path):
        directory = path / 'predictions'
        if not directory.is_dir():
            return {'files': {}, 'n_xml': 0, 'n_geojson': 0, 'scanned_at_utc': '', 'inventory_source': 'directory_missing', 'prediction_directory': str(directory)}
        return core._scan_prediction_names(directory)
    print(f'[INVENTORY] Scanning {len(paths)} source folders ({len(gt)} GT slides); cached metrics will not be recomputed', flush=True)
    with ThreadPoolExecutor(max_workers=core.INVENTORY_WORKERS) as pool:
        futures = {pool.submit(scan, p): p for p in paths}
        found = {futures[f]: f.result() for f in as_completed(futures)}
    for run, choices in sources.items():
        report = []
        for source in ('official', 'unofficial'):
            p = choices[source]
            if p is None:
                continue
            info = found[p]
            count = sum((slide_key in info['files'] for slide_key in gt))
            report.append(f'{source}={count}/{len(gt)}')
        print(f'[INVENTORY] {run}: ' + ', '.join(report), flush=True)
    return found

def make_candidates(sources, logical, names, annotations):
    pairs = {}
    for run in logical:
        canonical = str(logical[run])
        for slide_key, (slide_id, gt_path) in annotations.items():
            options = {}
            for source, path in sources[run].items():
                if path is None:
                    continue
                filename = names[path]['files'].get(slide_key)
                if not filename:
                    continue
                options[source] = {'method': run, 'method_path': str(path), 'slide_id': slide_id, 'prediction_path': str(path / 'predictions' / filename), 'annotation_path': gt_path, 'status': 'pending', 'error': ''}
            if options:
                pairs[canonical, slide_key] = options
    return pairs

def same_failed_input(old, entry):
    if old is None or old.get('status') != 'error' or old.get('protocol_key') != core.CACHE_KEY or (old.get('prediction_path') != entry['prediction_path']) or (old.get('annotation_path') != entry['annotation_path']):
        return False
    gt_error = str(old.get('error') or '').startswith('Ground truth:')
    path = entry['annotation_path'] if gt_error else entry['prediction_path']
    expected = (old.get('gt_size'), old.get('gt_mtime_ns')) if gt_error else (old.get('pred_size'), old.get('pred_mtime_ns'))
    if expected[1] in (None, 0):
        return False
    try:
        return core.signature(path) == expected
    except OSError:
        return False

def state(entry, cache):
    if entry is None:
        return 'absent'
    old = cache.get(core.entry_key(entry))
    if core.cached_ok(old, entry):
        return 'ok'
    if same_failed_input(old, entry):
        return 'error'
    return 'pending'

def choose(options, cache):
    official = options.get('official')
    unofficial = options.get('unofficial')
    os_ = state(official, cache)
    us_ = state(unofficial, cache)
    if os_ == 'ok':
        return ('official', official, 'ok')
    if os_ == 'pending' and us_ == 'ok':
        return ('unofficial', unofficial, 'ok')
    if os_ == 'pending':
        return ('official', official, 'pending')
    if us_ in {'ok', 'error', 'pending'}:
        return ('unofficial', unofficial, us_)
    if official is not None:
        return ('official', official, os_)
    return ('unofficial', unofficial, us_)

def output_rows(pairs, cache):
    result = {}
    for key, options in pairs.items():
        source, selected, stat = choose(options, cache)
        if selected is None:
            continue
        item = dict(selected)
        item['method'] = Path(key[0]).name
        old = cache.get(core.entry_key(selected)) if stat != 'pending' else None
        if old:
            for name in core.SCORE_FIELDS:
                if name in old:
                    item[name] = old[name]
        item['method_path'] = key[0]
        item['prediction_source'] = source
        item['source_method_path'] = selected['method_path']
        item['source_run_id'] = Path(selected['method_path']).name
        item['source_config_match'] = 'exact_run_id' if item['source_run_id'] == item['method'] else 'device_only_equivalent'
        item['official_prediction_available'] = int('official' in options)
        item['unofficial_prediction_available'] = int('unofficial' in options)
        item['source_evaluated_at_utc'] = old.get('evaluated_at', '') if old else ''
        item['status'] = stat
        if stat == 'pending':
            item['error'] = ''
        result[key] = item
    return result

def protect_existing_active_csv(out_dir, cache):
    file = out_dir / 'per_slide.csv'
    if not file.is_file():
        return
    missing = []
    with file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if (row.get('status') or '').strip().lower() != 'ok':
                continue
            if is_excluded_run(row.get('method_path') or row.get('method') or ''):
                continue
            path = (row.get('source_method_path') or row.get('method_path') or '').strip()
            slide = (row.get('slide_id') or '').strip().casefold()
            old = cache.get((path, slide))
            if not old or old['status'] != 'ok' or old['prediction_path'] != row.get('prediction_path'):
                if len(missing) < 5:
                    missing.append((Path(path).name, slide))
    if missing:
        raise RuntimeError(f'SAFETY STOP: successful per_slide.csv rows are missing from the original SQLite cache. Examples: {missing}. Restore the correct cache.sqlite3 before continuing.')

class SourceTimingCollector:

    def __init__(self, conn):
        self.delegate = core.TimingCollector(conn) if core.COLLECT_INFERENCE_TIMINGS else None

    def enrich(self, logical_rows):
        if self.delegate is None:
            return
        source_rows = [{**row, 'method_path': row['source_method_path']} for row in logical_rows]
        self.delegate.enrich(source_rows)
        for selected, source_row in zip(logical_rows, source_rows):
            selected.update({f: source_row.get(f, '') for f in core.TIMING_EXPORT_FIELDS})

def merged_counts(logical, pairs, result_rows, names, gt_count):
    counts = {}
    for run, method in logical.items():
        key = str(method)
        active = [result_rows[k] for k in pairs if k[0] == key and k in result_rows]
        counts[key] = {'all_predictions': len(active), 'all_xml': sum((Path(row['prediction_path']).suffix.lower() == '.xml' for row in active)), 'all_geojson': sum((Path(row['prediction_path']).suffix.lower() == '.geojson' for row in active)), 'predictions': len(active), 'missing_predictions': gt_count - len(active), 'inventory_source': 'merged_official_first', 'inventory_scanned_at_utc': datetime.now(timezone.utc).isoformat()}
    return counts

def atomic_csv(path, fields, rows):
    core.export_csv(path, fields, rows)

def write_source_metadata(out_dir, sources, logical, pairs, cache, rows, annotations):
    stamp = datetime.now(timezone.utc).isoformat()
    selected, totals = ([], [])
    for run, logical_path in logical.items():
        counts = Counter()
        for slide_key, (slide, _) in annotations.items():
            key = (str(logical_path), slide_key)
            options = pairs.get(key, {})
            item = rows.get(key, {})
            status = item.get('status', 'missing_prediction')
            source = 'official' if 'official' in options else 'missing'
            counts[source] += 1
            if status == 'ok':
                counts['official_ok'] += 1
            entry = options.get('official')
            is_cached = bool(entry and cache.get(core.entry_key(entry), {}).get('status') == 'ok')
            selected.append({'method': run, 'method_path': str(logical_path), 'slide_id': slide, 'prediction_source': source, 'status': status, 'source_method_path': item.get('source_method_path', ''), 'source_run_id': item.get('source_run_id', ''), 'source_config_match': 'exact_run_id' if entry else '', 'prediction_path': item.get('prediction_path', ''), 'official_prediction_available': int(entry is not None), 'unofficial_prediction_available': 0, 'official_prediction_path': entry['prediction_path'] if entry else '', 'unofficial_prediction_path': '', 'official_cache_status': state(entry, cache), 'unofficial_cache_status': 'absent', 'source_evaluated_at_utc': item.get('source_evaluated_at_utc', ''), 'cached_success': int(is_cached and status == 'ok')})
        totals.append({'method': run, 'method_path': str(logical_path), 'n_gt_slides': len(annotations), 'n_selected_official': counts['official'], 'n_selected_unofficial': 0, 'n_no_prediction': counts['missing'], 'n_successful_official': counts['official_ok'], 'n_successful_unofficial': 0, 'official_method_path': str(sources[run]['official'] or ''), 'official_run_id': sources[run]['official'].name if sources[run]['official'] else '', 'unofficial_method_path': '', 'unofficial_run_id': '', 'cross_source_config_match': 'not_applicable'})
    if selected:
        atomic_csv(out_dir / 'source_selection.csv', tuple(selected[0]), selected)
    if totals:
        atomic_csv(out_dir / 'source_completeness.csv', tuple(totals[0]), totals)
    manifest = {'schema': 'segmenteer-cpg-official-only-v1', 'generated_at_utc': stamp, 'official_methods_file': str(OFFICIAL_METHODS_FILE), 'official_methods_sha256': hashlib.sha256(OFFICIAL_METHODS_FILE.read_bytes()).hexdigest(), 'ground_truth_dir': str(core.GROUND_TRUTH_DIR), 'results_dir': str(out_dir), 'cache_file': str(out_dir / 'cache.sqlite3'), 'prediction_source': 'official only', 'method_matching': 'exact run-directory ID; no unofficial fallback', 'method_list_policy': 'explicit list only; 5-MPP runs included when listed', 'metadata_workbook': str(METADATA_FILE), 'methods': totals}
    target = out_dir / 'source_provenance.json'
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, target)

def annotate_existing_completeness_reports(out_dir, sources, active):
    summary_path = out_dir / 'method_completeness.csv'
    slide_path = out_dir / 'slide_completeness.csv'
    by_run = defaultdict(Counter)
    for row in active.values():
        by_run[row['method']][row['prediction_source']] += 1
    if summary_path.is_file():
        with summary_path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
        for field in ('n_selected_official', 'n_selected_unofficial'):
            if field not in fields:
                fields.append(field)
        for row in rows:
            run = row['method']
            selected = by_run[run]
            row['n_selected_official'] = selected['official']
            row['n_selected_unofficial'] = selected['unofficial']
            if selected['official'] and selected['unofficial']:
                row['prediction_dir'] = 'MIXED (see source_selection.csv)'
            elif selected['official']:
                row['prediction_dir'] = str(sources[run]['official'] / 'predictions')
            elif selected['unofficial']:
                row['prediction_dir'] = str(sources[run]['unofficial'] / 'predictions')
            else:
                row['prediction_dir'] = 'NO PREDICTIONS'
        atomic_csv(summary_path, fields, rows)
        provenance_file = out_dir / 'completeness_provenance.json'
        if provenance_file.is_file():
            provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
            provenance['method_completeness'] = rows
            provenance['source_selection_report'] = 'source_selection.csv'
            tmp = provenance_file.with_name(provenance_file.name + '.tmp')
            tmp.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
            os.replace(tmp, provenance_file)
    if slide_path.is_file():
        with slide_path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
        added = ('prediction_source', 'source_method_path', 'official_prediction_available', 'unofficial_prediction_available')
        fields.extend((field for field in added if field not in fields))
        for row in rows:
            selected = active.get((row['method_path'], row['slide_id'].casefold()), {})
            for field in added:
                row[field] = selected.get(field, '')
        atomic_csv(slide_path, fields, rows)

def report(out_dir, sources, logical, pairs, cache, cached_successes, gt, collector):
    active = output_rows(pairs, cache)
    counts = merged_counts(logical, pairs, active, None, len(gt))
    reused_active = {key for key, row in active.items() if row['status'] == 'ok' and (row['source_method_path'], key[1]) in cached_successes}
    core.write_reports(out_dir, list(logical.values()), active, reused_active, len(gt), counts, gt, timing_collector=collector)
    annotate_existing_completeness_reports(out_dir, sources, active)
    write_source_metadata(out_dir, sources, logical, pairs, cache, active, gt)
    return active

def _initialize_geometry_worker():
    global core
    core = load_core()

def _evaluate_in_worker(job):
    return core.evaluate_slide(job)

def run_jobs(jobs_to_do, cache, conn, out_dir, sources, logical, pairs, gt, cached_successes, collector, phase_name):
    if not jobs_to_do:
        print(f'[{phase_name}] No new geometry evaluation needed', flush=True)
        return 0
    by_slide = defaultdict(list)
    for item in jobs_to_do:
        by_slide[item['slide_id'].casefold()].append(item)
    jobs = [(s, group[0]['annotation_path'], group, core.REPAIR_ON_FIRST_FAILURE) for s, group in sorted(by_slide.items())]
    print(f'[{phase_name}] Evaluating {len(jobs_to_do)} previously uncached/changed source predictions on {len(jobs)} slides; workers={core.WORKERS}', flush=True)
    last_report = time.monotonic()
    count = 0
    ok = errors = 0

    def accept(batch):
        nonlocal count, ok, errors, last_report
        for entry, status, values, error, pred_sig, gt_sig, repaired, details in batch:
            entry.update(pred_size=pred_sig[0], pred_mtime_ns=pred_sig[1], gt_size=gt_sig[0], gt_mtime_ns=gt_sig[1])
            core.store_result(conn, entry, status, values, error, repaired, details)
            record = dict(entry)
            record.update(values, slide_key=entry['slide_id'].casefold(), protocol_key=core.CACHE_KEY, status=status, error=error, geometry_repaired=int(repaired), repair_details=details, repair_policy_key=core.REPAIR_POLICY_KEY if repaired else '', evaluated_at=datetime.now(timezone.utc).isoformat())
            cache[core.entry_key(entry)] = record
            ok += int(status == 'ok')
            errors += int(status != 'ok')
            count += 1
        conn.commit()
        if time.monotonic() - last_report > core.CSV_UPDATE_SECONDS:
            report(out_dir, sources, logical, pairs, cache, cached_successes, gt, collector)
            last_report = time.monotonic()
            print(f'[{phase_name}] Committed {count}/{len(jobs_to_do)} (ok={ok}, failures={errors}); CSV snapshot refreshed', flush=True)
    if core.WORKERS == 1:
        for job in jobs:
            accept(core.evaluate_slide(job))
    else:
        with ProcessPoolExecutor(max_workers=core.WORKERS, initializer=_initialize_geometry_worker) as executor:
            futures = [executor.submit(_evaluate_in_worker, job) for job in jobs]
            for future in as_completed(futures):
                accept(future.result())
    print(f'[{phase_name}] Completed {count} source predictions (successes={ok}, errors={errors})', flush=True)
    return count

@contextmanager
def single_writer(out_dir, paths):
    import fcntl
    core.ensure_output_isolated(out_dir, paths, core.ANNOTATIONS_CSV)
    if out_dir == SCRIPT_DIR or out_dir in SCRIPT_DIR.parents:
        raise ValueError('CPG results must be separate from the script source folder')
    if out_dir == GROUND_TRUTH_DIR_CPG or out_dir in GROUND_TRUTH_DIR_CPG.parents:
        raise ValueError('CPG results must not overwrite ground truth')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / '.evaluator.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluator is writing this CPG results folder') from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def validate_resume_paths(pairs, cache):
    for options in pairs.values():
        for entry in options.values():
            old = cache.get(entry_key(entry))
            if old and old['status'] == 'ok' and old['annotation_path'] != entry['annotation_path']:
                raise ValueError('Cached annotation path differs from current input. Restore the original annotation path before resuming: ' + entry['slide_id'])


def main():
    global core
    core = load_core()
    if not core.GROUND_TRUTH_DIR.is_dir():
        raise FileNotFoundError(f'Missing CPG ground-truth directory: {core.GROUND_TRUTH_DIR}')
    sources, logical = method_sources()
    physical = sorted({p for s in sources.values() for p in s.values() if p is not None})
    out_dir = core.RESULTS_DIR.expanduser().resolve()
    print(f'[START] CPG OFFICIAL-only benchmark: {len(logical)} listed runs; {len(physical)} available method directories', flush=True)
    print(f'[GT] {core.GROUND_TRUTH_DIR}', flush=True)
    print(f'[OUT] {out_dir} (separate cache, unchanged evaluation metrics)', flush=True)
    print('[CACHE] Successful rows reused; new predictions evaluated incrementally', flush=True)
    if core.VERIFY_CACHED_FILE_CHANGES:
        print('[CACHE] Changed previously cached inputs can be recomputed', flush=True)
    annotations, _, _ = core.read_annotations(core.GROUND_TRUTH_DIR, core.ANNOTATIONS_CSV)
    if core.LIMIT_SLIDES is not None:
        annotations = dict(sorted(annotations.items())[:core.LIMIT_SLIDES])
        print(f'[TEST] LIMIT_SLIDES={len(annotations)}')
    if not annotations:
        raise ValueError(f'No ground-truth XML annotations found in {core.GROUND_TRUTH_DIR}')
    source_inventories = inventory(sources, annotations)
    pairs = make_candidates(sources, logical, source_inventories, annotations)
    print(f'[SETUP] {len(annotations)} ground truths; {len(pairs)} run+slide pairs have predictions', flush=True)
    if core.INVENTORY_ONLY:
        print('[DONE] Inventory only; no cache/CSV modifications', flush=True)
        return
    if not pairs:
        print('[WAIT] No matching predictions yet. Add runs/predictions and invoke again.', flush=True)
        return
    with single_writer(out_dir, physical):
        conn = core.connect_db(out_dir / 'cache.sqlite3')
        try:
            cache = core.cache_lookup(conn)
            protect_existing_active_csv(out_dir, cache)
            validate_resume_paths(pairs, cache)
            print(f'[CACHE] {len(cache)} source-specific rows in this CPG SQLite cache', flush=True)
            cached_successes = {k for k, v in cache.items() if v.get('status') == 'ok'}
            collector = SourceTimingCollector(conn)
            report(out_dir, sources, logical, pairs, cache, cached_successes, annotations, collector)
            jobs = [options['official'] for options in pairs.values() if 'official' in options and state(options['official'], cache) == 'pending']
            n = run_jobs(jobs, cache, conn, out_dir, sources, logical, pairs, annotations, cached_successes, collector, 'OFFICIAL')
            active = report(out_dir, sources, logical, pairs, cache, cached_successes, annotations, collector)
            statuses = Counter((row['status'] for row in active.values()))
            print(f'[DONE] New evaluations={n}; scored pairs={len(active)}; statuses={dict(statuses)}', flush=True)
            for filename in ('per_slide.csv', 'summary.csv', 'cache.sqlite3', 'method_completeness.csv', 'source_selection.csv', 'source_completeness.csv', 'source_provenance.json'):
                print(' ', out_dir / filename, flush=True)
            print('[SAFE] Original datasets and existing benchmark outputs were not modified', flush=True)
        finally:
            conn.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
