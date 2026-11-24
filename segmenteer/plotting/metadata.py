import csv
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _xlsx_rows(archive, sheet_name):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        node.get("Id"): node.get("Target")
        for node in relations.findall(f"{PKG}Relationship")
    }
    selected = next(
        (
            s
            for s in workbook.findall(f"{MAIN}sheets/{MAIN}sheet")
            if s.get("name", "").casefold() == sheet_name.casefold()
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"XLSX worksheet not found: {sheet_name}")
    rid = selected.get(f"{REL}id")
    target = targets.get(rid)
    if not target:
        raise ValueError(f"Missing XLSX worksheet relationship: {sheet_name}")
    path = target.lstrip("/")
    if not path.startswith("xl/"):
        path = "xl/" + path
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(t.text or "" for t in item.iter(f"{MAIN}t"))
            for item in shared.findall(f"{MAIN}si")
        ]
    root = ET.fromstring(archive.read(path))
    rows = []
    for row in root.findall(f"{MAIN}sheetData/{MAIN}row"):
        cells = {}
        for cell in row.findall(f"{MAIN}c"):
            coord = cell.get("r", "")
            match = re.match(r"([A-Z]+)", coord)
            if not match:
                continue
            col = 0
            for character in match.group(1):
                col = 26 * col + ord(character) - 64
            value = cell.find(f"{MAIN}v")
            if cell.get("t") == "inlineStr":
                value_text = "".join(t.text or "" for t in cell.iter(f"{MAIN}t"))
            else:
                value_text = (
                    value.text if value is not None and value.text is not None else ""
                )
                if cell.get("t") == "s" and value_text:
                    value_text = strings[int(value_text)]
            cells[col - 1] = value_text.strip()
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    if not rows:
        return []
    header = [str(s).strip() for s in rows[0]]
    return [
        {
            key: values[i] if i < len(values) else ""
            for i, key in enumerate(header)
            if key
        }
        for values in rows[1:]
    ]


def _read_workbook(path):
    with ZipFile(path) as archive:
        general = _xlsx_rows(archive, "general info")
        try:
            centers = _xlsx_rows(archive, "centers")
        except ValueError as exc:
            if "worksheet not found" not in str(exc):
                raise
            centers = []
    return general, centers


def normalize_slide(value):
    name = Path(str(value).replace("\\", "/")).name.strip()
    return re.sub(
        r"\.(xml|geojson|svs|tif|tiff|ndpi|mrxs|scn|qptiff|ome\.tiff)$",
        "",
        name,
        flags=re.I,
    ).casefold()


def load_metadata(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        general, centers = _read_workbook(path)
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            general = list(csv.DictReader(stream))
        centers = []
    else:
        raise ValueError("Metadata must be .xlsx or .csv")
    output = {}
    for row in general:
        slide = row.get("slide_id") or row.get("Slide id")
        if not slide:
            continue
        key = normalize_slide(slide)
        normalized = {
            "slide_id": str(slide),
            "source": (row.get("data_source") or row.get("Data source") or "").strip(),
            "stain": (row.get("staining") or row.get("Staining") or "").strip(),
            "organ": (row.get("organ") or row.get("Organ") or "").strip(),
            "center": (row.get("source_site") or row.get("Source Site") or "").strip(),
        }
        if key in output and any(
            output[key][dim] and normalized[dim] and output[key][dim] != normalized[dim]
            for dim in ("source", "stain", "organ", "center")
        ):
            raise ValueError(f"Conflicting metadata for slide {slide}")
        output[key] = normalized
    for row in centers:
        slide = row.get("Slide id") or row.get("slide_id")
        if not slide:
            continue
        key = normalize_slide(slide)
        if key not in output:
            continue
        site = (row.get("Source Site") or row.get("source_site") or "").strip()
        if output[key]["center"] and site and output[key]["center"] != site:
            raise ValueError(f"Conflicting center metadata for slide {slide}")
        output[key]["center"] = site
    return output
