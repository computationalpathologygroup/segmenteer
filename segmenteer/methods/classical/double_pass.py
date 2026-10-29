"""Adapter for the upstream Double-Pass tissue detector (Apache-2.0).

Source: https://github.com/aimas-upb/efficient-tissue-detection-wsi
Calls ``DoublePass.detect_tissue`` with both FilterGrays and DownKMeans.
The upstream implementation is fetched lazily or supplied locally; it is not
vendored. GrandQC comparison code is excluded from inference.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from segmenteer.core.base import NumpySegmenter
from segmenteer.model_cache import get_method_model_dir
from segmenteer.external_assets import require_network, _file_lock


UPSTREAM_URL = "https://github.com/aimas-upb/efficient-tissue-detection-wsi.git"
# Pin the audited upstream implementation to an immutable commit.
VERIFIED_UPSTREAM_COMMIT = "a3dc8861502e5de6e68edb366b845933bdc9383e"


def _resolved_revision(revision: str) -> str:
    if revision == "master":
        return VERIFIED_UPSTREAM_COMMIT
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        require_network("Double-Pass non-immutable Git ref", pinned=False)
    return revision


def _fetch_upstream(destination: Path, revision: str) -> Path:
    """Clone atomically to avoid incomplete cached checkouts."""
    if (destination / ".git").is_dir():
        return destination
    if destination.exists():
        raise RuntimeError(
            f"Double-Pass upstream cache exists but is not a Git checkout: {destination}. "
            "Remove it or pass upstream_dir=... to use a verified checkout."
        )
    require_network(
        "Double-Pass official Git checkout",
        pinned=bool(re.fullmatch(r"[a-fA-F0-9]{40}", revision)),
    )
    if shutil.which("git") is None:
        raise RuntimeError(
            "Double-Pass needs git to fetch the official research code. "
            "Install git, or provide upstream_dir=... with a checkout."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".double-pass-", dir=destination.parent))
    clone = temp_root / "upstream"
    try:
        result = subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", UPSTREAM_URL, str(clone)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(
                f"Cannot obtain official Double-Pass code: {result.stderr.strip()}"
            )
        result = subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", "--detach", revision],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(
                f"Double-Pass revision {revision!r} is unavailable: "
                f"{result.stderr.strip()}"
            )
        try:
            clone.rename(destination)
        except FileExistsError:

            if not (destination / ".git").is_dir():
                raise
        return destination
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _upstream_candidates(repo: Path) -> list[dict]:
    """Inspect upstream callable definitions without importing research scripts."""
    algorithms = repo / "algorithms"
    if not algorithms.is_dir():
        raise RuntimeError(
            f"Official Double-Pass algorithms directory not found under {repo}."
        )
    entries: list[dict] = []
    excluded = {
        "grandqc",
        "comparison",
        "comparisons",
        "benchmark",
        "benchmarks",
        "evaluation",
        "evaluations",
        "eval",
        "plots",
        "plotting",
    }
    for path in sorted(algorithms.rglob("*.py")):
        parts = [
            re.sub(r"[^a-z0-9]", "", x.lower())
            for x in path.relative_to(algorithms).parts
        ]
        if any(any(word in part for word in excluded) for part in parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError, OSError):
            continue
        filename = re.sub(r"[^a-z0-9]", "", str(path.relative_to(algorithms)).lower())
        # Require both detection passes and their fusion in the verified class.
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != "DoublePass":
                continue
            methods = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "detect_tissue"
            ]
            if len(methods) != 1:
                continue
            method = methods[0]
            names = {n.id for n in ast.walk(method) if isinstance(n, ast.Name)}
            calls = {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)}
            if not (
                {"FilterGrays", "DownsampleKMeansDetector"} <= names
                and "bitwise_or" in calls
            ):
                continue
            if (
                path.relative_to(algorithms).as_posix()
                != "combined_detector_algorithm.py"
            ):
                continue
            entries.append(
                {
                    "path": path,
                    "name": node.name,
                    "score": 150,
                    "strong": True,
                    "signature": "DoublePass().detect_tissue(thumbnail)",
                    "double_path": True,
                    "class": True,
                }
            )
        # Exclude single-pass algorithms from Double-Pass discovery.
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function = re.sub(r"[^a-z0-9]", "", node.name.lower())
            if function in {
                "filtergrays",
                "downkmeans",
                "kmeans",
                "otsu",
                "colorstatistics",
                "grayfilter",
                "filtergray",
            }:
                continue
            positional = list(node.args.posonlyargs) + list(node.args.args)
            mandatory = len(positional) - len(node.args.defaults)
            if (
                not positional
                or mandatory > 1
                or node.args.kwonlyargs
                and any(default is None for default in node.args.kw_defaults)
            ):
                continue
            if positional[0].arg.lower() in {
                "self",
                "cls",
                "path",
                "filepath",
                "file_path",
                "directory",
                "folder",
                "root",
            }:
                continue
            # Include indirect helper calls when inspecting both passes.
            refs = {
                re.sub(r"[^a-z0-9]", "", n.id.lower())
                for n in ast.walk(node)
                if isinstance(n, ast.Name)
            }
            refs.update(
                re.sub(r"[^a-z0-9]", "", n.attr.lower())
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute)
            )
            has_filter = any(
                "filtergray" in name or "grayfilter" in name for name in refs
            )
            has_down_kmeans = any(
                ("kmeans" in name or "kmean" in name)
                and ("down" in name or "resize" in name or "small" in name)
                for name in refs
            )
            # Generic K-means names require explicit Double-Pass fusion.
            has_kmeans = any("kmeans" in name for name in refs)
            has_fusion = any(
                isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)
                for n in ast.walk(node)
            ) or any(
                isinstance(n, ast.Call)
                and (isinstance(n.func, ast.Attribute) and n.func.attr == "logical_or")
                for n in ast.walk(node)
            )
            obvious_name = function in {
                "doublepass",
                "doublepassmask",
                "doublepasssegmentation",
                "doublepasstissuedetector",
                "doublepasstissuedetection",
                "doublepassalgorithm",
                "doublepasstissuemask",
                "doublepassdetector",
            }
            path_is_double = "doublepass" in filename or "hybrid" in filename
            generic_name = function in {
                "segment",
                "segmentation",
                "segmentimage",
                "detecttissue",
                "detecttissues",
                "tissuemask",
                "getmask",
                "getmaskfromimage",
                "tissuesegmentation",
                "generatemask",
                "predictmask",
                "predict",
                "runsegmentation",
                "tissuedetection",
            }
            confirmed_composition = has_filter and (
                has_down_kmeans or (has_kmeans and has_fusion)
            )
            score = (
                100
                if obvious_name
                else (
                    90
                    if confirmed_composition
                    else (40 if path_is_double and generic_name else 0)
                )
            )
            entries.append(
                {
                    "path": path,
                    "name": node.name,
                    "score": score,
                    "strong": bool(obvious_name or confirmed_composition),
                    "signature": f"{node.name}({', '.join(a.arg for a in positional)})",
                    "double_path": path_is_double,
                }
            )
    return entries


def _entrypoint_for_repo(repo: Path, explicit: str | None) -> tuple[Path, str]:
    if explicit:
        if ":" not in explicit:
            raise ValueError(
                'entrypoint must look like "algorithms/module.py:function_name".'
            )
        relative, function_name = explicit.rsplit(":", 1)
        if not function_name.isidentifier():
            raise ValueError("Invalid upstream callable name in entrypoint.")
        path = (repo / relative).resolve()
        if (
            not path.is_relative_to(repo.resolve())
            or not path.is_file()
            or path.suffix != ".py"
        ):
            raise ValueError(
                f"Double-Pass entrypoint must point to a Python file inside {repo}: {relative}"
            )
        if any(
            term in part.casefold()
            for part in path.relative_to(repo).parts
            for term in ("grandqc", "comparison", "eval_algorithms")
        ):
            raise ValueError(
                "GrandQC or comparison scripts cannot be Double-Pass entrypoints."
            )
        return path, function_name

    inventory = _upstream_candidates(repo)
    matches = [entry for entry in inventory if entry["score"] >= 40]
    if matches:
        highest = max(item["score"] for item in matches)
        best = [entry for entry in matches if entry["score"] == highest]
        if len(best) == 1:
            item = best[0]
            return item["path"], item["name"]
        paths = ", ".join(
            f"{item['path'].relative_to(repo)}:{item['name']}" for item in best[:8]
        )
        raise RuntimeError(
            "Ambiguous upstream Double-Pass candidates: "
            + paths
            + '. Specify a verified entrypoint="relative/path.py:function".'
        )

    # Reject unverified scripts rather than substituting a different method.
    sample = (
        ", ".join(
            f"{item['path'].relative_to(repo)}:{item['signature']}"
            for item in inventory[:15]
        )
        or "(no top-level callable functions in the algorithms folder)"
    )
    raise RuntimeError(
        "No verified image-to-mask Double-Pass callable or class found in upstream source. "
        "Expected algorithms/combined_detector_algorithm.py:DoublePass. "
        "Found: " + sample + ". "
        'Set entrypoint="algorithms/<file>.py:<class-or-function>" only after confirming '
        "that the function performs BOTH FilterGrays and DownKMeans; "
        "GrandQC comparison code is not a substitute."
    )


def _run_upstream(repo: Path, path: Path, function_name: str, image: np.ndarray):
    """Run the upstream detector on the unmodified RGB thumbnail."""
    # Restrict temporary import paths to the upstream invocation.
    extra = [str(path.parent), str(repo), str(repo / "algorithms")]
    original_path = sys.path[:]
    try:
        for item in reversed(extra):
            sys.path.insert(0, item)
        spec = importlib.util.spec_from_file_location(
            f"_segmenteer_upstream_double_pass_{os.getpid()}",
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to import original Double-Pass source file {path}"
            )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"The original Double-Pass script needs Python module "
                f"{exc.name!r}. Install its upstream runtime dependencies "
                "(openslide-python, opencv-python, scikit-image, matplotlib, tqdm) "
                "in this environment. No GrandQC model is required."
            ) from exc
        algorithm = getattr(module, function_name, None)
        if not callable(algorithm):
            raise RuntimeError(
                f"Upstream callable {function_name!r} was not defined in {path}"
            )
        if isinstance(algorithm, type):
            # The upstream class exposes 0/1 and 0/255 mask representations.
            detector = algorithm()
            detect = getattr(detector, "detect_tissue", None)
            if not callable(detect):
                raise RuntimeError(
                    f"Upstream class {function_name!r} has no detect_tissue method"
                )
            mask = detect(image)
            return detector.mask if mask is None else mask
        return algorithm(image)
    finally:
        sys.path[:] = original_path


class DoublePassSegmenter(NumpySegmenter):
    """Run the official Double-Pass detector at 10 µm/px.

    Uses a pinned upstream Git commit by default. Explicit mutable revisions
    require unpinned-download approval. Model configuration IDs are preserved.
    """

    APPLY_TO_GRAYSCALE = False
    UPSTREAM_URL = UPSTREAM_URL

    def __init__(
        self,
        mpp: float = 10.0,
        upstream_dir: str | Path | None = None,
        revision: str = "master",
        entrypoint: str | None = None,
        auto_checkout: bool = True,
        *args,
        **kwargs,
    ):
        if not np.isfinite(mpp) or abs(float(mpp) - 10.0) > 1e-6:
            raise ValueError(
                "The published Double-Pass benchmark uses mpp=10; "
                "other resolutions require a separate experimental configuration."
            )
        if not revision or not revision.strip():
            raise ValueError("revision must be non-empty")
        super().__init__(mpp=10.0, *args, **kwargs)
        self.upstream_dir = (
            str(Path(upstream_dir).expanduser()) if upstream_dir is not None else None
        )
        self.revision = revision
        self.entrypoint = entrypoint
        self.auto_checkout = bool(auto_checkout)

    @property
    def name(self) -> str:
        return "double_pass_mpp10"

    def _resolve_upstream(self) -> Path:
        if self.upstream_dir is not None:
            # Local upstream sources are used without network access.
            root = Path(self.upstream_dir).expanduser().resolve()
            if not (root / "algorithms").is_dir():
                raise FileNotFoundError(
                    f"Cannot find official Double-Pass algorithms under {root}"
                )
            return root
        root = get_method_model_dir("double-pass") / "upstream"
        selected = _resolved_revision(self.revision)
        with _file_lock(root.parent / ".upstream.checkout.lock"):
            if (root / "algorithms").is_dir() and (root / ".git").is_dir():
                result = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"Cannot verify cached Double-Pass revision: {root}"
                    )
                if (
                    re.fullmatch(r"[a-fA-F0-9]{40}", selected)
                    and result.stdout.strip().lower() != selected.lower()
                ):
                    raise RuntimeError(
                        f"Cached Double-Pass checkout at {root} differs from pinned revision {selected}. "
                        "Refusing to execute an unexpected source revision. "
                        "Move aside the old checkout to allow a fresh pinned clone."
                    )
                # Reject working-tree changes to the audited source.
                status = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                changes = [
                    line
                    for line in status.stdout.splitlines()
                    if not (
                        "/__pycache__/" in line
                        or line.endswith(".pyc")
                        or line.endswith("/.DS_Store")
                    )
                ]
                if status.returncode or changes:
                    raise RuntimeError(
                        "Double-Pass cached source contains uncommitted or untracked files. "
                        "Refusing to execute potentially modified upstream Python source."
                    )
                return root
            if root.exists():
                raise RuntimeError(
                    f"Double-Pass cache is not a complete verified Git checkout: {root}"
                )
            if not self.auto_checkout:
                raise FileNotFoundError(
                    "Double-Pass upstream code has not been installed. "
                    "Enable auto_checkout or provide upstream_dir."
                )
            return _fetch_upstream(root, selected)

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Double-Pass requires an RGB uint8 image (H,W,3).")
        repo = self._resolve_upstream()
        path, function_name = _entrypoint_for_repo(repo, self.entrypoint)
        result = _run_upstream(repo, path, function_name, image)
        if isinstance(result, tuple):
    
            result = result[0]
        mask = np.asarray(result)
        if mask.shape != image.shape[:2] or mask.dtype.kind not in "bui":
            raise RuntimeError(
                "Official Double-Pass code returned an invalid mask: "
                f"expected {image.shape[:2]} integer/bool, "
                f"got {mask.shape} {mask.dtype}."
            )
        unique = np.unique(mask)
        if not np.all(np.isin(unique, [0, 1, 255])):
            raise RuntimeError(
                f"Unexpected values returned by Double-Pass: {unique[:10]}"
            )
        return mask != 0


def _main():
    """List cached upstream callables without importing them."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect official Double-Pass checkout (no code execution)"
    )
    parser.add_argument("--inspect", action="store_true", required=True)
    parser.add_argument(
        "--upstream-dir", type=Path, help="Alternate path to official Git checkout"
    )
    args = parser.parse_args()
    root = args.upstream_dir or get_method_model_dir("double-pass") / "upstream"
    if not root.is_dir():
        parser.error(
            f"Upstream checkout not found: {root}. Run Double-Pass once to fetch it."
        )
    print(f"Double-Pass source: {root}")
    if (root / ".git").exists():
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"Git commit: {rev.stdout.strip() if rev.returncode == 0 else 'unknown'}")
    for item in _upstream_candidates(root):
        print(
            f"{item['path'].relative_to(root)}:{item['signature']} "
            f"[candidate score={item['score']}]"
        )
    try:
        path, function = _entrypoint_for_repo(root, None)
    except RuntimeError as exc:
        print(f"Auto-resolution: unavailable — {exc}")
    else:
        print(f"Auto-resolution: {path.relative_to(root)}:{function}")


if __name__ == "__main__":
    _main()
