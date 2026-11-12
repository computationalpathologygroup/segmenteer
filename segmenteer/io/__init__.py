from importlib import import_module

_LAZY_EXPORTS = {
    "convert_output_directory": ("segmenteer.io.convert", "convert_output_directory"),
    "write_text_atomic": ("segmenteer.io.atomic", "write_text_atomic"),
    "write_json_atomic": ("segmenteer.io.atomic", "write_json_atomic"),
    "load_asap_xml": ("segmenteer.io.asap", "load_asap_xml"),
    "save_asap_xml": ("segmenteer.io.asap", "save_asap_xml"),
    "PREDICTION_FORMATS": ("segmenteer.io.predictions", "PREDICTION_FORMATS"),
    "normalize_prediction_format": (
        "segmenteer.io.predictions",
        "normalize_prediction_format",
    ),
    "prediction_path": ("segmenteer.io.predictions", "prediction_path"),
    "read_prediction": ("segmenteer.io.predictions", "read_prediction"),
    "save_prediction": ("segmenteer.io.predictions", "save_prediction"),
    "load_image": ("segmenteer.io.loader", "load_image"),
    "save_geojson": ("segmenteer.io.loader", "save_geojson"),
    "load_geojson": ("segmenteer.io.loader", "load_geojson"),
    "GroundTruthPairing": ("segmenteer.io.loader", "GroundTruthPairing"),
    "inspect_ground_truths": ("segmenteer.io.loader", "inspect_ground_truths"),
    "format_ground_truth_pairing_error": (
        "segmenteer.io.loader",
        "format_ground_truth_pairing_error",
    ),
    "load_ground_truths": ("segmenteer.io.loader", "load_ground_truths"),
    "create_timestamped_output_dir": (
        "segmenteer.io.utils",
        "create_timestamped_output_dir",
    ),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)
