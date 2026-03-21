import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Union

import numpy as np

from segmenteer.benchmark.runner import BenchmarkResult


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


def export_results_json(results: List[BenchmarkResult], output_path: Union[str, Path]):
    output_path = Path(output_path)

    data = []
    for result in results:
        result_dict = {
            "method_name": result.method_name,
            "execution_time": result.execution_time,
            "seconds_per_pixel": result.seconds_per_pixel,
            "unsupervised_metrics": asdict(result.unsupervised_metrics),
        }

        if result.supervised_metrics:
            result_dict["supervised_metrics"] = asdict(result.supervised_metrics)
        if result.error:
            result_dict["error"] = result.error

        data.append(result_dict)

    data = _sanitize_for_json(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_results_csv(results: List[BenchmarkResult], output_path: Union[str, Path]):
    output_path = Path(output_path)

    lines = []

    if results and results[0].supervised_metrics:
        header = (
            "method_name,execution_time,seconds_per_pixel,"
            "num_objects,total_area,mean_area,coverage_ratio,"
            "dice,iou,precision,recall,hausdorff,over_seg,under_seg,"
            "pixel_accuracy,mae,balanced_error_rate"
        )
        lines.append(header)

        for result in results:
            u = result.unsupervised_metrics
            s = result.supervised_metrics
            if result.failed or s is None:
                line = (
                    f"{result.method_name},{result.execution_time},,"
                    f"{u.num_objects},{u.total_area},{u.mean_area},{u.coverage_ratio},"
                    f",,,,,,,,"
                )
            else:
                line = (
                    f"{result.method_name},"
                    f"{result.execution_time},"
                    f"{result.seconds_per_pixel},"
                    f"{u.num_objects},"
                    f"{u.total_area},"
                    f"{u.mean_area},"
                    f"{u.coverage_ratio},"
                    f"{s.dice},"
                    f"{s.iou},"
                    f"{s.precision},"
                    f"{s.recall},"
                    f"{s.hausdorff},"
                    f"{s.over_segmentation_rate},"
                    f"{s.under_segmentation_rate},"
                    f"{s.pixel_accuracy},"
                    f"{s.mae},"
                    f"{s.balanced_error_rate}"
                )
            lines.append(line)
    else:
        header = (
            "method_name,execution_time,seconds_per_pixel,"
            "num_objects,total_area,mean_area,std_area,median_area,"
            "mean_compactness,mean_solidity,coverage_ratio"
        )
        lines.append(header)

        for result in results:
            u = result.unsupervised_metrics
            line = (
                f"{result.method_name},"
                f"{result.execution_time},"
                f"{result.seconds_per_pixel},"
                f"{u.num_objects},"
                f"{u.total_area},"
                f"{u.mean_area},"
                f"{u.std_area},"
                f"{u.median_area},"
                f"{u.mean_compactness},"
                f"{u.mean_solidity},"
                f"{u.coverage_ratio}"
            )
            lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
