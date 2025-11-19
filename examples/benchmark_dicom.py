#!/usr/bin/env python3
"""
Benchmark segmentation methods on DICOM images with annotations.
Matches DICOM images with their corresponding GeoJSON annotation files.
"""
from pathlib import Path
import segmenteer as seg


def find_matching_pairs(image_dir: Path, anno_dir: Path):
    """Find DICOM directories with corresponding GeoJSON annotations."""
    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)
    
    top_level_dirs = [d for d in image_dir.iterdir() if d.is_dir()]
    
    matching_pairs = []
    for top_dir in top_level_dirs:
        base_name = top_dir.name
        anno_file = anno_dir / f"{base_name}.geojson"
        
        if not anno_file.exists():
            print(f"  ✗ {top_dir.name} (no annotation found)")
            continue
        
        dicom_subdir = top_dir / f"{base_name}-dicom"
        if dicom_subdir.exists() and dicom_subdir.is_dir():
            matching_pairs.append((dicom_subdir, anno_file))
            print(f"  ✓ {dicom_subdir.relative_to(image_dir)} -> {anno_file.name}")
        else:
            print(f"  ✗ {top_dir.name} (no -dicom subdirectory found)")
    
    return matching_pairs


def main():
    image_dir = Path("data")
    anno_dir = Path("data_anno")
    
    print("Scanning for matching DICOM-annotation pairs...")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {anno_dir}")
    print()
    
    matching_pairs = find_matching_pairs(image_dir, anno_dir)
    
    if not matching_pairs:
        print("\nNo matching pairs found!")
        return
    
    print(f"\nFound {len(matching_pairs)} matching pairs")
    print("=" * 80)
    
    output_dir = seg.create_timestamped_output_dir("outputs/benchmark_dicom")
    print(f"\nOutput directory: {output_dir}\n")
    
    segmenters = [
        seg.EntropyMaskerSegmenter(),
        seg.BackgroundSubtractorMOG2Segmenter(
            history=50,
            var_threshold=16.0,
            detect_shadows=True,
            min_area=10,
        ),
    ]
    
    downsample_factor = 8
    print(f"Downsampling factor: {downsample_factor}x")
    print()
    
    all_results = {}
    
    for idx, (image_path, anno_path) in enumerate(matching_pairs, 1):
        print(f"{'=' * 80}")
        print(f"Processing [{idx}/{len(matching_pairs)}]: {image_path.parent.name}/{image_path.name}")
        print(f"{'=' * 80}")
        
        print("Loading DICOM image...")
        image = seg.load_image(image_path, level=4)
        print(f"  Loaded at level 3: {image.shape[0]}x{image.shape[1]} pixels ({image.shape[0]*image.shape[1]/1_000_000:.1f}MP)")
        
        original_level_factor = 8
        print(f"  Downsampling by {downsample_factor}x...")
        image_downsampled = seg.downsample_image(image, downsample_factor)
        print(f"  Processing at: {image_downsampled.shape[0]}x{image_downsampled.shape[1]} pixels ({image_downsampled.shape[0]*image_downsampled.shape[1]/1_000_000:.1f}MP)")
        
        print("Loading ground truth annotation...")
        ground_truth = seg.load_geojson(anno_path)
        
        image_output_dir = output_dir / image_path.parent.name
        image_output_dir.mkdir(exist_ok=True)
        
        seg.save_thumbnail(image, image_output_dir / "original_thumbnail.png")
        
        image_results = []
        
        for i, segmenter in enumerate(segmenters, 1):
            print(f"\n[{i}/{len(segmenters)}] Running {segmenter.name}...")
            
            try:
                runner = seg.BenchmarkRunner(verbose=False)
                result = runner.run_single(segmenter, image_downsampled, ground_truth)
                
                combined_scale = original_level_factor * downsample_factor
                for feature in result.geojson['features']:
                    coords = feature['geometry']['coordinates']
                    if feature['geometry']['type'] == 'Polygon':
                        feature['geometry']['coordinates'] = [
                            [[x * combined_scale, y * combined_scale] for x, y in ring]
                            for ring in coords
                        ]
                    elif feature['geometry']['type'] == 'MultiPolygon':
                        feature['geometry']['coordinates'] = [
                            [[[x * combined_scale, y * combined_scale] for x, y in ring] for ring in polygon]
                            for polygon in coords
                        ]
                
                from segmenteer.metrics.evaluation import compute_all_supervised_metrics
                from segmenteer.metrics.unsupervised import compute_unsupervised_metrics
                
                full_res_shape = (image.shape[0] * original_level_factor, image.shape[1] * original_level_factor)
                image_area = float(full_res_shape[0] * full_res_shape[1])
                result.unsupervised_metrics = compute_unsupervised_metrics(result.geojson, image_area)
                result.supervised_metrics = compute_all_supervised_metrics(result.geojson, ground_truth, full_res_shape)
                
                image_results.append(result)
                
                full_res_shape = (image.shape[0] * original_level_factor, image.shape[1] * original_level_factor)
                total_pixels = full_res_shape[0] * full_res_shape[1]
                megapixels = total_pixels / 1_000_000
                megapixels_per_second = megapixels / result.execution_time
                
                print(f"  Full resolution: {full_res_shape[0]}x{full_res_shape[1]} = {megapixels:.1f}MP")
                print(f"  Execution time: {result.execution_time:.2f}s ({megapixels_per_second:.2f} MP/s)")
                print(f"  Objects: {result.unsupervised_metrics.num_objects}")
                if result.supervised_metrics:
                    print(f"  Dice: {result.supervised_metrics.dice:.4f}, IoU: {result.supervised_metrics.iou:.4f}")
                
                seg.save_geojson(
                    result.geojson,
                    image_output_dir / f"{result.method_name}.geojson"
                )
                
                try:
                    seg.save_heatmap_thumbnail(
                        image, 
                        result.geojson, 
                        image_output_dir / f"{result.method_name}_thumbnail.png",
                        max_size=1024,
                        alpha=0.4
                    )
                    saved_viz = True
                except Exception as viz_error:
                    print(f"    Warning: Could not save thumbnail: {viz_error}")
                    saved_viz = False
                
                all_results_flat = []
                for img_name, img_results in all_results.items():
                    all_results_flat.extend(img_results)
                all_results_flat.extend(image_results)
                
                seg.export_results_csv(all_results_flat, output_dir / "benchmark_results.csv")
                seg.export_results_json(all_results_flat, output_dir / "benchmark_results.json")
                
                viz_msg = " + thumbnail" if saved_viz else ""
                print(f"  ✓ Saved {result.method_name} results (GeoJSON{viz_msg})")
                
            except Exception as e:
                print(f"  ✗ {segmenter.name} failed: {e}")
                continue
        
        all_results[image_path.parent.name] = image_results
        
        full_res_shape = (image.shape[0] * original_level_factor, image.shape[1] * original_level_factor)
        md_lines = [
            f"# Benchmark Results: {image_path.parent.name}",
            "",
            f"**Image:** `{image_path.parent.name}/{image_path.name}`  ",
            f"**Size (full res):** {full_res_shape[0]}x{full_res_shape[1]} pixels ({full_res_shape[0]*full_res_shape[1]/1_000_000:.1f}MP)  ",
            f"**Processed at:** level 3 + {downsample_factor}x downsampling  ",
            f"**Methods tested:** {len(image_results)}  ",
            "",
            "## Results",
            "",
            "| Method | Time (s) | MP/s | Dice | IoU | Precision | Recall | Hausdorff | Objects |",
            "|--------|----------|------|------|-----|-----------|--------|-----------|---------|"
        ]
        
        for result in image_results:
            full_res_pixels = full_res_shape[0] * full_res_shape[1]
            megapixels = full_res_pixels / 1_000_000
            mp_per_s = megapixels / result.execution_time
            
            if result.supervised_metrics:
                haus_str = f"{result.supervised_metrics.hausdorff:.2f}" if result.supervised_metrics.hausdorff != float('inf') else "inf"
                md_lines.append(
                    f"| {result.method_name} | {result.execution_time:.2f} | {mp_per_s:.2f} | "
                    f"{result.supervised_metrics.dice:.4f} | {result.supervised_metrics.iou:.4f} | "
                    f"{result.supervised_metrics.precision:.4f} | {result.supervised_metrics.recall:.4f} | "
                    f"{haus_str} | {result.unsupervised_metrics.num_objects} |"
                )
            else:
                md_lines.append(
                    f"| {result.method_name} | {result.execution_time:.2f} | {mp_per_s:.2f} | "
                    f"N/A | N/A | N/A | N/A | N/A | {result.unsupervised_metrics.num_objects} |"
                )
        
        md_lines.extend([
            "",
            "## Additional Supervised Metrics",
            "",
            "| Method | Pixel Accuracy | MAE | Balanced Error | Over-seg % | Under-seg % |",
            "|--------|----------------|-----|----------------|------------|-------------|"
        ])
        
        for result in image_results:
            if result.supervised_metrics:
                md_lines.append(
                    f"| {result.method_name} | "
                    f"{result.supervised_metrics.pixel_accuracy:.4f} | "
                    f"{result.supervised_metrics.mae:.4f} | "
                    f"{result.supervised_metrics.balanced_error_rate:.4f} | "
                    f"{result.supervised_metrics.over_segmentation_rate:.4f} | "
                    f"{result.supervised_metrics.under_segmentation_rate:.4f} |"
                )
            else:
                md_lines.append(
                    f"| {result.method_name} | N/A | N/A | N/A | N/A | N/A |"
                )
        
        md_lines.extend([
            "",
            "## Unsupervised Metrics",
            "",
            "| Method | Coverage % | Mean Area | Compactness | Solidity | Total Objects |",
            "|--------|------------|-----------|-------------|----------|---------------|"
        ])
        
        for result in image_results:
            md_lines.append(
                f"| {result.method_name} | "
                f"{result.unsupervised_metrics.coverage_ratio:.4f} | "
                f"{result.unsupervised_metrics.mean_area:.1f} | "
                f"{result.unsupervised_metrics.mean_compactness:.4f} | "
                f"{result.unsupervised_metrics.mean_solidity:.4f} | "
                f"{result.unsupervised_metrics.num_objects} |"
            )
        
        md_lines.extend([
            "",
            "## Visualizations",
            "",
            "### Original Image",
            "",
            '<img src="original_thumbnail.png" width="400">',
            "",
            "### Segmentation Results (Side-by-Side)",
            "",
            "| Method | Visualization |",
            "|--------|---------------|"
        ])
        
        for result in image_results:
            thumbnail_path = f"{result.method_name}_thumbnail.png"
            if (image_output_dir / thumbnail_path).exists():
                md_lines.append(
                    f'| {result.method_name} | <img src="{thumbnail_path}" width="400"> |'
                )
        
        md_content = "\n".join(md_lines)
        (image_output_dir / "results.md").write_text(md_content)
        
        print(f"\nResults for {image_path.parent.name}:")
        print("-" * 120)
        print(f"{'Method':30s} {'Time':>8s} {'Dice':>8s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'Haus':>8s} {'Over%':>8s} {'Under%':>8s} {'Objects':>8s}")
        print(f"{'':30s} {'':>8s} {'PixAcc':>8s} {'MAE':>8s} {'BER':>8s}")
        print("-" * 120)
        
        if not image_results:
            print("  No results for this image")
        
        for result in image_results:
            if result.supervised_metrics:
                haus_str = f"{result.supervised_metrics.hausdorff:8.2f}" if result.supervised_metrics.hausdorff != float('inf') else "     inf"
                print(
                    f"{result.method_name:30s} "
                    f"{result.execution_time:8.2f}s "
                    f"{result.supervised_metrics.dice:8.4f} "
                    f"{result.supervised_metrics.iou:8.4f} "
                    f"{result.supervised_metrics.precision:8.4f} "
                    f"{result.supervised_metrics.recall:8.4f} "
                    f"{haus_str} "
                    f"{result.supervised_metrics.over_segmentation_rate:8.4f} "
                    f"{result.supervised_metrics.under_segmentation_rate:8.4f} "
                    f"{result.unsupervised_metrics.num_objects:8d}"
                )
                print(
                    f"{'':30s} {'':>8s} "
                    f"{result.supervised_metrics.pixel_accuracy:8.4f} "
                    f"{result.supervised_metrics.mae:8.4f} "
                    f"{result.supervised_metrics.balanced_error_rate:8.4f}"
                )
            else:
                print(f"{result.method_name:30s} {result.execution_time:8.2f}s (no supervised metrics)")
        print()
    
    print(f"\n\n{'=' * 120}")
    print(f"OVERALL SUMMARY - {len(matching_pairs)} images")
    print(f"{'=' * 120}\n")
    
    method_stats = {}
    for method in segmenters:
        method_name = method.name
        method_stats[method_name] = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'hausdorff': [],
            'over_seg': [],
            'under_seg': [],
            'pixel_acc': [],
            'mae': [],
            'ber': [],
            'time': [],
            'objects': []
        }
    
    for image_name, results in all_results.items():
        for result in results:
            if result.supervised_metrics:
                method_stats[result.method_name]['dice'].append(result.supervised_metrics.dice)
                method_stats[result.method_name]['iou'].append(result.supervised_metrics.iou)
                method_stats[result.method_name]['precision'].append(result.supervised_metrics.precision)
                method_stats[result.method_name]['recall'].append(result.supervised_metrics.recall)
                method_stats[result.method_name]['hausdorff'].append(result.supervised_metrics.hausdorff)
                method_stats[result.method_name]['over_seg'].append(result.supervised_metrics.over_segmentation_rate)
                method_stats[result.method_name]['under_seg'].append(result.supervised_metrics.under_segmentation_rate)
                method_stats[result.method_name]['pixel_acc'].append(result.supervised_metrics.pixel_accuracy)
                method_stats[result.method_name]['mae'].append(result.supervised_metrics.mae)
                method_stats[result.method_name]['ber'].append(result.supervised_metrics.balanced_error_rate)
            method_stats[result.method_name]['time'].append(result.execution_time)
            method_stats[result.method_name]['objects'].append(result.unsupervised_metrics.num_objects)
    
    print(f"{'Method':30s} {'Dice':>8s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'Haus':>8s} {'Over%':>8s} {'Under%':>8s} {'Time':>8s} {'Objects':>8s}")
    print(f"{'':30s} {'PixAcc':>8s} {'MAE':>8s} {'BER':>8s}")
    print("-" * 120)
    
    for method_name, stats in method_stats.items():
        avg_dice = sum(stats['dice']) / len(stats['dice']) if stats['dice'] else 0
        avg_iou = sum(stats['iou']) / len(stats['iou']) if stats['iou'] else 0
        avg_prec = sum(stats['precision']) / len(stats['precision']) if stats['precision'] else 0
        avg_recall = sum(stats['recall']) / len(stats['recall']) if stats['recall'] else 0
        avg_haus = sum(stats['hausdorff']) / len(stats['hausdorff']) if stats['hausdorff'] else 0
        avg_over = sum(stats['over_seg']) / len(stats['over_seg']) if stats['over_seg'] else 0
        avg_under = sum(stats['under_seg']) / len(stats['under_seg']) if stats['under_seg'] else 0
        avg_pixel_acc = sum(stats['pixel_acc']) / len(stats['pixel_acc']) if stats['pixel_acc'] else 0
        avg_mae = sum(stats['mae']) / len(stats['mae']) if stats['mae'] else 0
        avg_ber = sum(stats['ber']) / len(stats['ber']) if stats['ber'] else 0
        avg_time = sum(stats['time']) / len(stats['time'])
        avg_objects = sum(stats['objects']) / len(stats['objects'])
        
        print(
            f"{method_name:30s} "
            f"{avg_dice:8.4f} "
            f"{avg_iou:8.4f} "
            f"{avg_prec:8.4f} "
            f"{avg_recall:8.4f} "
            f"{avg_haus:8.2f} "
            f"{avg_over:8.4f} "
            f"{avg_under:8.4f} "
            f"{avg_time:8.2f}s "
            f"{avg_objects:8.1f}"
        )
        print(
            f"{'':30s} "
            f"{avg_pixel_acc:8.4f} "
            f"{avg_mae:8.4f} "
            f"{avg_ber:8.4f}"
        )
    
    summary_md = [
        "# Benchmark Summary",
        "",
        f"**Total images:** {len(matching_pairs)}  ",
        f"**Methods tested:** {len(segmenters)}  ",
        f"**Output directory:** `{output_dir}`  ",
        "",
        "## Average Performance Across All Images",
        "",
        "| Method | Avg Dice | Avg IoU | Avg Precision | Avg Recall | Avg Time (s) | Avg Objects |",
        "|--------|----------|---------|---------------|------------|--------------|-------------|"
    ]
    
    for method_name, stats in method_stats.items():
        avg_dice = sum(stats['dice']) / len(stats['dice']) if stats['dice'] else 0
        avg_iou = sum(stats['iou']) / len(stats['iou']) if stats['iou'] else 0
        avg_prec = sum(stats['precision']) / len(stats['precision']) if stats['precision'] else 0
        avg_recall = sum(stats['recall']) / len(stats['recall']) if stats['recall'] else 0
        avg_time = sum(stats['time']) / len(stats['time'])
        avg_objects = sum(stats['objects']) / len(stats['objects'])
        
        summary_md.append(
            f"| {method_name} | {avg_dice:.4f} | {avg_iou:.4f} | {avg_prec:.4f} | "
            f"{avg_recall:.4f} | {avg_time:.2f} | {avg_objects:.1f} |"
        )
    
    summary_md.extend([
        "",
        "## Additional Metrics",
        "",
        "| Method | Pixel Accuracy | MAE | Balanced Error | Over-seg % | Under-seg % |",
        "|--------|----------------|-----|----------------|------------|-------------|"
    ])
    
    for method_name, stats in method_stats.items():
        avg_pixel_acc = sum(stats['pixel_acc']) / len(stats['pixel_acc']) if stats['pixel_acc'] else 0
        avg_mae = sum(stats['mae']) / len(stats['mae']) if stats['mae'] else 0
        avg_ber = sum(stats['ber']) / len(stats['ber']) if stats['ber'] else 0
        avg_over = sum(stats['over_seg']) / len(stats['over_seg']) if stats['over_seg'] else 0
        avg_under = sum(stats['under_seg']) / len(stats['under_seg']) if stats['under_seg'] else 0
        
        summary_md.append(
            f"| {method_name} | {avg_pixel_acc:.4f} | {avg_mae:.4f} | {avg_ber:.4f} | "
            f"{avg_over:.4f} | {avg_under:.4f} |"
        )
    
    summary_md.extend([
        "",
        "## Per-Image Results",
        ""
    ])
    
    for image_name in all_results.keys():
        summary_md.append(f"- [{image_name}](./{image_name}/results.md)")
    
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_md))
    
    print(f"\nAll results saved to: {output_dir}")
    print("  - SUMMARY.md (overall benchmark summary)")
    print("  - benchmark_results.csv")
    print("  - benchmark_results.json")
    print("  - Individual per-image results.md files")
    print("  - Individual GeoJSON files per method")
    print("\nBenchmarking complete! ✓")


if __name__ == "__main__":
    main()
