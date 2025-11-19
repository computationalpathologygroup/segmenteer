#!/usr/bin/env python3
"""
Benchmark segmentation methods on cropped regions with annotations.
Matches TIFF images with their corresponding GeoJSON annotation files.
"""
from pathlib import Path
import segmenteer as seg


def find_matching_pairs(image_dir: Path, anno_dir: Path):
    """Find TIFF images with corresponding GeoJSON annotations."""
    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)
    
    # Find all TIFF files
    tiff_files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
    
    matching_pairs = []
    for tiff_file in tiff_files:
        # Try to find corresponding annotation file
        base_name = tiff_file.stem
        anno_file = anno_dir / f"{base_name}.geojson"
        
        if anno_file.exists():
            matching_pairs.append((tiff_file, anno_file))
            print(f"  ✓ {tiff_file.name} -> {anno_file.name}")
        else:
            print(f"  ✗ {tiff_file.name} (no annotation found)")
    
    return matching_pairs


def main():
    # Paths to data
    image_dir = Path("/Users/agatapolejowska/histopathobiome-s/data/cropped_regions")
    anno_dir = Path("/Users/agatapolejowska/histopathobiome-s/data/cropped_regions_anno")
    
    print("Scanning for matching image-annotation pairs...")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {anno_dir}")
    print()
    
    matching_pairs = find_matching_pairs(image_dir, anno_dir)
    
    if not matching_pairs:
        print("\nNo matching pairs found!")
        return
    
    print(f"\nFound {len(matching_pairs)} matching pairs")
    print("=" * 80)
    
    # Create output directory
    output_dir = seg.create_timestamped_output_dir("outputs/benchmark_cropped_regions")
    print(f"\nOutput directory: {output_dir}\n")
    
    # Define segmentation methods to benchmark
    segmenters = [
        #seg.BackgroundSubtractorMOG2Segmenter(
        #     history=50,
        #     var_threshold=16.0,
        #     detect_shadows=True,
        #     min_area=10,
        # ),
        #seg.EntropyMaskerSegmenter(),
        # seg.BackgroundSubtractorMOG2Segmenter(),  # Fails on large images (>300M pixels)
        #seg.GrandQCSegmenter(confidence_threshold=0.5, min_area=10),
        seg.HESTSegmenter(mpp=1.0, confidence_threshold=0.5, min_area=10),
    ]
    
    print(f"Testing {len(segmenters)} segmentation methods:")
    for i, segmenter in enumerate(segmenters, 1):
        print(f"  {i}. {segmenter.name}")
    print()
    
    # Run benchmarking on each image
    all_results = {}
    
    for idx, (image_path, anno_path) in enumerate(matching_pairs, 1):
        print(f"{'=' * 80}")
        print(f"Processing [{idx}/{len(matching_pairs)}]: {image_path.name}")
        print(f"{'=' * 80}")
        
        # Load image and annotation
        print("Loading image...")
        image = seg.load_image(image_path)
        print(f"  Image size: {image.shape[0]}x{image.shape[1]} pixels")
        
        print("Loading ground truth annotation...")
        ground_truth = seg.load_geojson(anno_path)
        
        # Save individual image results as we go
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(exist_ok=True)
        
        # Save thumbnail
        seg.save_thumbnail(image, image_output_dir / "original_thumbnail.png")
        
        # Run each segmenter individually and save results immediately
        image_results = []
        
        for i, segmenter in enumerate(segmenters, 1):
            print(f"\n[{i}/{len(segmenters)}] Running {segmenter.name}...")
            
            try:
                # Check if this is BackgroundSubtractor and image is too large
                is_bg_subtractor = 'bg_subtractor' in segmenter.name.lower() or 'background' in segmenter.name.lower()
                total_pixels = image.shape[0] * image.shape[1] * (image.shape[2] if len(image.shape) > 2 else 1)
                needs_downsampling = is_bg_subtractor and total_pixels > 100_000_000
                
                if needs_downsampling:
                    downsample_factor = 8
                    print(f"  Large image detected - downsampling by {downsample_factor}x for BackgroundSubtractor")
                    image_downsampled = seg.downsample_image(image, downsample_factor)
                    print(f"  Processing at: {image_downsampled.shape[0]}x{image_downsampled.shape[1]} pixels")
                    
                    runner = seg.BenchmarkRunner(verbose=False)
                    result = runner.run_single(segmenter, image_downsampled, ground_truth)
                    
                    # Scale GeoJSON back to full resolution
                    for feature in result.geojson['features']:
                        coords = feature['geometry']['coordinates']
                        if feature['geometry']['type'] == 'Polygon':
                            feature['geometry']['coordinates'] = [
                                [[x * downsample_factor, y * downsample_factor] for x, y in ring]
                                for ring in coords
                            ]
                        elif feature['geometry']['type'] == 'MultiPolygon':
                            feature['geometry']['coordinates'] = [
                                [[[x * downsample_factor, y * downsample_factor] for x, y in ring] for ring in polygon]
                                for polygon in coords
                            ]
                    
                    # Recompute metrics at full resolution
                    from segmenteer.metrics.evaluation import compute_all_supervised_metrics
                    from segmenteer.metrics.unsupervised import compute_unsupervised_metrics
                    
                    image_area = float(image.shape[0] * image.shape[1])
                    result.unsupervised_metrics = compute_unsupervised_metrics(result.geojson, image_area)
                    result.supervised_metrics = compute_all_supervised_metrics(result.geojson, ground_truth, (image.shape[0], image.shape[1]))
                else:
                    # Run the segmenter normally
                    runner = seg.BenchmarkRunner(verbose=False)
                    result = runner.run_single(segmenter, image, ground_truth)
                
                image_results.append(result)
                
                # Calculate and display performance metrics
                total_pixels = image.shape[0] * image.shape[1]
                megapixels = total_pixels / 1_000_000
                megapixels_per_second = megapixels / result.execution_time
                
                print(f"  Image: {image.shape[0]}x{image.shape[1]} = {megapixels:.1f}MP")
                print(f"  Execution time: {result.execution_time:.2f}s ({megapixels_per_second:.2f} MP/s)")
                print(f"  Objects: {result.unsupervised_metrics.num_objects}")
                if result.supervised_metrics:
                    print(f"  Dice: {result.supervised_metrics.dice:.4f}, IoU: {result.supervised_metrics.iou:.4f}")
                
                # Save this method's GeoJSON
                seg.save_geojson(
                    result.geojson,
                    image_output_dir / f"{result.method_name}.geojson"
                )
                
                # Create and save visualization thumbnail
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
                
                # Save incremental CSV/JSON after each method
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
        
        # Store results for this image
        all_results[image_path.stem] = image_results
        
        # Generate per-image markdown report
        md_lines = [
            f"# Benchmark Results: {image_path.name}",
            f"",
            f"**Image:** `{image_path.name}`  ",
            f"**Size:** {image.shape[0]}x{image.shape[1]} pixels ({image.shape[0]*image.shape[1]/1_000_000:.1f}MP)  ",
            f"**Methods tested:** {len(image_results)}  ",
            f"",
            f"## Results",
            f"",
            f"| Method | Time (s) | MP/s | Dice | IoU | Precision | Recall | Hausdorff | Objects |",
            f"|--------|----------|------|------|-----|-----------|--------|-----------|---------|"
        ]
        
        for result in image_results:
            megapixels = (image.shape[0] * image.shape[1]) / 1_000_000
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
            f"",
            f"## Additional Supervised Metrics",
            f"",
            f"| Method | Pixel Accuracy | MAE | Balanced Error | Over-seg % | Under-seg % |",
            f"|--------|----------------|-----|----------------|------------|-------------|"
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
        
        # Add unsupervised metrics section
        md_lines.extend([
            f"",
            f"## Unsupervised Metrics",
            f"",
            f"| Method | Coverage % | Mean Area | Compactness | Solidity | Total Objects |",
            f"|--------|------------|-----------|-------------|----------|---------------|"
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
        
        # Add visualizations section
        md_lines.extend([
            f"",
            f"## Visualizations",
            f"",
            f"### Original Image",
            f"",
            f'<img src="original_thumbnail.png" width="400">',
            f"",
            f"### Segmentation Results (Side-by-Side)",
            f"",
            f"| Method | Visualization |",
            f"|--------|---------------|"
        ])
        
        for result in image_results:
            thumbnail_path = f"{result.method_name}_thumbnail.png"
            if (image_output_dir / thumbnail_path).exists():
                md_lines.append(
                    f'| {result.method_name} | <img src="{thumbnail_path}" width="400"> |'
                )
        
        md_content = "\n".join(md_lines)
        (image_output_dir / "results.md").write_text(md_content)
        
        # Print summary for this image
        print(f"\nResults for {image_path.name}:")
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
    
    # Print overall summary
    print(f"\n\n{'=' * 120}")
    print(f"OVERALL SUMMARY - {len(matching_pairs)} images")
    print(f"{'=' * 120}\n")
    
    # Compute average metrics across all images for each method
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
    
    # Generate overall summary markdown report
    summary_md = [
        f"# Benchmark Summary",
        f"",
        f"**Total images:** {len(matching_pairs)}  ",
        f"**Methods tested:** {len(segmenters)}  ",
        f"**Output directory:** `{output_dir}`  ",
        f"",
        f"## Average Performance Across All Images",
        f"",
        f"| Method | Avg Dice | Avg IoU | Avg Precision | Avg Recall | Avg Time (s) | Avg Objects |",
        f"|--------|----------|---------|---------------|------------|--------------|-------------|"
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
        f"",
        f"## Additional Metrics",
        f"",
        f"| Method | Pixel Accuracy | MAE | Balanced Error | Over-seg % | Under-seg % |",
        f"|--------|----------------|-----|----------------|------------|-------------|"
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
        f"",
        f"## Per-Image Results",
        f""
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
