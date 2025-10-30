import segmenteer as seg
from pathlib import Path


def process_single_tiff(
    tiff_path: Path,
    output_base_dir: Path,
    segmenters: list,
    pyramid_level: int = 4,
    downsample_factor: int = 4,
):
    tiff_name = tiff_path.stem if tiff_path.is_file() else tiff_path.name
    output_dir = output_base_dir / tiff_name
    
    if output_dir.exists():
        expected_files = ["original_thumbnail.png", "results.csv", "results.json"]
        for segmenter in segmenters:
            expected_files.extend([
                f"{segmenter.name}_fullres.geojson",
                f"{segmenter.name}_downsampled.geojson",
                f"{segmenter.name}_heatmap.png",
            ])
        
        all_files_exist = all((output_dir / f).exists() for f in expected_files)
        
        if all_files_exist:
            print(f"Skipping {tiff_name} - all outputs already exist")
            return True
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing: {tiff_name}")
    print(f"{'='*80}")

    try:
        original_image = seg.load_image(path=tiff_path, level=pyramid_level)

        print(
            f"Loaded pyramid level {pyramid_level}: {original_image.shape[0]}x{original_image.shape[1]} pixels"
        )
        print(f"Downsampling by {downsample_factor}x for processing...")

        image = seg.downsample_image(original_image, downsample_factor)
        print(f"Working with: {image.shape[0]}x{image.shape[1]} pixels")

        runner = seg.BenchmarkRunner()
        results = runner.run_multiple(segmenters, image)

        print("\nUnsupervised Benchmarking Results")
        print("-" * 80)
        print(
            f"{'Method':35s} {'Time':>8s} {'Objects':>8s} {'Coverage':>10s} {'Mean Area':>12s}"
        )
        print("-" * 80)

        for result in results:
            u = result.unsupervised_metrics
            print(
                f"{result.method_name:35s} "
                f"{result.execution_time:8.2f}s "
                f"{u.num_objects:8d} "
                f"{u.coverage_ratio * 100:9.2f}% "
                f"{u.mean_area:12.2f}"
            )

        print("\nSaving results...")

        seg.save_thumbnail(image, output_dir / "original_thumbnail.png")
        print("  ✓ original_thumbnail.png")

        total_scale_factor = downsample_factor * (2 ** pyramid_level)
        for result in results:
            seg.save_geojson(
                result.geojson,
                output_dir / f"{result.method_name}_fullres.geojson",
                scale_factor=total_scale_factor,
            )
            print(f"  ✓ {result.method_name}_fullres.geojson")

        for result in results:
            seg.save_geojson(
                result.geojson,
                output_dir / f"{result.method_name}_downsampled.geojson",
            )
            print(f"  ✓ {result.method_name}_downsampled.geojson")

        for result in results:
            seg.save_heatmap_thumbnail(
                image,
                result.geojson,
                output_dir / f"{result.method_name}_heatmap.png",
                max_size=1024,
                alpha=0.4,
            )
            print(f"  ✓ {result.method_name}_heatmap.png")

        seg.export_results_csv(results, output_dir / "results.csv")
        seg.export_results_json(results, output_dir / "results.json")
        print("  ✓ results.csv")
        print("  ✓ results.json")

        print(f"\n✓ Successfully processed {tiff_name}")
        return True

    except Exception as e:
        print(f"\n✗ Error processing {tiff_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    input_dir = Path("/Users/agatapolejowska/histopathobiome-s/data/cropped_regions")
    output_base_dir = Path("outputs_unsupervised_benchmark")
    pyramid_level = 0
    downsample_factor = 4

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    tiff_files = sorted(list(input_dir.glob("*.tiff"))) + sorted(
        list(input_dir.glob("*.tif"))
    )
    dicom_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if tiff_files and dicom_dirs:
        print("Warning: Found both TIFF files and directories (potential DICOMs)")
        print("Prioritizing TIFF files")
        all_inputs = tiff_files
        file_type = "TIFF"
    elif tiff_files:
        all_inputs = tiff_files
        file_type = "TIFF"
    elif dicom_dirs:
        all_inputs = dicom_dirs
        file_type = "DICOM"
    else:
        print(f"No TIFF files or DICOM directories found in {input_dir}")
        return

    print(f"Found {len(all_inputs)} {file_type} files to process")
    print(f"Output directory: {output_base_dir.absolute()}")
    print(f"Downsample factor: {downsample_factor}x")

    segmenters = [
        seg.OtsuSegmenter(min_area=10),
        seg.EntropyMaskerSegmenter(min_area=10),
        seg.BackgroundSubtractorMOG2Segmenter(
            history=50, var_threshold=16.0, detect_shadows=True, min_area=10
        ),
        seg.FastSAMSegmenter(
            model_name="FastSAM-x.pt",
            text_prompt="foreground",
            conf=0.4,
            iou=0.9,
            min_area=10,
        ),
        seg.GrandQCSegmenter(confidence_threshold=0.5, min_area=10),
        seg.HESTSegmenter(mpp=1.0, confidence_threshold=0.5, min_area=10),
    ]

    print("\nMethods to benchmark:")
    for s in segmenters:
        print(f"  - {s.name}")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0

    for i, input_path in enumerate(all_inputs, 1):
        input_name = input_path.stem if input_path.is_file() else input_path.name
        output_dir = output_base_dir / input_name
        
        was_skipped = False
        if output_dir.exists() and (output_dir / "results.csv").exists():
            expected_files = ["original_thumbnail.png", "results.csv", "results.json"]
            for segmenter in segmenters:
                expected_files.extend([
                    f"{segmenter.name}_fullres.geojson",
                    f"{segmenter.name}_downsampled.geojson",
                    f"{segmenter.name}_heatmap.png",
                ])
            
            if all((output_dir / f).exists() for f in expected_files):
                was_skipped = True
        
        print(f"\n[{i}/{len(all_inputs)}] Processing {input_path.name}...")

        success = process_single_tiff(
            input_path, output_base_dir, segmenters, pyramid_level, downsample_factor
        )

        if was_skipped:
            skipped += 1
        elif success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*80}")
    print("BATCH BENCHMARKING COMPLETE")
    print(f"{'='*80}")
    print(f"Total files: {len(all_inputs)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_base_dir.absolute()}")


if __name__ == "__main__":
    main()
