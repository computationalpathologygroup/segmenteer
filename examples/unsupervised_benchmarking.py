import segmenteer as seg


def main():
    # original_image = seg.load_image("data_small/013_M1.tiff")
    original_image = seg.load_image(
        path="data_small/1d4c700d-07d1-4553-ac65-820559bf8e94", dicom_level=2
    )

    print(
        f"Original image: {original_image.shape[0]}x{original_image.shape[1]} pixels ({original_image.shape[0] * original_image.shape[1]:,} total)"
    )

    downsample_factor = 8
    print(f"Downsampling by {downsample_factor}x for faster processing...")
    image = seg.downsample_image(original_image, downsample_factor)
    print(
        f"Working with: {image.shape[0]}x{image.shape[1]} pixels ({image.shape[0] * image.shape[1]:,} total)\n"
    )

    output_dir = seg.create_timestamped_output_dir("outputs")
    print(f"Output directory: {output_dir}\n")

    segmenters = [
        seg.OtsuSegmenter(),
        seg.LiSegmenter(),
        seg.YenSegmenter(),
        seg.MorphologicalSegmenter(disk_size=3),
        seg.EntropyMaskerSegmenter(),
    ]

    runner = seg.BenchmarkRunner()
    results = runner.run_multiple(segmenters, image)

    print("Unsupervised Segmentation Benchmarking")
    print("=" * 80)

    print("\nPerformance Metrics")
    print("-" * 80)
    for result in results:
        print(
            f"{result.method_name:30s} {result.execution_time:8.4f}s  "
            f"{result.seconds_per_pixel:.10f}s/px"
        )

    print()
    print("Segmentation Quality Metrics (Unsupervised)")
    print("-" * 80)
    print(
        f"{'Method':30s} {'Objects':>8s} {'Coverage':>10s} {'Mean Area':>12s} {'Compactness':>12s}"
    )
    print("-" * 80)

    for result in results:
        u = result.unsupervised_metrics
        print(
            f"{result.method_name:30s} "
            f"{u.num_objects:8d} "
            f"{u.coverage_ratio * 100:9.2f}% "
            f"{u.mean_area:12.2f} "
            f"{u.mean_compactness:12.4f}"
        )

    print()
    print("Detailed Statistics")
    print("-" * 80)

    for result in results:
        u = result.unsupervised_metrics
        print(f"\n{result.method_name}:")
        print(f"  Objects: {u.num_objects}")
        print(f"  Total area: {u.total_area:.2f}")
        print(
            f"  Area stats: mean={u.mean_area:.2f}, std={u.std_area:.2f}, median={u.median_area:.2f}"
        )
        print(f"  Area range: [{u.min_area:.2f}, {u.max_area:.2f}]")
        print(
            f"  Shape metrics: compactness={u.mean_compactness:.4f}, solidity={u.mean_solidity:.4f}"
        )
        print(f"  Coverage: {u.coverage_ratio * 100:.2f}%")

    print()
    print("Saving results...")

    seg.save_thumbnail(image, output_dir / "original_thumbnail.png")
    print(f"  Saved: original_thumbnail.png")

    print(
        f"  Saving full resolution annotations (original size: {original_image.shape[0]}x{original_image.shape[1]})..."
    )
    for result in results:
        seg.save_geojson(
            result.geojson,
            output_dir / f"{result.method_name}_fullres.geojson",
            scale_factor=downsample_factor,
        )
        print(f"    - {result.method_name}_fullres.geojson")

    print(f"  Saving downsampled annotations ({image.shape[0]}x{image.shape[1]})...")
    for result in results:
        seg.save_geojson(
            result.geojson, output_dir / f"{result.method_name}_downsampled.geojson"
        )
        print(f"    - {result.method_name}_downsampled.geojson")

    print(f"  Saving heatmap thumbnails...")
    for result in results:
        seg.save_heatmap_thumbnail(
            image,
            result.geojson,
            output_dir / f"{result.method_name}_heatmap.png",
            max_size=1024,
            alpha=0.4,
        )
        print(f"    - {result.method_name}_heatmap.png")

    seg.export_results_csv(results, output_dir / "results.csv")
    seg.export_results_json(results, output_dir / "results.json")
    print(f"  - results.csv")
    print(f"  - results.json")

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
