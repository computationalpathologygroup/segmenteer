from pathlib import Path

import segmenteer as seg


def main():
    path = Path("CMU-1-Small-Region.svs")

    output_dir = seg.create_timestamped_output_dir("outputs")
    print(f"Output directory: {output_dir}\n")

    segmenters = [
        seg.OtsuSegmenter(),
        # seg.EntropyMaskerSegmenter(),
        # seg.GrandQCSegmenter(confidence_threshold=0.5, min_area=10),
        # seg.HESTSegmenter(mpp=1.0, confidence_threshold=0.5, min_area=10),
        # seg.CPGSegmenter(docker_image="cpg-tissuemasker:latest", min_area=10),
    ]

    runner = seg.BenchmarkRunner()
    results = runner.run_multiple(segmenters, path)

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

    seg.save_thumbnail(path, output_dir / "original_thumbnail.png")
    print(f"  Saved: original_thumbnail.png")

    print(f"  Saving annotations...")
    for result in results:
        seg.save_geojson(
            result.geojson, output_dir / f"{result.method_name}.geojson"
        )
        print(f"    - {result.method_name}.geojson")

    print(f"  Saving heatmap thumbnails...")
    for result in results:
        seg.save_heatmap_thumbnail(
            path,
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
