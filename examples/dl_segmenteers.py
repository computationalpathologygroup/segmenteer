import segmenteer as seg


def main():
    wsi_path = "data/ce1e4a10-d4e7-4524-97ac-9f88fe971778"
    
    metadata = seg.get_wsi_metadata(wsi_path)
    print(f"\nWSI Information:")
    print(f"  Dimensions: {metadata.width}x{metadata.height}")
    print(f"  Native MPP: {metadata.mpp}")
    print(f"  Pyramid levels: {metadata.num_levels}")
    
    output_dir = seg.create_timestamped_output_dir("outputs")
    print(f"\nOutput directory: {output_dir}\n")
    
    target_mpp = 4.0
    print(f"Loading image at target MPP: {target_mpp}")
    image, _ = seg.load_wsi_at_mpp(wsi_path, target_mpp=target_mpp, verbose=True)
    
    segmenters = [
        seg.GrandQCSegmenter(confidence_threshold=0.5, min_area=10),
        seg.HESTSegmenter(mpp=1.0, confidence_threshold=0.5, min_area=10),
    ]
    
    runner = seg.BenchmarkRunner()
    results = runner.run_multiple(segmenters, image)
    
    print("\n" + "=" * 80)
    print("Deep Learning Segmentation Benchmarking")
    print("=" * 80)
    
    print("\nPerformance Metrics")
    print("-" * 80)
    for result in results:
        print(
            f"{result.method_name:40s} {result.execution_time:8.4f}s  "
            f"{result.seconds_per_pixel:.10f}s/px"
        )
    
    print("\nSegmentation Quality Metrics")
    print("-" * 80)
    print(
        f"{'Method':40s} {'Objects':>8s} {'Coverage':>10s} "
        f"{'Mean Area':>12s} {'Compactness':>12s}"
    )
    print("-" * 80)
    
    for result in results:
        u = result.unsupervised_metrics
        print(
            f"{result.method_name:40s} "
            f"{u.num_objects:8d} "
            f"{u.coverage_ratio * 100:9.2f}% "
            f"{u.mean_area:12.2f} "
            f"{u.mean_compactness:12.4f}"
        )
    
    print("\nSaving results...")
    
    seg.save_thumbnail(image, output_dir / "original_thumbnail.png")
    print(f"  - original_thumbnail.png")
    
    scale_factor = seg.calculate_scale_factor_for_coordinates(target_mpp, metadata.mpp)
    
    for result in results:
        seg.save_geojson(
            result.geojson,
            output_dir / f"{result.method_name}_at_{target_mpp}mpp.geojson",
        )
        seg.save_geojson(
            result.geojson,
            output_dir / f"{result.method_name}_native_resolution.geojson",
            scale_factor=scale_factor,
        )
        print(f"  - {result.method_name}_at_{target_mpp}mpp.geojson")
        print(f"  - {result.method_name}_native_resolution.geojson")
    
    for result in results:
        seg.save_heatmap_thumbnail(
            image,
            result.geojson,
            output_dir / f"{result.method_name}_heatmap.png",
            max_size=1024,
            alpha=0.4,
        )
        print(f"  - {result.method_name}_heatmap.png")
    
    seg.export_results_csv(results, output_dir / "results.csv")
    seg.export_results_json(results, output_dir / "results.json")
    print(f"  - results.csv")
    print(f"  - results.json")
    
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()