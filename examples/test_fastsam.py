import segmenteer as seg


def main():
    original_image = seg.load_image(
        path="data/001_M2.tiff", level=0
    )

    print(f"Original image: {original_image.shape[0]}x{original_image.shape[1]} pixels")

    downsample_factor = 2
    print(f"Downsampling by {downsample_factor}x for faster processing...")
    image = seg.downsample_image(original_image, downsample_factor)
    print(f"Working with: {image.shape[0]}x{image.shape[1]} pixels\n")

    output_dir = seg.create_timestamped_output_dir("outputs")
    print(f"Output directory: {output_dir}\n")

    segmenter = seg.FastSAMSegmenter(
        model_name="FastSAM-x.pt",
        text_prompt="human tissue",
        conf=0.75,
        iou=0.9,
        min_area=10,
    )

    runner = seg.BenchmarkRunner()
    results = runner.run_multiple([segmenter], image)

    print("\nFastSAM Segmentation Results")
    print("=" * 80)

    result = results[0]
    print(f"Execution time: {result.execution_time:.4f}s")
    print(f"Objects detected: {result.unsupervised_metrics.num_objects}")
    print(f"Coverage: {result.unsupervised_metrics.coverage_ratio * 100:.2f}%")
    print(f"Mean area: {result.unsupervised_metrics.mean_area:.2f}")
    print(f"Mean compactness: {result.unsupervised_metrics.mean_compactness:.4f}")

    print("\nSaving results...")
    
    seg.save_thumbnail(image, output_dir / "original_thumbnail.png")
    print("  ✓ original_thumbnail.png")

    seg.save_geojson(
        result.geojson,
        output_dir / f"{result.method_name}_fullres.geojson",
        scale_factor=downsample_factor,
    )
    print(f"  ✓ {result.method_name}_fullres.geojson")

    seg.save_geojson(
        result.geojson,
        output_dir / f"{result.method_name}_downsampled.geojson"
    )
    print(f"  ✓ {result.method_name}_downsampled.geojson")

    seg.save_heatmap_thumbnail(
        image,
        result.geojson,
        output_dir / f"{result.method_name}_heatmap.png",
        max_size=1024,
        alpha=0.4,
    )
    print(f"  ✓ {result.method_name}_heatmap.png")

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
