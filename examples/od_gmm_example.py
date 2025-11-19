import segmenteer as seg
from pathlib import Path


def process_slide_with_od_gmm(
    slide_path: Path,
    output_dir: Path,
    pyramid_level: int = 4,
    n_samples: int = 100000,
    n_components: int = 2,
):
    slide_name = slide_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing: {slide_name}")
    print("OD-GMM Slide-Level Tissue Segmentation")
    print(f"{'='*80}")

    fullres_geojson_path = output_dir / f"{slide_name}_od_gmm_fullres.geojson"
    downsampled_geojson_path = output_dir / f"{slide_name}_od_gmm_downsampled.geojson"
    thumbnail_path = output_dir / f"{slide_name}_thumbnail.png"
    heatmap_path = output_dir / f"{slide_name}_od_gmm_heatmap.png"

    print(f"Loading slide at pyramid level {pyramid_level}...")
    image = seg.load_image(path=slide_path, level=pyramid_level)
    print(f"Image dimensions: {image.shape[0]}x{image.shape[1]} pixels")

    segmenter = seg.ODGMMSlideSegmenter(
        n_samples=n_samples,
        n_components=n_components,
        min_area=10,
        morph_kernel_size=5,
    )

    print(f"Running OD-GMM with {n_samples} samples and {n_components} components...")
    runner = seg.BenchmarkRunner()
    results = runner.run_multiple([segmenter], image)
    result = results[0]

    print("\nResults:")
    print(f"  Execution time: {result.execution_time:.4f}s")
    print(f"  Objects detected: {result.unsupervised_metrics.num_objects}")
    print(f"  Coverage: {result.unsupervised_metrics.coverage_ratio * 100:.2f}%")

    print("\nSaving outputs...")
    seg.save_thumbnail(image, thumbnail_path)
    print(f"  ✓ {thumbnail_path.name}")

    scale_factor = 2 ** pyramid_level
    seg.save_geojson(
        result.geojson,
        fullres_geojson_path,
        scale_factor=scale_factor,
    )
    print(f"  ✓ {fullres_geojson_path.name}")

    seg.save_geojson(result.geojson, downsampled_geojson_path)
    print(f"  ✓ {downsampled_geojson_path.name}")

    seg.save_heatmap_thumbnail(
        image,
        result.geojson,
        heatmap_path,
    )
    print(f"  ✓ {heatmap_path.name}")

    print(f"\n{'='*80}")
    print(f"Completed: {slide_name}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    output_base_dir = Path(__file__).parent.parent / "outputs" / "od_gmm_example"

    slide_paths = [
        data_dir / "147_03_14708" / "147_03_14708-dicom",
        data_dir / "147_03_9683-4" / "147_03_9683-4-dicom",
        data_dir / "147_20_2096_A" / "147_20_2096_A-dicom",
    ]

    for slide_path in slide_paths:
        if slide_path.exists():
            process_slide_with_od_gmm(
                slide_path=slide_path,
                output_dir=output_base_dir,
                pyramid_level=4,
                n_samples=100000,
                n_components=2,
            )
        else:
            print(f"Warning: {slide_path} does not exist, skipping...")
