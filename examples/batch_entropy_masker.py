import segmenteer as seg
from pathlib import Path


def process_single_slide(slide_path: Path, output_base_dir: Path, pyramid_level: int = 4, downsample_factor: int = 4):
    slide_name = slide_path.stem
    output_dir = output_base_dir / slide_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing: {slide_name}")
    print(f"{'='*80}")
    
    fullres_geojson_path = output_dir / "entropy_masker_fullres.geojson"
    downsampled_geojson_path = output_dir / "entropy_masker_downsampled.geojson"
    thumbnail_path = output_dir / "original_thumbnail.png"
    heatmap_path = output_dir / "entropy_masker_heatmap.png"
    
    if all(p.exists() for p in [fullres_geojson_path, downsampled_geojson_path, thumbnail_path, heatmap_path]):
        print(f"Skipping {slide_name} - all outputs already exist")
        return True
    
    try:
        original_image = seg.load_image(path=slide_path, level=pyramid_level)
        
        print(f"Loaded pyramid level {pyramid_level}: {original_image.shape[0]}x{original_image.shape[1]} pixels")
        print(f"Downsampling by {downsample_factor}x for processing...")
        
        image = seg.downsample_image(original_image, downsample_factor)
        print(f"Working with: {image.shape[0]}x{image.shape[1]} pixels")
        
        total_scale_factor = downsample_factor * (2 ** pyramid_level)
        
        segmenter = seg.EntropyMaskerSegmenter()
        runner = seg.BenchmarkRunner()
        results = runner.run_multiple([segmenter], image)
        result = results[0]
        
        print(f"Execution time: {result.execution_time:.4f}s")
        print(f"Objects detected: {result.unsupervised_metrics.num_objects}")
        print(f"Coverage: {result.unsupervised_metrics.coverage_ratio * 100:.2f}%")
        
        print("Saving outputs...")
        seg.save_thumbnail(image, thumbnail_path)
        print(f"  ✓ {thumbnail_path.name}")
        
        total_scale_factor = downsample_factor * (2 ** pyramid_level)
        seg.save_geojson(
            result.geojson,
            fullres_geojson_path,
            scale_factor=total_scale_factor,
        )
        print(f"  ✓ {fullres_geojson_path.name}")
        
        seg.save_geojson(result.geojson, downsampled_geojson_path)
        print(f"  ✓ {downsampled_geojson_path.name}")
        
        seg.save_heatmap_thumbnail(
            image,
            result.geojson,
            heatmap_path,
            max_size=1024,
            alpha=0.4,
        )
        print(f"  ✓ {heatmap_path.name}")
        
        print(f"✓ Successfully processed {slide_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {slide_name}: {str(e)}")
        return False


def main():
    input_dir = Path("/Users/agatapolejowska/histopathobiome-s/data/imgs")
    output_base_dir = Path("outputs_entropy_masker")
    pyramid_level = 2
    downsample_factor = 8
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    mrxs_files = sorted(list(input_dir.glob("*.mrxs")))
    
    if not mrxs_files:
        print(f"No .mrxs files found in {input_dir}")
        return
    
    print(f"Found {len(mrxs_files)} MRXS files to process")
    print(f"Output directory: {output_base_dir.absolute()}")
    print(f"Pyramid level: {pyramid_level} (lower resolution to save memory)")
    print(f"Additional downsample factor: {downsample_factor}x")
    print(f"Total scale factor to full resolution: {downsample_factor * (2 ** pyramid_level)}x")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, slide_path in enumerate(mrxs_files, 1):
        print(f"\n[{i}/{len(mrxs_files)}] Processing {slide_path.name}...")
        
        success = process_single_slide(slide_path, output_base_dir, pyramid_level, downsample_factor)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total slides: {len(mrxs_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_base_dir.absolute()}")


if __name__ == "__main__":
    main()
