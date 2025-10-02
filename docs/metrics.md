# Segmentation Metrics Explained

## Unsupervised Metrics
These metrics can be computed without ground truth annotations. They characterize the intrinsic properties of the segmentation.

### Count Metrics
- **num_objects**: Total number of segmented regions/objects detected
  - *What it means*: How many distinct objects your segmentation algorithm found
  - *Good values*: Depends on application - more isn't always better

### Area Metrics
- **total_area**: Sum of all segmented object areas (in pixels²)
  - *What it means*: Total tissue/object coverage in the image
  
- **mean_area**: Average area per object
  - *What it means*: Typical size of detected objects
  - *Interpretation*: Low mean_area = many small objects; high mean_area = fewer large objects

- **std_area**: Standard deviation of object areas
  - *What it means*: Variability in object sizes
  - *High value*: Very heterogeneous objects (mix of small and large)
  - *Low value*: Similar-sized objects

- **median_area**: Middle value of object areas
  - *What it means*: More robust measure of typical size than mean (less affected by outliers)
  
- **min_area** / **max_area**: Smallest and largest object detected
  - *What it means*: Range of object sizes in your segmentation

### Shape Quality Metrics
- **mean_compactness**: Average circularity (range: 0-1, perfect circle = 1)
  - *Formula*: 4π × area / perimeter²
  - *What it means*: How circular/round your objects are
  - *High value (>0.8)*: Round, compact objects
  - *Low value (<0.4)*: Irregular, elongated, or fragmented objects
  - *Use case*: Quality check - fragmented segmentations have low compactness

- **mean_solidity**: Average convexity (range: 0-1, convex hull = 1)
  - *Formula*: object_area / convex_hull_area
  - *What it means*: How "filled in" objects are (measures concavities/holes)
  - *High value (>0.9)*: Smooth, convex objects without indentations
  - *Low value (<0.7)*: Objects with deep concavities or holes
  - *Use case*: Detects over-segmentation or rough boundaries

### Coverage Metrics
- **coverage_ratio**: Fraction of image covered by segmented objects (range: 0-1)
  - *Formula*: total_segmented_area / total_image_area
  - *What it means*: Percentage of image classified as foreground
  - *Example*: 0.35 = 35% of image is segmented tissue
  - *Use case*: QC check - too high (>0.95) or too low (<0.05) may indicate issues

---

## Supervised Metrics
These metrics require ground truth annotations for comparison.

### Overlap Metrics
- **Dice Coefficient** (range: 0-1, perfect = 1)
  - *Formula*: 2 × |A ∩ B| / (|A| + |B|)
  - *What it means*: How much predicted and ground truth overlap
  - *Good value*: >0.8 (excellent), 0.6-0.8 (good), <0.5 (poor)
  - *Properties*: Weights false positives and false negatives equally

- **IoU (Jaccard Index)** (range: 0-1, perfect = 1)
  - *Formula*: |A ∩ B| / |A ∪ B|
  - *What it means*: Intersection over union of predictions and ground truth
  - *Good value*: >0.7 (excellent), 0.5-0.7 (good), <0.5 (poor)
  - *Properties*: More strict than Dice, penalizes both errors strongly

### Precision & Recall
- **Precision** (range: 0-1, perfect = 1)
  - *Formula*: True Positives / (True Positives + False Positives)
  - *What it means*: Of all predicted tissue, what fraction is actually tissue?
  - *High precision*: Few false positives (clean segmentation)
  - *Low precision*: Over-segmentation (predicting too much)

- **Recall** (Sensitivity) (range: 0-1, perfect = 1)
  - *Formula*: True Positives / (True Positives + False Negatives)
  - *What it means*: Of all actual tissue, what fraction did you detect?
  - *High recall*: Few false negatives (captures most tissue)
  - *Low recall*: Under-segmentation (missing tissue)

### Segmentation Error Metrics
- **over_segmentation_rate**: Ratio of false positives to actual positives
  - *What it means*: How much extra tissue did you segment?
  - *Value > 0.2*: Significant over-segmentation
  - *Use case*: Identify methods that segment too aggressively

- **under_segmentation_rate**: Ratio of false negatives to actual positives
  - *What it means*: How much tissue did you miss?
  - *Value > 0.2*: Significant under-segmentation
  - *Use case*: Identify methods that are too conservative

### Boundary Accuracy
- **Hausdorff Distance** (in pixels, lower is better)
  - *What it means*: Maximum distance between predicted and true boundaries
  - *Low value (<5 pixels)*: Accurate boundary delineation
  - *High value (>20 pixels)*: Poor boundary matching
  - *Use case*: Sensitive to outliers, useful for detecting large boundary errors

---

## Interpretation Guide

### For Quality Assessment (Unsupervised)
```
Good Segmentation:
- Compactness: 0.6-0.9 (compact, smooth objects)
- Solidity: 0.8-0.95 (filled objects without holes)
- Coverage: 0.2-0.6 (depends on tissue type)
- std_area / mean_area < 5 (reasonable size variation)

Poor Segmentation:
- Compactness < 0.3 (fragmented)
- Solidity < 0.6 (too many holes)
- Coverage > 0.9 or < 0.05 (over/under-segmentation)
```

### For Accuracy Assessment (Supervised)
```
Excellent: Dice > 0.85, IoU > 0.75
Good:      Dice 0.7-0.85, IoU 0.5-0.75
Fair:      Dice 0.5-0.7, IoU 0.3-0.5
Poor:      Dice < 0.5, IoU < 0.3

Balanced:  |Precision - Recall| < 0.1
Over-seg:  Precision < Recall - 0.15
Under-seg: Recall < Precision - 0.15
```
