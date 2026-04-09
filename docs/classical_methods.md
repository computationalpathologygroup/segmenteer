# Classical Methods

This document describes the classical (non-deep learning) segmentation methods supported in the segmenteer framework.

## Common Parameters

All classical methods inherit from `NumpySegmenter` and share the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mpp` | float | 10 | Microns-per-pixel resolution at which the WSI is analyzed |
| `min_area` | int | 10 | Minimum polygon area in pixels at the specified mpp resolution |
| `to_gray_func` | callable | `skimage.color.rgb2gray` | Function to convert RGB images to grayscale |

Most methods operate on grayscale images automatically (unless `APPLY_TO_GRAYSCALE=False` is set). RGB images are converted using the `to_gray_func` parameter.

## Methods

### Otsu (`OtsuSegmenter`)

Automatic global thresholding using Otsu's method. Computes an optimal threshold by maximizing the between-class variance of the image histogram.

**Parameters:**
- Inherits only common parameters

**How it works:**
1. Computes histogram of the grayscale image
2. Calculates optimal threshold that maximizes variance between background and foreground classes
3. Returns binary mask where pixels below threshold are foreground (tissue is assumed to be darker than background)

---

### Connected Components (`ConnectedComponentSegmenter`)

Extracts foreground regions by Otsu thresholding followed by connected component labeling and area filtering.

**Parameters:**
- `connectivity` : int (default: `2`)
  - Type of connectivity for component labeling
  - `1` for 4-connectivity (neighbors share edges)
  - `2` for 8-connectivity (neighbors share edges or corners)
- Inherits common parameters

**How it works:**
1. Applies Otsu thresholding to convert grayscale image to binary
2. Labels connected components using specified connectivity rule
3. Filters components: keeps only those with area ≥ `min_area` pixels
4. Combines filtered components into final foreground mask
5. Returns binary mask of all valid components

**Notes:** Useful for separating distinct tissue regions and removing noise fragments below the size threshold. 8-connectivity produces fewer, larger components; 4-connectivity produces more fragmented results.

---

### Edge-Based (`EdgeBasedSegmenter`)

Extracts foreground regions by detecting edges and identifying regions between them via connected component labeling.

**Parameters:**
- `method` : str (default: `"canny"`)
  - Edge detection algorithm: `"canny"` or `"sobel"`
- `sigma` : float (default: `1.0`)
  - Standard deviation for Gaussian blur (Canny only)
  - Controls edge smoothness and thickness
- `sobel_threshold` : float or None (default: `None`)
  - Manual threshold for Sobel magnitude (Sobel only)
  - If `None`, automatically applies Otsu thresholding to magnitude
- Inherits common parameters

**How it works:**
1. Detects edges using selected method (Canny or Sobel)
   - **Canny**: Gaussian blur, gradient computation, non-maximum suppression, hysteresis
   - **Sobel**: Computes x and y gradients, calculates magnitude
2. Inverts edge map (edges become 0, non-edges become foreground candidates)
3. Labels connected components in inverted edge space
4. Filters components by minimum area
5. Returns mask of valid regions between edges

**Notes:** Useful for extracting regions separated by distinct boundaries. Canny provides cleaner, thinner edges; Sobel edges are thicker but may better capture gradual transitions. Sigma parameter affects edge detection sensitivity.

---

### Li (`LiSegmenter`)

Automatic thresholding using Li's minimum cross entropy method.

**Parameters:**
- Inherits only common parameters

**How it works:**
1. Computes threshold minimizing cross entropy between original and thresholded images
2. Iterative algorithm finding optimal division between background and foreground
3. Returns binary mask where pixels above threshold are foreground

---

### Yen (`YenSegmenter`)

Automatic thresholding using Yen's maximum correlation criterion.

**Parameters:**
- Inherits only common parameters

**How it works:**
1. Computes correlation coefficient between original and thresholded images
2. Finds threshold maximizing this correlation
3. Returns binary mask where pixels above threshold are foreground

---

### Entropy Masker (`EntropyMaskerSegmenter`)

Local entropy-based foreground extraction specifically designed for histopathological images.

**Parameters:**
- `footprint` : numpy.ndarray or None (default: `skimage.morphology.disk(9)`)
  - Circular or custom structuring element for local entropy calculation
  - Default disk size of 9 pixels
- Inherits common parameters

**How it works:**
1. Computes local entropy for each pixel using provided footprint
2. Applies Otsu thresholding on the entropy map
3. Returns binary mask where entropy is above threshold

**Notes:** Uses `skimage.util.apply_parallel` for faster parallel processing via dask. Falls back to sequential processing if dask is unavailable.


---

### HSV Threshold (`HSVThresholdSegmenter`)

Color-space thresholding targeting H&E haematoxylin and eosin stained tissue.

**Parameters:**
- `lower` : array-like of 3 uint8 (default: `[90, 8, 103]`)
  - Lower bound of HSV range as [Hue, Saturation, Value]
  - Default targets the purple-to-pink range of haematoxylin stain
- `upper` : array-like of 3 uint8 (default: `[180, 255, 255]`)
  - Upper bound of HSV range as [Hue, Saturation, Value]
- Inherits common parameters

**How it works:**
1. Converts RGB image to HSV color space
2. Applies range mask using OpenCV `cv2.inRange`
3. Returns binary mask where pixels fall within the specified HSV bounds

**Notes:** Requires RGB input; does not apply grayscale conversion. Default bounds reject white/gray backgrounds (low saturation) and dark artifacts (low value).

**Dependencies:** OpenCV (opencv-python or opencv-python-headless)

---

### FESI and Improved FESI (`FESISegmenter`)

Foreground Extraction for Histopathological Whole-Slide Imaging (FESI) algorithms targeting H&E-stained tissue.

**Parameters:**
- `improved` : bool (default: `True`)
  - If `True`, uses Improved FESI algorithm
  - If `False`, uses original FESI algorithm
- Inherits common parameters

**How it works:**

*Original FESI:*
1. Applies Gaussian blur and thresholding on grayscale image
2. Computes distance transform to find tissue seed points
3. Uses flood-fill seeding and distance-based region growing
4. Connects regions within 500 pixels to form connected tissue mask

*Improved FESI:*
1. Converts image to LAB color space and sets lightness/red-green channels to maximum
2. Computes Laplacian edges of grayscale image
3. Multiplies edge map with saturation from HSV space
4. Thresholds and applies distance-based region growing as in original FESI
5. Combines approaches for more robust tissue extraction

**References:** 
- Original FESI: https://www.lfb.rwth-aachen.de/bibtexupload/pdf/BUG15fesi.pdf
- Improved FESI: https://arxiv.org/pdf/2006.06531.pdf

---

### HistomicsTK (`HistomicsTKSegmenter`)

Tissue detection using the HistomicsTK library.

**Parameters:**
- `mask_type` : str (default: `"saliency"`)
  - `"saliency"`: Uses saliency-based tissue detection (recommended)
  - `"simple"`: Uses simple magnitude threshold without saliency
- Inherits common parameters

**How it works:**
- Delegates to HistomicsTK's tissue detection functions
- Saliency method uses attention mechanisms to highlight tissue regions
- Simple method uses direct magnitude thresholding

**Dependencies:** histomicstk

---

### Morphological Segmenter (`MorphologicalSegmenter`)

Combines Otsu thresholding with morphological filtering (opening and closing).

**Parameters:**
- `disk_size` : int (default: `3`)
  - Radius of the disk structuring element for opening and closing operations
- Inherits common parameters

**How it works:**
1. Applies Otsu thresholding to convert grayscale image to binary
2. Applies binary opening (erosion followed by dilation) with disk structuring element
   - Removes small noise and thin protrusions
3. Applies binary closing (dilation followed by erosion) with same element
   - Fills small holes and connects nearby regions
4. Returns the cleaned binary mask

---

### Watershed Segmenter (`WatershedSegmenter`)

Marker-based watershed segmentation for tissue boundary delineation.

**Parameters:**
- `min_distance` : int (default: `10`)
  - Minimum distance between detected peaks (tissue markers) in pixels
- Inherits common parameters

**How it works:**
1. Applies Otsu thresholding on grayscale image
2. Computes distance transform of binary image (Euclidean)
3. Detects local maxima in distance map with `min_distance` spacing constraint
4. Creates markers from detected peaks
5. Applies watershed algorithm using negative distance as topography
6. Returns segmented regions

**Notes:** Useful for separating closely adjacent tissue regions.

---

### Background Subtractor MOG2 (`BackgroundSubtractorMOG2Segmenter`)

OpenCV's Mixture of Gaussians (version 2) background/foreground separation.

**Parameters:**
- `history` : int (default: `500`)
  - Number of frames used to build the background model
  - Larger values increase model stability but reduce responsiveness
- `var_threshold` : float (default: `16.0`)
  - Threshold on the squared Mahalanobis distance for pixel classification
  - Lower values classify more pixels as foreground
- `detect_shadows` : bool (default: `True`)
  - If `True`, detects and removes shadow pixels (marked as 127 in output)
  - If `False`, treats shadows as foreground
- Inherits common parameters

**How it works:**
1. Initializes background model by learning from a series of white background images
2. Learns a mixture of K Gaussians for each pixel location
3. Classifies each test image pixel based on fit to learned Gaussian models
4. Optionally identifies and removes shadow pixels using color distortion ratio
5. Returns binary foreground mask

**Dependencies:** OpenCV (opencv-python or opencv-python-headless)

**Notes:** Designed for sequential frames. Implementation initializes model using synthetic white background frames before application to test images.

---

### OD GMM Slide (`ODGMMSlideSegmenter`)

Tissue detection using Gaussian Mixture Modeling in optical density (OD) space.

**Parameters:**
- `n_components` : int (default: `2`)
  - Number of Gaussian components in the mixture model
  - Typically 2 for background/tissue separation
- `n_samples` : int (default: `100000`)
  - Number of random pixels sampled from the image for model fitting
  - Larger values increase robustness but slow computation
- `morph_kernel_size` : int (default: `5`)
  - Size of the square structuring element for morphological cleaning
  - If 0, skips morphological operations
- `epsilon` : float (default: `1e-6`)
  - Small value added during OD conversion to avoid log(0)
- Inherits common parameters

**How it works:**
1. Converts RGB image to optical density (OD) space: `OD = -log((RGB + epsilon) / 255)`
2. Samples random pixels from the image
3. Sums OD values across channels and fits Gaussian Mixture Model
4. Identifies background cluster as the one with lower mean OD (lighter appearance)
5. Computes posterior probability each pixel belongs to background component
6. Classifies tissue as pixels with background probability < 0.5
7. Optionally applies morphological opening and closing for smoothing
8. Filters out small regions below `min_area` threshold
9. Returns binary tissue mask

**Notes:** Requires RGB input; does not apply grayscale conversion. OD space normalization makes the method robust to stain variations in H&E images.

---

## Selection Guidelines

| Use Case | Recommended Method |
|----------|-------------------|
| Fast baseline | Otsu, Li, or Yen |
| Region separation by size | Connected Components |
| Boundary detection | Edge-Based |
| H&E stain-specific | HSV Threshold, FESI/Improved FESI, OD GMM |
| Robust to stain variation | Improved FESI, OD GMM Slide |
| Edge detection | Entropy Masker, Edge-Based |
| Morphology-based | Morphological, Watershed |
| Multi-frame sequences | Background Subtractor MOG2 |
| Library-based | HistomicsTK |
