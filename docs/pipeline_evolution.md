# Computer Vision Pipeline for Root Length Prediction

## Competition Results

**Public Leaderboard:** 9.704% sMAPE  
**Private Leaderboard:** 10.687% sMAPE

## Project Overview

Pipeline for predicting primary root lengths from petri dish images, combining semantic segmentation (U-Net), instance segmentation, and root system architecture (RSA) extraction. Takes raw images from the NPEC Hades phenotyping platform and outputs per-plant root length estimates.

Started at 37.6% sMAPE with a complex multi-stage approach. Ended at 10.687% sMAPE after stripping most of it out and using what we actually know about the experimental setup instead of trying to build a fully general solution.

## Pipeline Evolution

### Initial Approach (Submission 1 & 2)

The baseline pipeline achieved 37.606-34.684% sMAPE on the public leaderboard.

**Architecture:**
- U-Net semantic segmentation for root detection
- Petri dish detection with adaptive circular masking  
- Multi-stage instance segmentation with merging and watershed splitting
- Bottom-up RSA extraction with multiple tip-finding strategies

**The Complexity Trap:**

Building the first version, I developed sophisticated algorithms to handle edge cases: fragmented roots, overlapping plants, irregular growth patterns. Each component worked well in isolation, but when combined, they created a fragile system with many failure modes.

**Instance Segmentation (v1 approach):**

The first version used a multi-stage pipeline to address fragmentation and overlap:

1. Morphological closing (kernel size 7) to reconnect fragments
2. Connected components with minimum area filtering (500 pixels)
3. Vertical alignment merging (20% overlap threshold, 300 pixel max gap)
4. Proximity-based merging (5000 pixel threshold, 400 pixel max distance)
5. Watershed-based splitting of oversized components (1.8x average area)

**Root Length Extraction (bottom-up approach):**

The bottom-up method attempted to identify primary roots through:

1. Skeletonization of individual plant masks
2. Endpoint and branch point detection
3. Primary tip identification using three strategies:
   - **Strategy A:** Deepest endpoint (maximum y-coordinate)
   - **Strategy B:** Depth + vertical alignment with origin
   - **Strategy C:** Depth + alignment + thickness weighting
4. Graph-based tracing from primary tip upward to origin
5. Branch pruning based on minimum length (5-10 pixels)

**Terminology note:** Strategies A/B/C here refer to different methods for identifying the primary root tip within the bottom-up tracing framework. None of them survive into the final pipeline.

**Limitations and Why They Failed:**

**Petri Dish Detection Issues:**
The circular detection worked about 90% of the time, but that 10% failure rate was devastating. When it failed, everything downstream failed. Even when it worked, coordinate transformations introduced small errors that accumulated through the pipeline.

**Segmentation Complexity:**
The merging and splitting heuristics were a maintenance nightmare. Each of six parameters affected the others in non-obvious ways. Tuning one parameter to fix one image would break another. The watershed splitting occasionally divided single root systems into unnatural pieces.

**RSA Extraction Ambiguity:**
The bottom-up approach required determining which endpoint was the "primary tip." This proved ambiguous:
- Strategy A (deepest) failed when lateral roots extended further down
- Strategy B (depth + alignment) failed on curved roots  
- Strategy C (depth + alignment + thickness) helped but added more parameters

The fundamental problem: trying to extract semantic information (which structure is "primary") from purely geometric features.

**Error Propagation:**
Each stage introduced errors that compounded through the pipeline:
- Petri dish detection errors shifted all downstream coordinates
- Shifted coordinates caused incorrect fragment merging in segmentation
- Incorrect merging produced wrong masks for RSA extraction
- RSA extraction then tried to find primary roots in corrupted data

These weren't independent — they cascaded. A single petri dish detection failure could produce completely wrong length estimates for all five plants in an image.

### Final Approach (Submission 7)

The improved pipeline achieved 9.704% public and 10.687% private sMAPE through fundamental architectural changes.

**Architecture:**
- U-Net semantic segmentation (unchanged)
- Fixed spatial ROI detection using predetermined plant positions
- Contour-based instance segmentation with spatial assignment
- RSA extraction via skeletonization and geodesic distance along the skeleton graph

**The Paradigm Shift:**

The breakthrough came from asking: "What if I stop trying to be clever and just use what I know?"

I knew the experimental setup. The NPEC phenotyping platform uses standardized petri dishes with consistent plant spacing. Why was I discarding this information and trying to rediscover it in every image?

This realization led to the first major change: **fixed spatial configuration**. Instead of detecting the petri dish, I used predetermined plant positions determined once via calibrate_plants.py, then hardcoded in config.py.

Specialization isn't a weakness when you have domain knowledge. Medical imaging pipelines are specialized for CT scanners. This pipeline is specialized for controlled phenotyping setups.

**Key Improvements:**

**1. Fixed Spatial Configuration**

**Implementation:**
   - Plant positions determined ONCE using calibrate_plants.py (automatic shoot detection)
   - Values hardcoded in config.py: first plant at x=1168, spacing 499 pixels, ROI width 224 pixels
   - calibrate_plants.py is a ONE-TIME configuration utility, not part of the runtime pipeline
   - run_pipeline.py reads these fixed values from config.py

**Why This Worked:**
The U-Net already provided excellent root/background segmentation. The problem wasn't detection quality - it was assignment ambiguity. By dividing the image into fixed spatial regions, I converted an ambiguous assignment problem into a trivial spatial lookup.

Each detected root structure simply asks: "which plant's region am I mostly in?"

**2. Simplified Instance Segmentation**

Contour detection replaced the entire multi-stage pipeline:
   - Find all contours in the binary mask using cv2.findContours
   - For each contour, calculate overlap with each plant's search window
   - Assign to plant with maximum overlap
   - No morphological merging, no watershed splitting, no complex heuristics

**Why This Worked:**
The complex v1 pipeline was solving problems that didn't exist in this dataset:
- Fragmentation? U-Net output was already continuous after basic morphological operations
- Overlap between plants? They were spatially separated by experimental design
- Oversized components? Not an issue when measuring individual pre-positioned plants

Contour detection also provided cleaner boundaries than connected components, naturally following root morphology without pixelation artifacts.

**3. RSA Extraction via Geodesic Distance**

The bottom-up tracing with semantic root classification was replaced with a simplified RSA extraction approach: skeletonization of the root mask followed by geodesic path tracing to identify and measure the primary root.

**Why Bottom-Up Failed:**
The fundamental issue was requiring semantic understanding of root architecture. It needed to know which branch was "primary" and which were "lateral." This information isn't reliably encoded in geometry alone. A thick lateral root growing downward can look exactly like a primary root.

**Why Geodesic Distance Succeeded:**
The RSA extraction works by first reducing the root mask to its topological skeleton (single-pixel centerline), then performing a weighted graph traversal from the topmost point (root origin) to the bottommost point (deepest extent). This traces the primary root path through the skeleton graph without requiring explicit branch classification.

**The Algorithm:**
```python
def calculate_root_length(mask):
    # 1. Preprocessing
    binary = (mask > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Extract largest component (the main plant structure)
    labeled, num = label(binary)
    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest_label = np.argmax(sizes) + 1
    largest = (labeled == largest_label).astype(np.uint8)
    
    # 3. Skeletonize to single-pixel centerline (RSA topology)
    skeleton = skeletonize(largest > 0).astype(np.uint8)
    
    # 4. Find topmost and bottommost skeleton points
    y_coords, x_coords = np.where(skeleton)
    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)
    start = (y_coords[top_idx], x_coords[top_idx])
    end = (y_coords[bottom_idx], x_coords[bottom_idx])
    
    # 5. Weighted graph traversal to find geodesic distance along skeleton
    # Traces the unique path on the tree-shaped skeleton
    # NOT the sum of all skeleton pixels
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([(start, 0.0)])
    visited[start] = True
    
    while queue:
        (y, x), dist = queue.popleft()
        
        if (y, x) == end:
            return int(round(dist))
        
        # 8-connectivity with distance weighting
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                       (0,1), (1,-1), (1,0), (1,1)]:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    # Diagonal: √2 ≈ 1.414, Orthogonal: 1.0
                    step_dist = 1.414 if (dy != 0 and dx != 0) else 1.0
                    queue.append(((ny, nx), dist + step_dist))
    
    # Fallback: Euclidean distance if path not found
    return int(np.linalg.norm(np.array(end) - np.array(start)))
```

**Critical Implementation Detail:** The algorithm does NOT sum all skeleton pixels (which would give total root system length including all laterals). Instead, it performs a weighted graph traversal along the skeleton from the topmost point (origin) to the bottommost point (deepest extent), accumulating distance with 8-connectivity weighting (1.0 for cardinal steps, √2 for diagonal steps).

**Why This Approximates Primary Root Length:**
Root skeletons are tree-shaped structures — there are no loops, so there is exactly one path between any two points. The traversal from topmost to bottommost simply traces that unique path and accumulates its length. Lateral branches don't contribute to this path because they branch off without reconnecting. The algorithm essentially traces the "spine" of the root system.

**Note on Algorithm Choice:** The implementation uses a FIFO queue (BFS-style) with non-uniform edge weights. On a tree graph this produces the correct path (since only one path exists), though the distance accumulation order is not strictly optimal for weighted graphs. In practice, the distance error from traversal order is negligible on skeleton graphs.

**Known Limitation:** 
If a lateral root extends significantly deeper than the primary tip, the bottommost point might be on that lateral, causing the traversal to include it. A more sophisticated approach could use thickness-weighted path finding or explicit primary/lateral classification, but these additions would reintroduce the complexity that caused problems in the initial pipeline.

**Performance Comparison:**

| Metric | Initial Approach | Final Approach | Improvement |
|--------|-----------------|----------------|-------------|
| Public sMAPE | 34.7-37.6% | 9.7% | 72-74% reduction |
| Private sMAPE | Unknown | 10.7% | - |
| Processing time | 90-120 seconds | ~60 seconds | 33-50% faster |

## Lessons Learned

### On Complexity vs. Simplicity

The most important lesson: **complexity is not sophistication**. I started assuming a complex problem requires a complex solution. The initial pipeline had multiple stages with multiple parameters, creating a combinatorial configuration space.

The final pipeline is simpler but not simplistic. It's **specialized**. Every component does one thing well:
- U-Net handles pixel-level root detection
- Fixed ROIs handle spatial assignment
- Geodesic distance handles length measurement

The reduction in complexity was about removing unnecessary flexibility, not functionality.

### On Error Propagation

One of the most valuable realizations was understanding error propagation through pipelines.

**Initial Pipeline Error Cascade:**
- Petri dish detection error → incorrect coordinate transformation
- Incorrect coordinates → plants assigned to wrong regions
- Wrong plant regions → incorrect mask extraction
- Incorrect masks → garbage input to RSA extraction
- RSA extraction tried to find primary roots in noise

**Final Pipeline Error Isolation:**
- Fixed ROIs → no coordinate error possible
- Spatial assignment → binary decision (overlap or not)
- Geodesic distance → operates on whatever mask is present

Errors localize rather than propagate. A poor U-Net prediction produces a poor mask, which produces a poor length estimate, but it doesn't cascade through coordinate transformations.

### On Validation Methodology

The discipline of Kaggle competitions is objective feedback. You can't rationalize away poor performance.

The improvement from 37% to 10.7% had nothing to do with parameter tuning. The biggest gains came from removing petri dish detection, complex merging logic, and semantic root classification — none of which were needed for this dataset. When validation metrics are poor, the problem is usually the approach, not the parameters.

### On Domain Knowledge vs. Machine Learning

Machine learning is a tool, not a complete solution. The U-Net does what ML does best: learn complex mappings from data (pixels → root masks). But the U-Net doesn't know about plant spacing, experimental setups, or measurement conventions.

The initial approach tried to make the pipeline "learn" these things through algorithmic complexity. The final approach explicitly encodes them as domain knowledge.

### On Development Process

**What Didn't Work:**
- Small parameter adjustments hoping for big improvements
- Adding more sophisticated algorithms to fix problems caused by existing algorithms
- Assuming better U-Net training would solve downstream issues

**What Worked:**
- Visualizing failure modes (looking at where predictions were wrong)
- Removing components to understand their contribution
- Testing radical simplifications even when they seemed too simple

The breakthrough came when I deleted the petri dish detection entirely and hardcoded positions. Iterate by subtraction as much as by addition.

## Development Timeline

**Week 1-2 (Initial Development):**
- Built instance segmentation with a full merging/splitting pipeline
- Developed bottom-up RSA extraction
- Integrated components with petri dish detection
- First submission: 37.6% sMAPE

**Week 3 (Parameter Optimization):**
- Tuned morphological kernel sizes, merging thresholds, watershed parameters
- Second submission: 34.7% sMAPE
- Marginal improvement suggested fundamental issues, not parameter problems

**Week 4 (Analysis Phase):**
- Generated visualizations for all test images
- Identified petri dish detection as major error source
- Noticed merge/split logic produced inconsistent results
- Visualizations showed plants shifted from expected positions due to circle detection variance

**Week 5 (Radical Simplification):**
- Removed petri dish detection, hardcoded plant positions
- Replaced multi-stage segmentation with contour assignment
- Third submission: 15.2% sMAPE

**Week 6 (Length Calculation Revision):**
- Analyzed remaining errors (localized to length calculation)
- Realized bottom-up tracing was failing on ambiguous roots
- Implemented geodesic distance as simpler alternative
- Fourth submission: 11.3% sMAPE

**Week 7 (Final Refinement):**
- Fine-tuned morphological operations
- Adjusted distance calculations
- Verified generalization on validation data
- Final submission: 10.7% private sMAPE

Improvement came in discrete jumps, not gradual refinement:
- Week 1-3: Stuck around 35-37% despite extensive tuning
- Week 5: Drop to 15% from architectural change
- Week 6: Drop to 11% from algorithm change
- Week 7: Final to 10.7% from refinement

## Pipeline Implementation

### System Requirements

**Python Dependencies:**
- TensorFlow 2.13+
- OpenCV (cv2)
- NumPy
- SciPy (ndimage, label)
- scikit-image (skeletonize)
- Pandas
- Matplotlib (for calibrate_plants.py visualization only)

**Hardware:**
- GPU recommended for U-Net inference
- 8GB+ RAM for full-resolution images

### Directory Structure

```
segmentation/
├── config.py              # Configuration parameters
├── run_pipeline.py        # Main pipeline execution
├── segment_plants.py      # Instance segmentation module
├── inference.py           # U-Net inference wrapper
└── calibrate_plants.py    # ONE-TIME configuration utility
```

### Configuration

**Spatial Parameters (in config.py):**
```python
DATA_DIR = Path('Kaggle')                                   # Input images directory (relative)
MODEL_PATH = Path('unet_root_segmentation_256px.h5')        # Model file
PLANT_START = 1168          # First plant x-position (pixels)
PLANT_STEP = 499            # Spacing between plants (pixels)
PLANT_ROI_WIDTH = 224       # Half-width of search window (pixels)
NUM_PLANTS = 5              # Plants per image
```

These values were determined by running calibrate_plants.py once on a sample image. They are fixed for the NPEC experimental setup.

### Running the Pipeline

**One-time setup (optional, values are already configured):**
```bash
uv run python calibrate_plants.py
```

**Run the pipeline:**
```bash
uv run python run_pipeline.py
```

### Pipeline Stages

**Stage 1: U-Net Semantic Segmentation**
- Loads all images from `DATA_DIR`
- Extracts overlapping 256×256 patches with 64px overlap
- Runs batch inference through the trained U-Net
- Stitches predictions and applies threshold (0.5) + morphological opening
- Outputs binary root masks

**Stage 2: Instance Segmentation**
- Finds contours in the binary mask
- Assigns each contour to the nearest plant based on spatial overlap with fixed ROIs
- Decomposes each plant mask into individual root structures using connected component analysis
- Exports individual plant and root masks

**Stage 3: RSA Extraction and Length Calculation**
- For each plant mask: binarize → extract largest component → skeletonize
- Identify topmost (origin) and bottommost (deepest extent) skeleton points
- Weighted graph traversal along the skeleton to compute geodesic distance
- Output: primary root length in pixels

### Validation and Error Handling

The pipeline includes:
1. Configuration validation before execution (paths, parameter ranges)
2. Image loading verification with shape reporting
3. Plant detection rate monitoring (warns if < expected)
4. Zero prediction tracking (identifies non-germinated/missing plants)
5. Statistical summary (mean length, detection rate)
6. Visual inspection outputs

These validation steps were added after debugging failures in earlier iterations. Configuration validation catches issues in milliseconds instead of minutes into processing.

## Performance Considerations

**Processing Time:**
- U-Net inference: 2-3 seconds per image (GPU) or 15-20 seconds (CPU)
- Instance segmentation: <1 second per image
- Length calculation: <1 second per image
- Total pipeline: ~60 seconds for 19 images (GPU)

**Memory Usage:**
- Peak: ~4GB during U-Net inference
- Intermediate masks: ~50MB per image
- Final output: ~200MB total

## Submission Format

The pipeline generates `root_lengths.csv`:

```csv
Plant ID,Length (px)
test_image_01_plant_1,641
test_image_01_plant_2,1159
test_image_01_plant_3,0
test_image_01_plant_4,1596
test_image_01_plant_5,1623
...
```

Where:
- Plant ID: `{image_stem}_plant_{plant_number}`
- Length (px): Integer pixel length of primary root
- Zero values indicate non-detected or non-germinated plants

## Future Improvements

Potential enhancements to improve prediction accuracy:

**1. Thickness-Weighted Geodesic Distance**
- Weight the traversal path cost by root thickness at each pixel
- Primary roots are typically thicker than laterals
- This would help disambiguate primary vs. lateral when both extend deep

**2. Curvature Detection**
- Current approach assumes relatively vertical root growth
- Curved roots might have bottommost points off the main axis
- Add orientation detection to trace curved primary roots

**3. Lateral Root Filtering**
- Explicit detection and removal of lateral branches before length calculation
- Use branch point analysis from the bottom-up approach, but more robustly

**4. Error Analysis**
- Analyze which images have highest errors
- Look for patterns: curved roots? Heavy branching? Poor segmentation?
- Target specific morphological patterns

## References

- U-Net training and semantic segmentation: `segmentation/inference.py`
- Instance segmentation approaches: `segmentation/segment_plants.py`
- Root System Architecture extraction: see Pipeline Evolution section above
- Kaggle competition data and evaluation metrics

## License

This pipeline was developed as part of a computer vision course at Breda University of Applied Sciences. See the repository [LICENSE](../LICENSE) for terms.

## Code Clarifications

### calibrate_plants.py Role

calibrate_plants.py is a **configuration utility**, not part of the runtime pipeline. It runs ONCE to detect plant positions and writes values to config.py. The runtime pipeline (run_pipeline.py) reads these hardcoded values.

**Setup flow:**
1. Run calibrate_plants.py on first image → detects positions → writes to config.py
2. Run run_pipeline.py → reads config.py → processes all images

### Strategy Terminology

**"Strategy" has two meanings in this project:**

**In the v1 bottom-up approach:**
- Strategy A/B/C = different methods for finding the primary root tip within bottom-up tracing
- These are NOT used in the final pipeline

**In final pipeline:**
- No "strategies" - just one method: geodesic distance from top to bottom

### Graph Traversal vs. Skeleton Sum

**The code does NOT sum all skeleton pixels.** 

The traversal algorithm finds the **unique path through the tree-shaped skeleton graph** from topmost to bottommost point. This is not the same as total skeleton length.

Example:
```
Skeleton has 500 pixels total (including laterals)
Traversal path from top to bottom: 320 pixels
Output: 320 (not 500)
```