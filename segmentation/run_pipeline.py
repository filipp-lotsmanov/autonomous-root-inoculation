import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, Tuple
import sys
from scipy.ndimage import label
from skimage.morphology import skeletonize
from collections import deque

# Import configuration
import config

# Import pipeline modules
from inference import generate_masks
from segment_plants import process_all


def print_header():
    """Print pipeline header."""
    config.print_config()


def load_images() -> Dict[str, np.ndarray]:
    """Load all images from data directory."""
    images = {}
    extensions = ['.png', '.jpg', '.tif', '.tiff']
    
    for ext in extensions:
        for img_path in sorted(config.DATA_DIR.glob(f'*{ext}')):
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[img_path.name] = image
                print(f" Loaded: {img_path.name} ({image.shape[1]}×{image.shape[0]})")
    
    if len(images) == 0:
        print(f" No images in {config.DATA_DIR}")
        sys.exit(1)
    
    print(f" Loaded {len(images)} images")
    return images


def unet_inference(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Stage 1: Generate semantic segmentation masks with U-Net."""
    print(f"Processing {len(images)} images...")
    print(f"  Overlap: {config.UNET_OVERLAP} pixels")
    print(f"  Threshold: {config.UNET_THRESHOLD}")
    print(f"  Batch size: {config.UNET_BATCH_SIZE}")
    
    masks = generate_masks(
        images=images,
        model_path=str(config.MODEL_PATH),
        overlap=config.UNET_OVERLAP,
        threshold=config.UNET_THRESHOLD,
        batch_size=config.UNET_BATCH_SIZE,
        morphology=True,
        verbose=True
    )
    
    # Save masks if requested
    if config.SAVE_UNET_MASKS:
        mask_dir = config.OUTPUT_DIR / 'unet_masks'
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        for name, mask in masks.items():
            stem = Path(name).stem
            mask_path = mask_dir / f"{stem}_unet_mask.png"
            cv2.imwrite(str(mask_path), mask)
        
        print(f" Saved masks to: {mask_dir}")
    
    print(f"Model part complete")
    return masks


def instance_segmentation(images: Dict[str, np.ndarray],
                                  masks: Dict[str, np.ndarray]) -> Dict:
    """Stage 2: Instance segmentation (separate plants and roots)."""
    
    print(f"  First plant position: {config.PLANT_START}")
    print(f"  Plant spacing: {config.PLANT_STEP}")
    print(f"  ROI width: {config.PLANT_ROI_WIDTH}")
    print(f"  Expected plants: {config.NUM_PLANTS}")
 
    results = process_all(
        images=images,
        masks=masks,
        output_base_dir=str(config.OUTPUT_DIR),
        start=config.PLANT_START,
        step=config.PLANT_STEP,
        num_plants=config.NUM_PLANTS,
        roi_width=config.PLANT_ROI_WIDTH,
        save_plants=config.SAVE_PLANTS,
        save_roots=config.SAVE_ROOTS,
        plant_format='mask',
        roots_full_size=config.ROOTS_FULL_SIZE,
        roots_validate=config.ROOTS_VALIDATE,
        visualize=config.SAVE_VISUALIZATIONS
    )
    
    print(f"Instance Segmentation complete")
    return results


def analyse_root(mask: Optional[np.ndarray]) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Measure one plant mask.

    Returns:
        (length_px, tip_xy) where length_px is the geodesic length of the primary
        root along its skeleton, and tip_xy is the (x, y) pixel position of the
        bottom-most skeleton point — the growing tip, which is the inoculation
        target. tip_xy is None when no root was segmented.
    """
    if mask is None or mask.max() == 0:
        return 0, None

    # Binarize
    binary = (mask > 127).astype(np.uint8)

    # Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get largest component
    labeled, num = label(binary)
    if num == 0:
        return 0, None

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest_label = np.argmax(sizes) + 1
    largest = (labeled == largest_label).astype(np.uint8)

    # Skeletonize
    skeleton = skeletonize(largest > 0).astype(np.uint8)

    if skeleton.sum() == 0:
        return 0, None

    # Find endpoints
    y_coords, x_coords = np.where(skeleton)
    if len(y_coords) == 0:
        return 0, None

    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)

    start = (y_coords[top_idx], x_coords[top_idx])
    end = (y_coords[bottom_idx], x_coords[bottom_idx])
    tip = (int(x_coords[bottom_idx]), int(y_coords[bottom_idx]))

    # BFS for geodesic distance
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([(start, 0.0)])
    visited[start] = True

    while queue:
        (y, x), dist = queue.popleft()

        if (y, x) == end:
            return int(round(dist)), tip

        # 8-connectivity
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    step_dist = 1.414 if (dy != 0 and dx != 0) else 1.0
                    queue.append(((ny, nx), dist + step_dist))

    # Fallback to Euclidean
    return int(np.linalg.norm(np.array(end) - np.array(start))), tip


def length_calculation(images_processed: Dict) -> pd.DataFrame:
    """Stage 3: Calculate primary root lengths and locate root tips."""

    plant_masks_dir = config.OUTPUT_DIR / 'individual_plants'

    if not plant_masks_dir.exists():
        print(f" Masks directory not found: {plant_masks_dir}")
        return None

    expected_plants = []
    for image_name in sorted(images_processed.keys()):
        image_stem = Path(image_name).stem
        for plant_num in range(1, config.NUM_PLANTS + 1):
            plant_id = f"{image_stem}_plant_{plant_num}"
            expected_plants.append((image_name, plant_id))

    print(f"Processing {len(expected_plants)} plant masks...")

    # Process all expected plants
    results = []
    tip_records = []
    for i, (image_name, plant_id) in enumerate(expected_plants, 1):
        mask_path = plant_masks_dir / f"{plant_id}.png"

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        length, tip = analyse_root(mask)

        results.append({'Plant ID': plant_id, 'Length (px)': length})
        tip_records.append({
            'Image': image_name,
            'Plant ID': plant_id,
            'Tip X (pixels)': tip[0] if tip is not None else 0,
            'Tip Y (pixels)': tip[1] if tip is not None else 0,
            'Length (px)': length,
            'Detected': tip is not None
        })

        if i % 10 == 0:
            print(f"  Processed {i}/{len(expected_plants)} plants...")

    df = pd.DataFrame(results)

    # Save CSV
    results_dir = config.OUTPUT_DIR / 'results'
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / 'submission7.csv'
    df.to_csv(csv_path, index=False)

    # Save root tip coordinates. This is the handoff to integration/ — the
    # column names and the 'Detected' flag are what SpatialTransformationEngine
    # and the inoculation orchestrator read.
    tips_df = pd.DataFrame(tip_records)
    tips_df.to_csv(config.ROOT_TIPS_CSV, index=False)
    detected = int(tips_df['Detected'].sum())
    print(f" Root tips located: {detected}/{len(tips_df)}")
    print(f" Saved tip coordinates to: {config.ROOT_TIPS_CSV}")

    return df


def print_summary(images, masks, results, df):
    """Print final summary."""
    print(f"  Images processed: {len(images)}")
    print(f"  Masks generated: {len(masks)}")
    
    if results:
        total_plants = sum(r['summary']['total_found'] for r in results.values())
        total_roots = sum(r['summary']['total_roots'] for r in results.values())
        print(f"  Plants detected: {total_plants}")
        print(f"  Roots segmented: {total_roots}")
    
    if df is not None:
        print(f"  Root lengths calculated: {len(df)}")
        print(f"  Mean length: {df['Length (px)'].mean():.1f} pixels")
    
    print(f"\nOutput Directory: {config.OUTPUT_DIR}/")
    print(f"  Structure:")
    if config.SAVE_UNET_MASKS:
        print(f"    unet_masks/           - U-Net predictions")
    if config.SAVE_VISUALIZATIONS:
        print(f"    visualizations/       - Segmentation visualizations")
    if config.SAVE_PLANTS:
        print(f"    individual_plants/    - Individual plant masks")
    if config.SAVE_ROOTS:
        print(f"    individual_roots/     - Individual root masks")
    print(f"    root_tips_pixels.csv  - Root tip coordinates (input to integration/)")
    print(f"    results/              - submission7.csv")


def main():
    """Run complete pipeline."""
    # Print header
    print_header()
    
    # Validate configuration
    if not config.validate_config():
        print(" Configuration failed")
        sys.exit(1)
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Confirmation
    print(f" all images processed in: {config.DATA_DIR}")
    print(f"Output will be saved to: {config.OUTPUT_DIR}")
    
    
    try:
        # Load images
        images = load_images()
        
        # Stage 1: U-Net inference
        masks = unet_inference(images)
        
        # Stage 2: Instance segmentation
        results = instance_segmentation(images, masks)
        
        # Stage 3: Length calculation
        df = length_calculation(images)
       
        # Print summary
        print_summary(images, masks, results, df)
        
    except KeyboardInterrupt:
        print(" Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f" Pipeline failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
