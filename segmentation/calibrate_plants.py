import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Import configuration
import config


def check_environment():
    """Check if environment is set up correctly."""
    
    print("Checking Environment")
    
    issues = []
    
    # Check data directory
    if not config.DATA_DIR.exists():
        issues.append(f" Data directory not found: {config.DATA_DIR}")
    else:
        images = list(config.DATA_DIR.glob('*.png')) + list(config.DATA_DIR.glob('*.jpg'))
        print(f" Data directory found: {config.DATA_DIR}")
        print(f"Found {len(images)} images")
        
        if len(images) == 0:
            issues.append("No images in data directory")
    
    # Check model file
    if not config.MODEL_PATH.exists():
        issues.append(f" Model file not found: {config.MODEL_PATH}")
    else:
        print(f" Model file found: {config.MODEL_PATH}")
        size_mb = config.MODEL_PATH.stat().st_size / (1024*1024)
        print(f" Model size: {size_mb:.1f} MB")
    
    # Check required modules
    try:
        import tensorflow as tf
        print(f" TensorFlow installed: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f" GPUs available: {len(gpus)}")
        else:
            print("  No GPU detected")
    except ImportError:
        issues.append(" TensorFlow not installed")
    
    try:
        from scipy.ndimage import label  # noqa: F401  availability probe
        print(" SciPy installed")
    except ImportError:
        issues.append(" SciPy not installed")
    
    try:
        from skimage.morphology import skeletonize  # noqa: F401  availability probe
        print(" scikit-image installed")
    except ImportError:
        issues.append(" scikit-image not installed")
    
    if issues:
        for issue in issues:
            print(issue)
        return False
    
    print(" Environment check passed")
    return True


def detect_plant_positions_auto(image_path):
    """
    Automatically detect plant positions using shoot detection.
    """
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f" Could not load: {image_path}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"\nAnalyzing: {image_path.name}")
    print(f"Image size: {w} × {h} pixels")
    
    # Strategy: Find dark regions (shoots) at top of image
    top_height = int(h * 0.25)
    top_region = gray[:top_height, :]
    
    # Invert so dark becomes bright
    inverted = cv2.bitwise_not(top_region)
    
    # Try multiple thresholds to find best one
    best_threshold = None
    best_count = 999
    best_binary = None
    
    for thresh in range(120, 200, 10):
        _, binary = cv2.threshold(inverted, thresh, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Count components with min area
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid_count = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 300)
        
        print(f"  Threshold {thresh}: {valid_count} shoots detected")
        
        if abs(valid_count - 5) < abs(best_count - 5):
            best_threshold = thresh
            best_count = valid_count
            best_binary = binary.copy()
    
    print(f" Using threshold {best_threshold} found {best_count} shoots")
    
    # Use best threshold
    _, binary = cv2.threshold(inverted, best_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Get valid shoots
    shoots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 300:
            cx, cy = centroids[i]
            shoots.append((int(cx), int(cy), area))
    
    # Sort by area, take top 5
    shoots.sort(key=lambda x: x[2], reverse=True)
    shoots = shoots[:5]
    
    # Sort by x-coordinate
    shoots.sort(key=lambda x: x[0])
    
    detected_positions = [x for x, y, a in shoots]
    
    print("Detected plant x-positions:")
    for i, x in enumerate(detected_positions, 1):
        print(f"  Plant {i}: x = {x}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original with detected positions
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, (x, y, _) in enumerate(shoots):
        ax.plot(x, y, 'g*', markersize=20)
        ax.text(x, y-50, f'P{i+1}\nx={x}', color='green', fontsize=11, 
               ha='center', weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f'Detected Positions (n={len(shoots)})', fontsize=12, weight='bold')
    ax.axis('off')
    
    # Shoot detection visualization
    ax = axes[0, 1]
    ax.imshow(best_binary, cmap='gray')
    ax.set_title(f'Shoot Detection (threshold={best_threshold})', fontsize=12)
    ax.axis('off')
    
    # Calculate parameters
    if len(detected_positions) >= 2:
        spacings = [detected_positions[i+1] - detected_positions[i] 
                   for i in range(len(detected_positions)-1)]
        avg_spacing = int(np.mean(spacings))
        start_pos = detected_positions[0]
        roi_width = int(avg_spacing * 0.45)
        
        # Show calculated parameters
        ax = axes[1, 0]
        ax.axis('off')
        params_text = f"""
DETECTED PARAMETERS:

Number of plants: {len(detected_positions)}
First plant position: {start_pos}
Average spacing: {avg_spacing}
Recommended ROI width: {roi_width}

Plant positions:
{detected_positions}

Spacings:
{spacings}
"""
        ax.text(0.1, 0.5, params_text, fontsize=11, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_title('Calculated Parameters', fontsize=12, weight='bold')
        
        # Comparison with typical setup
        ax = axes[1, 1]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detected (green)
        for i, (x, y, _) in enumerate(shoots):
            ax.axvline(x, color='green', linestyle='-', linewidth=3, alpha=0.7, 
                      label='Detected' if i == 0 else '')
            ax.text(x, 100, f'{x}', color='green', fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('Detected Plant Positions', fontsize=12, weight='bold')
        ax.axis('off')
    else:
        # Not enough plants
        ax = axes[1, 0]
        ax.axis('off')
        ax.text(0.5, 0.5, ' Not enough plants detected', 
               ha='center', va='center', fontsize=14, color='red')
        
        ax = axes[1, 1]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('setup_plant_detection.png', dpi=150, bbox_inches='tight')
    print(" Visualization saved: setup_plant_detection.png")
    plt.close()
    
    if len(detected_positions) < 2:
        return None
    
    # Calculate parameters
    spacings = [detected_positions[i+1] - detected_positions[i] 
               for i in range(len(detected_positions)-1)]
    avg_spacing = int(np.mean(spacings))
    start_pos = detected_positions[0]
    roi_width = int(avg_spacing * 0.45)
    
    result = {
        'start': start_pos,
        'step': avg_spacing,
        'roi_width': roi_width,
        'num_plants': len(detected_positions),
        'positions': detected_positions
    }
    
    print(f"  Number of plants: {result['num_plants']}")
    print(f"  First plant position: {result['start']}")
    print(f"  Average spacing: {result['step']}")
    print(f"  Recommended ROI width: {result['roi_width']}")
    print(f"  Plant positions: {detected_positions}")
    if len(spacings) > 1:
        print(f"  Spacing range: {min(spacings)} - {max(spacings)} pixels")
    
    return result


def update_config_file(params):
    """Update config.py with detected parameters."""
    
    config_path = Path('config.py')
    
    if not config_path.exists():
        print(" config.py not found!")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update parameters. Replacement lines must keep the trailing newline that
    # readlines() preserved, otherwise the rewritten file collapses into one
    # statement and config.py becomes a syntax error.
    updated_lines = []
    for line in lines:
        if line.strip().startswith('PLANT_START ='):
            updated_lines.append(f"PLANT_START = {params['start']}\n")
        elif line.strip().startswith('PLANT_STEP ='):
            updated_lines.append(f"PLANT_STEP = {params['step']}\n")
        elif line.strip().startswith('PLANT_ROI_WIDTH ='):
            updated_lines.append(f"PLANT_ROI_WIDTH = {params['roi_width']}\n")
        elif line.strip().startswith('NUM_PLANTS ='):
            updated_lines.append(f"NUM_PLANTS = {params['num_plants']}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("config.py updated successfully")
    print(f"  PLANT_START = {params['start']}")
    print(f"  PLANT_STEP = {params['step']}")
    print(f"  PLANT_ROI_WIDTH = {params['roi_width']}")
    print(f"  NUM_PLANTS = {params['num_plants']}")
    
    return True


def test_single_image():
    """Quick test on one image to verify setup."""
    
    try:
        from inference import generate_masks
        
        # Load one image
        images = list(config.DATA_DIR.glob('*.png')) + list(config.DATA_DIR.glob('*.jpg'))
        test_img = cv2.imread(str(images[0]))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        print(f"Testing on: {images[0].name}")
        
        # Generate mask
        masks = generate_masks(
            images={images[0].name: test_img},
            model_path=str(config.MODEL_PATH),
            overlap=config.UNET_OVERLAP,
            threshold=config.UNET_THRESHOLD,
            batch_size=config.UNET_BATCH_SIZE,
            verbose=False
        )
        
        mask = masks[images[0].name]
        
        # Check mask
        root_pixels = (mask > 0).sum()
        total_pixels = mask.size
        root_pct = root_pixels / total_pixels * 100
        
        print("Test successful")
        print(f" Mask shape: {mask.shape}")
        print(f" Root coverage: {root_pct:.2f}%")
        
        # Quick visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(test_img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('U-Net Mask')
        axes[1].axis('off')
        
        axes[2].imshow(test_img)
        axes[2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('setup_test_result.png', dpi=150, bbox_inches='tight')
        print(" Visualization saved: setup_test_result.png")
        plt.close()
        
        return True
        
    except Exception as e:
        print(" Test failed with error:")
        print(f"   {str(e)}")
        return False


def main():
    """Run complete setup process."""
    
    #Check environment
    if not check_environment():
        print(" Setup failed at environment check")
        return
    

    #Detect plant positions automatically
    images = list(config.DATA_DIR.glob('*.png')) + list(config.DATA_DIR.glob('*.jpg'))
    if not images:
        print("No images in data directory")
        return
    
    # Use first image for detection
    sample_image = images[0]
    params = detect_plant_positions_auto(sample_image)
    
    if params is None:
        print("Automatic detection failed")
        return
    
    #Update config
    if not update_config_file(params):
        print("Failed to update config.py")
        return
    
    # Reload config
    import importlib
    importlib.reload(config)
    
    
    # Step 4: Test
    if not test_single_image():
        print("Test failed")

if __name__ == "__main__":
    main()
