from pathlib import Path


DATA_DIR = Path('Kaggle')
MODEL_PATH = Path('unet_root_segmentation_256px.h5')
OUTPUT_DIR = Path('pipeline_results')

# Root tip coordinates consumed by integration/. Must stay in sync with
# system_config.ROOT_TIP_COORDINATES_CSV.
ROOT_TIPS_CSV = OUTPUT_DIR / 'root_tips_pixels.csv'



PLANT_START = 1168
PLANT_STEP = 499
PLANT_ROI_WIDTH = 224  # half-width in pixels; the search window spans 448
NUM_PLANTS = 5



# U-Net inference settings
UNET_OVERLAP = 64
UNET_THRESHOLD = 0.5
UNET_BATCH_SIZE = 8  

# What to save
SAVE_UNET_MASKS = True
SAVE_PLANTS = True
SAVE_ROOTS = True
SAVE_VISUALIZATIONS = True

# Root processing
ROOTS_FULL_SIZE = True
ROOTS_VALIDATE = True



def validate_config():
    """Check if configuration is valid."""
    errors = []
    
    if not DATA_DIR.exists():
        errors.append(f"DATA_DIR does not exist: {DATA_DIR}")
    
    if not MODEL_PATH.exists():
        errors.append(f"MODEL_PATH does not exist: {MODEL_PATH}")
    
    if PLANT_START <= 0:
        errors.append(f"PLANT_START should be > 0, got {PLANT_START}")
    
    if PLANT_STEP <= 0:
        errors.append(f"PLANT_STEP should be > 0, got {PLANT_STEP}")
    
    if not (0 <= UNET_THRESHOLD <= 1):
        errors.append(f"UNET_THRESHOLD should be 0-1, got {UNET_THRESHOLD}")
    
    if errors:
        print("\n".join(errors))
        return False
    
    print("Configuration is valid")
    return True


def print_config():
    """Print current configuration."""
    
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Model file: {MODEL_PATH}")
    print(f"  Output directory: {OUTPUT_DIR}")
 
    print(f"  First plant position: {PLANT_START}")
    print(f"  Plant spacing: {PLANT_STEP}")
    print(f"  ROI width: {PLANT_ROI_WIDTH}")
    print(f"  Number of plants: {NUM_PLANTS}")
 
    print(f"  Overlap: {UNET_OVERLAP} pixels")
    print(f"  Threshold: {UNET_THRESHOLD}")
    print(f"  Batch size: {UNET_BATCH_SIZE}")


if __name__ == "__main__":
    print_config()
    validate_config()
