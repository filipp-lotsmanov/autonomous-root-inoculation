import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
from typing import Dict, Tuple

CONFIG = {
    'patch_size': 256,
    'overlap': 64,
    'threshold': 0.5,
    'batch_size': 8,
    'morphology': True
}


def crop_to_match(large_tensor, small_tensor):
    """Crop large tensor to match small tensor dimensions."""
    large_shape = tf.shape(large_tensor)
    small_shape = tf.shape(small_tensor)

    offsets = (large_shape - small_shape) // 2
    cropped = large_tensor[:,
    offsets[1]:offsets[1] + small_shape[1],
    offsets[2]:offsets[2] + small_shape[2],
    :]
    return cropped


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient for evaluation."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    """Dice loss for training."""
    return 1 - dice_coef(y_true, y_pred, smooth)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal Loss to handle class imbalance."""
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Calculate focal loss
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    focal_loss_1 = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
    focal_loss_0 = -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_1 + focal_loss_0


def combined_loss(y_true, y_pred):
    """Combination of Focal Loss and Dice Loss."""
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1):
    """Intersection over Union metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def f1_score(y_true, y_pred, smooth=1):
    """F1 score metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Calculate TP, FP, FN
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class ModelManager:
    """Handles model loading and inference."""

    def __init__(self, model_path: str, config: dict = None):
        self.config = config or CONFIG
        self.patch_size = self.config['patch_size']
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model with ALL custom objects from training.

        Includes compatibility patches for models trained with Keras 2:
        - safe_mode=False enables Lambda layer deserialization (crop_to_match)
        - Strips 'groups=1' from Conv2DTranspose (Keras 3 incompatibility)
        - Replaces Lambda(crop_to_match) with a proper Layer subclass
        """
        custom_objects = {
            'crop_to_match': crop_to_match,
            'combined_loss': combined_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'iou_metric': iou_metric,
            'f1_score': f1_score
        }

        # Patch Conv2DTranspose to ignore 'groups' parameter
        original_conv_init = keras.layers.Conv2DTranspose.__init__

        def patched_conv_init(self, *args, **kwargs):
            kwargs.pop("groups", None)
            original_conv_init(self, *args, **kwargs)

        keras.layers.Conv2DTranspose.__init__ = patched_conv_init

        # Replace Lambda with a proper layer that declares output shape
        class CropToMatchLayer(keras.layers.Layer):
            def call(self, inputs):
                return crop_to_match(inputs[0], inputs[1])

            def compute_output_shape(self, input_shape):
                return input_shape[1]

        original_lambda_from_config = keras.layers.Lambda.from_config.__func__

        def patched_lambda_from_config(cls, config):
            try:
                return original_lambda_from_config(cls, config)
            except (ValueError, TypeError, NotImplementedError):
                return CropToMatchLayer(name=config.get("name", "crop_to_match"))

        keras.layers.Lambda.from_config = classmethod(patched_lambda_from_config)

        try:
            return keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                safe_mode=False,
                compile=False
            )
        finally:
            keras.layers.Conv2DTranspose.__init__ = original_conv_init
            keras.layers.Lambda.from_config = classmethod(original_lambda_from_config)

    def predict_patches(self, patches: np.ndarray) -> np.ndarray:
        """Predict on batch of patches."""
        batch_size = self.config['batch_size']
        predictions = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            pred = self.model.predict(batch, verbose=0)
            predictions.append(pred)

        return np.vstack(predictions)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to model input format."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return image


def extract_patches(image: np.ndarray, patch_size: int, overlap: int) -> Tuple[np.ndarray, dict]:
    """Extract overlapping patches from image."""
    h, w = image.shape[:2]
    step = patch_size - overlap

    patches = []
    positions = []

    y_starts = list(range(0, h - overlap, step))
    x_starts = list(range(0, w - overlap, step))

    # Ensure we cover the entire image
    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)

            # Adjust if at edge
            if y_end - y_start < patch_size:
                y_start = max(0, y_end - patch_size)
            if x_end - x_start < patch_size:
                x_start = max(0, x_end - patch_size)

            patch = image[y_start:y_start + patch_size, x_start:x_start + patch_size]

            # Pad if needed
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded

            patches.append(patch)
            positions.append((y_start, x_start))

    metadata = {
        'positions': positions,
        'shape': (h, w),
        'overlap': overlap
    }

    return np.array(patches), metadata


def stitch_predictions(patches: np.ndarray, metadata: dict, patch_size: int) -> np.ndarray:
    """Stitch patches back into full image."""
    h, w = metadata['shape']
    positions = metadata['positions']

    output = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for patch, (y_start, x_start) in zip(patches, positions):
        y_end = min(y_start + patch_size, h)
        x_end = min(x_start + patch_size, w)

        patch_h = y_end - y_start
        patch_w = x_end - x_start

        output[y_start:y_end, x_start:x_end] += patch[:patch_h, :patch_w, 0]
        counts[y_start:y_end, x_start:x_end] += 1

    # out= is required: without it np.divide allocates an uninitialised buffer
    # and only writes where the condition holds, leaving garbage in any pixel
    # no patch covered.
    output = np.divide(output, counts, out=np.zeros_like(output), where=counts > 0)
    return output


def postprocess_mask(pred: np.ndarray, threshold: float, morphology: bool) -> np.ndarray:
    """Convert prediction to binary mask."""
    binary = (pred > threshold).astype(np.uint8) * 255

    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def predict_image(image: np.ndarray, model_manager: ModelManager, config: dict = None) -> np.ndarray:
    """
    Complete prediction pipeline for single image.

    Args:
        image: RGB image (H, W, 3) uint8 [0, 255]
        model_manager: Initialized ModelManager
        config: Optional config overrides

    Returns:
        Binary mask (H, W) uint8 [0, 255]
    """
    cfg = config or CONFIG

    # Preprocess
    img_norm = preprocess_image(image)

    # Extract patches
    patches, metadata = extract_patches(
        img_norm,
        cfg['patch_size'],
        cfg['overlap']
    )

    # Predict
    predictions = model_manager.predict_patches(patches)

    # Stitch
    full_pred = stitch_predictions(predictions, metadata, cfg['patch_size'])

    # Postprocess
    mask = postprocess_mask(full_pred, cfg['threshold'], cfg['morphology'])

    return mask


def predict_batch(images: Dict[str, np.ndarray],
                  model_manager: ModelManager,
                  config: dict = None,
                  verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Predict on multiple images.

    Args:
        images: Dictionary {name: image_array}
        model_manager: Initialized ModelManager
        config: Optional config overrides
        verbose: Print progress

    Returns:
        Dictionary {name: mask_array}
    """
    masks = {}

    for i, (name, img) in enumerate(images.items(), 1):
        if verbose:
            print(f"[{i}/{len(images)}] Processing {name}")

        mask = predict_image(img, model_manager, config)
        masks[name] = mask

    if verbose:
        print(f" Generated {len(masks)} masks")

    return masks


def generate_masks(images: Dict[str, np.ndarray],
                   model_path: str,
                   **kwargs) -> Dict[str, np.ndarray]:
    """
    One-line function to generate all masks.
    Usage:
        masks = generate_masks(images, 'unet_root_segmentation_256px.h5', overlap=64, threshold=0.5)
    """
    # Update config with kwargs
    config = CONFIG.copy()
    config.update(kwargs)

    # Load model
    model_manager = ModelManager(model_path, config)

    # Generate masks
    masks = predict_batch(images, model_manager, config, verbose=kwargs.get('verbose', True))

    return masks


if __name__ == "__main__":
    print("U-Net Inference Module")