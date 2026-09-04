import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import label as scipy_label
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class PlantSegmenter:
    """
    Handles segmentation of individual plants from binary mask.
    """

    def __init__(self,
                 expected_specimen_count: int = 5,
                 first_specimen_x: int = 335,
                 specimen_interval: int = 500,
                 focal_width: int = 240):
        """
        Initialize segmenter with spatial configuration.
        Args:
            expected_specimen_count: How many plants to look for
            first_specimen_x: Horizontal position of leftmost plant (pixels)
            specimen_interval: Horizontal spacing between plant centers (pixels)
            focal_width: Half-width of detection window (pixels)
        """
        self.expected_count = expected_specimen_count
        self.first_x = first_specimen_x
        self.interval = specimen_interval
        self.radius = focal_width

        # Pre-calculate plant centers
        self.specimen_centers = [
            self.first_x + i * self.interval
            for i in range(self.expected_count)
        ]

    def get_search_window(self, specimen_index: int, image_width: int) -> Tuple[int, int]:
        """
        Calculate search boundaries for a specific plant.
        """
        center = self.specimen_centers[specimen_index]
        left = max(0, center - self.radius)
        right = min(image_width, center + self.radius)
        return left, right

    def segment(self, binary_mask: np.ndarray) -> List[Dict]:
        """
        Main segmentation method using contour detection
        """
        height, width = binary_mask.shape

        # Detect all contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build region information from contours
        region_data = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Create mask for this region
            region_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(region_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            region_data.append({
                'id': idx,
                'contour': cnt,
                'mask': region_mask,
                'area': area,
                'bbox': (x, y, w, h)
            })

        # Initialize storage for each plant
        specimen_data = []
        individual_masks = [
            np.zeros((height, width), dtype=np.uint8)
            for _ in range(self.expected_count)
        ]

        # Assign each detected region to nearest plant using ROI overlap
        for region in region_data:
            region_pixels = (region['mask'] == 255)

            # Find which plant this region belongs to
            best_match_score = 0
            best_match_index = -1

            for specimen_idx in range(self.expected_count):
                window_left, window_right = self.get_search_window(specimen_idx, width)

                # Count pixels in this region that fall within this plant's window
                overlap_pixels = region_pixels[:, window_left:window_right].sum()

                if overlap_pixels > best_match_score:
                    best_match_score = overlap_pixels
                    best_match_index = specimen_idx

            # Assign region to best matching plant
            if best_match_index >= 0 and best_match_score > 0:
                individual_masks[best_match_index][region_pixels] = 255

        # Build plant data structures
        for idx in range(self.expected_count):
            window_left, window_right = self.get_search_window(idx, width)
            specimen_mask = individual_masks[idx]

            has_content = specimen_mask.sum() > 0

            specimen_info = {
                'index': idx,
                'name': f'Plant_{idx + 1}',
                'detected': has_content,
                'center_x': self.specimen_centers[idx],
                'window': (window_left, window_right),
                'mask': specimen_mask,
                'root_count': 0,
                'total_pixels': 0
            }

            specimen_data.append(specimen_info)

        return specimen_data


class RootAnalyzer:
    """
    Analyzes individual roots within a plant mask.
    """

    def __init__(self, noise_floor: int = 50):
        """
        Initialize analyzer with quality thresholds.
        Args:
            noise_floor: Smallest acceptable root size
        """
        self.min_pixels = noise_floor

    def decompose_into_roots(self, specimen_mask: np.ndarray, specimen_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Break down plant mask into individual root structures.
        Args:
            specimen_mask: Binary mask containing all roots for one plant
            specimen_id: Identifier for parent plant

        Returns:
            labeled_map: Each root has unique integer ID
            structure_registry: List of dictionaries with root measurements
        """
        # Use scipy instead of cv2
        labeled_map, root_count = scipy_label(specimen_mask)

        structure_registry = []

        for root_idx in range(1, root_count + 1):
            root_region = (labeled_map == root_idx)
            pixel_count = root_region.sum()

            # Quality filter
            if pixel_count < self.min_pixels:
                labeled_map[labeled_map == root_idx] = 0
                continue

            # Extract spatial properties
            y_positions, x_positions = np.where(root_region)

            root_record = {
                'parent_plant': specimen_id,
                'root_index': root_idx,
                'identifier': f'Plant_{specimen_id}_Root_{root_idx}',
                'pixel_count': pixel_count,
                'center_of_mass_x': float(x_positions.mean()),
                'center_of_mass_y': float(y_positions.mean()),
                'bounding_box': {
                    'x_min': int(x_positions.min()),
                    'x_max': int(x_positions.max()),
                    'y_min': int(y_positions.min()),
                    'y_max': int(y_positions.max())
                }
            }

            structure_registry.append(root_record)

        return labeled_map, structure_registry


class SegmentationVisualizer:
    """
    Creates visualization of segmentation results.
    """

    def __init__(self, color_scheme: str = 'bright'):
        """
        Initialize visualizer with color preferences.
        """
        self.scheme = color_scheme
        self.specimen_colors = self._generate_color_palette()
        # Fixed seed: per-root colours are arbitrary but should be stable so
        # that re-running the pipeline produces identical visualizations.
        self._color_rng = np.random.default_rng(0)

    def _generate_color_palette(self) -> List[Tuple]:
        """
        Generate distinct colors for visualization.
        """
        if self.scheme == 'bright':
            return [
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]
        else:
            return [
                (150, 50, 50),
                (50, 150, 50),
                (50, 50, 150),
                (150, 150, 50),
                (150, 50, 150),
                (50, 150, 150),
            ]

    def render_complete_figure(self,
                               original_image: np.ndarray,
                               binary_mask: np.ndarray,
                               specimen_list: List[Dict],
                               title: str,
                               save_path: Path) -> None:
        """
        Create comprehensive visualization figure.
        Expects original_image in RGB format.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)

        # Original image (already RGB from load_images)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Binary mask
        axes[0, 1].imshow(binary_mask, cmap='gray')
        axes[0, 1].set_title('Binary Mask')
        axes[0, 1].axis('off')

        # Individual plants
        height, width = binary_mask.shape
        combined_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for specimen in specimen_list:
            if specimen['detected']:
                color = self.specimen_colors[specimen['index'] % len(self.specimen_colors)]
                mask_bool = specimen['mask'] > 0
                combined_mask[mask_bool] = color

        axes[1, 0].imshow(combined_mask)
        axes[1, 0].set_title('Segmented Plants')
        axes[1, 0].axis('off')

        # Root visualization
        root_composite = np.zeros((height, width, 3), dtype=np.uint8)

        for specimen in specimen_list:
            if specimen['detected'] and 'root_map' in specimen:
                root_map = specimen['root_map']
                unique_roots = np.unique(root_map)

                for root_id in unique_roots:
                    if root_id == 0:
                        continue

                    color = tuple(self._color_rng.integers(50, 255, 3).tolist())
                    root_composite[root_map == root_id] = color

        axes[1, 1].imshow(root_composite)
        axes[1, 1].set_title('Individual Roots')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_overlay(self,
                       original_image: np.ndarray,
                       specimen_list: List[Dict],
                       alpha: float = 0.4) -> np.ndarray:
        """
        Create overlay of segmentation on original image.
        Expects original_image in RGB format.
        """
        overlay = original_image.copy()
        height, width = overlay.shape[:2]
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for specimen in specimen_list:
            if specimen['detected']:
                color = self.specimen_colors[specimen['index'] % len(self.specimen_colors)]
                mask_bool = specimen['mask'] > 0
                color_mask[mask_bool] = color

        result = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
        return result


class ResultExporter:
    """
    Handles export of results to files.
    """

    def __init__(self, base_directory: Path):
        """
        Initialize exporter with output directory.
        """
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_plant_masks(self,
                         results: Dict,
                         format_type: str = 'binary') -> Dict[str, List[str]]:
        """
        Export individual plant masks.
        """
        plants_folder = self.base_dir / 'individual_plants'
        plants_folder.mkdir(exist_ok=True)

        saved_paths = {}

        for image_name, result in results.items():
            image_stem = Path(image_name).stem
            plant_files = []

            for plant in result['plants']:
                filename = f"{image_stem}_plant_{plant['index'] + 1}.png"
                filepath = plants_folder / filename

                if format_type == 'binary':
                    cv2.imwrite(str(filepath), plant['mask'])

                plant_files.append(str(filepath))

            saved_paths[image_name] = plant_files

        return saved_paths

    def save_root_masks(self,
                        results: Dict,
                        full_resolution: bool = True) -> Dict[str, Dict]:
        """
        Export individual root masks.
        """
        roots_folder = self.base_dir / 'individual_roots'
        roots_folder.mkdir(exist_ok=True)

        saved_paths = {}

        for image_name, result in results.items():
            image_stem = Path(image_name).stem
            image_roots = {}

            for plant in result['plants']:
                if not plant['detected'] or 'root_details' not in plant:
                    continue

                plant_key = f"plant_{plant['index'] + 1}"
                root_files = []

                root_map = plant['root_map']

                for root in plant['root_details']:
                    root_idx = root['root_index']

                    if full_resolution:
                        # Full-size approach
                        height, width = root_map.shape
                        isolated_root = np.zeros((height, width), dtype=np.uint8)
                        isolated_root[root_map == root_idx] = 255
                    else:
                        # Cropped approach
                        bbox = root['bounding_box']
                        y_coords, x_coords = np.where(root_map == root_idx)

                        pad = 10
                        crop_y1 = max(0, bbox['y_min'] - pad)
                        crop_y2 = min(root_map.shape[0], bbox['y_max'] + pad + 1)
                        crop_x1 = max(0, bbox['x_min'] - pad)
                        crop_x2 = min(root_map.shape[1], bbox['x_max'] + pad + 1)

                        cropped_map = root_map[crop_y1:crop_y2, crop_x1:crop_x2]
                        isolated_root = (cropped_map == root_idx).astype(np.uint8) * 255

                    filename = f"{image_stem}_plant_{plant['index'] + 1}_root_{root_idx}.png"
                    filepath = roots_folder / filename

                    cv2.imwrite(str(filepath), isolated_root)
                    root_files.append(str(filepath))

                if root_files:
                    image_roots[plant_key] = root_files

            if image_roots:
                saved_paths[image_name] = image_roots

        return saved_paths


def process_all(images: Dict[str, np.ndarray],
                masks: Dict[str, np.ndarray],
                output_base_dir: str,
                start: int = 335,
                step: int = 500,
                num_plants: int = 5,
                roi_width: int = 240,  # half-width; window spans 2*roi_width
                save_plants: bool = True,
                save_roots: bool = False,
                plant_format: str = 'mask',
                roots_full_size: bool = True,
                roots_validate: bool = True,
                visualize: bool = True,
                **kwargs) -> Dict:
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    segmenter = PlantSegmenter(
        expected_specimen_count=num_plants,
        first_specimen_x=start,
        specimen_interval=step,
        focal_width=roi_width
    )

    root_analyzer = RootAnalyzer(
        noise_floor=kwargs.get('min_root_area', 50)
    )

    results = {}

    # Process each image
    for img_name, mask in masks.items():
        img = images.get(img_name)

        # Segment plants
        specimen_list = segmenter.segment(mask)

        # Analyze roots
        for specimen_record in specimen_list:
            if specimen_record['detected']:
                root_map, root_details = root_analyzer.decompose_into_roots(
                    specimen_record['mask'],
                    specimen_record['index']
                )
                specimen_record['root_map'] = root_map
                specimen_record['root_details'] = root_details
                specimen_record['root_count'] = len(root_details)
                specimen_record['total_pixels'] = sum(r['pixel_count'] for r in root_details)

        # Calculate stats
        detected = sum(1 for p in specimen_list if p['detected'])
        total_roots = sum(p['root_count'] for p in specimen_list)

        results[img_name] = {
            'plants': specimen_list,
            'summary': {
                'total_expected': num_plants,
                'total_found': detected,
                'detection_rate': detected / num_plants,
                'total_roots': total_roots,
                'avg_roots_per_plant': total_roots / detected if detected > 0 else 0,
                'total_area': sum(p['total_pixels'] for p in specimen_list),
                'avg_area_per_plant': sum(p['total_pixels'] for p in specimen_list) / detected if detected > 0 else 0,
                'avg_area_per_root': sum(
                    p['total_pixels'] for p in specimen_list) / total_roots if total_roots > 0 else 0
            }
        }

        # Map to compatible format
        for plant in specimen_list:
            plant['Plant ID'] = plant['index']
            plant['plant_label'] = plant['name']
            plant['found'] = plant['detected']
            plant['expected_x'] = plant['center_x']
            plant['roi_bounds'] = plant['window']
            plant['num_roots'] = plant.get('root_count', 0)
            plant['total_area'] = plant.get('total_pixels', 0)

            if 'root_details' in plant:
                plant['root_info'] = []
                for root in plant['root_details']:
                    plant['root_info'].append({
                        'plant_id': root['parent_plant'],
                        'root_id': root['root_index'],
                        'root_label': root['identifier'],
                        'area': root['pixel_count'],
                        'centroid_x': root['center_of_mass_x'],
                        'centroid_y': root['center_of_mass_y'],
                        'bbox': (
                            root['bounding_box']['x_min'],
                            root['bounding_box']['x_max'],
                            root['bounding_box']['y_min'],
                            root['bounding_box']['y_max']
                        )
                    })

                if 'root_map' in plant:
                    plant['labeled_roots'] = plant['root_map']
            else:
                plant['root_info'] = []

        # Console output
        print(f"\n{img_name}:")
        print(f"  Found: {detected}/{num_plants} plants")
        for p in specimen_list:
            status = "FOUND" if p['detected'] else "NOT FOUND"
            info = f", {p['root_count']} roots" if p['detected'] else ""
            print(f"    Plant {p['index']}: {status}{info}")
        print(f"  Total roots: {total_roots}")

        # Visualize
        if visualize:
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)

            visualizer = SegmentationVisualizer(color_scheme='bright')
            save_path = viz_dir / f"{Path(img_name).stem}_segmentation.png"

            visualizer.render_complete_figure(
                img, mask, specimen_list, img_name, save_path
            )

    # Export individual files
    exporter = ResultExporter(output_dir)

    if save_plants:
        plant_paths = exporter.save_plant_masks(results, format_type='binary')
        print(f" Saved {sum(len(paths) for paths in plant_paths.values())} plant masks")

    if save_roots:
        root_paths = exporter.save_root_masks(results, full_resolution=roots_full_size)
        print(f" Saved root masks")

    return results


if __name__ == "__main__":
    print("Plant Instance Segmentation Module")