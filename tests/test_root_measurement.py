"""Tests for root length measurement and tip localisation.

Masks are synthesised so the true length and tip position are known exactly.
Skeletonization insets the centreline slightly from a thick stroke, so the
tolerances below allow a couple of pixels.
"""
import cv2
import numpy as np
import pytest

from run_pipeline import analyse_root


def vertical_root(top_y=200, bottom_y=1200, x=1500, thickness=7,
                  shape=(3006, 4202)):
    mask = np.zeros(shape, np.uint8)
    cv2.line(mask, (x, top_y), (x, bottom_y), 255, thickness)
    return mask


class TestEmptyAndDegenerateInput:
    def test_none_mask_reports_nothing_detected(self):
        assert analyse_root(None) == (0, None)

    def test_all_zero_mask_reports_nothing_detected(self):
        assert analyse_root(np.zeros((100, 100), np.uint8)) == (0, None)

    def test_sub_threshold_noise_is_not_a_root(self):
        """Values at or below the 127 binarisation threshold are background."""
        mask = np.full((100, 100), 120, np.uint8)
        assert analyse_root(mask) == (0, None)


class TestTipLocalisation:
    def test_tip_is_the_bottom_most_point(self):
        _, tip = analyse_root(vertical_root(top_y=200, bottom_y=1200, x=1500))
        assert tip is not None
        tip_x, tip_y = tip
        assert tip_x == pytest.approx(1500, abs=4)
        assert tip_y == pytest.approx(1200, abs=4)

    def test_tip_is_returned_as_x_y_not_row_col(self):
        """Guards the axis-order convention the robot transform depends on."""
        _, tip = analyse_root(vertical_root(top_y=100, bottom_y=400, x=2500))
        tip_x, tip_y = tip
        assert tip_x > tip_y, "expected (x, y); got what looks like (row, col)"

    @pytest.mark.parametrize("x", [800, 1500, 2600, 3400])
    def test_tip_tracks_horizontal_position(self, x):
        _, tip = analyse_root(vertical_root(x=x))
        assert tip[0] == pytest.approx(x, abs=4)


class TestLengthMeasurement:
    @pytest.mark.parametrize("span", [300, 700, 1500])
    def test_straight_root_length_matches_span(self, span):
        length, _ = analyse_root(vertical_root(top_y=200, bottom_y=200 + span))
        assert length == pytest.approx(span, abs=4)

    def test_length_is_geodesic_not_euclidean(self):
        """A zigzag root must measure along its path, not end-to-end.

        The path descends monotonically so the bottom-most point is
        unambiguously the far end.
        """
        mask = np.zeros((3006, 4202), np.uint8)
        cv2.line(mask, (1500, 200), (1500, 600), 255, 7)    # 400 down
        cv2.line(mask, (1500, 600), (2400, 1000), 255, 7)   # ~985 down-right
        cv2.line(mask, (2400, 1000), (1500, 1400), 255, 7)  # ~985 down-left
        length, _ = analyse_root(mask)
        path_length = 400 + 2 * np.hypot(900, 400)  # ~2370
        euclidean = 1200                            # (1500,200) -> (1500,1400)
        assert length > euclidean * 1.5
        assert length == pytest.approx(path_length, rel=0.1)

    def test_lateral_branch_does_not_inflate_primary_length(self):
        mask = np.zeros((3006, 4202), np.uint8)
        cv2.line(mask, (1500, 200), (1500, 1200), 255, 7)  # primary, 1000
        cv2.line(mask, (1500, 700), (1800, 800), 255, 5)   # lateral branch
        length, tip = analyse_root(mask)
        assert length == pytest.approx(1000, abs=40)
        assert tip[1] == pytest.approx(1200, abs=6)

    def test_largest_component_wins_over_detached_noise(self):
        mask = vertical_root(top_y=200, bottom_y=1200, x=1500)
        cv2.circle(mask, (3000, 2500), 12, 255, -1)  # detached blob
        length, tip = analyse_root(mask)
        assert length == pytest.approx(1000, abs=40)
        assert tip[0] == pytest.approx(1500, abs=6), "tip jumped to the noise blob"


class TestDeterminism:
    def test_repeated_calls_agree(self):
        mask = vertical_root()
        assert analyse_root(mask) == analyse_root(mask)


class TestKnownLimitations:
    """Documents behaviour that is a consequence of the algorithm's domain
    assumption: primary roots grow downward, so the deepest skeleton point is
    the tip. Where that assumption does not hold the measurement is truncated.
    docs/pipeline_evolution.md notes the same caveat for curved roots.
    """

    def test_horizontal_arm_at_the_lowest_extent_is_not_traced(self):
        """A root turning sideways at its deepest point measures short.

        The tip is chosen as the first pixel at maximum y, which for a
        horizontal run is its corner rather than its far end, so the arm is
        excluded from the length.
        """
        mask = np.zeros((3006, 4202), np.uint8)
        cv2.line(mask, (1500, 200), (1500, 1000), 255, 7)   # 800 down
        cv2.line(mask, (1500, 1000), (2100, 1000), 255, 7)  # 600 across
        length, tip = analyse_root(mask)

        # Only the vertical run is measured; the 600px arm is dropped.
        assert length == pytest.approx(800, abs=20)
        assert tip[0] == pytest.approx(1500, abs=10)
