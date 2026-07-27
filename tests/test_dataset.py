import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import rsna_to_yolo


class TestRsnaToYolo:
    '''Tests for the RSNA -> YOLO bounding box format converter.'''

    def test_center_box_conversion(self):
        '''A box exactly in the center of a 1024x1024 image.'''
        x, y, w, h = 412, 412, 200, 200
        xc, yc, bw, bh = rsna_to_yolo(x, y, w, h, img_w=1024, img_h=1024)

        assert xc == pytest.approx(0.5, abs=0.01)
        assert yc == pytest.approx(0.5, abs=0.01)
        assert bw == pytest.approx(200 / 1024)
        assert bh == pytest.approx(200 / 1024)

    def test_top_left_box(self):
        '''A box at the top-left corner.'''
        xc, yc, bw, bh = rsna_to_yolo(0, 0, 100, 100, img_w=1024, img_h=1024)

        # center of a box at (0,0) with size 100x100 should be at (50/1024, 50/1024)
        assert xc == pytest.approx(50 / 1024)
        assert yc == pytest.approx(50 / 1024)

    def test_output_values_are_normalized(self):
        '''All YOLO outputs must be between 0 and 1 for valid boxes.'''
        xc, yc, bw, bh = rsna_to_yolo(264, 152, 213, 379, img_w=1024, img_h=1024)

        for value in (xc, yc, bw, bh):
            assert 0.0 <= value <= 1.0

    def test_known_value_from_real_data(self):
        '''
        Regression test using a real box from the RSNA dataset
        (patient 00436515, first box) to catch any future
        accidental changes to the conversion formula.
        '''
        xc, yc, bw, bh = rsna_to_yolo(264, 152, 213, 379, img_w=1024, img_h=1024)

        assert xc == pytest.approx(0.36181640625, abs=0.0001)
        assert yc == pytest.approx(0.33349609375, abs=0.0001)
        assert bw == pytest.approx(0.2080078125, abs=0.0001)
        assert bh == pytest.approx(0.3701171875, abs=0.0001)


class TestLoadLabels:
    '''Tests for label loading and deduplication logic.'''

    def test_deduplication_keeps_one_row_per_patient(self, tmp_path):
        '''
        A patient with multiple bounding boxes should produce exactly
        one row in df_patients but multiple rows in df_boxes.
        '''
        import pandas as pd
        from src.dataset import load_labels

        csv_path = tmp_path / 'test_labels.csv'
        pd.DataFrame({
            'patientId': ['p1', 'p1', 'p2'],
            'x':         [10, 50, None],
            'y':         [10, 50, None],
            'width':     [20, 20, None],
            'height':    [20, 20, None],
            'Target':    [1, 1, 0],
        }).to_csv(csv_path, index=False)

        df_patients, df_boxes = load_labels(csv_path)

        assert len(df_patients) == 2          # one row per unique patient
        assert len(df_boxes) == 2             # both boxes for p1 preserved
        assert df_patients['Target'].sum() == 1  # only p1 is positive