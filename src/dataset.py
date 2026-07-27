from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def dicom_to_array(dicom_path: Path) -> np.ndarray:
    """Read a DICOM file and return a normalized uint8 numpy array."""
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)

def load_labels(labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split labels into:
    - df_patients: one row per patient with Target label (deduplicated)
    - df_boxes: all bounding boxes for positive patients
    """
    df = pd.read_csv(labels_path)

    # deduplication
    df_patients = df.drop_duplicates(
        subset='patientId', keep='first'
    ).reset_index(drop=True)[['patientId', 'Target']]

    # all boxes for positive patients
    df_boxes = df[df['Target'] == 1][
        ['patientId', 'x', 'y', 'width', 'height']
    ].reset_index(drop=True)

    return df_patients, df_boxes

def rsna_to_yolo(x: float, y: float, w: float, h: float, img_w: int = 1024, img_h: int = 1024) -> tuple:
    """
    Convert RSNA bounding box format to YOLO format.

    RSNA:   (x_top_left, y_top_left, width, height) in pixels
    YOLO:   (x_center, y_center, width, height) normalized 0-1
    """
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height

def build_yolo_dataset(
        config: dict,
        output_dir: Path,
        seed: int = 42
) -> None:
    """
    Convert raw RSNA data into YOLO directory structure:

    data/processed/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    raw_dir     = Path(config['paths']['data_raw'])
    img_dir     = raw_dir / 'stage_2_train_images'
    labels_csv  = raw_dir / 'stage_2_train_labels.csv'
    img_size    = config['data']['image_size']
    val_split   = config['data']['val_split']

    # create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # load labels
    df_patients, df_boxes = load_labels(labels_csv)

    # train / val split stratified on Target
    train_df, val_df = train_test_split(
        df_patients,
        test_size=val_split,
        random_state=seed,
        stratify=df_patients['Target']
    )

    print(f'Train patients: {len(train_df)} | Val patients: {len(val_df)}')
    print(f'Train positives: {train_df["Target"].sum()}' f'({train_df["Target"].mean()*100:.1f}%)')
    print(f'Train positives: {train_df["Target"].sum()}' f'({val_df["Target"].mean()*100:.1f}%)')

    # process each split
    for split, split_df in [('train', train_df), ('val', val_df)]:
        print(f'\nProcessing {split}...')
        img_out_dir = output_dir / 'images' / split
        lbl_out_dir = output_dir / 'labels' / split

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            pid     = row['patientId']
            target  = row['Target']

            # convert DICOM to PNG
            dicom_path = img_dir / f'{pid}.dcm'
            img = dicom_to_array(dicom_path)

            # resize to model imput size
            img_resized = cv2.resize(img, (img_size, img_size))

            # convert grayscale to rgb (YOLO expects 3 channels)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

            # save image
            img_path = img_out_dir / f'{pid}.png'
            cv2.imwrite(str(img_path), img_rgb)

            # write label file
            lbl_path = lbl_out_dir / f'{pid}.txt'
            if target == 0:
                # negative case: empty label file
                lbl_path.touch()
            else:
                boxes = df_boxes[df_boxes['patientId'] == pid]
                with open(lbl_path, 'w') as f:
                    for _, box in boxes.iterrows():
                        xc, yc, w, h = rsna_to_yolo(
                            box.x, box.y, box.width, box.height
                        )
                        # class 0 = pneumonia (only one class)
                        f.write(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')

def write_dataset_yaml(output_dir: Path) -> Path:
    """
    Write the dataset.yaml file YOLO requires.
    """

    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['pneumonia']
    }
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f'\nDataset YAML written to {yaml_path}')
    return yaml_path

if __name__ == '__main__':
    config      = load_config()
    output_dir  = Path(config['paths']['data_processed'])

    build_yolo_dataset(config, output_dir, seed=config['project']['seed'])
    write_dataset_yaml(output_dir)

    print('\nDone. Dataset ready for YOLO training.')