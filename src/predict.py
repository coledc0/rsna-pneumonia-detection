from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path: Path) -> YOLO:
    '''
    load YOLO model from weights path.
    '''
    if not model_path.exists():
        raise FileNotFoundError(f'Model weights not found at {model_path}')
    return YOLO(str(model_path))

def preprocess_image(image_input, img_size: int) -> np.ndarray:
    '''
    Accept either:
    - a file path (str or Path)
    - a numpy array (already loaded image)
    - raw bytes (from API upload)
    
    Returns a resized RGB numpy array ready for inference.
    '''

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f'Could not load image at {image_input}')

    elif isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Could not decode image bytes')
        
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()

    else:
        raise TypeError(f'Unsupported image input type: {type(image_input)}')
    
    # convert grayscale to TGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # resize to model input size
    img = cv2.resize(img, (img_size, img_size))

    return img

def predict(
        model: YOLO,
        image_input,
        img_size: int,
        conf_threshold: float = 0.25,
) -> dict:
    '''
    Run inference on a single image.

    Returns a dict with:
    - predicted:    bool, whether pneumonia was detected
    - confidence:   flaot,highest confidence score acreoss all boxes
    - boxes:        list of dicts with box coordinates and confidence
    -n_detections:  int, number of boxes above threshold
    '''
    img = preprocess_image(image_input, img_size)

    results = model(img, verbose = False, conf = conf_threshold)[0]
    boxes = results.boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append({
                'x1':       x1,
                'y1':       y1,
                'x2':       x2,
                'y2':       y2,
                'confidence': round(conf, 4),
            })
    
    predicted = len(detections) > 0
    max_conf = max((d['confidence'] for d in detections), default = 0.0)

    return {
        'predicted':    predicted,
        'confidence':   round(max_conf, 4),
        'n_detections': len(detections),
        'boxes':        detections,
    }

def predict_batch(
        model: YOLO,
        image_paths: list[Path],
        img_size: int,
        conf_threshold: float = 0.25,
) -> list[dict]:
    '''
    Run inference on a list of image paths.
    returns a list of prediction dicts in the same order as input.
    '''
    results = []
    for path in image_paths:
        try:
            result = predict(model, path, img_size, conf_threshold)
            result['path'] = str(path)
            results.append(result)
        except Exception as e:
            results.append({
                'path':         str(path),
                'predicted':    None,
                'confidence':   None,
                'n_detections': None,
                'boxes':        [],
                'error':        str(e)
            })
    return results

if __name__ == '__main__':

    config      = load_config()
    img_size    = config['data']['image_size']
    conf_thresh = config['api']['confidence_threshold']

    # point to best weights
    model_path = Path('runs/detect/weights/yolov8m/weights/best.pt')
    model      = load_model(model_path)

    # get a mix of positive and negative val images
    val_img_dir   = Path(config['paths']['data_processed']) / 'images' / 'val'
    val_label_dir = Path(config['paths']['data_processed']) / 'labels' / 'val'

    positive_imgs = [
        val_img_dir / f.name.replace('.txt', '.png')
        for f in val_label_dir.glob('*.txt')
        if f.stat().st_size > 0
    ][:3]

    negative_imgs = [
        val_img_dir / f.name.replace('.txt', '.png')
        for f in val_label_dir.glob('*.txt')
        if f.stat().st_size == 0
    ][:2]

    sample_imgs = positive_imgs + negative_imgs

    print(f'Running inference on {len(sample_imgs)} sample validation images:\n')
    print(f'  ({len(positive_imgs)} positive, {len(negative_imgs)} negative)\n')

    results = predict_batch(model, sample_imgs, img_size, conf_thresh)

    for r in results:
        print(f"Image: {Path(r['path']).name}")
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Predicted pneumonia: {r['predicted']}")
            print(f"  Max confidence:      {r['confidence']}")
            print(f"  N detections:        {r['n_detections']}")
            if r['boxes']:
                for box in r['boxes']:
                    print(f"    Box: {box}")
            else:
                print("    No boxes detected")
        print()