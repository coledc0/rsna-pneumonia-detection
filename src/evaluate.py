from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_ground_truth(label_path: Path, img_size: int) -> list[dict]:
    '''
    Load YOLO format labels and convert back to pixel coordinates
    '''
    boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path, 'r') as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.strip().split())
                x1 = int((xc - w/2) * img_size)
                y1 = int((yc - h/2) * img_size)
                x2 = int((xc + w/2) * img_size)
                y2 = int((yc + h/2) * img_size)
                boxes.append({
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'class': int(cls)
                })
    return boxes

def draw_boxes_on_image(
        img: np.ndarray,
        gt_boxes: list[dict],
        pred_boxes: list,
        conf_threshold: float = 0.25
) -> np.ndarray:
    '''
    Draw ground truth (green) and predicted (red) boxes on image.
    '''
    img_draw = (
        cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if len(img.shape) == 2
        else img.copy()
    )

    #ground truth - green
    for box in gt_boxes:
        cv2.rectangle(
            img_draw,
            (box['x1'], box['y1']), 
             (box['x2'], box['y2']),
             (0, 255, 0), 2
        )
        cv2.putText(
            img_draw, 'GT',
            (box['x1'], box['y1'] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    #predictions - red
    if pred_boxes is not None:
        for box in pred_boxes:
            if box.conf[0] >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(
                    img_draw,
                    (x1, y1), (x2, y2),
                    (255, 0, 0), 2
                )
                cv2.putText(
                    img_draw,
                    f'Pred {conf:.2f}',
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )
    return img_draw

def evaluate_and_visualize(
        model_path: Path,
        config: dict,
        n_samples: int = 12,
        conf_threshold: float = 0.25,
        split: str = 'val'
) -> None:
    '''
    Run inference on a sample of validation images and visualize ground truth vs predictions side by side.
    '''
    img_size    = config['data']['image_size']
    data_dir    = Path(config['paths']['data_processed'])
    img_dir     = data_dir / 'images' / split
    label_dir   = data_dir / 'labels' / split

    # load model
    print(f'Loading model from {model_path}')
    model = YOLO(str(model_path))

    # get sample of positive cases(more interesting to visualize)
    all_labels  = list(label_dir.glob('*.txt'))
    positive    = [f for f in all_labels if f.stat().st_size > 0]
    negative    = [f for f in all_labels if f.stat().st_size == 0]

    # mix of positive and negative
    n_pos = int(n_samples * 0.75)
    n_neg = n_samples - n_pos
    import random
    sample_labels = (
        random.sample(positive, min(n_pos, len(positive))) + random.sample(negative, min(n_neg, len(negative)))
    )
    random.shuffle(sample_labels)

    # run inference and plot
    cols = 3
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*6))
    axes = axes.flatten()

    for i, label_path in enumerate(sample_labels):
        pid         = label_path.stem
        img_path    = img_dir / f'{pid}.png'

        if not img_path.exists():
            continue

        # load image
        img = cv2.imread(str(img_path))

        # ground truth
        gt_boxes = load_ground_truth(label_path, img_size)

        # prediction
        results = model(img, verbose=False)[0]
        pred_boxes = results.boxes if results.boxes is not None else []

        # draw
        img_draw = draw_boxes_on_image(img, gt_boxes, pred_boxes, conf_threshold)
        img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

        n_gt = len(gt_boxes)
        n_pred = len([b for b in pred_boxes if b.conf[0] >= conf_threshold]) if len(pred_boxes) > 0 else 0

        axes[i].imshow(img_rgb)
        axes[i].set_title(f'{pid[:12]}...\nGT boxes: {n_gt} | Pred boxes: {n_pred}', fontsize=9)

        axes[i].axis('off')

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f'Ground truth (green) vs Predictions (red)\n'
        f'Model: {model_path.name} | Conf threshold: {conf_threshold}',
        fontsize = 13, fontweight = 'bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('reports/eval_visualization.png', dpi = 150, bbox_inches='tight')
    plt.show()
    print('Saved to reports/eval_visualizations.png')

if __name__ == '__main__':
    config = load_config()

    # point to best weights from run 1
    model_path = Path('runs/detect/weights/yolov8m/weights/best.pt')

    if not model_path.exists():
        raise FileNotFoundError(f'Weights not found at {model_path}')
    
    evaluate_and_visualize(
        model_path      = model_path,
        config          = config,
        n_samples       = 12,
        conf_threshold  = 0.25
    )