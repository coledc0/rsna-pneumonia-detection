from pathlib import Path

import mlflow
import torch
import yaml
from ultralytics import YOLO
import datetime

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_device() -> str:
    if torch.cuda.is_available():
        print(f'using GPU: {torch.cuda.get_device_name(0)}')
        return '0'
    print('GPU not available, using CPU')
    return 'CPU'

def train(config: dict) -> None:
    device = get_device()
    output_dir = Path(config['paths']['data_processed'])
    dataset_yaml = output_dir / 'dataset.yaml'
    weights_dir = Path(config['paths']['weights'])
    weights_dir.mkdir(parents=True, exist_ok=True)

    # verify yaml dataset exists
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f'Dataset YAML not found at {dataset_yaml}.' 'Run src/dataset.py first'
        )
    
    # mlflow setup
    mlflow_dir = Path(config['paths']['mlflow_tracking'])
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{mlflow_dir.resolve()}/mlflow.db')
    mlflow.set_experiment('rsna-pneumonia-yolo')

    # model
    arch = config['model']['architecture']
    model = YOLO(f'{arch}.pt') # downloads pretrained weights if not cached

    run_name = f"{arch}_bs{config['training']['batch_size']}_img{config['data']['image_size']}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run():
        # log config to mlflow
        mlflow.log_params({
            'architecture':     arch,
            'image_size':       config['data']['image_size'],
            'epochs':           config['training']['epochs'],
            'batch_size':       config['training']['batch_size'],
            'learning_rate':    config['training']['learning_rate'],
            'val_split':        config['data']['val_split'],
            'seed':             config['project']['seed'],
        })

        print(f'\nStarting training: {arch}')
        print(f'Dataset: {dataset_yaml}')
        print(f'Epochs: {config["training"]["epochs"]} | 'f'Batch: {config["training"]["batch_size"]} | ' f'Image size: {config["data"]["image_size"]}')
        print('-' * 60)

        results = model.train(
            data            = str(dataset_yaml),
            epochs          = config['training']['epochs'],
            imgsz           = config['data']['image_size'],
            batch           = config['training']['batch_size'],
            lr0             = config['training']['learning_rate'],
            patience        = config['training']['patience'],
            warmup_epochs   = config['training']['warmup_epochs'],
            warmup_momentum = config['training']['warmup_momentum'],
            weight_decay    = config['training']['weight_decay'],
            dropout         = 0.2,
            close_mosaic    = 10,
            device          = device,
            seed            = config['project']['seed'],
            project         = str(weights_dir),
            name            = run_name, 
            exist_ok        = False,
            verbose         = True,

            # class imbalance: weight positive samples higher
            # ~77.5% negative, ~22.5% positive
            cls         = 1.0,
            box         = 7.5,

            # augmentation - conservative for medical imaging
            fliplr      = 0.5,
            flipud      = 0.0,
            degrees     = 15.0,
            translate   = 0.15,
            scale       = 0.3,
            mosaic      = 0.2,
            mixup       = 0.0,
        )

        # log final metrics to mlflow
        metrics = results.results_dict
        mlflow.log_metrics({
            'mAP50':        metrics.get('metrics/mAP50(B)', 0),
            'mAP50_95':     metrics.get('metrics/mAP50-95(B)', 0),
            'precision':    metrics.get('metrics/precision(B)', 0),
            'recall':       metrics.get('metrics/recall(B)', 0),
            'box_loss':     metrics.get('train/box_loss', 0),
            'cls_loss':     metrics.get('train/cls_loss', 0),
        })

        # save best weights path to mlflow
        best_weights = weights_dir / arch / 'weights' / 'best.pt'
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights))
            print(f'\nBest weights saved to: {best_weights}')

        print('\nTraining complete')
        print(f'mAP50:      {metrics.get("metrics/mAP50(B)", 0):.4f}')
        print(f'mAP50-95:   {metrics.get("metrics/mAP50-95(B)", 0):.4f}')
        print(f'Precision:  {metrics.get("metrics/precision(B)", 0):.4f}')
        print(f'Recall:     {metrics.get("metrics/recall(B)", 0):.4f}')

if __name__ == '__main__':
    config = load_config()
    train(config)