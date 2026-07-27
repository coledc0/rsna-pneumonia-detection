from pathlib import Path
import pandas as pd
import torch


def _is_valid_checkpoint(weights_path: Path) -> bool:
    '''
    Quickly verify a .pt file is loadable before trusting it.
    Returns False if the file is corrupted or incomplete.
    '''
    try:
        torch.load(weights_path, map_location='cpu', weights_only=False)
        return True
    except Exception as e:
        print(f'  WARNING: {weights_path} failed to load ({e}) - skipping')
        return False


def find_best_run(weights_dir: Path, metric: str = 'metrics/mAP50(B)') -> Path:
    '''
    Scan all run folders under weights_dir, read each results.csv,
    and return the path to the best.pt of whichever *valid* run
    achieved the highest value of `metric`.

    Runs with corrupted or unreadable checkpoints are skipped
    and the next-best valid run is used instead.
    '''
    weights_dir = Path(weights_dir)

    # collect all candidate runs sorted by score, best first
    candidates = []

    for run_dir in weights_dir.iterdir():
        if not run_dir.is_dir():
            continue

        results_csv = run_dir / 'results.csv'
        weights_path = run_dir / 'weights' / 'best.pt'

        if not results_csv.exists() or not weights_path.exists():
            continue

        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            if metric not in df.columns:
                continue

            run_best_score = df[metric].max()
            candidates.append((run_best_score, run_dir.name, weights_path))

        except Exception as e:
            print(f'Skipping {run_dir.name}: could not read results.csv ({e})')
            continue

    if not candidates:
        raise FileNotFoundError(
            f'No valid runs found in {weights_dir} with metric {metric}'
        )

    # sort descending by score, best first
    candidates.sort(key=lambda x: x[0], reverse=True)

    # walk down the sorted list until we find one that actually loads
    for score, run_name, weights_path in candidates:
        print(f'Checking {run_name} ({metric}={score:.4f})...')
        if _is_valid_checkpoint(weights_path):
            print(f'Selected run: {run_name} ({metric}={score:.4f})')
            return weights_path
        else:
            print(f'  {run_name} has a corrupted checkpoint, trying next best...')

    raise FileNotFoundError(
        'All candidate runs had corrupted checkpoints. '
        'No valid weights file could be loaded.'
    )


if __name__ == '__main__':
    best = find_best_run(Path('runs/detect/weights'))
    print(f'Best weights: {best}')