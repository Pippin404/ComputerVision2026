import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parents[1]))

from evaluate_model import ValidationDataset, build_transforms, build_model, evaluate_model, compute_metrics, compute_confusion_matrix, print_results, save_visualizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='resnet18_seven_segment_best.pt')
    parser.add_argument('--data-dir', type=str, default='data/dataset/dataset')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f'Checkpoint not found: {ckpt_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Determine class mapping
    class_to_idx = None
    idx_to_class = None
    if 'class_to_idx' in ckpt and 'idx_to_class' in ckpt:
        class_to_idx = ckpt['class_to_idx']
        idx_to_class = ckpt['idx_to_class']
    elif 'classes' in ckpt and isinstance(ckpt['classes'], (list, tuple)):
        classes = list(ckpt['classes'])
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        idx_to_class = {i: int(c) for i, c in enumerate(classes)}
    else:
        # fallback: build dataset mapping from validation (or train) folder
        val_root = Path(args.data_dir) / 'validation'
        if not val_root.exists():
            val_root = Path(args.data_dir) / 'train'
        ds_tmp = ValidationDataset(val_root, transform=build_transforms(args.image_size))
        class_to_idx = ds_tmp.class_to_idx
        idx_to_class = ds_tmp.idx_to_class

    num_classes = len(class_to_idx)
    print('Num classes:', num_classes)

    # Build model and load weights
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    # Dataset and loader
    val_root = Path(args.data_dir) / 'validation'
    if not val_root.exists():
        val_root = Path(args.data_dir) / 'train'
    ds = ValidationDataset(val_root, transform=build_transforms(args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels, all_probs, per_class_correct, per_class_total, false_positives, false_negatives = evaluate_model(model, loader, device, idx_to_class)

    metrics = compute_metrics(all_preds, all_labels, all_probs, per_class_correct, per_class_total, idx_to_class)
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)

    print_results(metrics, cm, false_positives, false_negatives, idx_to_class)
    save_visualizations(metrics, cm, false_positives, false_negatives, idx_to_class, output_dir=Path('evaluation_results'))


if __name__ == '__main__':
    main()
