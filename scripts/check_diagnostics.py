from pathlib import Path
from collections import Counter
import torch
import sys

sys.path.append(str(Path(__file__).parents[1]))

from evaluate_model import ValidationDataset, build_transforms
from torchvision import models
import numpy as np

print('=== Validation dataset check ===')
val_root = Path('data/dataset/dataset/validation')
# If the validation folder was moved into the train folder, use that as a fallback.
if not val_root.exists():
    alt = Path('data/dataset/dataset/train')
    if alt.exists():
        print(f'Validation path {val_root} not found; using fallback {alt}')
        val_root = alt
    else:
        print('Validation path not found:', val_root)

ds = ValidationDataset(val_root, transform=build_transforms())
print('class_to_idx:', ds.class_to_idx)
print('idx_to_class:', ds.idx_to_class)
counts = Counter([lbl for _, lbl in ds.samples])
print('per-class counts (dataset labels -> counts):')
for k in sorted(counts.keys()):
    print(f'  {k}: {counts[k]}')

print('\n=== Checkpoint contents ===')
ckpt_path = Path('resnet18_seven_segment_best.pt')
if not ckpt_path.exists():
    print('Checkpoint not found at', ckpt_path)
else:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('checkpoint keys:', list(ckpt.keys()))
    # print a few keys if present
    for key in ['class_to_idx', 'idx_to_class', 'classes', 'args', 'name']:
        if key in ckpt:
            print(f'{key}:', ckpt[key])

print('\n=== Model predictions vs labels (if checkpoint loadable) ===')
# Try to construct model and run predictions
try:
    if not ckpt_path.exists():
        raise FileNotFoundError('no checkpoint')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # determine num_classes
    if 'classes' in ckpt and isinstance(ckpt['classes'], (list, tuple)):
        num_classes = len(ckpt['classes'])
    elif 'args' in ckpt and 'num_classes' in ckpt['args']:
        num_classes = int(ckpt['args']['num_classes'])
    else:
        # fallback to dataset mapping length
        try:
            num_classes = len(ds.class_to_idx)
        except Exception:
            num_classes = None

    if num_classes is None:
        print('Could not determine num_classes for model loading; skipping preds check.')
    else:
        print('Attempting to build ResNet18 with', num_classes, 'output classes')
        model = models.resnet18(weights=None)
        import torch.nn as nn
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print('Model state_dict loaded successfully.')

            # build dataloader
            from torch.utils.data import DataLoader, Subset

            # Deterministic check: evaluate each class (0-9) up to 29 times.
            target_per_class = 29
            class_indices = {i: [] for i in range(10)}
            for sample_idx, (_, label_idx) in enumerate(ds.samples):
                if label_idx in class_indices and len(class_indices[label_idx]) < target_per_class:
                    class_indices[label_idx].append(sample_idx)

            selected_indices = []
            print(f'Selecting up to {target_per_class} samples per class (0-9):')
            for class_id in range(10):
                n_selected = len(class_indices[class_id])
                selected_indices.extend(class_indices[class_id])
                print(f'  class {class_id}: {n_selected} selected')

            if not selected_indices:
                print('No matching samples found for classes 0-9; skipping preds check.')
                raise RuntimeError('empty selected subset for deterministic check')

            subset = Subset(ds, selected_indices)
            loader = DataLoader(subset, batch_size=64, shuffle=False)
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in loader:
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            print('unique preds:', np.unique(all_preds))
            print('unique labels:', np.unique(all_labels))
            print('per_class_total from dataset mapping (label idx -> count):')
            for k in sorted(counts.keys()):
                print(f'  {k}: {counts[k]}')
        except Exception as e:
            print('Failed to load model_state_dict into constructed model:', e)
except Exception as e:
    print('Skipping model prediction check:', e)

print('\nDiagnostics complete.')
