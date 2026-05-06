import torch
from pathlib import Path
p = Path('resnet18_seven_segment_best.pt')
if not p.exists():
    print('Checkpoint not found:', p)
else:
    ckpt = torch.load(p, map_location='cpu')
    print('Loaded', p)
    print('keys:', list(ckpt.keys()))
    for k in ['class_to_idx','idx_to_class','classes','args','name']:
        if k in ckpt:
            print(k, ':', ckpt[k])
