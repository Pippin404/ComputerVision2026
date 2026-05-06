import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


class SevenSegmentDataset(Dataset):
	"""Dataset that reads digit labels from class folders or filename prefixes."""

	def __init__(self, root: Path, transform=None, max_per_label: int | None = None, seed: int = 42):
		self.root = root
		self.transform = transform
		self.samples: List[Tuple[Path, int]] = []
		self.class_to_idx: Dict[int, int] = {}

		supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
		raw_samples: List[Tuple[Path, int]] = []

		if not root.exists():
			raise FileNotFoundError(f"Dataset folder not found: {root}")

		class_folders = sorted(
			[path for path in root.iterdir() if path.is_dir() and path.name.isdigit()],
			key=lambda path: int(path.name),
		)

		if class_folders:
			# Preferred layout: one folder per class, named 0-9.
			for class_folder in class_folders:
				label = int(class_folder.name)
				for img_path in sorted(class_folder.rglob("*")):
					if img_path.is_file() and img_path.suffix.lower() in supported_suffixes:
						raw_samples.append((img_path, label))
		else:
			# Backward-compatible fallback for filename-based labels.
			for img_path in sorted(root.rglob("*")):
				if not img_path.is_file() or img_path.suffix.lower() not in supported_suffixes:
					continue

				filename = img_path.name
				if not filename:
					continue

				first_char = filename[0]
				if not first_char.isdigit():
					continue

				raw_samples.append((img_path, int(first_char)))

		if max_per_label is not None:
			grouped: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
			for path, label in raw_samples:
				grouped[label].append((path, label))

			rng = random.Random(seed)
			limited_samples: List[Tuple[Path, int]] = []
			for label, items in grouped.items():
				rng.shuffle(items)
				limited_samples.extend(items[:max_per_label])

			raw_samples = limited_samples

		if not raw_samples:
			raise RuntimeError(
				f"No valid images found in {root}. "
				"Expected image files with a leading digit in the filename."
			)

		unique_labels = sorted({label for _, label in raw_samples})
		self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
		self.samples = [(path, self.class_to_idx[label]) for path, label in raw_samples]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		img_path, label = self.samples[index]

		# Load as grayscale and convert to RGB so ResNet-18 can be used directly.
		image = Image.open(img_path).convert("L").convert("RGB")
		if self.transform is not None:
			image = self.transform(image)

		return image, label


def validate_dataset_root(
	root: Path,
	expected_classes: int = 10,
	expected_min_images_per_class: int | None = None,
	label: str = "dataset",
):
	"""Ensure a root contains digit-named class folders and optional per-class minimums."""
	if not root.exists():
		raise FileNotFoundError(f"{label.capitalize()} folder not found: {root}")

	class_folders = sorted(
		[path for path in root.iterdir() if path.is_dir() and path.name.isdigit()],
		key=lambda path: int(path.name),
	)

	if len(class_folders) != expected_classes:
		found = [folder.name for folder in class_folders]
		raise RuntimeError(
			f"Expected {expected_classes} digit-named class folders in {root}, found {len(class_folders)}: {found}"
		)

	counts = {}
	for class_folder in class_folders:
		label = int(class_folder.name)
		image_count = 0
		for img_path in class_folder.rglob("*"):
			if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
				image_count += 1
		counts[label] = image_count

	if expected_min_images_per_class is not None:
		missing = [class_label for class_label, count in counts.items() if count < expected_min_images_per_class]
		if missing:
			raise RuntimeError(
				f"{label.capitalize()} data under {root} has too few images in classes {missing}. "
				f"Expected at least {expected_min_images_per_class} images per class; counts={counts}"
			)

	print(f"Validated {label}: {root}")
	print(f"Found class folders: {sorted(counts.keys())}")
	print(f"Images per class: {counts}")
	return counts


def validate_training_dataset_root(root: Path, expected_classes: int = 10, expected_min_images_per_class: int = 1000):
	"""Backward-compatible wrapper for strict training validation."""
	return validate_dataset_root(
		root,
		expected_classes=expected_classes,
		expected_min_images_per_class=expected_min_images_per_class,
		label="training dataset",
	)


def build_transforms(img_size: int = 224):
	train_transform = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.RandomAffine(
				degrees=20,
				translate=(0.05, 0.05),
				scale=(0.95, 1.05),
				fill=0,
			),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)

	eval_transform = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)
	return train_transform, eval_transform


def build_model(num_classes: int, pretrained: bool = False):
	weights = models.ResNet18_Weights.DEFAULT if pretrained else None
	model = models.resnet18(weights=weights)
	in_features = model.fc.in_features
	model.fc = nn.Linear(in_features, num_classes)
	return model


def run_epoch(model, loader, criterion, optimizer, device, training: bool, epoch: int, epochs: int):
	if training:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_correct = 0
	total_count = 0

	total_batches = len(loader)
	phase = "train" if training else "val"

	for batch_idx, (images, labels) in enumerate(loader, start=1):
		images = images.to(device)
		labels = labels.to(device)

		if training:
			optimizer.zero_grad()

		with torch.set_grad_enabled(training):
			logits = model(images)
			loss = criterion(logits, labels)

			if training:
				loss.backward()
				optimizer.step()

		total_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		total_correct += (preds == labels).sum().item()
		total_count += images.size(0)

		if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
			percent = (batch_idx / max(total_batches, 1)) * 100
			print(
				f"Epoch {epoch}/{epochs} [{phase}] "
				f"batch {batch_idx}/{total_batches} ({percent:.1f}%)"
			)

	avg_loss = total_loss / max(total_count, 1)
	avg_acc = total_correct / max(total_count, 1)
	return avg_loss, avg_acc


def make_data_loaders(
	train_root: Path,
	val_root: Path,
	batch_size: int,
	img_size: int,
	seed: int,
):
	train_tf, eval_tf = build_transforms(img_size=img_size)
	train_ds = SevenSegmentDataset(train_root, transform=train_tf, max_per_label=25, seed=seed)

	if val_root is not None and val_root.exists():
		# Use explicit validation folder when available.
		val_ds = SevenSegmentDataset(val_root, transform=eval_tf)
	else:
		# Fallback to split training data if no validation folder is provided.
		eval_train_ds = SevenSegmentDataset(train_root, transform=eval_tf)
		val_len = max(1, int(0.2 * len(eval_train_ds)))
		train_len = len(eval_train_ds) - val_len
		generator = torch.Generator().manual_seed(seed)
		train_subset, val_subset = random_split(eval_train_ds, [train_len, val_len], generator=generator)

		# Keep strong augmentation only on training split.
		train_indices = train_subset.indices
		val_indices = val_subset.indices

		train_ds_full = SevenSegmentDataset(train_root, transform=train_tf)
		val_ds_full = SevenSegmentDataset(train_root, transform=eval_tf)

		train_ds = torch.utils.data.Subset(train_ds_full, train_indices)
		val_ds = torch.utils.data.Subset(val_ds_full, val_indices)

	num_workers = 0 if not torch.cuda.is_available() else 2
	pin_memory = torch.cuda.is_available()
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
	return train_loader, val_loader


def load_checkpoint(checkpoint_path: Path, device: torch.device):
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location=device)
	class_to_idx = checkpoint["class_to_idx"]
	idx_to_class = checkpoint["idx_to_class"]
	img_size = checkpoint.get("img_size", 224)

	model = build_model(num_classes=len(class_to_idx), pretrained=False).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	return model, class_to_idx, idx_to_class, img_size


def evaluate_checkpoint(args, device: torch.device):
	model, _, _, ckpt_img_size = load_checkpoint(args.checkpoint_path, device)
	img_size = args.img_size if args.img_size is not None else ckpt_img_size
	_, eval_tf = build_transforms(img_size=img_size)

	val_ds = SevenSegmentDataset(args.val_dir, transform=eval_tf)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
	criterion = nn.CrossEntropyLoss()

	val_loss, val_acc = run_epoch(
		model,
		val_loader,
		criterion,
		optimizer=None,
		device=device,
		training=False,
		epoch=1,
		epochs=1,
	)
	print(f"Test results | loss={val_loss:.4f}, acc={val_acc:.4f}")


def predict_single_image(args, device: torch.device):
	if args.image_path is None:
		raise ValueError("--image-path is required when --mode predict")

	model, _, idx_to_class, ckpt_img_size = load_checkpoint(args.checkpoint_path, device)
	img_size = args.img_size if args.img_size is not None else ckpt_img_size
	_, eval_tf = build_transforms(img_size=img_size)

	image = Image.open(args.image_path).convert("L").convert("RGB")
	tensor = eval_tf(image).unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(tensor)
		pred_idx = int(torch.argmax(logits, dim=1).item())
		probs = torch.softmax(logits, dim=1)
		confidence = float(probs[0, pred_idx].item())

	pred_label = idx_to_class.get(pred_idx, pred_idx)
	print(f"Prediction: {pred_label} (confidence={confidence:.4f})")


def parse_args():
	parser = argparse.ArgumentParser(description="Train ResNet-18 on seven-segment images.")
	parser.add_argument(
		"--mode",
		type=str,
		choices=["train", "test", "predict"],
		default="train",
		help="train: fit model, test: evaluate checkpoint, predict: classify one image",
	)
	parser.add_argument(
		"--train-dir",
		type=Path,
		default=Path(r"data/dataset/dataset/train"),
		help="Path to training image root.",
	)
	parser.add_argument(
		"--val-dir",
		type=Path,
		default=Path(r"data/dataset/dataset/validation"),
		help="Path to validation image root. If missing, script uses train split.",
	)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--img-size", type=int, default=None, help="Override image size. Uses checkpoint size for test/predict.")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
	parser.add_argument(
		"--save-path",
		type=Path,
		default=Path("resnet18_seven_segment_best.pt"),
		help="Where to save the best model checkpoint.",
	)
	parser.add_argument(
		"--checkpoint-path",
		type=Path,
		default=Path("resnet18_seven_segment_best.pt"),
		help="Checkpoint path to load for test/predict.",
	)
	parser.add_argument(
		"--image-path",
		type=Path,
		default=None,
		help="Single image path for --mode predict.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	if args.mode == "test":
		evaluate_checkpoint(args, device)
		return

	if args.mode == "predict":
		predict_single_image(args, device)
		return

	img_size = args.img_size if args.img_size is not None else 224
	validate_training_dataset_root(args.train_dir, expected_classes=10, expected_min_images_per_class=1000)
	validate_dataset_root(args.val_dir, expected_classes=10, expected_min_images_per_class=1, label="validation dataset")

	train_loader, val_loader = make_data_loaders(
		train_root=args.train_dir,
		val_root=args.val_dir,
		batch_size=args.batch_size,
		img_size=img_size,
		seed=args.seed,
	)

	# Infer class count from labels found in filenames.
	sample_dataset = SevenSegmentDataset(args.train_dir)
	num_classes = len(sample_dataset.class_to_idx)
	idx_to_class = {v: k for k, v in sample_dataset.class_to_idx.items()}
	print(f"Detected labels: {sorted(sample_dataset.class_to_idx.keys())}")

	model = build_model(num_classes=num_classes, pretrained=args.pretrained).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	best_val_acc = 0.0

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True, epoch=epoch, epochs=args.epochs)
		val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, training=False, epoch=epoch, epochs=args.epochs)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
		)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			checkpoint = {
				"model_state_dict": model.state_dict(),
				"class_to_idx": sample_dataset.class_to_idx,
				"idx_to_class": idx_to_class,
				"img_size": img_size,
			}
			torch.save(checkpoint, args.save_path)
			print(f"Saved new best model to {args.save_path} (val_acc={best_val_acc:.4f})")

	print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
	main()
