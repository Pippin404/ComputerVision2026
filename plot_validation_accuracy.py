import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ValidationDataset(Dataset):
	"""Dataset that loads images from class folders (0-9)."""

	def __init__(self, root: Path, transform=None):
		self.root = root
		self.transform = transform
		self.samples = []
		self.class_to_idx = {}
		self.idx_to_class = {}

		supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

		# Load images from class folders
		class_folders = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])

		if not class_folders:
			raise RuntimeError(
				f"No class folders found in {root}. "
				"Expected folders named 0-9 containing images."
			)

		for class_idx, class_folder in enumerate(class_folders):
			class_label = int(class_folder.name)
			self.class_to_idx[class_label] = class_idx
			self.idx_to_class[class_idx] = class_label

			for img_path in sorted(class_folder.rglob("*")):
				if not img_path.is_file() or img_path.suffix.lower() not in supported_suffixes:
					continue
				self.samples.append((img_path, class_idx))

		if not self.samples:
			raise RuntimeError(f"No valid images found in {root}")

		print(f"Loaded {len(self.samples)} images from {len(class_folders)} classes")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		img_path, label = self.samples[index]
		image = Image.open(img_path).convert("L").convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, label


def build_transforms(img_size: int = 224):
	eval_transform = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)
	return eval_transform


def build_model(num_classes: int, pretrained: bool = False):
	weights = models.ResNet18_Weights.DEFAULT if pretrained else None
	model = models.resnet18(weights=weights)
	in_features = model.fc.in_features
	model.fc = nn.Linear(in_features, num_classes)
	return model


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


def evaluate_model(model, loader, device, idx_to_class):
	"""Evaluate model and return overall and per-class accuracy."""
	model.eval()

	all_preds = []
	all_labels = []
	per_class_correct = defaultdict(int)
	per_class_total = defaultdict(int)

	with torch.no_grad():
		for batch_idx, (images, labels) in enumerate(loader):
			print(f"  Batch {batch_idx + 1}/{len(loader)}...", end='\r')
			images = images.to(device)
			labels = labels.to(device)

			logits = model(images)
			preds = logits.argmax(dim=1)

			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

			for pred, label in zip(preds, labels):
				label_digit = idx_to_class[label.item()]
				per_class_total[label_digit] += 1

				if pred == label:
					per_class_correct[label_digit] += 1

	print()  # New line after progress
	all_preds = np.array(all_preds)
	all_labels = np.array(all_labels)

	# Overall accuracy
	overall_accuracy = (all_preds == all_labels).sum() / len(all_labels)

	# Per-class accuracy
	per_class_accuracy = {}
	for class_label in idx_to_class.values():
		if per_class_total[class_label] > 0:
			per_class_accuracy[class_label] = per_class_correct[class_label] / per_class_total[class_label]
		else:
			per_class_accuracy[class_label] = 0.0

	return overall_accuracy, per_class_accuracy


def run_multiple_evaluations(model, val_dataset, batch_size, num_runs, device, idx_to_class):
	"""Run evaluation multiple times with different batch shuffles."""
	overall_accuracies = []
	per_class_accuracies_list = []

	for run in range(num_runs):
		print(f"\nRun {run + 1}/{num_runs}...")
		# Create loader with shuffle=True each time to get different batch orders
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
		overall_acc, per_class_acc = evaluate_model(model, val_loader, device, idx_to_class)
		overall_accuracies.append(overall_acc)
		per_class_accuracies_list.append(per_class_acc)
		print(f"  Overall Accuracy: {overall_acc:.4f}")

	return overall_accuracies, per_class_accuracies_list


def plot_accuracy_over_runs(overall_accuracies, per_class_accuracies_list, idx_to_class, output_dir: Path = Path("evaluation_results")):
	"""Generate visualizations for accuracy across runs."""
	output_dir.mkdir(exist_ok=True)
	num_runs = len(overall_accuracies)
	run_numbers = list(range(1, num_runs + 1))

	# 1. Overall accuracy line plot
	print(f"\nGenerating visualizations in {output_dir}...")
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(run_numbers, overall_accuracies, marker='o', linewidth=2, markersize=8, color='steelblue', label='Validation Accuracy')
	ax.axhline(y=np.mean(overall_accuracies), color='r', linestyle='--', linewidth=2, label=f"Mean: {np.mean(overall_accuracies):.4f}")
	ax.axhline(y=np.max(overall_accuracies), color='g', linestyle=':', linewidth=2, alpha=0.7, label=f"Max: {np.max(overall_accuracies):.4f}")
	ax.axhline(y=np.min(overall_accuracies), color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f"Min: {np.min(overall_accuracies):.4f}")
	
	ax.set_xlabel("Evaluation Run", fontsize=12, fontweight='bold')
	ax.set_ylabel("Overall Accuracy", fontsize=12, fontweight='bold')
	ax.set_title("Validation Accuracy Across Evaluation Runs", fontsize=14, fontweight='bold')
	ax.set_xticks(run_numbers)
	ax.set_ylim([0, 1])
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "01_overall_accuracy_over_runs.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 2. Per-class accuracy heatmap across runs
	class_labels = sorted(idx_to_class.values())
	per_class_matrix = np.zeros((num_runs, len(class_labels)))
	
	for run_idx, per_class_dict in enumerate(per_class_accuracies_list):
		for col_idx, class_label in enumerate(class_labels):
			per_class_matrix[run_idx, col_idx] = per_class_dict.get(class_label, 0.0)

	fig, ax = plt.subplots(figsize=(12, 6))
	sns.heatmap(per_class_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
				xticklabels=class_labels, yticklabels=[f"Run {i+1}" for i in range(num_runs)],
				ax=ax, cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
	ax.set_xlabel("Class", fontsize=12, fontweight='bold')
	ax.set_ylabel("Evaluation Run", fontsize=12, fontweight='bold')
	ax.set_title("Per-Class Accuracy Across Runs", fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_dir / "02_per_class_accuracy_heatmap.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 3. Box plot for per-class accuracy distribution
	fig, ax = plt.subplots(figsize=(12, 6))
	per_class_data = [per_class_matrix[:, i] for i in range(len(class_labels))]
	bp = ax.boxplot(per_class_data, labels=[str(c) for c in class_labels], patch_artist=True)
	
	for patch in bp['boxes']:
		patch.set_facecolor('lightblue')
	
	ax.set_xlabel("Class", fontsize=12, fontweight='bold')
	ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
	ax.set_title("Per-Class Accuracy Distribution Across Runs", fontsize=14, fontweight='bold')
	ax.set_ylim([0, 1])
	ax.grid(True, alpha=0.3, axis='y')
	plt.tight_layout()
	plt.savefig(output_dir / "03_per_class_boxplot.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 4. Statistics summary
	print("\n" + "=" * 80)
	print("VALIDATION ACCURACY STATISTICS")
	print("=" * 80)
	print(f"Number of Evaluation Runs: {num_runs}")
	print(f"Overall Mean Accuracy: {np.mean(overall_accuracies):.4f}")
	print(f"Overall Std Dev: {np.std(overall_accuracies):.4f}")
	print(f"Overall Min: {np.min(overall_accuracies):.4f}")
	print(f"Overall Max: {np.max(overall_accuracies):.4f}")
	print(f"Overall Range: {np.max(overall_accuracies) - np.min(overall_accuracies):.4f}")
	print("\n" + "=" * 80)
	print("PER-CLASS STATISTICS")
	print("=" * 80)
	print(f"{'Class':<8} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
	print("-" * 56)
	for col_idx, class_label in enumerate(class_labels):
		class_accs = per_class_matrix[:, col_idx]
		print(f"{class_label:<8} {np.mean(class_accs):<12.4f} {np.std(class_accs):<12.4f} {np.min(class_accs):<12.4f} {np.max(class_accs):<12.4f}")
	print("=" * 80)

	print(f"\n✓ Visualizations saved to {output_dir}/")
	print(f"  - 01_overall_accuracy_over_runs.png")
	print(f"  - 02_per_class_accuracy_heatmap.png")
	print(f"  - 03_per_class_boxplot.png")


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate model accuracy across multiple runs.")
	parser.add_argument(
		"--checkpoint-path",
		type=Path,
		default=Path("resnet18_seven_segment_best.pt"),
		help="Path to the trained model checkpoint.",
	)
	parser.add_argument(
		"--val-dir",
		type=Path,
		default=Path(r"data/dataset/dataset/validation"),
		help="Path to validation data root (with class folders 0-9).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size for evaluation.",
	)
	parser.add_argument(
		"--num-runs",
		type=int,
		default=5,
		help="Number of evaluation runs (with different batch shuffles).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("evaluation_results"),
		help="Directory to save visualizations.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Load checkpoint and model
	print(f"\nLoading checkpoint from {args.checkpoint_path}...")
	model, class_to_idx, idx_to_class, img_size = load_checkpoint(args.checkpoint_path, device)
	print(f"Model loaded. Classes: {sorted(idx_to_class.values())}")

	# Load validation dataset
	print(f"\nLoading validation data from {args.val_dir}...")
	eval_tf = build_transforms(img_size=img_size)
	val_dataset = ValidationDataset(args.val_dir, transform=eval_tf)

	# Run multiple evaluations
	print(f"\nRunning {args.num_runs} evaluation runs...")
	overall_accuracies, per_class_accuracies_list = run_multiple_evaluations(
		model, val_dataset, args.batch_size, args.num_runs, device, idx_to_class
	)

	# Generate visualizations
	plot_accuracy_over_runs(overall_accuracies, per_class_accuracies_list, idx_to_class, args.output_dir)


if __name__ == "__main__":
	main()
