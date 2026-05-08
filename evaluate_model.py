import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

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

	def __init__(self, root: Path, transform=None, samples_per_class: int | None = None):
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

		# Deterministic evaluation subset: classes 0-9, up to N samples per class.
		if samples_per_class is not None:
			samples_by_idx = defaultdict(list)
			for img_path, class_idx in self.samples:
				samples_by_idx[class_idx].append((img_path, class_idx))

			selected_samples = []
			print(f"Selecting up to {samples_per_class} samples per class for digits 0-9...")
			for digit in range(10):
				if digit not in self.class_to_idx:
					print(f"  class {digit}: not present")
					continue
				class_idx = self.class_to_idx[digit]
				chosen = samples_by_idx[class_idx][:samples_per_class]
				selected_samples.extend(chosen)
				print(f"  class {digit}: {len(chosen)} selected")

			if not selected_samples:
				raise RuntimeError("No samples selected for classes 0-9.")

			self.samples = selected_samples

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
	"""Evaluate model and compute comprehensive metrics."""
	model.eval()

	all_preds = []
	all_labels = []
	all_probs = []
	per_class_correct = defaultdict(int)
	per_class_total = defaultdict(int)
	false_positives = defaultdict(lambda: defaultdict(int))  # predicted -> actual
	false_negatives = defaultdict(lambda: defaultdict(int))  # actual -> predicted

	with torch.no_grad():
		for batch_idx, (images, labels) in enumerate(loader):
			print(f"Processing batch {batch_idx + 1}/{len(loader)}...", end='\r')
			images = images.to(device)
			labels = labels.to(device)

			logits = model(images)
			probs = torch.softmax(logits, dim=1)
			preds = logits.argmax(dim=1)

			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
			all_probs.extend(probs.cpu().numpy())

			for pred, label in zip(preds, labels):
				pred_digit = idx_to_class[pred.item()]
				label_digit = idx_to_class[label.item()]
				per_class_total[label_digit] += 1

				if pred == label:
					per_class_correct[label_digit] += 1
				else:
					false_positives[pred_digit][label_digit] += 1
					false_negatives[label_digit][pred_digit] += 1

	print()  # New line after progress
	all_preds = np.array(all_preds)
	all_labels = np.array(all_labels)
	all_probs = np.array(all_probs)

	return all_preds, all_labels, all_probs, per_class_correct, per_class_total, false_positives, false_negatives


def compute_metrics(all_preds, all_labels, all_probs, per_class_correct, per_class_total, idx_to_class):
	"""Compute overall and per-class metrics."""
	num_classes = len(idx_to_class)

	# Overall accuracy
	overall_accuracy = (all_preds == all_labels).sum() / len(all_labels)

	# Per-class metrics
	per_class_accuracy = {}
	per_class_precision = {}
	per_class_recall = {}
	per_class_f1 = {}

	for class_idx in range(num_classes):
		class_label = idx_to_class[class_idx]

		# Accuracy
		if per_class_total[class_label] > 0:
			per_class_accuracy[class_label] = per_class_correct[class_label] / per_class_total[class_label]
		else:
			per_class_accuracy[class_label] = 0.0

		# Precision: TP / (TP + FP)
		tp = per_class_correct[class_label]
		fp = (all_preds == class_idx).sum() - tp
		if (tp + fp) > 0:
			per_class_precision[class_label] = tp / (tp + fp)
		else:
			per_class_precision[class_label] = 0.0

		# Recall: TP / (TP + FN)
		fn = per_class_total[class_label] - tp
		if (tp + fn) > 0:
			per_class_recall[class_label] = tp / (tp + fn)
		else:
			per_class_recall[class_label] = 0.0

		# F1 Score
		precision = per_class_precision[class_label]
		recall = per_class_recall[class_label]
		if (precision + recall) > 0:
			per_class_f1[class_label] = 2 * (precision * recall) / (precision + recall)
		else:
			per_class_f1[class_label] = 0.0

	return {
		"overall_accuracy": overall_accuracy,
		"per_class_accuracy": per_class_accuracy,
		"per_class_precision": per_class_precision,
		"per_class_recall": per_class_recall,
		"per_class_f1": per_class_f1,
	}


def compute_confusion_matrix(all_preds, all_labels, num_classes):
	"""Compute confusion matrix."""
	cm = np.zeros((num_classes, num_classes), dtype=int)
	for pred, label in zip(all_preds, all_labels):
		cm[label, pred] += 1
	return cm


def print_results(metrics, confusion_matrix, false_positives, false_negatives, idx_to_class):
	print("\n" + "=" * 80)
	print("OVERALL PERFORMANCE")
	print("=" * 80)
	print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")

	print("\n" + "=" * 80)
	print("PER-CLASS METRICS")
	print("=" * 80)
	print(f"{'Class':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
	print("-" * 56)

	for class_label in sorted(idx_to_class.values()):
		acc = metrics["per_class_accuracy"][class_label]
		prec = metrics["per_class_precision"][class_label]
		rec = metrics["per_class_recall"][class_label]
		f1 = metrics["per_class_f1"][class_label]
		print(f"{class_label:<8} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

	print("\n" + "=" * 80)
	print("CONFUSION MATRIX")
	print("=" * 80)
	num_classes = len(idx_to_class)
	print(f"{'Pred/True':<12}", end="")
	for i in range(num_classes):
		print(f"{i:<6}", end="")
	print()
	print("-" * (12 + 6 * num_classes))

	for i in range(num_classes):
		print(f"{i:<12}", end="")
		for j in range(num_classes):
			print(f"{confusion_matrix[i, j]:<6}", end="")
		print()

	print("\n" + "=" * 80)
	print("FALSE POSITIVES (Predicted → Actual counts)")
	print("=" * 80)
	for pred_digit in sorted(false_positives.keys()):
		fp_dict = false_positives[pred_digit]
		if fp_dict:
			print(f"\nPredicted as {pred_digit}:")
			for actual_digit, count in sorted(fp_dict.items()):
				print(f"  Actually {actual_digit}: {count} instances")

	print("\n" + "=" * 80)
	print("FALSE NEGATIVES (Actual → Predicted counts)")
	print("=" * 80)
	for actual_digit in sorted(false_negatives.keys()):
		fn_dict = false_negatives[actual_digit]
		if fn_dict:
			print(f"\nActually {actual_digit}:")
			for pred_digit, count in sorted(fn_dict.items()):
				print(f"  Predicted as {pred_digit}: {count} instances")

	print("\n" + "=" * 80)


def save_visualizations(metrics, confusion_matrix, false_positives, false_negatives, idx_to_class, output_dir: Path = Path("evaluation_results")):
	"""Generate and save matplotlib visualizations."""
	output_dir.mkdir(exist_ok=True)
	
	class_labels = sorted(idx_to_class.values())
	num_classes = len(class_labels)

	# 1. Per-class accuracy bar plot
	print(f"\nGenerating visualizations in {output_dir}...")
	fig, ax = plt.subplots(figsize=(12, 6))
	accuracies = [metrics["per_class_accuracy"][c] for c in class_labels]
	bars = ax.bar([str(c) for c in class_labels], accuracies, color='steelblue', edgecolor='black')
	ax.set_xlabel("Class", fontsize=12, fontweight='bold')
	ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
	ax.set_title("Per-Class Accuracy", fontsize=14, fontweight='bold')
	ax.set_ylim([0, 1])
	ax.axhline(y=metrics["overall_accuracy"], color='r', linestyle='--', linewidth=2, label=f"Overall: {metrics['overall_accuracy']:.4f}")
	ax.legend()
	for bar in bars:
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width()/2., height,
			f'{height:.3f}', ha='center', va='bottom', fontsize=10)
	plt.tight_layout()
	plt.savefig(output_dir / "01_per_class_accuracy.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 2. Precision, Recall, F1 comparison
	fig, ax = plt.subplots(figsize=(14, 6))
	x = np.arange(num_classes)
	width = 0.25
	precisions = [metrics["per_class_precision"][c] for c in class_labels]
	recalls = [metrics["per_class_recall"][c] for c in class_labels]
	f1_scores = [metrics["per_class_f1"][c] for c in class_labels]
	
	ax.bar(x - width, precisions, width, label='Precision', color='skyblue', edgecolor='black')
	ax.bar(x, recalls, width, label='Recall', color='lightcoral', edgecolor='black')
	ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen', edgecolor='black')
	
	ax.set_xlabel("Class", fontsize=12, fontweight='bold')
	ax.set_ylabel("Score", fontsize=12, fontweight='bold')
	ax.set_title("Precision, Recall, and F1-Score by Class", fontsize=14, fontweight='bold')
	ax.set_xticks(x)
	ax.set_xticklabels([str(c) for c in class_labels])
	ax.set_ylim([0, 1])
	ax.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "02_precision_recall_f1.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 3. Confusion matrix heatmap
	fig, ax = plt.subplots(figsize=(10, 8))
	sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
				xticklabels=class_labels, yticklabels=class_labels, ax=ax, cbar_kws={'label': 'Count'})
	ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
	ax.set_ylabel("Actual", fontsize=12, fontweight='bold')
	ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_dir / "03_confusion_matrix.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 4. False positives heatmap
	fp_matrix = np.zeros((num_classes, num_classes), dtype=int)
	for pred_digit in false_positives.keys():
		for actual_digit, count in false_positives[pred_digit].items():
			pred_idx = class_labels.index(pred_digit)
			actual_idx = class_labels.index(actual_digit)
			fp_matrix[actual_idx, pred_idx] = count
	
	if fp_matrix.sum() > 0:
		fig, ax = plt.subplots(figsize=(10, 8))
		sns.heatmap(fp_matrix, annot=True, fmt='d', cmap='Reds', 
					xticklabels=class_labels, yticklabels=class_labels, ax=ax, cbar_kws={'label': 'Count'})
		ax.set_xlabel("Predicted (Incorrect)", fontsize=12, fontweight='bold')
		ax.set_ylabel("Actual (True Class)", fontsize=12, fontweight='bold')
		ax.set_title("False Positives Heatmap", fontsize=14, fontweight='bold')
		plt.tight_layout()
		plt.savefig(output_dir / "04_false_positives_heatmap.png", dpi=150, bbox_inches='tight')
		plt.close()

	# 5. Overall accuracy gauge
	fig, ax = plt.subplots(figsize=(8, 6))
	accuracy = metrics["overall_accuracy"]
	colors = ['#2ecc71' if accuracy > 0.8 else '#f39c12' if accuracy > 0.6 else '#e74c3c']
	wedges, texts, autotexts = ax.pie([accuracy, 1-accuracy], 
										labels=['Correct', 'Incorrect'],
										colors=[colors[0], '#ecf0f1'],
										autopct='%1.2f%%',
										startangle=90,
										textprops={'fontsize': 12, 'fontweight': 'bold'})
	ax.set_title(f"Overall Accuracy: {accuracy:.4f}", fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_dir / "05_overall_accuracy.png", dpi=150, bbox_inches='tight')
	plt.close()

	# 6. Misclassification breakdown
	total_samples = confusion_matrix.sum()
	correct = np.trace(confusion_matrix)
	incorrect = total_samples - correct
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	# Pie chart
	colors_pie = ['#2ecc71', '#e74c3c']
	ax1.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
			autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
	ax1.set_title(f"Classification Results\n(Total: {total_samples} samples)", fontsize=12, fontweight='bold')
	
	# Bar chart
	categories = ['Correct', 'Incorrect']
	counts = [correct, incorrect]
	bars = ax2.bar(categories, counts, color=colors_pie, edgecolor='black', linewidth=2)
	ax2.set_ylabel("Count", fontsize=12, fontweight='bold')
	ax2.set_title("Misclassification Breakdown", fontsize=12, fontweight='bold')
	for bar in bars:
		height = bar.get_height()
		ax2.text(bar.get_x() + bar.get_width()/2., height,
			f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
	
	plt.tight_layout()
	plt.savefig(output_dir / "06_misclassification_breakdown.png", dpi=150, bbox_inches='tight')
	plt.close()

	print(f"✓ Visualizations saved to {output_dir}/")
	print(f"  - 01_per_class_accuracy.png")
	print(f"  - 02_precision_recall_f1.png")
	print(f"  - 03_confusion_matrix.png")
	if fp_matrix.sum() > 0:
		print(f"  - 04_false_positives_heatmap.png")
	print(f"  - 05_overall_accuracy.png")
	print(f"  - 06_misclassification_breakdown.png")


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate ResNet-18 seven-segment model.")
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
		"--samples-per-class",
		type=int,
		default=None,
		help="Optional limit per class. Omit to evaluate every image in each class folder.",
	)
	parser.add_argument(
		"--save-results",
		type=Path,
		default=None,
		help="Optional: save results to JSON file.",
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
	val_dataset = ValidationDataset(args.val_dir, transform=eval_tf, samples_per_class=args.samples_per_class)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

	# Evaluate model
	print("\nEvaluating model...")
	all_preds, all_labels, all_probs, per_class_correct, per_class_total, false_positives, false_negatives = evaluate_model(
		model, val_loader, device, idx_to_class
	)

	# Compute metrics
	print("Computing metrics...")
	metrics = compute_metrics(all_preds, all_labels, all_probs, per_class_correct, per_class_total, idx_to_class)

	# Compute confusion matrix
	num_classes = len(idx_to_class)
	confusion_matrix = compute_confusion_matrix(all_preds, all_labels, num_classes)

	# Print results
	print_results(metrics, confusion_matrix, false_positives, false_negatives, idx_to_class)

	# Generate and save visualizations
	save_visualizations(metrics, confusion_matrix, false_positives, false_negatives, idx_to_class)

	# Save results if requested
	if args.save_results is not None:
		results_dict = {
			"overall_accuracy": float(metrics["overall_accuracy"]),
			"per_class_accuracy": {str(k): float(v) for k, v in metrics["per_class_accuracy"].items()},
			"per_class_precision": {str(k): float(v) for k, v in metrics["per_class_precision"].items()},
			"per_class_recall": {str(k): float(v) for k, v in metrics["per_class_recall"].items()},
			"per_class_f1": {str(k): float(v) for k, v in metrics["per_class_f1"].items()},
			"confusion_matrix": confusion_matrix.tolist(),
			"false_positives": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in false_positives.items()},
			"false_negatives": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in false_negatives.items()},
		}
		with open(args.save_results, "w") as f:
			json.dump(results_dict, f, indent=2)
		print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
	main()
