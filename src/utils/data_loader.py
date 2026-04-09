import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _get_pickle_value(obj, key):
	if key in obj:
		return obj[key]
	bkey = key.encode("utf-8")
	if bkey in obj:
		return obj[bkey]
	raise KeyError(f"Missing key '{key}' in pickle object")


def _resolve_kaggle_paths(kaggle_root):
	# Expected structure follows Kaggle's ArtBench-10 dump layout.
	root = Path(kaggle_root)
	csv_path = root / "ArtBench-10.csv"
	batch_dir = root / "artbench-10-python" / "artbench-10-batches-py"
	return root, csv_path, batch_dir


def _load_kaggle_batches(batch_dir):
	def _load_batch(path):
		with open(path, "rb") as f:
			batch = pickle.load(f)
		data = np.asarray(_get_pickle_value(batch, "data"), dtype=np.uint8)
		labels = np.asarray(_get_pickle_value(batch, "labels"), dtype=np.int64)
		if data.ndim != 2 or data.shape[1] != 3072:
			raise ValueError(f"Unexpected data shape in {path}: {data.shape}")
		# Stored format is flat CIFAR-like vectors; reshape into HWC RGB images.
		images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		return images, labels

	train_images_chunks = []
	train_labels_chunks = []
	for batch_idx in range(1, 6):
		images, labels = _load_batch(batch_dir / f"data_batch_{batch_idx}")
		train_images_chunks.append(images)
		train_labels_chunks.append(labels)

	train_images = np.concatenate(train_images_chunks, axis=0)
	train_labels = np.concatenate(train_labels_chunks, axis=0)
	test_images, test_labels = _load_batch(batch_dir / "test_batch")

	return train_images, train_labels, test_images, test_labels


def load_kaggle_artbench10(kaggle_root):
	root, csv_path, batch_dir = _resolve_kaggle_paths(kaggle_root)
	if not csv_path.exists():
		raise FileNotFoundError(
			f"Kaggle CSV not found: {csv_path}. "
			"Expected the original ArtBench-10 folder structure."
		)
	if not batch_dir.exists():
		raise FileNotFoundError(
			f"Kaggle CIFAR batches not found: {batch_dir}. "
			"Expected ArtBench-10/artbench-10-python/artbench-10-batches-py"
		)

	with open(batch_dir / "meta", "rb") as f:
		meta = pickle.load(f)
	styles = _get_pickle_value(meta, "styles")
	if not isinstance(styles, list) or len(styles) == 0:
		raise ValueError(f"Could not read class names from {batch_dir / 'meta'}")
	styles = [str(s).strip() for s in styles]

	train_images, train_labels, test_images, test_labels = _load_kaggle_batches(batch_dir)

	print(f"Dataset source: kaggle root='{root}'")
	return train_images, train_labels, test_images, test_labels, styles


def _read_subset_csv_indices(csv_path):
	# Accept both naming conventions used in provided subset CSVs.
	indices = []
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames:
			raise ValueError(f"Subset CSV has no header: {csv_path}")
		if "train_id_original" in reader.fieldnames:
			id_key = "train_id_original"
		elif "train_id_filtered" in reader.fieldnames:
			id_key = "train_id_filtered"
		else:
			raise ValueError(
				f"Subset CSV missing train_id_original/train_id_filtered: {csv_path}"
			)

		for row in reader:
			raw = row.get(id_key, "").strip()
			if raw == "":
				continue
			indices.append(int(raw))
	if not indices:
		raise ValueError(f"Subset CSV yielded zero indices: {csv_path}")
	return indices


def _apply_subset(images, labels, subset_indices):
	# Subset is applied only to train split; test split remains untouched.
	subset_indices = np.asarray(subset_indices, dtype=np.int64)
	if subset_indices.min() < 0 or subset_indices.max() >= images.shape[0]:
		raise ValueError("Subset indices out of bounds for train split")
	return images[subset_indices], labels[subset_indices]


class ArtBenchKaggleDataset(Dataset):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels

	def __len__(self):
		return int(self.images.shape[0])

	def __getitem__(self, idx):
		image = self.images[idx]
		label = int(self.labels[idx])
		if image.ndim != 3 or image.shape[-1] != 3:
			raise ValueError(f"Expected HxWx3 image, got {image.shape}")
		image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
		# Normalize to [-1, 1] to match the generators' tanh output range.
		image = image.float().div(255.0)
		image = image.mul(2.0).sub(1.0)
		return image, label


def get_dataloaders(
	batch_size,
	num_workers=2,
	use_subset=True,
	subset_mode="csv",
	subset_csv_path="provided/student_start_pack/training_20_percent.csv",
	subset_seed=42,
	subset_fraction=0.2,
	kaggle_root="ArtBench-10",
	shuffle_train=True,
):
	# Load raw train/test splits once, then optionally filter train data.
	train_images, train_labels, test_images, test_labels, class_names = load_kaggle_artbench10(
		kaggle_root
	)

	if use_subset:
		if subset_mode == "csv":
			subset_indices = _read_subset_csv_indices(subset_csv_path)
			train_images, train_labels = _apply_subset(
				train_images, train_labels, subset_indices
			)
		elif subset_mode == "random":
			rng = np.random.default_rng(int(subset_seed))
			fraction = float(subset_fraction)
			if fraction <= 0.0 or fraction > 1.0:
				raise ValueError("subset_fraction must be in (0, 1]")
			count = max(1, int(fraction * train_images.shape[0]))
			subset_indices = rng.choice(train_images.shape[0], size=count, replace=False)
			train_images, train_labels = _apply_subset(
				train_images, train_labels, subset_indices
			)
		else:
			raise ValueError("subset_mode must be 'csv' or 'random'")

	train_ds = ArtBenchKaggleDataset(train_images, train_labels)
	test_ds = ArtBenchKaggleDataset(test_images, test_labels)

	effective_num_workers = int(num_workers)
	if os.name == "nt" and effective_num_workers > 0:
		# Windows uses spawn-based workers, which is less robust for large in-memory
		# datasets inside sweep/agent subprocesses. Default to single-process loading.
		print(
			f"WARNING: num_workers={effective_num_workers} on Windows can cause "
			"multiprocessing/pickle crashes during long runs. Falling back to num_workers=0."
		)
		effective_num_workers = 0

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=shuffle_train,
		num_workers=effective_num_workers,
		pin_memory=True,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=effective_num_workers,
		pin_memory=True,
	)

	return train_loader, test_loader, class_names
