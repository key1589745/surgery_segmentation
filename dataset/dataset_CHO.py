from pathlib import Path
from bisect import bisect_left
from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import functional as F
from torchvision import tv_tensors


def load_video_ids(split_file: Path) -> List[int]:
    video_ids = []
    with split_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            video_ids.append(int(float(line)))
    return sorted(set(video_ids))


def parse_sample_id(stem: str) -> Tuple[int, int]:
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {stem}")
    return int(parts[0]), int(parts[1])


class EndoscapesSegClipDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        frame_dirs: Sequence[Path],
        mask_dir: Path,
        clip_len: int = 8,
        clip_stride: int = 2,
        transform=None,
    ):
        self.allowed_video_ids = load_video_ids(split_file)
        self.allowed_video_ids_set = set(self.allowed_video_ids)
        self.frame_dirs = [Path(d) for d in frame_dirs]
        self.mask_dir = Path(mask_dir)
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.transform = transform

        self.frame_lookup = {}
        self.available_indices = defaultdict(list)
        for frame_dir in self.frame_dirs:
            for pattern in ("*.jpg", "*.png"):
                for img_path in frame_dir.glob(pattern):
                    vid, frame_idx = parse_sample_id(img_path.stem)
                    if vid not in self.allowed_video_ids_set:
                        continue
                    key = (vid, frame_idx)
                    if key not in self.frame_lookup:
                        self.frame_lookup[key] = img_path
                        self.available_indices[vid].append(frame_idx)
        for vid, indices in self.available_indices.items():
            self.available_indices[vid] = sorted(set(indices))

        self.samples = []
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            vid, frame_idx = parse_sample_id(mask_path.stem)
            if vid not in self.allowed_video_ids_set:
                continue
            clip_indices = self._build_clip_indices(frame_idx)
            clip_paths = [self._resolve_or_nearest(vid, idx) for idx in clip_indices]
            if any(path is None for path in clip_paths):
                continue
            self.samples.append(
                {
                    "video_id": vid,
                    "frame_idx": frame_idx,
                    "clip_paths": clip_paths,
                    "mask_path": mask_path,
                }
            )

    def _build_clip_indices(self, center_idx: int) -> List[int]:
        start = center_idx - (self.clip_len - 1) * self.clip_stride
        return [start + i * self.clip_stride for i in range(self.clip_len)]

    def _nearest_index(self, video_id: int, target_idx: int) -> int | None:
        indices = self.available_indices.get(video_id)
        if not indices:
            return None
        pos = bisect_left(indices, target_idx)
        if pos < len(indices) and indices[pos] == target_idx:
            return indices[pos]
        candidates = []
        if pos < len(indices):
            candidates.append(indices[pos])
        if pos > 0:
            candidates.append(indices[pos - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda idx: abs(idx - target_idx))

    def _resolve_or_nearest(self, video_id: int, target_idx: int) -> Path | None:
        key = (video_id, target_idx)
        if key in self.frame_lookup:
            return self.frame_lookup[key]
        nearest_idx = self._nearest_index(video_id, target_idx)
        if nearest_idx is None:
            return None
        return self.frame_lookup.get((video_id, nearest_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames = []
        for frame_path in sample["clip_paths"]:
            with Image.open(frame_path) as img:
                frames.append(F.pil_to_tensor(img.convert("RGB")))
        video = torch.stack(frames)  # T, C, H, W
        video = tv_tensors.Video(video)

        with Image.open(sample["mask_path"]) as mask_img:
            mask_arr = np.array(mask_img, dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        
        # Sanitize mask values: treat >= 7 as ignore_index (255)
        # assuming 7 classes (0-6) are valid.
        mask_arr[mask_arr >= 7] = 255

        mask = tv_tensors.Mask(torch.from_numpy(mask_arr.astype(np.int64)))

        if self.transform:
            video, mask = self.transform(video, mask)

        return {
            "video": video.as_subclass(torch.Tensor),
            "mask": mask.as_subclass(torch.Tensor).long(),
            "video_id": sample["video_id"],
            "frame_idx": sample["frame_idx"],
            "clip_paths": [str(p) for p in sample["clip_paths"]],
            "mask_path": str(sample["mask_path"]),
        }


def prepare_transforms(image_size: tuple[int, int] = (256, 256)):
    train_transforms = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.9,1.1)),
        v2.ColorJitter(contrast=0.5,brightness=0.5,saturation=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transforms = v2.Compose([v2.Resize(image_size, antialias=True),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                 )
    return train_transforms,  test_transforms


class DataLoaders:
    """
    A container for train, val, and test datasets and dataloaders.
    """
    def __init__(self, data_root: str, split_files: dict, 
                 frame_dirs: list, mask_dir: str, clip_len: int, 
                 clip_stride: int, transforms: callable,
                 batch_size: int = 2, num_workers: int = 4, num_classes: int = 7, dataset_name="CHO"):
        
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.data_root = Path(data_root)
        self.frame_dirs = [self.data_root / d for d in frame_dirs]
        self.mask_dir = self.data_root / mask_dir
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.train_transforms, self.test_transforms = transforms

        common_dataset_args = dict(
            frame_dirs=self.frame_dirs,
            mask_dir=self.mask_dir,
            clip_len=self.clip_len,
            clip_stride=self.clip_stride,
        )
        
        self.train = EndoscapesSegClipDataset(
            split_file=self.data_root / split_files['train'],
            transform=self.train_transforms,
            **common_dataset_args,
        )
        self.val = EndoscapesSegClipDataset(
            split_file=self.data_root / split_files['val'],
            transform=self.test_transforms,
            **common_dataset_args,
        )
        self.test = EndoscapesSegClipDataset(
            split_file=self.data_root / split_files['test'],
            transform=self.test_transforms,
            **common_dataset_args,
        )
        
        self.train_loader = DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
