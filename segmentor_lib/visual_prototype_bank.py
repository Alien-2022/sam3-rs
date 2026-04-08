"""Visual Prototype Bank: extract class-wise visual prototypes from calibration images.

Uses sliding window with 288x288 crop size to extract high-resolution backbone
features (FPN level 0: 288x288), then masked average pooling per class.

Key design:
- Uses backbone_fpn[0] (288x288) instead of vision_features (72x72) for 16x
  finer spatial resolution
- Default crop size 288 pixels matches backbone_fpn[0] resolution for ~1:1
  GT mask alignment (no resize interpolation needed)
- Sliding window for large images to avoid extreme downscaling
- Each crop's feature map directly aligns with its GT mask crop
"""

import gc
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


class VisualPrototypeBank:
    """Build and manage per-class visual prototypes from GT calibration data."""

    def __init__(
        self,
        processor,
        device: torch.device,
        num_classes: int,
        max_crops_per_class: int = 50,
        crop_size: int = 288,
        stride: Optional[int] = None,
        feat_level: int = -1,
    ):
        """Init Visual Prototype Bank.

        Args:
            processor: SAM3 processor with model.backbone access
            device: torch device
            num_classes: number of semantic classes
            max_crops_per_class: max number of crop-features to accumulate per class
            crop_size: sliding window crop size in original image pixels.
                       288 matches backbone_fpn[0] for ~1:1 GT alignment.
            stride: sliding window stride. Default=crop_size (no overlap).
            feat_level: which FPN level to use (-1=last/semantic 72x72, 0=288x288, 1=144x144).
        """
        self.processor = processor
        self.device = device
        self.num_classes = num_classes
        self.max_crops_per_class = max_crops_per_class
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size
        self.feat_level = feat_level

        # Will be filled after build: class_id -> prototype [1, 1, 256]
        self.prototypes: Dict[int, torch.Tensor] = {}
        # Store per-crop features for analysis
        self.crop_features: Dict[int, List[torch.Tensor]] = {}

    def _extract_backbone_features(self, image: Image.Image) -> torch.Tensor:
        """Extract visual features from an image using SAM3 backbone.

        Returns:
            features: [1, 256, H_feat, W_feat] from backbone_fpn[feat_level]
        """
        import torchvision.transforms.v2 as v2

        # Same transform as Sam3Processor
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img_tensor = transform(v2.functional.to_image(image).to(self.device)).unsqueeze(0)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            backbone_out = self.processor.model.backbone.forward_image(img_tensor)

        # Use high-level FPN features (last level, 72x72) for semantic discrimination
        features = backbone_out["backbone_fpn"][self.feat_level].float()
        return features  # [1, 256, 288, 288]

    def _generate_crop_positions(self, img_w: int, img_h: int) -> List[tuple]:
        """Generate sliding window crop positions.

        Returns:
            List of (x1, y1, x2, y2) crop positions.
        """
        if img_w <= self.crop_size and img_h <= self.crop_size:
            return [(0, 0, img_w, img_h)]

        crops = []
        for y in range(0, img_h, self.stride):
            for x in range(0, img_w, self.stride):
                x2 = min(x + self.crop_size, img_w)
                y2 = min(y + self.crop_size, img_h)
                # Skip tiny edge crops (< half crop_size in either dim)
                cw, ch = x2 - x, y2 - y
                if cw < self.crop_size // 2 or ch < self.crop_size // 2:
                    continue
                crops.append((x, y, x2, y2))
        return crops

    def _pool_class_from_crop(
        self,
        features: torch.Tensor,
        gt_crop: np.ndarray,
        class_id: int,
    ) -> Optional[torch.Tensor]:
        """Masked average pooling for one class from one crop's features.

        Args:
            features: [1, C, H_feat, W_feat] backbone features
            gt_crop: [crop_h, crop_w] GT mask for this crop region
            class_id: class to extract

        Returns:
            pooled: [C] feature vector, or None if no pixels of this class
        """
        H_feat, W_feat = features.shape[-2:]

        # For 288x288 crop + fpn[0]=288x288: gt_crop is 288x288 → no resize needed
        # For other sizes: nearest resize to match feature map
        if gt_crop.shape[0] != H_feat or gt_crop.shape[1] != W_feat:
            gt_tensor = torch.from_numpy(gt_crop).float().to(features.device)
            gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, crop_h, crop_w]
            gt_resized = F.interpolate(gt_tensor, (H_feat, W_feat), mode='nearest').squeeze()
        else:
            gt_resized = torch.from_numpy(gt_crop).float().to(features.device)

        # Binary mask for this class
        class_mask = (gt_resized == class_id).float()
        if class_mask.sum() == 0:
            return None

        # Masked average pooling
        masked_feats = features.squeeze(0) * class_mask.unsqueeze(0)  # [C, H, W]
        num_pixels = class_mask.sum()
        pooled = masked_feats.sum(dim=(1, 2)) / num_pixels  # [C]
        return pooled

    def build(
        self,
        dataset,
        num_calib: int = 4,
        save_path: Optional[str] = None,
    ) -> Dict[int, torch.Tensor]:
        """Build visual prototype bank from calibration images using sliding window.

        Args:
            dataset: dataset with 'image_path' and 'mask' fields
            num_calib: number of calibration images to use
            save_path: optional path to save the bank

        Returns:
            Dict[class_id, prototype] where prototype is [1, 1, 256]
        """
        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Building Visual Prototype Bank (Sliding Window)")
        print(f"  Classes: {self.num_classes}, Calib images: {num_calib}")
        print(f"  Crop size: {self.crop_size}, Stride: {self.stride}")
        print(f"  FPN level: {self.feat_level} (-1=semantic 72x72, 0=288x288, 1=144x144)")
        print(f"{'=' * 60}")

        # Accumulators: class_id -> list of [C] vectors
        class_features: Dict[int, List[torch.Tensor]] = {
            c: [] for c in range(self.num_classes)
        }
        crop_counts = {c: 0 for c in range(self.num_classes)}

        # Process calibration images
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        calib_count = 0

        for batch in tqdm(loader, desc="Extracting prototypes"):
            if calib_count >= num_calib:
                break

            image_path = batch["image_path"][0]
            gt_mask = batch["mask"][0].numpy() if torch.is_tensor(batch["mask"]) else np.asarray(batch["mask"])
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size

            print(f"\n  Image {calib_count}: {os.path.basename(image_path)}")
            print(f"    Size: {img_w}x{img_h}, GT classes: {np.unique(gt_mask).tolist()}")

            # Generate crop positions
            crops = self._generate_crop_positions(img_w, img_h)
            print(f"    Crops: {len(crops)} ({img_w}x{img_h}, crop={self.crop_size}, stride={self.stride})")

            img_crop_count = 0
            for x1, y1, x2, y2 in tqdm(crops, desc=f"  Image {calib_count} crops",
                                         leave=False, disable=len(crops) <= 1):
                # Check if all classes are satisfied
                all_done = all(crop_counts[c] >= self.max_crops_per_class for c in range(self.num_classes))
                if all_done:
                    break

                crop_img = image.crop((x1, y1, x2, y2))
                gt_crop = gt_mask[y1:y2, x1:x2]  # [crop_h, crop_w]

                # Skip crops with no valid GT classes
                gt_classes_in_crop = np.unique(gt_crop)
                if not any(c in gt_classes_in_crop and c < self.num_classes for c in range(self.num_classes)):
                    continue

                try:
                    features = self._extract_backbone_features(crop_img)  # [1, C, H_feat, W_feat]
                except Exception as e:
                    print(f"    Error extracting features for crop ({x1},{y1})-({x2},{y2}): {e}")
                    continue

                C = features.shape[1]
                img_crop_count += 1

                # Extract per-class prototypes from this crop
                for class_id in range(self.num_classes):
                    if crop_counts[class_id] >= self.max_crops_per_class:
                        continue

                    pooled = self._pool_class_from_crop(features, gt_crop, class_id)
                    if pooled is not None:
                        class_features[class_id].append(pooled.float().cpu())
                        crop_counts[class_id] += 1

                # Free memory
                del features

            print(f"    Processed {img_crop_count}/{len(crops)} crops, "
                  f"class crop_counts: {crop_counts}")
            calib_count += 1

            # Periodic GPU cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average pool across crops per class
        elapsed = time.time() - t_start
        print(f"\n  Prototype extraction done in {elapsed:.1f}s")
        print(f"  Final crop counts: {crop_counts}")

        prototypes = {}
        for class_id in range(self.num_classes):
            feats = class_features[class_id]
            if feats:
                stacked = torch.stack(feats)  # [N, C]
                proto = stacked.mean(dim=0)  # [C]
                # L2 normalize to ensure consistent scale regardless of FPN level
                proto = proto / (proto.norm(p=2) + 1e-8)
                proto = proto.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                prototypes[class_id] = proto.to(self.device)
                self.crop_features[class_id] = feats
                print(f"  Class {class_id}: {len(feats)} crops, "
                      f"proto norm={proto.norm().item():.4f}, "
                      f"proto mean={proto.mean().item():.4f}")
            else:
                print(f"  Class {class_id}: WARNING - no crops found!")

        self.prototypes = prototypes
        print(f"\n  Built {len(prototypes)}/{self.num_classes} class prototypes")

        # Print inter-class cosine similarity matrix
        if len(prototypes) > 1:
            self._print_similarity_matrix()

        if save_path:
            self.save(save_path)

        return prototypes

    def _print_similarity_matrix(self):
        """Print cosine similarity matrix between all class prototypes."""
        class_ids = sorted(self.prototypes.keys())
        n = len(class_ids)
        # Stack all prototypes: [n, 256]
        all_protos = torch.cat([self.prototypes[cid].reshape(1, -1) for cid in class_ids], dim=0)

        # Cosine similarity: [n, n]
        sim = torch.mm(all_protos, all_protos.mT)

        print(f"\n  Inter-class Cosine Similarity (diagonal=1.0):")
        header = "        " + "".join(f"  Cls{j:<3d}" for j in class_ids)
        print(header)
        for i, ci in enumerate(class_ids):
            row = f"  Cls{ci}  "
            for j, cj in enumerate(class_ids):
                if i == j:
                    row += "   1.00 "
                else:
                    row += f" {sim[i, j]:+.3f}"
            print(row)

        # Print off-diagonal stats
        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag.append(sim[i, j].item())
        off_diag = torch.tensor(off_diag)
        print(f"  Off-diagonal: min={off_diag.min():.4f}, max={off_diag.max():.4f}, "
              f"mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")

    def save(self, save_path: str):
        """Save prototype bank to disk."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_dict = {
            "prototypes": {
                str(k): v.cpu() for k, v in self.prototypes.items()
            },
            "num_classes": self.num_classes,
            "config": {
                "max_crops_per_class": self.max_crops_per_class,
                "crop_size": self.crop_size,
                "stride": self.stride,
                "feat_level": self.feat_level,
            },
        }
        torch.save(save_dict, save_path)
        print(f"Visual Prototype Bank saved to {save_path}")

    @staticmethod
    def load(save_path: str, device: torch.device) -> Dict[int, torch.Tensor]:
        """Load prototype bank from disk."""
        data = torch.load(save_path, map_location="cpu", weights_only=False)
        prototypes = {}
        for k, v in data["prototypes"].items():
            prototypes[int(k)] = v.to(device)
        return prototypes
