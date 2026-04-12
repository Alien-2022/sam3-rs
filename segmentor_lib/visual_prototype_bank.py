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
        feat_levels: Optional[List[int]] = None,
        feat_source: str = "backbone_fpn",
        num_clusters: int = 1,
        label_offset: int = 0,
        dinov3_weights: Optional[str] = None,
        dinov3_input_size: int = 544,
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
            feat_levels: list of FPN levels for multi-level extraction. Overrides feat_level.
            feat_source: feature source for prototype construction.
                - "backbone_fpn": use raw backbone FPN features (default)
                - "pixel_decoder": use PixelDecoder output (multi-scale FPN fusion, 288x288)
                - "dinov3_sat": use DINOv3 ViT-L/16 SAT (493M remote sensing) features (1024-dim)
            num_clusters: maximum number of sub-prototypes per class via adaptive K-means (1=weighted avg, >1=auto-select K in [1, num_clusters] by silhouette score).
            label_offset: offset between class_id and GT label values.
                For datasets where valid GT labels start at 1 (e.g., OpenEarthMap: 0=no-data, 1-8=classes),
                set label_offset=1 so that class_id=0 looks for GT label=1.
                For datasets where GT labels start at 0 (e.g., Potsdam), use default 0.
            dinov3_weights: path to DINOv3 SAT weights (safetensors or pth). Required when feat_source="dinov3_sat".
            dinov3_input_size: input resolution for DINOv3 (must be multiple of 16). Default 544 → 34x34 patches.
        """
        if feat_levels is not None:
            self.feat_levels = list(feat_levels)
        else:
            self.feat_levels = [feat_level]
        self.feat_level = self.feat_levels[0]  # backward compat
        self.multi_level = len(self.feat_levels) > 1
        self.feat_source = feat_source
        self.num_clusters = num_clusters
        self.label_offset = label_offset
        self.dinov3_input_size = dinov3_input_size

        self.processor = processor
        self.device = device
        self.num_classes = num_classes
        self.max_crops_per_class = max_crops_per_class
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size

        # DINOv3 model (lazy loaded on first feature extraction)
        self._dinov3_model = None
        self._dinov3_weights = dinov3_weights

        # Will be filled after build:
        # Single-level: class_id -> [1, 1, C]  (C=256 for SAM3, C=1024 for DINOv3)
        # Multi-level:  class_id -> {level: [1, 1, C]}
        self.prototypes: Dict = {}
        # Store per-crop features for analysis
        self.crop_features: Dict = {}

    def _load_dinov3(self):
        """Lazy-load DINOv3 ViT-L/16 SAT model using HuggingFace transformers."""
        if self._dinov3_model is not None:
            return

        from transformers import AutoModel

        print(f"  Loading DINOv3 ViT-L/16 SAT from {self._dinov3_weights} ...")
        weights_path = self._dinov3_weights
        if os.path.isdir(weights_path):
            weights_path = os.path.join(weights_path, "model.safetensors")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(self._dinov3_weights, "pytorch_model.bin")

        self._dinov3_hf_model = AutoModel.from_pretrained(
            self._dinov3_weights, local_files_only=True
        ).to(self.device).eval()
        self._dinov3_model = self._dinov3_hf_model  # alias for compatibility
        self._dinov3_n_reg = getattr(self._dinov3_hf_model.config, "num_register_tokens", 0)
        self._dinov3_patch_size = self._dinov3_hf_model.config.patch_size
        patch_res = self.dinov3_input_size // self._dinov3_patch_size
        print(f"  DINOv3 loaded: input={self.dinov3_input_size}x{self.dinov3_input_size}, "
              f"patches={patch_res}x{patch_res}, dim=1024")

    def _extract_dinov3_features(self, image: Image.Image) -> Dict[int, torch.Tensor]:
        """Extract features using DINOv3 ViT-L/16 SAT (HuggingFace).

        Returns:
            features: {0: [1, 1024, H_patch, W_patch]}  (single level, keyed as 0)
        """
        import torchvision.transforms.v2 as v2

        self._load_dinov3()

        # DINOv3 SAT normalization (from preprocessor_config.json)
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.dinov3_input_size, self.dinov3_input_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143]),
        ])

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self._dinov3_hf_model(img_tensor, output_hidden_states=True)

        # Use last_hidden_state (includes final LayerNorm) for best discriminative features
        n_skip = 1 + self._dinov3_n_reg  # skip CLS + register tokens
        patch_tokens = outputs.last_hidden_state[:, n_skip:, :]  # [1, H*W, 1024]

        B, N, C = patch_tokens.shape
        patch_res = int(N ** 0.5)
        spatial_feats = patch_tokens.reshape(B, patch_res, patch_res, C).permute(0, 3, 1, 2)

        # DINOv3 has single output level; store under level 0
        return {0: spatial_feats.float()}

    def _extract_backbone_features(self, image: Image.Image) -> Dict[int, torch.Tensor]:
        """Extract visual features from an image.

        Dispatches to SAM3 backbone or DINOv3 based on feat_source.

        Returns:
            features: {level: [1, C, H_feat, W_feat]} for each requested FPN level
        """
        if self.feat_source == "dinov3_sat":
            return self._extract_dinov3_features(image)

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

        features = {}

        if self.feat_source == "pixel_decoder":
            # Use PixelDecoder: top-down multi-scale FPN fusion → 288x288
            fpn_feats = backbone_out["backbone_fpn"]
            # PixelDecoder expects all FPN levels as a list, output is 288x288
            pixel_decoder = self.processor.model.segmentation_head.pixel_decoder
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ):
                pixel_embed = pixel_decoder(fpn_feats)
            # Store under the primary feat_level key so rest of pipeline works unchanged
            features[self.feat_levels[0]] = pixel_embed.float()
        else:
            for level in self.feat_levels:
                features[level] = backbone_out["backbone_fpn"][level].float()
        return features

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

    def _cluster_prototypes(
        self,
        stacked: torch.Tensor,
        weights: List[float],
        max_K: int,
    ) -> torch.Tensor:
        """Adaptive clustering: try K=2..max_K, pick best by silhouette score.

        Falls back to weighted average (K=1) if no K>=2 achieves silhouette > 0.15.

        Args:
            stacked: [N, C] per-crop feature vectors
            weights: per-crop pixel-count weights
            max_K: maximum number of clusters to try

        Returns:
            [K, 1, 256] sub-prototype tensor (K determined adaptively, min 1)
        """
        N, C = stacked.shape
        if N < 2:
            w = torch.tensor(weights, dtype=stacked.dtype)
            w = w / w.sum()
            proto = (stacked * w.unsqueeze(1)).sum(dim=0).unsqueeze(0).unsqueeze(0)
            return proto

        normalized = F.normalize(stacked, p=2, dim=1).numpy()  # [N, C]

        # Weighted average as K=1 fallback
        w = torch.tensor(weights, dtype=stacked.dtype)
        w = w / w.sum()
        proto_k1 = (stacked * w.unsqueeze(1)).sum(dim=0).unsqueeze(0).unsqueeze(0)  # [1,1,C]

        if max_K <= 1:
            return proto_k1

        # Try K=2..max_K, compute silhouette score for each
        best_K = 1
        best_silhouette = -1.0
        best_result = None  # (assignments, centers, cluster_sizes)

        for K in range(2, max_K + 1):
            if N <= K:
                break
            centers, assignments, sizes = self._kmeans(normalized, K)
            if sizes is None:
                continue
            sil = self._silhouette_score(normalized, assignments)
            # Penalize overly fragmented clusters: require each cluster >= min_size
            min_size = max(3, N * 0.05)
            if min(sizes) < min_size:
                continue
            if sil > best_silhouette:
                best_silhouette = sil
                best_K = K
                best_result = (assignments, centers, sizes)

        # Minimum silhouette threshold to accept clustering
        SIL_THRESHOLD = 0.15
        if best_K == 1 or best_silhouette < SIL_THRESHOLD:
            print(f"    K=1 (sil={best_silhouette:.4f} < {SIL_THRESHOLD}, no meaningful clusters)")
            return proto_k1

        assignments, centers, sizes = best_result
        print(f"    K={best_K} (sil={best_silhouette:.4f}), sizes={sizes}")

        # Raw magnitude centroids
        centroids = torch.zeros(best_K, C)
        for k in range(best_K):
            mask = assignments == k
            centroids[k] = stacked[mask].mean(dim=0)
        return centroids.unsqueeze(1)  # [K, 1, 256]

    def _kmeans(self, np_feats: np.ndarray, K: int, max_iter: int = 30):
        """Run K-means on normalized features.

        Returns:
            centers: [K, C], assignments: [N], sizes: list of int (or None if failed)
        """
        N = np_feats.shape[0]

        # K-means++ init
        centers = [np_feats[np.random.randint(N)].copy()]
        for _ in range(1, K):
            dists = np.min([np.sum((np_feats - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (dists.sum() + 1e-8)
            idx = np.random.choice(N, p=probs)
            centers.append(np_feats[idx].copy())
        centers = np.array(centers)

        for _ in range(max_iter):
            dists = np.sum((np_feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers[k] = np.mean(np_feats[mask], axis=0)
                else:
                    new_centers[k] = centers[k]
            norms = np.linalg.norm(new_centers, axis=1, keepdims=True)
            new_centers = new_centers / (norms + 1e-8)
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        dists = np.sum((np_feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        sizes = [int((assignments == k).sum()) for k in range(K)]
        return centers, assignments, sizes

    @staticmethod
    def _silhouette_score(np_feats: np.ndarray, assignments: np.ndarray) -> float:
        """Compute mean silhouette score.

        For efficiency on large N, subsample if N > 500.
        """
        N = np_feats.shape[0]
        unique_labels = np.unique(assignments)
        K = len(unique_labels)
        if K <= 1 or N <= K:
            return 0.0

        # Subsample for efficiency
        if N > 500:
            idx = np.random.choice(N, 500, replace=False)
            feats = np_feats[idx]
            asgn = assignments[idx]
        else:
            feats = np_feats
            asgn = assignments

        # Pairwise distance matrix [M, M]
        dists = np.sum((feats[:, None, :] - feats[None, :, :]) ** 2, axis=2)

        sil_values = []
        for i in range(len(feats)):
            ci = asgn[i]
            same = asgn == ci
            same[i] = False
            if same.sum() == 0:
                sil_values.append(0.0)
                continue
            a_i = dists[i, same].mean()  # mean intra-cluster distance

            diff_mask = ~same
            if diff_mask.sum() == 0:
                sil_values.append(0.0)
                continue
            # Mean inter-cluster distance to nearest other cluster
            b_i = float('inf')
            for cj in unique_labels:
                if cj == ci:
                    continue
                mask_j = asgn == cj
                if mask_j.sum() > 0:
                    b_i = min(b_i, dists[i, mask_j].mean())

            sil_values.append((b_i - a_i) / (max(a_i, b_i) + 1e-8))

        return float(np.mean(sil_values))

    def _pool_class_from_crop(
        self,
        features: torch.Tensor,
        gt_crop: np.ndarray,
        class_id: int,
    ) -> Optional[tuple]:
        """Masked average pooling for one class from one crop's features.

        Args:
            features: [1, C, H_feat, W_feat] backbone features
            gt_crop: [crop_h, crop_w] GT mask for this crop region
            class_id: class to extract

        Returns:
            (pooled, num_pixels): ([C] feature vector, int pixel count), or None
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
        num_pixels = class_mask.sum()
        if num_pixels == 0:
            return None

        # Masked average pooling
        masked_feats = features.squeeze(0) * class_mask.unsqueeze(0)  # [C, H, W]
        pooled = masked_feats.sum(dim=(1, 2)) / num_pixels  # [C]
        return (pooled, num_pixels.item())

    def build(
        self,
        dataset,
        num_calib: int = 4,
        save_path: Optional[str] = None,
    ) -> Dict:
        """Build visual prototype bank from calibration images using sliding window.

        Args:
            dataset: dataset with 'image_path' and 'mask' fields
            num_calib: number of calibration images to use
            save_path: optional path to save the bank

        Returns:
            Single-level: Dict[class_id, [1, 1, 256]]
            Multi-level:  Dict[class_id, {level: [1, 1, 256]}]
        """
        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Building Visual Prototype Bank (Sliding Window)")
        print(f"  Classes: {self.num_classes}, Calib images: {num_calib}")
        print(f"  Crop size: {self.crop_size}, Stride: {self.stride}")
        print(f"  Feature source: {self.feat_source}")
        if self.feat_source == "dinov3_sat":
            print(f"  DINOv3 input size: {self.dinov3_input_size}x{self.dinov3_input_size}")
        print(f"  FPN levels: {self.feat_levels} (multi={'yes' if self.multi_level else 'no'})")
        if self.label_offset != 0:
            print(f"  Label offset: {self.label_offset} (GT labels [{self.label_offset}, {self.label_offset + self.num_classes - 1}] ↔ class_id [0, {self.num_classes - 1}])")
        print(f"{'=' * 60}")

        # Accumulators: class_id -> level -> list of [C] vectors
        class_features: Dict[int, Dict[int, List[torch.Tensor]]] = {
            c: {l: [] for l in self.feat_levels} for c in range(self.num_classes)
        }
        # Pixel-count weights for weighted averaging across crops
        class_pixel_weights: Dict[int, Dict[int, List[float]]] = {
            c: {l: [] for l in self.feat_levels} for c in range(self.num_classes)
        }
        crop_counts: Dict[int, Dict[int, int]] = {
            c: {l: 0 for l in self.feat_levels} for c in range(self.num_classes)
        }

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
                # Check if all classes at all levels are satisfied
                all_done = all(
                    crop_counts[c][l] >= self.max_crops_per_class
                    for c in range(self.num_classes)
                    for l in self.feat_levels
                )
                if all_done:
                    break

                crop_img = image.crop((x1, y1, x2, y2))
                gt_crop = gt_mask[y1:y2, x1:x2]  # [crop_h, crop_w]

                # Skip crops with no valid GT classes
                gt_classes_in_crop = np.unique(gt_crop)
                if not any(c in gt_classes_in_crop and c < self.num_classes for c in range(self.num_classes)):
                    continue

                try:
                    features_per_level = self._extract_backbone_features(crop_img)
                except Exception as e:
                    print(f"    Error extracting features for crop ({x1},{y1})-({x2},{y2}): {e}")
                    continue

                img_crop_count += 1

                # Extract per-class prototypes from this crop, at each level
                for level in self.feat_levels:
                    feats = features_per_level[level]
                    C = feats.shape[1]
                    for class_id in range(self.num_classes):
                        if crop_counts[class_id][level] >= self.max_crops_per_class:
                            continue

                        # Apply label_offset: map class_id [0, N) to GT label [offset, offset+N)
                        # E.g., OpenEarthMap: class_id=0 → GT label 1, class_id=7 → GT label 8
                        gt_label = class_id + self.label_offset
                        result = self._pool_class_from_crop(feats, gt_crop, gt_label)
                        if result is not None:
                            pooled, num_px = result
                            class_features[class_id][level].append(pooled.float().cpu())
                            class_pixel_weights[class_id][level].append(num_px)
                            crop_counts[class_id][level] += 1

                # Free memory
                del features_per_level

            print(f"    Processed {img_crop_count}/{len(crops)} crops, "
                  f"class crop_counts: {crop_counts}")
            calib_count += 1

            # Periodic GPU cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Weighted average pool across crops per class per level
        elapsed = time.time() - t_start
        print(f"\n  Prototype extraction done in {elapsed:.1f}s")
        print(f"  Final crop counts: {crop_counts}")
        if self.num_clusters > 1:
            print(f"  Clustering: K={self.num_clusters} sub-prototypes per class")

        prototypes = {}
        for class_id in range(self.num_classes):
            lv_protos = {}
            for level in self.feat_levels:
                feats = class_features[class_id][level]
                weights = class_pixel_weights[class_id][level]
                if feats:
                    stacked = torch.stack(feats)  # [N, C]
                    if self.num_clusters > 1:
                        proto = self._cluster_prototypes(stacked, weights, self.num_clusters)
                    else:
                        w = torch.tensor(weights, dtype=stacked.dtype)  # [N]
                        w = w / w.sum()  # normalize weights
                        proto = (stacked * w.unsqueeze(1)).sum(dim=0)  # [C], weighted average
                        proto = proto.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                    lv_protos[level] = proto.to(self.device)
            if lv_protos:
                if self.multi_level:
                    prototypes[class_id] = lv_protos
                    self.crop_features[class_id] = class_features[class_id]
                    print(f"  Class {class_id}: {', '.join(f'lv{l}:{lv_protos[l].shape}' for l in lv_protos)}")
                else:
                    level0 = self.feat_levels[0]
                    prototypes[class_id] = lv_protos[level0]
                    self.crop_features[class_id] = class_features[class_id][level0]
                    p = lv_protos[level0]
                    if self.num_clusters > 1:
                        norms = [p[i].norm().item() for i in range(p.shape[0])]
                        print(f"  Class {class_id}: {len(feats)} crops → {p.shape[0]} clusters, "
                              f"norms={[f'{n:.3f}' for n in norms]}")
                    else:
                        print(f"  Class {class_id}: {len(feats)} crops, "
                              f"proto norm={p.norm().item():.4f}, "
                              f"proto mean={p.mean().item():.4f}")
            else:
                print(f"  Class {class_id}: WARNING - no crops found!")

        self.prototypes = prototypes
        print(f"\n  Built {len(prototypes)}/{self.num_classes} class prototypes "
              f"({'multi-level' if self.multi_level else 'single-level'})")

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

        def _get_proto(cid, level=None):
            """Get prototype tensor for a class, handling multi-prototype [K,1,C] and dict formats."""
            p = self.prototypes[cid]
            if isinstance(p, dict):
                p = p[level]
            # Multi-prototype: average sub-prototypes for similarity display
            if p.shape[0] > 1:
                return p.mean(dim=0)  # [1, C]
            return p

        levels = self.feat_levels if self.multi_level else [self.feat_levels[0]]

        for level in levels:
            print(f"\n  FPN Level {level} Inter-class Cosine Similarity (diagonal=1.0):")
            all_protos = torch.cat([_get_proto(cid, level).reshape(1, -1) for cid in class_ids], dim=0)
            # Normalize for cosine similarity (prototypes are no longer pre-normalized)
            all_protos_norm = F.normalize(all_protos, p=2, dim=1)
            sim = torch.mm(all_protos_norm, all_protos_norm.mT)
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
            off_diag_vals = sim[~torch.eye(n, dtype=bool)].tolist()
            if off_diag_vals:
                od = torch.tensor(off_diag_vals)
                print(f"  Off-diagonal: min={od.min():.4f}, max={od.max():.4f}, "
                      f"mean={od.mean():.4f}, std={od.std():.4f}")

    def save(self, save_path: str):
        """Save prototype bank to disk."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        if self.multi_level:
            proto_dict = {
                str(k): {str(l): v.cpu() for l, v in lv.items()}
                for k, lv in self.prototypes.items()
            }
        else:
            proto_dict = {
                str(k): v.cpu() for k, v in self.prototypes.items()
            }

        save_dict = {
            "prototypes": proto_dict,
            "num_classes": self.num_classes,
            "config": {
                "max_crops_per_class": self.max_crops_per_class,
                "crop_size": self.crop_size,
                "stride": self.stride,
                "feat_levels": self.feat_levels,
                "feat_source": self.feat_source,
                "num_clusters": self.num_clusters,
                "label_offset": self.label_offset,
                "dinov3_input_size": self.dinov3_input_size if self.feat_source == "dinov3_sat" else None,
            },
            "multi_level": self.multi_level,
        }
        torch.save(save_dict, save_path)
        print(f"Visual Prototype Bank saved to {save_path}")

    @staticmethod
    def load(save_path: str, device: torch.device) -> Dict:
        """Load prototype bank from disk.

        Returns:
            Single-level: Dict[class_id, [1, 1, 256]]
            Multi-level:  Dict[class_id, {level: [1, 1, 256]}]
        """
        data = torch.load(save_path, map_location="cpu", weights_only=False)
        multi_level = data.get("multi_level", False)
        prototypes = {}
        if multi_level:
            for k, v in data["prototypes"].items():
                prototypes[int(k)] = {int(l): t.to(device) for l, t in v.items()}
        else:
            for k, v in data["prototypes"].items():
                prototypes[int(k)] = v.to(device)
        return prototypes
