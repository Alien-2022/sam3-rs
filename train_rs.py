"""Remote sensing fine-tuning script for SAM3-RS.

Fine-tunes a SAM3 model on a labelled remote sensing dataset (segmentation masks)
using lightweight LoRA-style or full-parameter optimisation.

Directory structure expected:
    <data_root>/
        images/        ← RGB images (.png / .jpg / .tif)
        masks/         ← label maps as uint8 PNGs (class IDs starting from 0)

Prompts file:
    One class name per line (synonyms comma-separated), same format as inference.
    Line index = class ID.

Usage:
    python train_rs.py \\
        --data_root data/LoveDA/train \\
        --prompts_file configs/loveda_classes.txt \\
        --checkpoint weights/sam3.pt \\
        --output_dir outputs/finetune_loveda \\
        --epochs 10 \\
        --batch_size 2 \\
        --lr 1e-4 \\
        --device cuda

Note:
    This script requires the SAM3 training infrastructure available under
    ``sam3/train/``.  Adapt ``--freeze_backbone`` and ``--freeze_text_encoder``
    flags to control which parts of the model are updated.
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RSSegDataset(Dataset):
    """Simple remote sensing segmentation dataset.

    Scans ``<root>/images/`` and ``<root>/masks/`` for matching file stems.
    Images are loaded as RGB; masks as single-channel uint8 label maps.

    Args:
        root: Dataset root directory.
        image_size: Spatial size ``(H, W)`` to resize images and masks to.
        image_exts: Accepted image file extensions.
        mask_ext: Mask file extension.
        ignore_index: Label value to ignore during loss computation.
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int] = (512, 512),
        image_exts: Tuple[str, ...] = IMAGE_EXTS,
        mask_ext: str = ".png",
        ignore_index: int = 255,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.image_exts = image_exts
        self.mask_ext = mask_ext
        self.ignore_index = ignore_index

        img_dir = self.root / "images"
        mask_dir = self.root / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Build list of (image_path, mask_path) pairs
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in self.image_exts:
                continue
            mask_path = mask_dir / (img_path.stem + self.mask_ext)
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No matching image/mask pairs found under {root}. "
                "Check that images/ and masks/ directories are populated."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize
        h, w = self.image_size
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        # To tensors
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask_tensor, str(img_path)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def seg_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Pixel-wise cross-entropy between ``logits`` and ``targets``.

    Args:
        logits: ``[B, C, H, W]`` unnormalised class scores.
        targets: ``[B, H, W]`` integer class labels.
        ignore_index: Label to exclude from the loss.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    processor,
    prompts,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    amp_dtype: str = "bfloat16",
) -> float:
    """Run one training epoch.

    Args:
        model: SAM3 backbone model.
        processor: ``Sam3Processor`` instance.
        prompts: Dict with ``'names'`` and ``'indices'`` from ``load_prompts``.
        dataloader: Training data loader.
        optimizer: PyTorch optimiser.
        device: Target device.
        num_classes: Number of semantic classes.
        ignore_index: Label to ignore in loss.
        amp_dtype: AMP precision – ``"bfloat16"`` (Ampere+), ``"float16"``
            (older GPUs), or ``"float32"`` (no AMP).

    Returns:
        Mean loss over the epoch.

    Note:
        The SAM3Processor API currently expects a PIL Image per sample, so
        per-image forward passes are unavoidable here.  Batching is applied
        at the dataset level (DataLoader) to benefit from parallel data
        loading and vectorised loss computation.
    """
    _torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype)
    use_amp = _torch_dtype is not None

    model.train()
    total_loss = 0.0

    for images, masks, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)       # [B, 3, H, W]
        masks = masks.to(device)         # [B, H, W]

        optimizer.zero_grad()

        B, _, H, W = images.shape
        class_logits = torch.zeros((B, num_classes, H, W), device=device)

        for prompt_idx, prompt_word in enumerate(prompts["names"]):
            class_id = prompts["indices"][prompt_idx]

            for b in range(B):
                # SAM3Processor requires PIL input; conversion is unavoidable
                # given the current processor API.
                pil_img = Image.fromarray(
                    (images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )

                ctx = (
                    torch.autocast(device_type=device.type, dtype=_torch_dtype)
                    if use_amp
                    else torch.no_grad.__class__()  # null context
                )
                with ctx:
                    state = processor.set_image(pil_img)
                    output = processor.set_text_prompt(state, prompt=prompt_word)

                    # Use semantic head logits
                    sem_logit = output["semantic_seg"]  # [1, 1, H_orig, W_orig]
                    sem_logit = F.interpolate(
                        sem_logit, size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze()  # [H, W]

                    class_logits[b, class_id] = torch.max(
                        class_logits[b, class_id], sem_logit
                    )

        loss = seg_cross_entropy(class_logits, masks, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(
    model,
    processor,
    prompts,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU on the validation split.

    Returns:
        mIoU as a float.
    """
    from eval.metrics.seg_metrics import SegmentationMetric

    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)

    for images, masks, _ in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        B, _, H, W = images.shape
        class_logits = torch.zeros((B, num_classes, H, W), device=device)

        for prompt_idx, prompt_word in enumerate(prompts["names"]):
            class_id = prompts["indices"][prompt_idx]
            for b in range(B):
                pil_img = Image.fromarray(
                    (images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                state = processor.set_image(pil_img)
                output = processor.set_text_prompt(state, prompt=prompt_word)
                sem_logit = output["semantic_seg"]
                sem_logit = F.interpolate(
                    sem_logit, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze()
                class_logits[b, class_id] = torch.max(class_logits[b, class_id], sem_logit)

        preds = class_logits.argmax(dim=1).cpu().numpy()
        labels = masks.cpu().numpy()

        for b in range(B):
            metric.update(preds[b], labels[b])

    results = metric.compute()
    return results["mIoU"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3-RS Remote Sensing Fine-tuning")

    # Data
    parser.add_argument("--data_root", required=True,
                        help="Root of training dataset (must contain images/ and masks/)")
    parser.add_argument("--val_root", default=None,
                        help="Validation dataset root (optional)")
    parser.add_argument("--prompts_file", required=True,
                        help="Path to class prompts file")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Crop / resize size for training (square)")
    parser.add_argument("--ignore_index", type=int, default=255,
                        help="Label value to ignore in loss computation")

    # Model
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SAM3 pre-trained checkpoint")
    parser.add_argument("--bpe_path",
                        default="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocabulary file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the image backbone (train only decoders)")
    parser.add_argument("--freeze_text_encoder", action="store_true",
                        help="Freeze the text encoder")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--amp_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help=(
            "Automatic mixed precision dtype.  Use 'bfloat16' on Ampere+ GPUs, "
            "'float16' on older GPUs, or 'float32' to disable AMP."
        ),
    )

    # Output
    parser.add_argument("--output_dir", default="outputs/finetune",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("SAM3-RS Remote Sensing Fine-tuning")
    print(f"{'=' * 60}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Prompts     : {args.prompts_file}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # ---- Load prompts ----
    from segmentor_lib.prompts import load_prompts
    prompts = load_prompts(args.prompts_file)
    if prompts is None:
        raise RuntimeError(f"Failed to load prompts from {args.prompts_file}")
    num_classes = max(prompts["indices"]) + 1
    print(f"✓ Loaded {num_classes} classes, {len(prompts['names'])} prompts")

    # ---- Build model ----
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.image_encoder.parameters():
            p.requires_grad_(False)
        print("✓ Image backbone frozen")

    if args.freeze_text_encoder:
        for p in model.backbone.text_encoder.parameters():
            p.requires_grad_(False)
        print("✓ Text encoder frozen")

    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

    # ---- Datasets ----
    image_size = (args.image_size, args.image_size)
    train_dataset = RSSegDataset(
        root=args.data_root,
        image_size=image_size,
        ignore_index=args.ignore_index,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"✓ Training set: {len(train_dataset)} images")

    val_loader = None
    if args.val_root:
        val_dataset = RSSegDataset(
            root=args.val_root,
            image_size=image_size,
            ignore_index=args.ignore_index,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(f"✓ Validation set: {len(val_dataset)} images")

    # ---- Optimiser ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    best_miou = 0.0
    log_path = os.path.join(args.output_dir, "train_log.txt")

    with open(log_path, "w") as log_fh:
        log_fh.write("epoch,train_loss,val_miou\n")

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_loss = train_one_epoch(
                model, processor, prompts, train_loader,
                optimizer, device, num_classes, args.ignore_index,
                amp_dtype=args.amp_dtype,
            )
            scheduler.step()
            print(f"  Train loss : {train_loss:.4f}")

            val_miou = 0.0
            if val_loader is not None:
                val_miou = validate(
                    model, processor, prompts, val_loader,
                    device, num_classes, args.ignore_index,
                )
                print(f"  Val mIoU   : {val_miou:.4f}")

            log_fh.write(f"{epoch},{train_loss:.6f},{val_miou:.6f}\n")
            log_fh.flush()

            # Save checkpoint
            if epoch % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_miou": val_miou,
                    },
                    ckpt_path,
                )
                print(f"  Checkpoint : {ckpt_path}")

            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ New best mIoU {best_miou:.4f} → {best_path}")

    print(f"\n{'=' * 60}")
    print(f"Training complete.  Best val mIoU: {best_miou:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
