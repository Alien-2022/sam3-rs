"""Prompt management module for SAM3-RS.

This module provides core functionality for loading prompts and
pre-computing text features.
"""

import os
from typing import Dict, Optional
import torch


def load_prompts(prompts_file: Optional[str]) -> Optional[Dict]:
    """Load class names and their indices from config file.

    File format (same as SegEarthOV3):
        background
        bareland,barren
        grass
        road
        car,vehicle
        tree,forest
        water,river
        cropland,farmland
        building,roof,house

    Returns:
        Dict with 'names', 'indices', 'mapping' (synonym -> class_id)
    """
    if not prompts_file or not os.path.exists(prompts_file):
        return None

    names, indices, mapping = [], [], {}

    with open(prompts_file, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Split by comma for synonyms
            synonyms = [s.strip() for s in line.split(",") if s.strip()]
            class_id = line_idx

            names.extend(synonyms)
            indices.extend([class_id] * len(synonyms))

            # Create mapping: each synonym maps to class_id
            for synonym in synonyms:
                mapping[synonym] = class_id

    return {"names": names, "indices": indices, "mapping": mapping}


def precompute_text_features(processor, prompts: Dict, device, num_prompts: int) -> Optional[Dict]:
    """Pre-compute text features for all prompts to avoid repeated computation during inference.

    Args:
        processor: SAM3Processor instance
        prompts: Dict with 'names'
        device: torch.device
        num_prompts: Number of prompts

    Returns:
        Dictionary mapping prompt_idx to text features
    """
    if not prompts or num_prompts == 0:
        return None

    print(f"Pre-computing text features for {num_prompts} prompts...")

    # Initialize cache dictionary
    text_features_cache = {}

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        for prompt_idx, prompt_word in enumerate(prompts["names"]):
            # Compute text features for this prompt
            text_outputs = processor.model.backbone.forward_text(
                [prompt_word], device=device
            )

            # Store the features in cache
            text_features_cache[prompt_idx] = {
                "language_features": text_outputs.get("language_features"),
                "language_mask": text_outputs.get("language_mask"),
                "language_embeds": text_outputs.get("language_embeds"),
            }

            # Progress indicator
            if (prompt_idx + 1) % 10 == 0 or prompt_idx == num_prompts - 1:
                print(f"  Progress: {prompt_idx + 1}/{num_prompts}")

    print(f"✓ Text features pre-computed successfully!")
    return text_features_cache
