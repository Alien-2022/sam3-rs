"""Prompt management module for SAM3-RS."""

import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np


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


def apply_semantic_enhancement(processor, prompts: Dict, num_classes: int, device) -> Dict:
    """
    Apply semantic enhancement by selecting the most representative synonym for each class.

    For each class, compute embeddings for all synonyms, find the average embedding,
    and select the synonym closest to the average as the representative prompt.
    This reduces inference time by running only one prompt per class instead of
    one per synonym.

    Args:
        processor: SAM3Processor instance
        prompts: Dict with 'names' and 'indices' (will be modified in-place)
        num_classes: Number of classes
        device: torch.device

    Returns:
        Updated prompts dict with enhanced names/indices

    Example:
        Class 0: [road, highway, street, alley, path] -> compute average -> find closest -> "road"
        Class 1: [water, river] -> compute average -> find closest -> "water"
    """
    if not prompts or num_classes == 0:
        return prompts

    print("=" * 60)
    print("Applying semantic enhancement in text space...")
    print("=" * 60)

    # Group prompts by class ID
    class_prompts = {}
    for prompt_name, class_id in zip(prompts["names"], prompts["indices"]):
        if class_id not in class_prompts:
            class_prompts[class_id] = []
        class_prompts[class_id].append(prompt_name)

    # For each class, compute average embedding
    enhanced_prompts_names = []
    enhanced_prompts_indices = []
    class_synonym_counts = {}

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        for class_id in sorted(class_prompts.keys()):
            synonyms = class_prompts[class_id]
            class_synonym_counts[class_id] = len(synonyms)

            # Compute embeddings for all synonyms of this class
            all_embeddings = []
            for synonym in synonyms:
                text_outputs = processor.model.backbone.forward_text(
                    [synonym], device=device
                )
                # Extract pooled embedding: average language_features
                language_features = text_outputs['language_features']
                language_mask = text_outputs['language_mask']

                valid_mask = ~language_mask
                valid_mask = valid_mask.transpose(0, 1).float()
                valid_mask = valid_mask.unsqueeze(-1)
                embedding = (language_features * valid_mask).sum(dim=0) / (valid_mask.sum(dim=0) + 1e-8)
                all_embeddings.append(embedding)

            # Stack and average
            stacked_embeddings = torch.stack(all_embeddings, dim=0)  # [num_synonyms, d_model]
            avg_embedding = stacked_embeddings.mean(dim=0)  # [d_model]

            # Find the synonym closest to the average embedding
            # Compute cosine similarity between each synonym and average
            similarities = []
            for emb in all_embeddings:
                # Flatten embeddings to [seq_len * d_model] for comparison
                emb_flat = emb.flatten()
                avg_flat = avg_embedding.flatten()

                # Compute cosine similarity
                sim = F.cosine_similarity(emb_flat.unsqueeze(0), avg_flat.unsqueeze(0))
                similarities.append(sim.item())

            # Select the most representative synonym
            best_idx = int(np.argmax(similarities))
            best_synonym = synonyms[best_idx]

            print(f"  Class {class_id}: {len(synonyms)} synonyms -> using '{best_synonym}' (similarity: {similarities[best_idx]:.4f})")
            print(f"    Synonyms: {', '.join(synonyms)}")

            # Store enhanced prompt info
            enhanced_prompts_names.append(best_synonym)
            enhanced_prompts_indices.append(class_id)

    # Replace prompts with enhanced versions
    prompts["names"] = enhanced_prompts_names
    prompts["indices"] = enhanced_prompts_indices

    print(f"✓ Semantic enhancement applied: {num_classes} classes, {len(enhanced_prompts_names)} enhanced prompts")
    print(f"  Average synonyms per class: {sum(class_synonym_counts.values()) / len(class_synonym_counts):.2f}")
    print(f"  Selected most representative synonym for each class based on semantic similarity")
    print("=" * 60)

    return prompts
