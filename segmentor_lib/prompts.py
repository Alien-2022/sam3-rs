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


def apply_semantic_enhancement(
    processor,
    prompts: Dict,
    num_classes: int,
    device,
    sim_threshold: float = 0.7,
    min_synonyms: int = 1
) -> Dict:
    """
    Apply semantic enhancement by selecting the most representative synonym for each class.

    For each class:
    1. Compute embeddings for all synonyms
    2. Filter out synonyms with low similarity to the initial average (optional)
    3. Compute the average embedding of remaining synonyms
    4. Select the synonym closest to the average as the representative prompt

    This reduces inference time by running only one prompt per class instead of
    one per synonym.

    Args:
        processor: SAM3Processor instance
        prompts: Dict with 'names' and 'indices' (will be modified in-place)
        num_classes: Number of classes
        device: torch.device
        sim_threshold: Similarity threshold for filtering synonyms (0.0-1.0)
                     Synonyms with similarity < threshold will be filtered out.
                     Set to 0.0 to disable filtering.
        min_synonyms: Minimum number of synonyms to keep after filtering.
                     Ensures at least this many synonyms are used for averaging.

    Returns:
        Updated prompts dict with enhanced names/indices

    Example:
        Class 0: [road, highway, street, alley, path]
        -> filter outliers with sim_threshold=0.7
        -> compute average -> find closest -> "road"
        Class 1: [water, river]
        -> compute average -> find closest -> "water"
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

    # For each class, compute average embedding with optional filtering
    enhanced_prompts_names = []
    enhanced_prompts_indices = []
    class_synonym_counts = {}
    total_filtered = 0
    total_original = 0

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        for class_id in sorted(class_prompts.keys()):
            synonyms = class_prompts[class_id]
            n_original = len(synonyms)
            total_original += n_original
            class_synonym_counts[class_id] = n_original

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

            stacked_embeddings = torch.stack(all_embeddings, dim=0)  # [num_synonyms, d_model]

            # Optional: Filter synonyms with low similarity to main synonym
            if sim_threshold > 0.0 and n_original > 1:
                # Use first synonym as main (reference) synonym
                main_embedding = stacked_embeddings[0]

                # Compute similarity of other synonyms to main synonym
                other_embeddings = stacked_embeddings[1:]
                other_sims = F.cosine_similarity(other_embeddings, main_embedding.unsqueeze(0))

                # Main synonym always has similarity 1.0 to itself
                sims = torch.cat([torch.tensor([1.0]), other_sims])

                # Filter: keep main synonym + top-K others above threshold
                other_indices = list(range(1, n_original))

                # Determine how many others to keep
                n_others_above = (other_sims >= sim_threshold).sum().item()
                n_others_needed = min_synonyms - 1

                if n_others_above >= n_others_needed:
                    # Enough: keep only those above threshold
                    keep_other_indices = [i for i, sim in zip(other_indices, other_sims) if sim >= sim_threshold]
                else:
                    # Not enough: keep top-K most similar others
                    top_k = min(n_others_needed, n_original - 1)
                    topk_other_indices = torch.argsort(other_sims, descending=True)[:top_k].tolist()
                    keep_other_indices = [other_indices[i] for i in topk_other_indices]

                # Main synonym (index 0) must be kept
                keep_indices = [0] + keep_other_indices
                keep_indices = sorted(keep_indices)  # Restore original order

                # Filter embeddings and synonyms
                stacked_embeddings = stacked_embeddings[keep_indices]
                filtered_synonyms = [synonyms[i] for i in keep_indices]
                n_filtered = len(filtered_synonyms)

                if n_filtered < n_original:
                    print(f"  Class {class_id}: Filtered {n_original} -> {n_filtered} synonyms (threshold={sim_threshold})")
                    print(f"    Main synonym: '{synonyms[0]}' (always kept)")
                    for i, syn in enumerate(synonyms):
                        if i == 0:
                            print(f"    {syn:20s} sim=1.0000 [MAIN]")
                        else:
                            marker = " [KEEP]" if i in keep_indices else " [DROP]"
                            print(f"    {syn:20s} sim={sims[i]:.4f}{marker}")

                total_filtered += (n_original - n_filtered)
            else:
                filtered_synonyms = synonyms
                n_filtered = n_original

            # Compute final average embedding
            avg_embedding = stacked_embeddings.mean(dim=0)  # [d_model]

            # Find the synonym closest to the average embedding
            similarities = []
            for emb in stacked_embeddings:
                # Flatten embeddings to [seq_len * d_model] for comparison
                emb_flat = emb.flatten()
                avg_flat = avg_embedding.flatten()

                # Compute cosine similarity
                sim = F.cosine_similarity(emb_flat.unsqueeze(0), avg_flat.unsqueeze(0))
                similarities.append(sim.item())

            # Select the most representative synonym
            best_idx = int(np.argmax(similarities))
            best_synonym = filtered_synonyms[best_idx]

            print(f"  Class {class_id}: {n_original} -> {n_filtered} synonyms -> '{best_synonym}' (similarity: {similarities[best_idx]:.4f})")
            if n_filtered < n_original:
                print(f"    Original: {', '.join(synonyms)}")
                print(f"    Filtered: {', '.join(filtered_synonyms)}")
            else:
                print(f"    Synonyms: {', '.join(synonyms)}")

            # Store enhanced prompt info
            enhanced_prompts_names.append(best_synonym)
            enhanced_prompts_indices.append(class_id)

    # Replace prompts with enhanced versions
    prompts["names"] = enhanced_prompts_names
    prompts["indices"] = enhanced_prompts_indices

    print(f"✓ Semantic enhancement applied: {num_classes} classes, {len(enhanced_prompts_names)} enhanced prompts")
    print(f"  Total original synonyms: {total_original}")
    if sim_threshold > 0.0:
        print(f"  Total filtered out: {total_filtered} ({100*total_filtered/total_original:.1f}%)")
    print(f"  Average synonyms per class: {sum(class_synonym_counts.values()) / len(class_synonym_counts):.2f}")
    print(f"  Selected most representative synonym for each class based on semantic similarity")
    print(f"  Filtering threshold: sim_threshold={sim_threshold}, min_synonyms={min_synonyms}")
    print("=" * 60)

    return prompts
