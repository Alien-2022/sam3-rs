"""Semantic enhancement for prompt optimization.

EXPERIMENTAL: This module may be removed or significantly changed in future versions.

This module provides strategies for enhancing prompts by leveraging
synonym information in the text embedding space.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def apply_semantic_enhancement(
    processor,
    prompts: Dict,
    num_classes: int,
    device,
    sim_threshold: float = 0.5,
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

                # L2 normalize the embedding for correct cosine similarity computation
                embedding = F.normalize(embedding, p=2, dim=-1)

                all_embeddings.append(embedding)

            stacked_embeddings = torch.stack(all_embeddings, dim=0)  # [num_synonyms, seq_len, d_model]

            # Optional: Filter synonyms with low similarity to main synonym
            if sim_threshold > 0.0 and n_original > 1:
                # Use first synonym as main (reference) synonym
                main_embedding = stacked_embeddings[0]  # [seq_len, d_model]

                # Compute similarity of other synonyms to main synonym
                other_embeddings = stacked_embeddings[1:]  # [n_other, seq_len, d_model]

                # Flatten sequence dimension for comparison
                main_flat = main_embedding.flatten()  # [seq_len * d_model]
                other_flat = other_embeddings.flatten(start_dim=1)  # [n_other, seq_len * d_model]

                # Compute cosine similarity
                other_sims = F.cosine_similarity(other_flat, main_flat.unsqueeze(0)).flatten()  # Ensure 1D

                # Main synonym always has similarity 1.0 to itself
                # Ensure tensors are on the same device and same dimensions
                device_local = other_sims.device
                sims = torch.cat([torch.tensor([1.0], device=device_local), other_sims])

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


def apply_semantic_enhancement_with_avg_embedding(
    processor,
    prompts: Dict,
    num_classes: int,
    device,
    sim_threshold: float = 0.5,
    min_synonyms: int = 1
) -> Dict:
    """
    Apply semantic enhancement by using the average embedding of synonyms for each class.

    Unlike apply_semantic_enhancement which selects a representative word,
    this function computes the average embedding directly and stores it for inference.
    This avoids the bias of selecting a "closest" word and uses the full semantic information.

    For each class:
    1. Compute embeddings for all synonyms
    2. Filter out synonyms with low similarity to the main synonym (optional)
    3. Compute the average embedding of remaining synonyms
    4. Store the average embedding as the class text feature

    Args:
        processor: SAM3Processor instance
        prompts: Dict with 'names' and 'indices' (will be modified in-place)
        num_classes: Number of classes
        device: torch.device
        sim_threshold: Similarity threshold for filtering synonyms (0.0-1.0)
                     Synonyms with similarity < threshold will be filtered out.
                     Set to 0.0 to disable filtering.
        min_synonyms: Minimum number of synonyms to keep after filtering.

    Returns:
        Updated prompts dict with:
        - 'names': virtual names (e.g., "class_0", "class_1")
        - 'indices': class IDs
        - 'avg_embeddings': {class_id: avg_embedding_dict} for each class
        where avg_embedding_dict contains:
            - 'language_features': [seq_len, 1, d_model]
            - 'language_mask': [1, seq_len]
            - 'language_embeds': [seq_len, 1, d_embed]

    Example:
        Class 0: [road, highway, street, alley, path]
        -> filter with sim_threshold=0.7
        -> compute average embedding
        -> store average embedding for inference
        -> virtual name: "class_0"
    """
    if not prompts or num_classes == 0:
        return prompts

    print("=" * 60)
    print("Applying semantic enhancement with average embeddings...")
    print("=" * 60)

    # Group prompts by class ID
    class_prompts = {}
    for prompt_name, class_id in zip(prompts["names"], prompts["indices"]):
        if class_id not in class_prompts:
            class_prompts[class_id] = []
        class_prompts[class_id].append(prompt_name)

    # For each class, compute average embedding
    virtual_names = []
    virtual_indices = []
    avg_embeddings = {}
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
                # Extract text features for averaging
                language_features = text_outputs['language_features']  # [seq_len, 1, d_model]
                language_mask = text_outputs['language_mask']  # [1, seq_len]
                language_embeds = text_outputs['language_embeds']  # [seq_len, 1, d_embed]

                # Create valid mask for language_features
                valid_mask = ~language_mask
                valid_mask = valid_mask.transpose(0, 1).float()
                valid_mask = valid_mask.unsqueeze(-1)

                # Store for averaging
                all_embeddings.append({
                    'language_features': language_features,
                    'language_mask': language_mask,
                    'language_embeds': language_embeds,
                    'valid_mask': valid_mask
                })

            # Optional: Filter synonyms with low similarity to main synonym
            if sim_threshold > 0.0 and n_original > 1:
                # Compute pooled embeddings for similarity calculation
                pooled_embeddings = []
                for emb in all_embeddings:
                    # Average over valid tokens
                    # emb['language_features'] shape: [seq_len, batch, d_model] = [32, 1, 256]
                    # emb['valid_mask'] shape: [seq_len, batch, 1] = [32, 1, 1]
                    # We want to average over seq_len (dim 0), result should be [batch, d_model] = [1, 256]
                    weighted_sum = (emb['language_features'] * emb['valid_mask']).sum(dim=0)  # [1, 256]
                    mask_sum = emb['valid_mask'].sum(dim=0)  # [1, 1]
                    embedding = (weighted_sum / (mask_sum + 1e-8)).squeeze(0)  # [256]
                    # L2 normalize
                    embedding = F.normalize(embedding, p=2, dim=-1)
                    pooled_embeddings.append(embedding)

                stacked_pooled = torch.stack(pooled_embeddings, dim=0)  # [num_synonyms, d_model]

                # Use first synonym as main reference
                main_embedding = stacked_pooled[0]  # [d_model]
                other_embeddings = stacked_pooled[1:]  # [n_other, d_model]

                # Compute cosine similarity (handle case with no other synonyms)
                n_other = other_embeddings.shape[0]
                if n_other > 0:
                    # Ensure main_embedding is 1D [d_model]
                    if main_embedding.dim() > 1:
                        main_embedding = main_embedding.view(-1)
                    # Expand to [1, d_model] for broadcasting with [n, d_model]
                    main_embedding_expanded = main_embedding.unsqueeze(0)  # [1, d_model]
                    # Use cosine similarity function which handles dimensions correctly
                    other_sims = F.cosine_similarity(other_embeddings, main_embedding_expanded.expand(n_other, -1))
                    # Main synonym always has similarity 1.0
                    sims = torch.cat([torch.tensor([1.0], device=device), other_sims])
                else:
                    sims = torch.tensor([1.0], device=device)

                # Filter: keep main synonym + top-K others above threshold
                if other_embeddings.shape[0] > 0:
                    other_indices = list(range(1, n_original))

                    # Determine how many others to keep
                    n_others_above = (other_sims >= sim_threshold).sum().item()
                    n_others_needed = min_synonyms - 1

                    if n_others_above >= n_others_needed:
                        # Enough synonyms above threshold: keep only those above threshold
                        keep_other_indices = [i for i, sim in zip(other_indices, other_sims) if sim >= sim_threshold]
                    else:
                        # Not enough: keep top-K most similar (but still filter by threshold if possible)
                        top_k = min(n_others_needed, n_original - 1)
                        topk_other_indices = torch.argsort(other_sims, descending=True)[:top_k].tolist()
                        # Only keep those that meet the threshold
                        keep_other_indices = [other_indices[i] for i in topk_other_indices if other_sims[i] >= sim_threshold]
                        # If none meet threshold, fallback to just main synonym
                        if len(keep_other_indices) == 0:
                            keep_other_indices = []

                    keep_indices = [0] + keep_other_indices
                else:
                    keep_indices = [0]
                keep_indices = sorted(keep_indices)

                # Filter embeddings
                all_embeddings = [all_embeddings[i] for i in keep_indices]
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

            # Compute average embedding
            if len(all_embeddings) == 1:
                # Single synonym: use directly without averaging or normalization
                avg_language_features = all_embeddings[0]['language_features']
                avg_language_mask = all_embeddings[0]['language_mask']
                avg_language_embeds = all_embeddings[0]['language_embeds']
            else:
                # Multiple synonyms: average them
                avg_language_features = torch.stack([emb['language_features'] for emb in all_embeddings]).mean(dim=0)
                avg_language_mask = all_embeddings[0]['language_mask']  # Use first synonym's mask
                avg_language_embeds = torch.stack([emb['language_embeds'] for emb in all_embeddings]).mean(dim=0)

                # NOTE: Don't normalize averaged features!
                # These are the actual features used in the model, not for similarity calculation
                # Re-normalizing would change the model's expected input distribution

            # Store average embedding
            avg_embeddings[class_id] = {
                'language_features': avg_language_features,
                'language_mask': avg_language_mask,
                'language_embeds': avg_language_embeds
            }

            # Use virtual name for logging and result storage
            virtual_name = f"class_{class_id}"
            print(f"  Class {class_id}: {n_original} -> {n_filtered} synonyms -> avg_embedding stored")
            print(f"    Synonyms: {', '.join(synonyms)}")
            print(f"    Virtual name: '{virtual_name}'")

            # Store in prompts dict
            virtual_names.append(virtual_name)
            virtual_indices.append(class_id)

    # Update prompts with virtual names
    prompts["names"] = virtual_names
    prompts["indices"] = virtual_indices
    prompts["avg_embeddings"] = avg_embeddings

    print(f"✓ Semantic enhancement with avg embeddings applied: {num_classes} classes")
    print(f"  Total original synonyms: {total_original}")
    if sim_threshold > 0.0:
        print(f"  Total filtered out: {total_filtered} ({100*total_filtered/total_original:.1f}%)")
    print(f"  Average synonyms per class: {sum(class_synonym_counts.values()) / len(class_synonym_counts):.2f}")
    print(f"  Average embeddings stored for each class based on semantic similarity")
    print(f"  Filtering threshold: sim_threshold={sim_threshold}, min_synonyms={min_synonyms}")
    print("=" * 60)

    return prompts


def get_semantic_enhancer(mode: str):
    """
    Factory function to get semantic enhancement function by mode.

    Args:
        mode: One of 'select_word', 'avg_embedding'

    Returns:
        Enhancement function
    """
    enhancers = {
        'select_word': apply_semantic_enhancement,
        'avg_embedding': apply_semantic_enhancement_with_avg_embedding,
    }

    if mode not in enhancers:
        raise ValueError(
            f"Unknown enhancement mode '{mode}'. "
            f"Available modes: {list(enhancers.keys())}"
        )

    return enhancers[mode]
