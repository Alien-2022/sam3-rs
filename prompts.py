"""
Remote sensing prompt management and optimization.

Features:
- Prompt loading from config files
- Synonym handling (e.g., 'building,roof,house' -> class 8)
- Prompt generation strategies
- Class name to class ID mapping
"""

import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class PromptConfig:
    """Configuration for class prompts"""
    # Source
    config_file: Optional[str] = None  # Path to prompts file

    # Generation options (future: auto-generate prompts)
    use_synonyms: bool = True
    include_attributes: bool = False  # e.g., "large building"
    base_class_only: bool = True  # Or use variations

    # Optimization options (future: CLIP-based prompt ranking)
    temperature: float = 0.0


class PromptManager:
    """
    Manage text prompts for remote sensing segmentation.

    Key features:
    1. Load prompts from file (SegEarthOV3 format)
    2. Synonym mapping for better coverage
    3. Easy to extend with new classes
    4. Support for prompt optimization strategies
    """

    def __init__(self, config: PromptConfig):
        """
        Args:
            config: PromptConfig object
        """
        self.config = config
        self.class_names: List[str] = []
        self.class_indices: List[int] = []
        self.synonym_mapping: Dict[str, int] = {}
        self.class_to_synonyms: Dict[int, List[str]] = {}

        if config.config_file:
            self._load_from_file(config.config_file)
        else:
            print("Warning: No prompt config file provided")

    def _load_from_file(self, file_path: str):
        """
        Load prompts from config file.

        File format (SegEarthOV3 compatible):
            background
            bareland,barren
            grass
            road,route
            car,vehicle
            tree,forest
            water,river
            cropland,farmland
            building,roof,house

        Creates:
            - class_names: ['background', 'bareland', 'barren', 'grass', ...]
            - class_indices: [0, 1, 1, 2, ...]
            - synonym_mapping: {'bareland': 1, 'barren': 1, ...}
            - class_to_synonyms: {1: ['bareland', 'barren'], 2: ['grass'], ...}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt config not found: {file_path}")

        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()

                if not line:
                    continue

                # Split by comma for synonyms
                synonyms = [s.strip() for s in line.split(',') if s.strip()]

                class_id = line_idx
                primary_name = synonyms[0]  # First one is primary

                # Store all names
                self.class_names.extend(synonyms)

                # Each synonym maps to the same class ID
                for synonym in synonyms:
                    self.synonym_mapping[synonym] = class_id

                # Class ID maps to all its synonyms
                self.class_to_synonyms[class_id] = synonyms
                self.class_indices.extend([class_id] * len(synonyms))

        print(f"✓ Loaded {len(self.class_to_synonyms)} classes from {file_path}")
        for cls_id, synonyms in self.class_to_synonyms.items():
            print(f"  Class {cls_id}: {', '.join(synonyms)}")

    def get_class_name(self, class_id: int) -> str:
        """Get primary class name for given class ID."""
        if class_id not in self.class_to_synonyms:
            return f"class_{class_id}"
        return self.class_to_synonyms[class_id][0]

    def get_class_id(self, prompt: str) -> Optional[int]:
        """Get class ID for a given prompt (handles synonyms)."""
        return self.synonym_mapping.get(prompt)

    def get_all_prompts(self) -> List[str]:
        """Get all prompts (all synonyms)."""
        return self.class_names

    def get_prompts_by_class(self, class_id: int) -> List[str]:
        """Get all prompts for a given class ID."""
        return self.class_to_synonyms.get(class_id, [f"class_{class_id}"])

    def get_primary_prompts(self) -> List[str]:
        """
        Get only primary prompts (first synonym for each class).

        Returns:
            List of primary class names
        """
        return [synonyms[0] for synonyms in self.class_to_synonyms.values()]

    def generate_variations(self, base_class: str) -> List[str]:
        """
        Generate prompt variations for a class.

        Future: Use CLIP embeddings or GPT to generate effective variations.

        Args:
            base_class: Base class name (e.g., "building")

        Returns:
            List of variations
        """
        variations = [base_class]

        # Simple rule-based variations (can be extended)
        if self.config.include_attributes:
            variations.extend([
                f"large {base_class}",
                f"small {base_class}",
                f"dense {base_class}",
                f"sparse {base_class}"
            ])

        return variations

    def save_mapping(self, output_path: str):
        """
        Save class mapping as JSON for reference/visualization.

        Example output:
            {
                "class_ids": [0, 1, 2, ...],
                "class_names": ["background", "bareland", ...],
                "synonym_mapping": {
                    "bareland": 1,
                    "barren": 1,
                    ...
                }
            }
        """
        mapping = {
            'class_ids': list(self.class_to_synonyms.keys()),
            'class_names': [self.get_class_name(i) for i in self.class_to_synonyms.keys()],
            'synonym_mapping': self.synonym_mapping,
            'class_to_synonyms': self.class_to_synonyms
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"✓ Saved class mapping to {output_path}")


# ============ Convenience Functions ============

def create_default_prompts_file(output_path: str, classes: List[str]):
    """
    Create a default prompts config file.

    Args:
        output_path: Where to save the file
        classes: List of class names

    Example:
        classes = [
            'background',
            'building,roof,house',
            'road,route',
            'water,river',
            'tree,forest'
        ]
        create_default_prompts_file('configs/prompts.txt', classes)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for i, cls in enumerate(classes):
            if i > 0:
                f.write('\n')
            f.write(cls)

    print(f"✓ Created default prompts file at {output_path}")


def parse_prompts_from_directory(directory: str) -> PromptManager:
    """
    Parse prompt names from directory structure.

    Example:
        data/dataset/train/
            building/
                img1.png
            road/
                img2.png

    This creates prompts: ['building', 'road']

    Args:
        directory: Root directory with class subdirectories

    Returns:
        PromptManager instance
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    class_names = []
    for name in sorted(os.listdir(directory)):
        subpath = os.path.join(directory, name)
        if os.path.isdir(subpath):
            class_names.append(name)

    # Create temporary config file
    config = PromptConfig(config_file=None)
    manager = PromptManager(config)

    # Manually set class info
    for class_id, name in enumerate(class_names):
        manager.class_names.append(name)
        manager.class_indices.append(class_id)
        manager.synonym_mapping[name] = class_id
        manager.class_to_synonyms[class_id] = [name]

    return manager
