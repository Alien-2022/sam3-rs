"""Analyzer hooks for SAM3-RS inference."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAnalyzer(ABC):
    """Base class for inference analyzers."""

    @abstractmethod
    def on_before_inference(self, context: Dict[str, Any]):
        """Called before starting inference on an image/batch."""
        pass

    @abstractmethod
    def on_after_prompt(self, context: Dict[str, Any]):
        """Called after processing each prompt."""
        pass

    @abstractmethod
    def on_after_inference(self, context: Dict[str, Any]):
        """Called after completing inference on an image/batch."""
        pass

    def report(self):
        """Print analysis results."""
        pass


class PresenceScoreAnalyzer(BaseAnalyzer):
    """Analyzer for collecting and reporting presence score statistics."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stats = []

    def on_before_inference(self, context: Dict[str, Any]):
        """No action needed before inference."""
        pass

    def on_after_prompt(self, context: Dict[str, Any]):
        """Collect presence score after each prompt."""
        if not self.enabled:
            return

        output = context.get("output")
        image_name = context.get("image_name", "unknown")
        prompt_word = context.get("prompt_word", "unknown")

        # Extract presence score from output
        if output is not None:
            if "presence_score" in output:
                # Single view mode
                ps = output["presence_score"]
                self.stats.append({
                    "image": image_name,
                    "prompt": prompt_word,
                    "presence_score": ps
                })
            elif "presence_logit_dec" in output:
                # Batch mode
                img_presence = output["presence_logit_dec"].sigmoid().max(dim=1)[0]
                for img_idx, ps in enumerate(img_presence):
                    image_names = context.get("image_names")
                    img_name = image_names[img_idx] if image_names else f"img_{img_idx}"
                    self.stats.append({
                        "image": img_name,
                        "prompt": prompt_word,
                        "presence_score": ps.item()
                    })

    def on_after_inference(self, context: Dict[str, Any]):
        """No action needed after inference."""
        pass

    def report(self):
        """Print presence score statistics."""
        if not self.enabled or not self.stats:
            print("[DEBUG] Presence score analysis disabled or no data collected")
            return

        import numpy as np

        all_scores = [s["presence_score"] for s in self.stats]

        print("=" * 80)
        print("PRESENCE SCORE ANALYSIS")
        print("=" * 80)

        # Overall statistics
        print(f"\nTotal samples: {len(self.stats)}")
        print(f"Mean: {np.mean(all_scores):.4f}")
        print(f"Median: {np.median(all_scores):.4f}")
        print(f"Std: {np.std(all_scores):.4f}")
        print(f"Min: {np.min(all_scores):.4f}")
        print(f"Max: {np.max(all_scores):.4f}")

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p}%: {np.percentile(all_scores, p):.4f}")

        # Distribution
        bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        print("\nDistribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in all_scores if bins[i] <= s < bins[i+1])
            ratio = count / len(all_scores) if all_scores else 0
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} ({ratio:6.2%})")
        count = sum(1 for s in all_scores if s == 1.0)
        ratio = count / len(all_scores) if all_scores else 0
        print(f"  [1.0, 1.0]: {count:4d} ({ratio:6.2%})")

        # Per-prompt statistics
        prompt_stats = {}
        for s in self.stats:
            prompt = s["prompt"]
            if prompt not in prompt_stats:
                prompt_stats[prompt] = []
            prompt_stats[prompt].append(s["presence_score"])

        print("\nPer-prompt statistics (sorted by mean):")
        prompt_means = [(p, np.mean(v), len(v), np.min(v), np.max(v), np.std(v))
                        for p, v in prompt_stats.items()]
        prompt_means.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Prompt':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
        print("-" * 80)
        for p, mean, count, pmin, pmax, pstd in prompt_means:
            print(f"{p:<30} {mean:8.4f} {pstd:8.4f} {pmin:8.4f} {pmax:8.4f} {count:6d}")

        print("=" * 80)
