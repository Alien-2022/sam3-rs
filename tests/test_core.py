"""Tests for segmentor_lib modules (no SAM3 model required)."""
import numpy as np
import torch
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

class TestRSEvaluator:
    """Tests for the vectorised RSEvaluator.update() in metrics.py."""

    def _make_tmp_mask(self, data: np.ndarray, tmp_path, name: str) -> str:
        from PIL import Image
        path = str(tmp_path / name)
        Image.fromarray(data.astype(np.uint8), mode="L").save(path)
        return path

    def test_perfect_prediction(self, tmp_path):
        from metrics import RSEvaluator
        evaluator = RSEvaluator(num_classes=3)
        gt = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
        pred = gt.copy()
        gt_path = self._make_tmp_mask(gt, tmp_path, "gt.png")
        pred_path = self._make_tmp_mask(pred, tmp_path, "pred.png")
        evaluator.update(pred_path, gt_path)
        result = evaluator.compute()
        assert result.mean_iou == pytest.approx(1.0, abs=1e-6)
        assert result.pixel_accuracy == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self, tmp_path):
        from metrics import RSEvaluator
        evaluator = RSEvaluator(num_classes=2)
        gt   = np.zeros((4, 4), dtype=np.uint8)
        pred = np.ones((4, 4), dtype=np.uint8)
        gt_path   = self._make_tmp_mask(gt, tmp_path, "gt.png")
        pred_path = self._make_tmp_mask(pred, tmp_path, "pred.png")
        evaluator.update(pred_path, gt_path)
        result = evaluator.compute()
        # No correct pixels
        assert result.pixel_accuracy == pytest.approx(0.0, abs=1e-6)

    def test_ignore_index(self, tmp_path):
        from metrics import RSEvaluator
        evaluator = RSEvaluator(num_classes=3, ignore_index=255)
        gt   = np.array([[0, 255, 1], [2, 255, 0]], dtype=np.uint8)
        # Pixel (1,2): gt=0 but pred=1 → wrong; all other valid pixels are correct
        pred = np.array([[0,   0, 1], [2,   0, 1]], dtype=np.uint8)
        gt_path   = self._make_tmp_mask(gt, tmp_path, "gt.png")
        pred_path = self._make_tmp_mask(pred, tmp_path, "pred.png")
        evaluator.update(pred_path, gt_path)
        result = evaluator.compute()
        # 4 valid pixels: (0,0)✓ (0,2)✓ (1,0)✓ (1,2)✗  → 3/4 correct
        assert result.pixel_accuracy == pytest.approx(3 / 4, abs=1e-6)

    def test_confusion_matrix_shape(self, tmp_path):
        from metrics import RSEvaluator
        evaluator = RSEvaluator(num_classes=5)
        assert evaluator.confusion_matrix.shape == (5, 5)


# ---------------------------------------------------------------------------
# segmentor_lib.postprocess
# ---------------------------------------------------------------------------

class TestFusePromptsToClasses:
    def test_basic_max_fusion(self):
        from segmentor_lib.postprocess import fuse_prompts_to_classes
        # 3 prompts (index 0 and 1 both map to class 0; index 2 maps to class 1)
        query_indices = torch.tensor([0, 0, 1])
        seg_logits = torch.tensor([
            [[1.0, 0.5], [0.3, 0.8]],  # prompt 0 -> class 0
            [[0.2, 0.9], [0.7, 0.4]],  # prompt 1 -> class 0 (max)
            [[0.6, 0.6], [0.6, 0.6]],  # prompt 2 -> class 1
        ])  # [3, 2, 2]
        fused = fuse_prompts_to_classes(seg_logits, query_indices, num_classes=2, num_prompts=3)
        assert fused.shape == (2, 2, 2)
        # Class 0 is max of prompt 0 and prompt 1
        expected_class0 = torch.max(seg_logits[0], seg_logits[1])
        assert torch.allclose(fused[0], expected_class0)
        assert torch.allclose(fused[1], seg_logits[2])


class TestLogitsToPred:
    def test_background_assigned_below_threshold(self):
        from segmentor_lib.postprocess import logits_to_pred
        # 2 classes, 3x3 spatial
        logits = torch.zeros(2, 3, 3)
        logits[0, :, :] = 0.05  # class 0 logits very low
        logits[1, :, :] = 0.05  # class 1 logits also low
        pred = logits_to_pred(logits, use_prompted_background=False,
                              prob_threshold=0.1, bg_idx=0)
        assert (pred == 0).all()

    def test_high_confidence_class_selected(self):
        from segmentor_lib.postprocess import logits_to_pred
        logits = torch.zeros(2, 3, 3)
        logits[1, :, :] = 0.9  # class 1 (second class) is confident
        pred = logits_to_pred(logits, use_prompted_background=False,
                              prob_threshold=0.1, bg_idx=0)
        # Without prompted background: classes are shifted, logit index 1
        # corresponds to class 2 in the argmax (bg_pad + logits)
        assert (pred == 2).all()


class TestResizeLogits:
    def test_no_op_same_shape(self):
        from segmentor_lib.postprocess import resize_logits
        t = torch.ones(3, 64, 64)
        out = resize_logits(t, (64, 64))
        assert out.shape == (3, 64, 64)
        assert torch.allclose(t, out)

    def test_upsample(self):
        from segmentor_lib.postprocess import resize_logits
        t = torch.ones(4, 32, 32)
        out = resize_logits(t, (64, 64))
        assert out.shape == (4, 64, 64)
        # A uniform tensor should remain uniform after bilinear upsampling
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5)

    def test_downsample_values(self):
        from segmentor_lib.postprocess import resize_logits
        # Constant tensor: value should be preserved exactly after interpolation
        t = torch.full((2, 64, 64), 0.5)
        out = resize_logits(t, (32, 32))
        assert out.shape == (2, 32, 32)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)

    def test_batch_resize(self):
        from segmentor_lib.postprocess import resize_logits
        t = torch.ones(2, 3, 32, 32)
        out = resize_logits(t, (64, 64))
        assert out.shape == (2, 3, 64, 64)


# ---------------------------------------------------------------------------
# segmentor_lib.prompts
# ---------------------------------------------------------------------------

class TestLoadPrompts:
    def test_load_synonyms(self, tmp_path):
        from segmentor_lib.prompts import load_prompts
        content = "background\nbuilding,house\nroad\nwater"
        p = tmp_path / "prompts.txt"
        p.write_text(content)
        result = load_prompts(str(p))
        assert result is not None
        assert "background" in result["names"]
        assert "building" in result["names"]
        assert "house" in result["names"]
        # "house" is a synonym for class 1
        assert result["mapping"]["house"] == 1
        assert result["mapping"]["building"] == 1

    def test_returns_none_for_missing_file(self):
        from segmentor_lib.prompts import load_prompts
        assert load_prompts("/nonexistent/path.txt") is None

    def test_indices_correct(self, tmp_path):
        from segmentor_lib.prompts import load_prompts
        content = "cat,feline\ndog"
        p = tmp_path / "p.txt"
        p.write_text(content)
        result = load_prompts(str(p))
        # class 0: cat, feline; class 1: dog
        assert result["indices"] == [0, 0, 1]


# ---------------------------------------------------------------------------
# segmentor_lib.multi_scale
# ---------------------------------------------------------------------------

class TestMultiScaleInference:
    def _make_fake_inference(self, num_prompts=3, h=32, w=32):
        """Return a fake inference function that yields fixed logits."""
        def inference_func(image, detailed=False, image_name=""):
            seg = torch.rand(num_prompts, image.height, image.width)
            return seg, {}, None, None, None
        return inference_func

    def test_avg_merge(self):
        from segmentor_lib.multi_scale import MultiScaleInference
        from PIL import Image as PILImage
        import numpy as np
        device = torch.device("cpu")
        func = self._make_fake_inference(num_prompts=2)
        ms = MultiScaleInference(func, scales=[0.5, 1.0], merge_mode="avg", device=device)
        img = PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        logits, per_class, sem, inst, adaptive = ms(img)
        assert logits.shape == (2, 64, 64)
        assert per_class == {}

    def test_max_merge(self):
        from segmentor_lib.multi_scale import MultiScaleInference
        from PIL import Image as PILImage
        import numpy as np
        device = torch.device("cpu")
        func = self._make_fake_inference(num_prompts=2)
        ms = MultiScaleInference(func, scales=[1.0, 1.5], merge_mode="max", device=device)
        img = PILImage.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        logits, _, _, _, _ = ms(img)
        assert logits.shape == (2, 48, 48)

    def test_invalid_merge_mode(self):
        from segmentor_lib.multi_scale import MultiScaleInference
        with pytest.raises(ValueError, match="merge_mode"):
            MultiScaleInference(lambda *a, **k: None, scales=[1.0], merge_mode="unknown")


# ---------------------------------------------------------------------------
# segmentor_lib.config_loader
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_save_load_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        from segmentor_lib.config_loader import load_inference_config, save_inference_config
        from segmentor import InferenceConfig

        cfg = InferenceConfig(
            checkpoint_path="weights/sam3.pt",
            bpe_path="sam3/assets/bpe.txt.gz",
            device="cpu",
            prob_threshold=0.2,
            prompts_file="configs/prompts.txt",
        )
        yaml_path = str(tmp_path / "test.yaml")
        save_inference_config(cfg, yaml_path)
        loaded = load_inference_config(yaml_path)
        assert loaded.checkpoint_path == cfg.checkpoint_path
        assert loaded.prob_threshold == pytest.approx(0.2)
        assert loaded.device == "cpu"

    def test_save_load_json(self, tmp_path):
        from segmentor_lib.config_loader import load_inference_config, save_inference_config
        from segmentor import InferenceConfig

        cfg = InferenceConfig(
            checkpoint_path="weights/sam3.pt",
            bpe_path="sam3/assets/bpe.txt.gz",
            confidence_threshold=0.3,
        )
        json_path = str(tmp_path / "test.json")
        save_inference_config(cfg, json_path)
        loaded = load_inference_config(json_path)
        assert loaded.confidence_threshold == pytest.approx(0.3)

    def test_overrides(self, tmp_path):
        from segmentor_lib.config_loader import load_inference_config, save_inference_config
        from segmentor import InferenceConfig

        cfg = InferenceConfig(
            checkpoint_path="weights/sam3.pt",
            bpe_path="sam3/assets/bpe.txt.gz",
            device="cpu",
        )
        json_path = str(tmp_path / "cfg.json")
        save_inference_config(cfg, json_path)
        loaded = load_inference_config(json_path, overrides={"device": "cuda"})
        assert loaded.device == "cuda"

    def test_missing_file(self):
        from segmentor_lib.config_loader import load_inference_config
        with pytest.raises(FileNotFoundError):
            load_inference_config("/nonexistent/config.yaml")

    def test_unsupported_format(self, tmp_path):
        from segmentor_lib.config_loader import load_inference_config
        p = tmp_path / "config.toml"
        p.write_text("")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_inference_config(str(p))

    def test_unknown_keys_ignored(self, tmp_path):
        import json
        import warnings
        from segmentor_lib.config_loader import load_inference_config
        data = {
            "checkpoint_path": "weights/sam3.pt",
            "bpe_path": "sam3/assets/bpe.txt.gz",
            "unknown_key_xyz": "should_be_ignored",
        }
        p = str(tmp_path / "cfg.json")
        with open(p, "w") as fh:
            json.dump(data, fh)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_inference_config(p)
        assert loaded.checkpoint_path == "weights/sam3.pt"
        assert not hasattr(loaded, "unknown_key_xyz")
        assert any("unknown_key_xyz" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# segmentor_lib.sliding_window – adaptive threshold averaging
# ---------------------------------------------------------------------------

class TestSlidingWindowAdaptiveThresholds:
    """Verify that adaptive thresholds are averaged across crops."""

    def _make_fake_adaptive_inference(self, thresholds_per_call):
        """Return an inference func that cycles through given threshold dicts."""
        call_count = [0]

        def inference_func(image, detailed=False, image_name=""):
            from PIL import Image as PILImage
            import numpy as np
            h, w = image.height, image.width
            logits = torch.zeros(2, h, w)
            adaptive = thresholds_per_call[call_count[0] % len(thresholds_per_call)]
            call_count[0] += 1
            return logits, {}, None, None, adaptive

        return inference_func

    def test_thresholds_averaged(self):
        from segmentor_lib.sliding_window import SlidingWindowInference
        from PIL import Image as PILImage
        import numpy as np

        # Two crops will be generated; each has a different threshold for class 0
        thresholds = [{0: 0.1, 1: 0.2}, {0: 0.3, 1: 0.4}]
        func = self._make_fake_adaptive_inference(thresholds)

        sw = SlidingWindowInference(
            inference_func=func,
            crop_size=32,
            stride=32,
            device=torch.device("cpu"),
        )
        # 64-wide image → 2 crops horizontally
        img = PILImage.fromarray(np.zeros((32, 64, 3), dtype=np.uint8))
        _, _, _, _, adaptive = sw(img)

        assert adaptive is not None
        assert adaptive[0] == pytest.approx((0.1 + 0.3) / 2, abs=1e-6)
        assert adaptive[1] == pytest.approx((0.2 + 0.4) / 2, abs=1e-6)
