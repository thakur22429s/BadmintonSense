"""End-to-end tests for the Badminton-Sense pipeline."""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import pytest

from src.data.preprocessing import (
    select_keypoints,
    normalize_to_hip_center,
    scale_by_torso,
    temporal_resample,
    augment_horizontal_flip,
    preprocess_single,
)
from src.data.dataset import StrokeDataset
from src.models.lstm import StrokeLSTM, build_lstm
from src.models.transformer import StrokeTransformer, build_transformer
from src.models.utils import count_parameters
from src.training.losses import LabelSmoothingCrossEntropy
from src.data.download import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def dummy_keypoints():
    """Simulated MediaPipe output: (T=45, 33, 3)"""
    rng = np.random.RandomState(42)
    kps = rng.randn(45, 33, 3).astype(np.float32)
    # Make hips and shoulders have reasonable values
    kps[:, 23, :] = [0.4, 0.6, 0.0]  # left hip
    kps[:, 24, :] = [0.6, 0.6, 0.0]  # right hip
    kps[:, 11, :] = [0.4, 0.4, 0.0]  # left shoulder
    kps[:, 12, :] = [0.6, 0.4, 0.0]  # right shoulder
    return kps


class TestPreprocessing:
    def test_select_keypoints(self, dummy_keypoints, config):
        indices = config["data"]["selected_keypoints"]
        selected = select_keypoints(dummy_keypoints, indices)
        assert selected.shape == (45, len(indices), 3)

    def test_normalize_to_hip_center(self):
        kps = np.zeros((10, 5, 3))
        kps[:, 3, :] = [1.0, 2.0, 0.0]  # "left hip"
        kps[:, 4, :] = [3.0, 2.0, 0.0]  # "right hip"
        kps[:, 0, :] = [2.0, 1.0, 0.0]  # some joint

        normed = normalize_to_hip_center(kps, hip_left_idx=3, hip_right_idx=4)
        # Hip center should be at origin
        hip_center = (normed[:, 3, :] + normed[:, 4, :]) / 2
        np.testing.assert_allclose(hip_center, 0.0, atol=1e-6)

    def test_scale_by_torso(self):
        kps = np.zeros((10, 5, 3))
        kps[:, 0, :] = [0.0, 0.0, 0.0]  # left shoulder
        kps[:, 1, :] = [1.0, 0.0, 0.0]  # right shoulder
        kps[:, 2, :] = [0.0, 1.0, 0.0]  # left hip
        kps[:, 3, :] = [1.0, 1.0, 0.0]  # right hip
        kps[:, 4, :] = [0.5, 0.5, 0.0]  # some joint

        scaled = scale_by_torso(kps, 0, 1, 2, 3)
        # Torso length is 1.0 (both sides), so values should be unchanged
        np.testing.assert_allclose(scaled, kps, atol=1e-5)

    def test_temporal_resample(self):
        kps = np.random.randn(50, 15, 3)
        resampled = temporal_resample(kps, 30)
        assert resampled.shape == (30, 15, 3)

    def test_temporal_resample_identity(self):
        kps = np.random.randn(30, 15, 3)
        resampled = temporal_resample(kps, 30)
        np.testing.assert_allclose(resampled, kps)

    def test_augment_horizontal_flip(self, config):
        indices = config["data"]["selected_keypoints"]
        kps = np.random.randn(30, len(indices), 3)
        flipped = augment_horizontal_flip(kps, indices)
        assert flipped.shape == kps.shape
        # x coords should be negated
        np.testing.assert_allclose(flipped[:, 0, 0], -kps[:, 0, 0])

    def test_preprocess_single(self, dummy_keypoints, config):
        result = preprocess_single(dummy_keypoints, config, augment=False)
        expected_T = config["data"]["sequence_length"]
        expected_K = config["data"]["num_keypoints"]
        assert result.shape == (expected_T, expected_K, 3)

    def test_preprocess_with_augmentation(self, dummy_keypoints, config):
        result = preprocess_single(dummy_keypoints, config, augment=True)
        expected_T = config["data"]["sequence_length"]
        expected_K = config["data"]["num_keypoints"]
        assert result.shape == (expected_T, expected_K, 3)


class TestDataset:
    def test_stroke_dataset(self, config):
        N, T, K = 100, 30, 15
        sequences = np.random.randn(N, T, K, 3).astype(np.float32)
        labels = np.random.randint(0, 10, N)

        dataset = StrokeDataset(sequences, labels)
        assert len(dataset) == N

        x, y = dataset[0]
        assert x.shape == (T, K * 3)
        assert isinstance(y, int)


class TestModels:
    def test_lstm_forward(self, config):
        model = build_lstm(config)
        batch = torch.randn(4, 30, 45)
        out = model(batch)
        assert out.shape == (4, config["classes"]["num_classes"])

    def test_transformer_forward(self, config):
        model = build_transformer(config)
        batch = torch.randn(4, 30, 45)
        out = model(batch)
        assert out.shape == (4, config["classes"]["num_classes"])

    def test_transformer_embeddings(self, config):
        model = build_transformer(config)
        batch = torch.randn(4, 30, 45)
        emb = model.get_embeddings(batch)
        t_cfg = config["model"]["transformer"]["temporal"]
        assert emb.shape == (4, t_cfg["embed_dim"])

    def test_model_parameters(self, config):
        lstm = build_lstm(config)
        transformer = build_transformer(config)
        assert count_parameters(lstm) > 0
        assert count_parameters(transformer) > 0
        print(f"LSTM params: {count_parameters(lstm):,}")
        print(f"Transformer params: {count_parameters(transformer):,}")


class TestLoss:
    def test_label_smoothing_loss(self):
        criterion = LabelSmoothingCrossEntropy(num_classes=10, smoothing=0.1)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert loss.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
