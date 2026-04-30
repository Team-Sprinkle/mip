"""Tests for observation encoders.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch

from mip.encoders import (
    FrozenSiglip2VisionEncoder,
    IdentityEncoder,
    MultiImageObsEncoder,
)


class TestIdentityEncoder:
    """Test suite for the IdentityEncoder class."""

    def test_identity_encoder_creation(self):
        """Test creating an IdentityEncoder."""
        encoder = IdentityEncoder(dropout=0.25)
        assert encoder is not None

    def test_identity_encoder_forward_with_mask(self):
        """Test IdentityEncoder forward pass with mask."""
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        bs = 4
        obs_dim = 10
        condition = torch.randn(bs, obs_dim)
        mask = torch.ones(bs)

        with torch.no_grad():
            output = encoder(condition, mask)

        assert output.shape == condition.shape
        assert torch.allclose(output, condition)


class TestMultiImageObsEncoder:
    """Test suite for the MultiImageObsEncoder class."""

    def test_multi_image_encoder_creation_rgb_only(self):
        """Test creating a MultiImageObsEncoder with RGB input only."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
        )
        assert encoder is not None

    def test_multi_image_encoder_creation_mixed_inputs(self):
        """Test creating a MultiImageObsEncoder with mixed RGB and low-dim inputs."""
        shape_meta = {
            "obs": {
                "rgb": {"shape": [3, 224, 224], "type": "rgb"},
                "state": {"shape": [10], "type": "low_dim"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
        )
        assert encoder is not None

    def test_frozen_siglip2_encoder_forward(self):
        """Test frozen SigLIP2 adapter with a local random model."""
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        config = Siglip2VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
            num_channels=3,
        )
        model = Siglip2VisionModel(config)
        encoder = FrozenSiglip2VisionEncoder(
            model_name="test-siglip2",
            model=model,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

        output = encoder(torch.rand(2, 3, 16, 16))

        assert output.shape == (2, 16)
        assert not any(param.requires_grad for param in encoder.parameters())
        encoder.train()
        assert not encoder.model.training

    def test_multi_image_encoder_creation_siglip2_mixed_inputs(self, monkeypatch):
        """Test MultiImageObsEncoder can use a frozen shared SigLIP2-style backbone."""
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        def get_test_siglip2(model_name):
            config = Siglip2VisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=16,
                patch_size=8,
                num_channels=3,
            )
            model = Siglip2VisionModel(config)
            return FrozenSiglip2VisionEncoder(
                model_name=model_name,
                model=model,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )

        monkeypatch.setattr("mip.encoders.get_siglip2", get_test_siglip2)
        shape_meta = {
            "obs": {
                "rgb": {"shape": [3, 16, 16], "type": "rgb"},
                "state": {"shape": [10], "type": "low_dim"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="siglip2:test-siglip2",
            emb_dim=32,
            use_group_norm=True,
        )

        output = encoder(
            {
                "rgb": torch.rand(2, 3, 16, 16),
                "state": torch.rand(2, 10),
            }
        )

        assert output.shape == (2, 32)
        assert encoder.share_rgb_model
        assert not any(
            param.requires_grad
            for param in encoder.key_model_map["rgb"].model.parameters()
        )
