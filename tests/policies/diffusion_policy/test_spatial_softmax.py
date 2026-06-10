#!/usr/bin/env python

"""Scoped tests for the from-scratch SpatialSoftmax module."""

import pytest
import torch

from lerobot.policies.diffusion_policy.spatial_softmax import SpatialSoftmax


def test_output_shape_without_learnable_keypoint_projection():
    batch_size, channels, height, width = 2, 3, 4, 5
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=None)
    features = torch.randn(batch_size, channels, height, width)

    out = module(features)

    assert out.shape == (batch_size, channels, 2)


def test_output_shape_with_learnable_keypoint_projection():
    batch_size, channels, height, width, num_keypoints = 2, 3, 4, 5, 7
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=num_keypoints)
    features = torch.randn(batch_size, channels, height, width)

    out = module(features)

    assert out.shape == (batch_size, num_keypoints, 2)


def test_uniform_feature_map_returns_grid_center_without_projection():
    batch_size, channels, height, width = 2, 3, 3, 5
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=None)
    features = torch.zeros(batch_size, channels, height, width)

    out = module(features)

    torch.testing.assert_close(out, torch.zeros(batch_size, channels, 2), atol=1e-6, rtol=0.0)


def test_uniform_feature_map_returns_grid_center_with_projection():
    batch_size, channels, height, width, num_keypoints = 2, 3, 3, 5, 4
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=num_keypoints)
    assert module.keypoint_projection is not None
    with torch.no_grad():
        module.keypoint_projection.weight.zero_()
        module.keypoint_projection.bias.zero_()

    features = torch.randn(batch_size, channels, height, width)

    out = module(features)

    torch.testing.assert_close(out, torch.zeros(batch_size, num_keypoints, 2), atol=1e-6, rtol=0.0)


def test_dominant_activation_returns_expected_normalized_coordinate():
    batch_size, channels, height, width = 1, 2, 3, 5
    target_row, target_col = 2, 3
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=None)
    features = torch.full((batch_size, channels, height, width), -20.0)
    features[:, :, target_row, target_col] = 20.0

    out = module(features)

    expected_x = torch.linspace(-1.0, 1.0, steps=width)[target_col]
    expected_y = torch.linspace(-1.0, 1.0, steps=height)[target_row]
    expected = torch.tensor([expected_x, expected_y]).expand(batch_size, channels, 2)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.0)


def test_position_grid_is_registered_buffer_not_parameter_and_moves_with_module():
    module = SpatialSoftmax(input_shape=(2, 3, 4))

    assert "pos_grid" in dict(module.named_buffers())
    assert "pos_grid" not in dict(module.named_parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    assert module.pos_grid.device == device


@pytest.mark.parametrize("num_keypoints", [None, 4])
def test_gradient_flow_to_features_and_optional_projection(num_keypoints):
    batch_size, channels, height, width = 2, 3, 4, 5
    module = SpatialSoftmax(input_shape=(channels, height, width), num_keypoints=num_keypoints)
    features = torch.randn(batch_size, channels, height, width, requires_grad=True)

    out = module(features)
    out.sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    if module.keypoint_projection is not None:
        for parameter in module.keypoint_projection.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "input_shape",
    [
        (3, 4),
        (3, 4, 5, 6),
        (0, 4, 5),
        (3, 0, 5),
        (3, 4, 0),
        (-1, 4, 5),
        (3, -1, 5),
        (3, 4, -1),
    ],
)
def test_invalid_constructor_input_shape_raises_clear_error(input_shape):
    with pytest.raises(ValueError, match="input_shape"):
        SpatialSoftmax(input_shape=input_shape)


@pytest.mark.parametrize("num_keypoints", [0, -1])
def test_invalid_constructor_num_keypoints_raises_clear_error(num_keypoints):
    with pytest.raises(ValueError, match="num_keypoints"):
        SpatialSoftmax(input_shape=(3, 4, 5), num_keypoints=num_keypoints)


def test_forward_rejects_non_4d_input():
    module = SpatialSoftmax(input_shape=(3, 4, 5))

    with pytest.raises(ValueError, match="4D"):
        module(torch.randn(3, 4, 5))


def test_forward_rejects_channel_mismatch():
    module = SpatialSoftmax(input_shape=(3, 4, 5))

    with pytest.raises(ValueError, match="input_shape"):
        module(torch.randn(2, 4, 4, 5))


@pytest.mark.parametrize("shape", [(2, 3, 5, 5), (2, 3, 4, 6)])
def test_forward_rejects_height_or_width_mismatch(shape):
    module = SpatialSoftmax(input_shape=(3, 4, 5))

    with pytest.raises(ValueError, match="input_shape"):
        module(torch.randn(*shape))
