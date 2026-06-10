#!/usr/bin/env python

"""Spatial soft-argmax module for the from-scratch Diffusion Policy.

This file is intentionally independent from the future RGB encoder. SpatialSoftmax is the first real
vision-path building block, and keeping it standalone lets us verify the feature-map-to-keypoint contract
before adding a backbone, crops, or policy-level diffusion code.
"""

import torch
from torch import nn


class SpatialSoftmax(nn.Module):
    """Convert feature maps into differentiable spatial keypoint coordinates."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_keypoints: int | None = None,
    ) -> None:
        """Create a spatial soft-argmax layer for `[C, H, W]` feature maps."""

        super().__init__()

        self._input_channels, self._height, self._width = self._validate_input_shape(input_shape)

        if num_keypoints is not None and (type(num_keypoints) is not int or num_keypoints <= 0):
            raise ValueError(f"`num_keypoints` must be a positive integer or None. Got {num_keypoints!r}.")

        self.num_keypoints = num_keypoints
        self.output_channels = num_keypoints if num_keypoints is not None else self._input_channels

        if num_keypoints is None:
            self.keypoint_projection = None
        else:
            # A 1x1 convolution learns a per-pixel channel projection from CNN channels to K keypoint maps.
            self.keypoint_projection = nn.Conv2d(self._input_channels, num_keypoints, kernel_size=1)

        # The grid is fixed geometry, not learned state. Registering it as a buffer keeps it out of optimizer
        # parameters while still moving it with the module across devices and dtypes.
        pos_grid = self._make_position_grid(self._height, self._width)
        self.register_buffer("pos_grid", pos_grid)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return expected normalized xy coordinates with shape `[B, K, 2]`."""

        self._validate_features(features)

        if self.keypoint_projection is not None:
            features = self.keypoint_projection(features)

        batch_size = features.shape[0]

        # SpatialSoftmax is a differentiable soft-argmax: each channel becomes a probability distribution over
        # all H * W positions, instead of reducing with a hard max or pooling window.
        features = features.reshape(batch_size * self.output_channels, self._height * self._width)
        attention = torch.softmax(features, dim=-1)

        # Multiplying spatial probabilities by the normalized grid computes one expected xy coordinate per
        # channel/keypoint, so the RGB encoder can later emit compact K x 2 visual features.
        expected_xy = attention @ self.pos_grid.to(device=attention.device, dtype=attention.dtype)
        return expected_xy.view(batch_size, self.output_channels, 2)

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(input_shape) != 3:
            raise ValueError(f"`input_shape` must contain exactly 3 values `(C, H, W)`. Got {input_shape!r}.")

        channels, height, width = input_shape
        for name, value in (("C", channels), ("H", height), ("W", width)):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"`input_shape` values must be positive integers for `(C, H, W)`. "
                    f"Got {name}={value!r} in {input_shape!r}."
                )

        return channels, height, width

    @staticmethod
    def _make_position_grid(height: int, width: int) -> torch.Tensor:
        x_coordinates = torch.linspace(-1.0, 1.0, steps=width)
        y_coordinates = torch.linspace(-1.0, 1.0, steps=height)
        pos_y, pos_x = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        return torch.stack((pos_x.reshape(-1), pos_y.reshape(-1)), dim=1)

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 4:
            raise ValueError(
                "`features` must be a 4D tensor with shape `[B, C, H, W]`. "
                f"Got shape {tuple(features.shape)}."
            )

        _, channels, height, width = features.shape
        expected_shape = (self._input_channels, self._height, self._width)
        observed_shape = (channels, height, width)
        if observed_shape != expected_shape:
            raise ValueError(
                "`features` shape must match `input_shape` after the batch dimension. "
                f"Expected `(C, H, W)={expected_shape}`, got {observed_shape} from full shape "
                f"{tuple(features.shape)}."
            )
