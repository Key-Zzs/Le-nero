#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("source_kind", ["numpy", "torch"])
def test_add_frame_snapshots_mutable_non_image_arrays(
    tmp_path, empty_lerobot_dataset_factory, source_kind
):
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    original = np.array([1.0, 2.0], dtype=np.float32)
    source = original if source_kind == "numpy" else torch.from_numpy(original)

    dataset.add_frame({"state": source, "task": "Dummy task"})
    original[...] = -1

    buffered = dataset.episode_buffer["state"][0]
    np.testing.assert_array_equal(buffered, np.array([1.0, 2.0], dtype=np.float32))
    assert buffered.flags.owndata
