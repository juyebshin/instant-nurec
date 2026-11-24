# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from instant_nurec.config_schema.dataset import PMTIndexDatasetConfig
from instant_nurec.config_schema.instantnurec import InstantNuRecConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.utils.batch import InstantNuRecDataBatch


if TYPE_CHECKING:
    from instant_nurec.datasets.pmt_index import PMTIndexDataset


class InstantNuRecDataModule:
    """Predict datamodule configured from the public input schema."""

    def __init__(self, instantnurec_config: InstantNuRecConfig) -> None:
        self.instantnurec_config = instantnurec_config
        self.predict_dataset: NCoreInstantNuRecDataset | PMTIndexDataset | None = None

    def predict_dataloader(self) -> DataLoader:
        dataset_config = self.instantnurec_config.dataset.predict
        assert dataset_config is not None, "dataset.predict has to be specified in the config to use the predict mode"

        if isinstance(dataset_config, PMTIndexDatasetConfig):  # PMT index logs (AV2 / nuScenes / Waymo / OpenScene)
            from instant_nurec.datasets.pmt_index import PMTIndexDataset

            self.predict_dataset = PMTIndexDataset(dataset_config, global_seed=self.instantnurec_config.seed)
        else:
            self.predict_dataset = NCoreInstantNuRecDataset(
                dataset_config,
                frame_width=dataset_config.camera_subsampler.frame_width,
                frame_height=dataset_config.camera_subsampler.frame_height,
                n_frames_per_sample=dataset_config.frame_batch_sampler.n_frames_per_sample,
                global_seed=self.instantnurec_config.seed,
            )
        return DataLoader(
            self.predict_dataset,
            num_workers=self.instantnurec_config.system.predict_num_workers,
            persistent_workers=False,
            batch_size=self.instantnurec_config.system.predict_batch_size,
            pin_memory=True,
            collate_fn=InstantNuRecDataBatch.collate_fn,
        )
