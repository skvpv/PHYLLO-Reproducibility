"""MMSeg dataset adapter for authoritative PlantSeg image keys.

study workflow lists preserve public identities as ``split/name.jpg``. MMSeg's
standard annotation-file loader expects suffix-free stems. This adapter keeps
the authoritative keys unchanged and maps each image key to its corresponding
``split/name.png`` segmentation mask.
"""
from __future__ import annotations

import os.path as osp

import mmengine
from mmseg.datasets.plantseg115 import PlantSeg115Dataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PhylloPlantSeg115Dataset(PlantSeg115Dataset):
    """PlantSeg115 dataset accepting full ``split/name.jpg`` list entries."""

    def load_data_list(self):
        if not self.ann_file:
            return super().load_data_list()

        assert osp.isfile(self.ann_file), (
            f"Failed to load `ann_file` {self.ann_file}"
        )
        img_dir = self.data_prefix.get("img_path")
        ann_dir = self.data_prefix.get("seg_map_path")
        data_list = []

        for raw in mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args):
            key = raw.strip().replace("\\", "/")
            if not key:
                continue
            if "/" not in key:
                raise ValueError(f"expected split/name key, got {key!r}")

            split, name = key.split("/", 1)
            if not name.endswith(self.img_suffix):
                raise ValueError(
                    f"expected image suffix {self.img_suffix!r}: {key!r}"
                )
            stem = name[:-len(self.img_suffix)]

            data_info = {
                "img_path": osp.join(img_dir, split, name),
                "label_map": self.label_map,
                "reduce_zero_label": self.reduce_zero_label,
                "seg_fields": [],
            }
            if ann_dir is not None:
                data_info["seg_map_path"] = osp.join(
                    ann_dir, split, stem + self.seg_map_suffix
                )
            data_list.append(data_info)

        return data_list
