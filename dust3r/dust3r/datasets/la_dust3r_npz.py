# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""Dataset loader for the per-view NPZ files produced by assets/installer."""
import json
import os
import re

import cv2
import numpy as np
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


def _linear_to_srgb(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1 / 2.4) - 0.055)


class LADUSt3RNPZDataset(BaseStereoViewDataset):
    """Load generated scene NPZs and form same-scene view pairs.

    ``root`` may be the generated dataset directory (containing ``npz/``) or
    that ``npz/`` directory itself. The corresponding scene ``meta.json``
    supplies pair membership. ``image_gain`` is a fixed exposure conversion
    used only for the DUSt3R network input; the photometric target remains
    unclamped linear renderer RGB.

    Training configuration expression::

        LADUSt3RNPZDataset(root='assets/data', split='train', resolution=224)
    """

    def __init__(self, root, *, split=None, resolution=224, image_gain=1 / 1000, max_lights=8,
                 pairs_per_scene='all', **kwargs):
        super().__init__(split=split, resolution=resolution, **kwargs)
        root = os.path.abspath(os.path.expanduser(root))
        self.data_root = os.path.join(root, 'npz') if os.path.isdir(os.path.join(root, 'npz')) else root
        self.scene_root = os.path.dirname(self.data_root) if os.path.basename(self.data_root) == 'npz' else self.data_root
        if image_gain <= 0:
            raise ValueError('image_gain must be positive')
        if pairs_per_scene != 'all' and int(pairs_per_scene) < 1:
            raise ValueError("pairs_per_scene must be 'all' or a positive integer")
        self.image_gain = float(image_gain)
        self.max_lights = int(max_lights)
        if self.max_lights < 1:
            raise ValueError('max_lights must be positive')
        self.pairs_per_scene = pairs_per_scene
        scene_names = sorted(name for name in os.listdir(self.data_root)
                             if re.fullmatch(r'scene_\d+', name)
                             and os.path.isfile(os.path.join(self.scene_root, name, 'meta.json')))
        if split in ('train', 'val', 'test'):
            # Deterministic scene-level split prevents views from one scene
            # appearing in both training and evaluation.
            ids = [int(name.split('_')[-1]) for name in scene_names]
            unique_ids = sorted(set(ids))
            n = len(unique_ids)
            n_train = max(1, int(round(0.8 * n))) if n > 1 else n
            n_val = max(1, int(round(0.1 * n))) if n >= 4 else 0
            if n_train + n_val >= n and n >= 3:
                n_train = n - n_val - 1
            selected = {'train': set(unique_ids[:n_train]),
                        'val': set(unique_ids[n_train:n_train + n_val]),
                        'test': set(unique_ids[n_train + n_val:])}[split]
            scene_names = [name for name in scene_names if int(name.split('_')[-1]) in selected]
        elif split not in (None, 'all'):
            raise ValueError("split must be one of None, 'all', 'train', 'val', or 'test'")

        self.scenes = []
        for scene_name in scene_names:
            meta_path = os.path.join(self.scene_root, scene_name, 'meta.json')
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            views = [os.path.splitext(view['file'])[0] + '.npz' for view in meta['views']]
            view_pairs = [(a, b) for i, a in enumerate(views) for b in views[i + 1:]]
            if pairs_per_scene != 'all':
                view_pairs = view_pairs[:int(pairs_per_scene)]
            if view_pairs:
                self.scenes.append((scene_name, meta, view_pairs))
        self.pairs = [(scene_idx, pair_idx) for scene_idx, (_, _, pairs) in enumerate(self.scenes)
                      for pair_idx in range(len(pairs))]
        if not self.pairs:
            raise ValueError(f'No scene pairs found under {self.data_root!r} for split={split!r}')

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self.pairs)} pairs from {len(self.scenes)} scenes'

    def _load_view(self, scene_name, meta, npz_name, resolution):
        path = os.path.join(self.data_root, scene_name, npz_name)
        with np.load(path) as data:
            rgb = np.asarray(data['rgb'], dtype=np.float32)
            depth = np.asarray(data['depth'], dtype=np.float32)
            albedo = np.asarray(data['albedo'], dtype=np.float32)
            mask = np.asarray(data['mask'], dtype=bool)
            K = np.asarray(data['K'], dtype=np.float32).copy()
            c2w = np.asarray(data['c2w'], dtype=np.float32)
            light_pos = np.asarray(data['light_pos'], dtype=np.float32)
            light_E = np.asarray(data['light_E'], dtype=np.float32)

        height, width = depth.shape
        if rgb.shape != (height, width, 3) or albedo.shape != rgb.shape or mask.shape != depth.shape:
            raise ValueError(f'Inconsistent spatial fields in {path}')
        if not np.isfinite(rgb).all() or not np.isfinite(depth).all() or not np.isfinite(albedo).all():
            raise ValueError(f'Non-finite image/geometry values in {path}')
        if light_pos.ndim != 2 or light_pos.shape[-1] != 3 or light_E.shape != light_pos.shape:
            raise ValueError(f'Expected light_pos/light_E with shape (K,3) in {path}')
        if light_pos.shape[0] > self.max_lights:
            raise ValueError(f'{path} has {light_pos.shape[0]} lights, exceeding max_lights={self.max_lights}')
        count = light_pos.shape[0]
        light_mask = np.ones(count, dtype=bool)
        if count < self.max_lights:
            pad = self.max_lights - count
            light_pos = np.pad(light_pos, ((0, pad), (0, 0)))
            light_E = np.pad(light_E, ((0, pad), (0, 0)))
            light_mask = np.pad(light_mask, (0, pad))
        depth = np.where(mask, depth, 0).astype(np.float32)

        out_w, out_h = resolution
        # Resize to cover the requested shape, then center crop. This applies
        # the same pixel transform to every spatial target and to intrinsics.
        scale = max(out_w / width, out_h / height)
        new_w, new_h = max(out_w, int(round(width * scale))), max(out_h, int(round(height * scale)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        albedo = cv2.resize(albedo, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        K[0, :] *= new_w / width
        K[1, :] *= new_h / height
        left, top = (new_w - out_w) // 2, (new_h - out_h) // 2
        rgb = rgb[top:top + out_h, left:left + out_w]
        depth = depth[top:top + out_h, left:left + out_w]
        albedo = albedo[top:top + out_h, left:left + out_w]
        mask = mask[top:top + out_h, left:left + out_w]
        K[0, 2] -= left
        K[1, 2] -= top
        # build_dataset backprojects with pixel centers j+0.5; DUSt3R's
        # geometry utility uses integer pixel coordinates, so shift the
        # principal point by -0.5 to preserve the exact supplied pointmap.
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5

        photo_rgb = np.maximum(rgb, 0).astype(np.float32)
        display = np.clip(_linear_to_srgb(photo_rgb * self.image_gain), 0, 1)
        image = Image.fromarray(np.round(display * 255).astype(np.uint8), mode='RGB')
        view_id = os.path.splitext(npz_name)[0]
        return dict(img=image, depthmap=depth.astype(np.float32), camera_intrinsics=K,
                    camera_pose=c2w, dataset='LA-DUSt3R', label=scene_name, instance=view_id,
                    metric_scale=True, albedo=albedo.astype(np.float32), photo_rgb=photo_rgb,
                    photo_rgb_linear=True, photo_valid=True, albedo_valid=True,
                    lights={'positions': light_pos, 'intensities': light_E, 'mask': light_mask})

    def _get_views(self, idx, resolution, rng):
        scene_idx, pair_idx = self.pairs[idx]
        scene_name, meta, pairs = self.scenes[scene_idx]
        names = pairs[pair_idx]
        return [self._load_view(scene_name, meta, name, resolution) for name in names]
