# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""Adapt stereo datasets to LA-DUSt3R's training-view schema."""
import numpy as np
import torch

from .easy_dataset import EasyDataset


class LADUSt3RDatasetAdapter(EasyDataset):
    """Wrap a DUSt3R stereo dataset and normalize LA-DUSt3R supervision fields.

    Wrap a dataset in the existing dataset expression, for example::

        LADUSt3RDatasetAdapter(
            MyStereoDataset(...), max_lights=2, rgb_encoding='srgb')

    Input views must already contain images, camera data, depth and masks in
    the dataset's final resolution. Optional canonical fields are ``is_metric``
    (or ``metric_scale``), ``albedo`` (or ``gt_albedo``), camera-local lights
    as a ``lights``/``light_params`` dict or ``(K,6)`` array, and the
    renderer-space target ``photo_rgb`` (or an accepted RGB alias). The
    adapter converts these to batched-collatable tensors and pads light sets
    to ``max_lights`` with an explicit validity mask.

    ``rgb_encoding`` describes the supplied target: ``linear`` and ``encoded``
    preserve values, while ``srgb`` applies the standard sRGB-to-linear
    transfer before it is used for photometric supervision. Use ``encoded``
    for a fixed renderer encoding that is not sRGB.
    """
    RGB_ALIASES = ('photo_rgb', 'linear_rgb', 'renderer_rgb', 'photometric_rgb', 'rgb_linear', 'rgb')
    ALBEDO_ALIASES = ('albedo', 'gt_albedo', 'albedo_gt')

    def __init__(self, dataset, max_lights=8, rgb_encoding='linear',
                 field_map=None, require_lighting=False, require_albedo=False,
                 require_photo_rgb=False, rho_max=1.0):
        if max_lights < 1:
            raise ValueError('max_lights must be positive')
        if rgb_encoding not in ('linear', 'srgb', 'encoded'):
            raise ValueError("rgb_encoding must be 'linear', 'srgb', or 'encoded'")
        self.dataset = dataset
        self.max_lights = int(max_lights)
        self.rgb_encoding = rgb_encoding
        self.field_map = dict(field_map or {})
        self.require_lighting = require_lighting
        self.require_albedo = require_albedo
        self.require_photo_rgb = require_photo_rgb
        self.rho_max = float(rho_max)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (f'LADUSt3RDatasetAdapter({self.dataset!r}, max_lights={self.max_lights}, '
                f'rgb_encoding={self.rgb_encoding!r})')

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        views = self.dataset[index]
        if not isinstance(views, (tuple, list)) or len(views) != 2:
            raise ValueError('LA-DUSt3R adapter expects a pair of stereo views')
        return tuple(self._adapt_view(view) for view in views)

    def _lookup(self, view, target, aliases):
        source = self.field_map.get(target)
        candidates = (source,) if source is not None else aliases
        for candidate in candidates:
            value = view
            for part in candidate.split('.'):
                if not isinstance(value, dict) or part not in value:
                    value = None
                    break
                value = value[part]
            if value is not None:
                return value
        return None

    @staticmethod
    def _tensor(value, dtype=None):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype) if dtype is not None else value
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        return torch.as_tensor(value.copy() if not value.flags.writeable else value, dtype=dtype)

    @classmethod
    def _as_hwc(cls, value, name):
        tensor = cls._tensor(value)
        if tensor.ndim != 3:
            raise ValueError(f'{name} must be a rank-three color image, got {tuple(tensor.shape)}')
        if tensor.shape[-1] == 3:
            return tensor.contiguous()
        if tensor.shape[0] == 3:
            return tensor.permute(1, 2, 0).contiguous()
        raise ValueError(f'{name} must have three color channels, got {tuple(tensor.shape)}')

    @staticmethod
    def _scale_rgb(value):
        # Integer images use the conventional 8-bit range. Floating-point
        # renderer outputs (including radiance > 1) are already in scene units.
        if not torch.is_floating_point(value):
            return value.float() / 255.0
        return value.float()

    def _adapt_lights(self, view):
        raw = self._lookup(view, 'lights', ('lights', 'light_params', 'lighting'))
        mask = None
        if isinstance(raw, dict):
            positions = raw.get('positions', raw.get('position'))
            intensities = raw.get('intensities', raw.get('intensity', raw.get('rgb_intensity')))
            mask = raw.get('mask', raw.get('valid'))
        elif raw is not None:
            packed = self._tensor(raw, torch.float32)
            if packed.ndim == 1 and packed.numel() == 6:
                packed = packed.reshape(1, 6)
            if packed.ndim != 2 or packed.shape[-1] != 6:
                raise ValueError('packed light_params must have shape (K,6)')
            positions, intensities = packed[:, :3], packed[:, 3:]
        else:
            positions = self._lookup(view, 'light_positions', ('light_positions', 'light_pos'))
            intensities = self._lookup(view, 'light_intensities', ('light_intensities', 'light_rgb', 'light_energy', 'light_E'))
            mask = self._lookup(view, 'light_mask', ('light_mask', 'light_valid'))

        if positions is None or intensities is None:
            if self.require_lighting:
                raise KeyError('view is missing camera-frame light positions/intensities')
            return {'positions': torch.zeros(self.max_lights, 3),
                    'intensities': torch.zeros(self.max_lights, 3),
                    'mask': torch.zeros(self.max_lights, dtype=torch.bool)}
            positions = self._tensor(positions, torch.float32)
            intensities = self._tensor(intensities, torch.float32)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if intensities.ndim == 1:
            intensities = intensities.unsqueeze(0)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError('light positions must have shape (K,3)')
        if intensities.shape != positions.shape:
            raise ValueError('RGB light intensities must have shape (K,3), matching positions')
        if positions.shape[0] > self.max_lights:
            raise ValueError(f"view has {positions.shape[0]} lights, exceeding max_lights={self.max_lights}")
        count = positions.shape[0]
        if mask is None:
            mask = torch.ones(count, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
            if mask.numel() != count:
                raise ValueError('light validity mask must have shape (K,)')
        pad = self.max_lights - count
        positions = torch.nn.functional.pad(positions, (0, 0, 0, pad))
        intensities = torch.nn.functional.pad(intensities.clamp_min(0), (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad), value=False)
        return {'positions': positions, 'intensities': intensities, 'mask': mask}

    def _adapt_view(self, original):
        view = dict(original)
        image = view.get('img')
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("view['img'] must be a normalized tensor with shape (3,H,W)")
        height, width = image.shape[-2:]

        metric = self._lookup(view, 'metric_scale', ('metric_scale', 'is_metric', 'metric'))
        view['metric_scale'] = torch.tensor(bool(metric) if metric is not None else False, dtype=torch.bool)

        albedo = self._lookup(view, 'albedo', self.ALBEDO_ALIASES)
        if albedo is not None:
            albedo = self._as_hwc(albedo, 'albedo')
            albedo = self._scale_rgb(albedo)
            if albedo.shape[:2] != (height, width):
                raise ValueError('albedo target must be cropped/resized to the image resolution')
            if not torch.isfinite(albedo).all() or (albedo < 0).any() or (albedo > self.rho_max + 1e-4).any():
                raise ValueError(f'albedo target must be finite and within [0,{self.rho_max}]')
            view['albedo'] = albedo
            view['albedo_valid'] = torch.tensor(True)
        elif self.require_albedo:
            raise KeyError('view is missing its diffuse albedo target')
        else:
            view['albedo'] = torch.zeros(height, width, 3, dtype=torch.float32)
            view['albedo_valid'] = torch.tensor(False)

        rgb = self._lookup(view, 'photo_rgb', self.RGB_ALIASES)
        if rgb is not None:
            rgb = self._as_hwc(rgb, 'photo_rgb')
            rgb = self._scale_rgb(rgb)
            if rgb.shape[:2] != (height, width):
                raise ValueError('photometric RGB target must be cropped/resized to the image resolution')
            if not torch.isfinite(rgb).all():
                raise ValueError('photometric RGB target must contain only finite values')
            if self.rgb_encoding == 'srgb':
                rgb = torch.where(rgb <= 0.04045, rgb / 12.92,
                                  ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))
            view['photo_rgb'] = rgb
            view['photo_rgb_linear'] = torch.tensor(self.rgb_encoding == 'linear')
            view['photo_valid'] = torch.tensor(True)
        elif self.require_photo_rgb:
            raise KeyError('view is missing its renderer-consistent photometric RGB target')
        else:
            view['photo_rgb'] = (image.float().permute(1, 2, 0) + 1.0).mul(0.5).clamp(0, 1)
            view['photo_valid'] = torch.tensor(False)
            view['photo_rgb_linear'] = torch.tensor(False)

        lights = self._adapt_lights(view)
        view['lights'] = lights

        # Remove duplicate source fields so variable-size source light arrays
        # cannot break DataLoader collation after they have been padded.
        source_aliases = set(self.RGB_ALIASES + self.ALBEDO_ALIASES + (
            'metric', 'is_metric', 'light_params', 'lighting', 'light_positions',
            'light_pos', 'light_intensities', 'light_rgb', 'light_energy', 'light_E',
            'light_mask', 'light_valid'))
        for key in source_aliases:
            if key not in ('photo_rgb', 'albedo'):
                view.pop(key, None)
        for source in self.field_map.values():
            if '.' not in source and source not in ('metric_scale', 'albedo', 'photo_rgb', 'lights'):
                view.pop(source, None)
        return view
