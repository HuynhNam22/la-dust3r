# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from .heads.auxiliary_head import DenseAuxiliaryHead
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 light_conditioning=False,
                 light_dropout=0.1,
                 light_fourier_bands=6,
                 rho_max=1.0,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.light_conditioning = light_conditioning
        self.light_dropout = light_dropout
        self.light_fourier_bands = light_fourier_bands
        self.rho_max = rho_max
        if light_conditioning:
            self.light_token_mlp = torch.nn.Sequential(
                torch.nn.Linear(3 + 6 * light_fourier_bands + 3, self.enc_embed_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.enc_embed_dim, self.enc_embed_dim),
            )
            self.null_light_token = torch.nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
            torch.nn.init.normal_(self.null_light_token, std=0.02)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.auxiliary_head1 = DenseAuxiliaryHead(self, rho_max=self.rho_max)
        self.auxiliary_head2 = DenseAuxiliaryHead(self, rho_max=self.rho_max)
        self.auxiliary_head1_landscape = transpose_to_landscape(self.auxiliary_head1, activate=landscape_only)
        self.auxiliary_head2_landscape = transpose_to_landscape(self.auxiliary_head2, activate=landscape_only)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, lights=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        n_image_tokens = x.shape[1]
        key_mask = None
        if self.light_conditioning:
            light_tokens, light_mask = self._encode_lights(lights, x)
            x = torch.cat((x, light_tokens), dim=1)
            key_mask = torch.cat((torch.ones(x.shape[0], n_image_tokens, dtype=torch.bool, device=x.device),
                                  light_mask), dim=1)

        # Light tokens only participate in this view's self-attention. They
        # have no image position and are removed before downstream decoding.
        for blk in self.enc_blocks:
            x = blk(x, pos if not self.light_conditioning else
                    torch.cat((pos, pos.new_zeros((pos.shape[0], x.shape[1] - n_image_tokens, 2))), dim=1),
                    n_pos_tokens=n_image_tokens if self.light_conditioning else None,
                    key_mask=key_mask)

        x = self.enc_norm(x[:, :n_image_tokens])
        return x, pos, None

    def _encode_lights(self, lights, image_tokens):
        """Encode a view's lights from (..., 6) [position xyz, RGB intensity].

        A dict with ``positions``, ``intensities`` and optional ``mask`` is
        also accepted. Missing lights use one learned null token.
        """
        batch, _, _ = image_tokens.shape
        if lights is None:
            values = image_tokens.new_zeros((batch, 1, 6))
            valid = torch.zeros((batch, 1), dtype=torch.bool, device=image_tokens.device)
        else:
            if isinstance(lights, dict):
                positions = lights['positions']
                intensities = lights['intensities']
                values = torch.cat((positions, intensities), dim=-1)
                valid = lights.get('mask')
            else:
                values = lights
                valid = None
            values = torch.as_tensor(values, device=image_tokens.device, dtype=image_tokens.dtype)
            if values.ndim == 2:
                values = values.unsqueeze(0)
            if values.ndim != 3 or values.shape[0] != batch or values.shape[-1] != 6:
                raise ValueError('lights must have shape (B, K, 6) or provide positions/intensities')
            if valid is None:
                valid = torch.ones(values.shape[:2], dtype=torch.bool, device=image_tokens.device)
            else:
                valid = torch.as_tensor(valid, device=image_tokens.device, dtype=torch.bool)
            if valid.shape != values.shape[:2]:
                raise ValueError('light mask must have shape (B, K)')

        xyz, intensity = values[..., :3], values[..., 3:].clamp_min(0)
        frequencies = (2.0 ** torch.arange(self.light_fourier_bands, device=values.device,
                                             dtype=values.dtype)) * torch.pi
        phase = xyz[..., None] * frequencies
        fourier = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        encoded = self.light_token_mlp(torch.cat((xyz, fourier, torch.log1p(intensity)), dim=-1))
        no_lights = ~valid.any(dim=1)
        if no_lights.any():
            encoded = encoded.clone()
            valid = valid.clone()
            encoded[no_lights, 0] = self.null_light_token.to(encoded.dtype)
            valid[no_lights, 0] = True
        if self.training and self.light_dropout > 0:
            dropped = torch.rand((batch, 1), device=values.device) < self.light_dropout
            if dropped.any():
                encoded = encoded.clone()
                valid = valid.clone()
                encoded[dropped[:, 0], 0] = self.null_light_token.to(encoded.dtype)
                valid[dropped[:, 0]] = False
                valid[dropped[:, 0], 0] = True
        return encoded, valid

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, lights1=None, lights2=None):
        can_batch_lights = not self.light_conditioning or (
            lights1 is not None and lights2 is not None and
            self._light_batch_compatible(lights1, lights2))
        if img1.shape[-2:] == img2.shape[-2:] and can_batch_lights:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0),
                                             None if lights1 is None and lights2 is None else
                                             self._concat_light_batches(lights1, lights2, img1.shape[0]))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, lights1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, lights2)
        return out, out2, pos, pos2

    @staticmethod
    def _concat_light_batches(lights1, lights2, batch):
        if lights1 is None or lights2 is None:
            return None
        if isinstance(lights1, dict):
            return {key: torch.cat((lights1[key], lights2[key]), dim=0)
                    for key in ('positions', 'intensities', 'mask') if key in lights1 and key in lights2}
        return torch.cat((lights1, lights2), dim=0)

    @staticmethod
    def _light_batch_compatible(lights1, lights2):
        if isinstance(lights1, dict) != isinstance(lights2, dict):
            return False
        if isinstance(lights1, dict):
            return all(key in lights1 and key in lights2 and
                       lights1[key].shape[1:] == lights2[key].shape[1:]
                       for key in ('positions', 'intensities')) and (
                           ('mask' not in lights1 and 'mask' not in lights2) or
                           ('mask' in lights1 and 'mask' in lights2 and
                            lights1['mask'].shape[1:] == lights2['mask'].shape[1:]))
        return lights1.shape[1:] == lights2.shape[1:]

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        lights1 = view1.get('lights', view1.get('light_params'))
        lights2 = view2.get('lights', view2.get('light_params'))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2],
                self._slice_lights(lights1, slice(None, None, 2)),
                self._slice_lights(lights2, slice(None, None, 2)))
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, lights1, lights2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    @staticmethod
    def _slice_lights(lights, selection):
        if lights is None:
            return None
        if isinstance(lights, dict):
            return {key: value[selection] for key, value in lights.items()}
        return lights[selection]

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        result = head(decout, img_shape)
        auxiliary_head = getattr(self, f'auxiliary_head{head_num}_landscape')
        result.update(auxiliary_head(decout, img_shape))
        return result

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
