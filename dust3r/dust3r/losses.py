# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def _mask_mean(value, mask):
    mask = mask.bool()
    selected = value[mask]
    return selected.mean() if selected.numel() else value.sum() * 0


def _as_batch_mask(value, batch, device):
    if value is None:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if value.numel() == 1 and batch != 1:
        value = value.expand(batch)
    if value.numel() != batch:
        raise ValueError(f'expected one metric flag per sample, got {value.shape}')
    return value


def _subbatch(value, indices, batch_size):
    if isinstance(value, dict):
        return {key: _subbatch(item, indices, batch_size) for key, item in value.items()}
    if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
        return value[indices]
    if isinstance(value, (tuple, list)) and len(value) == batch_size:
        return type(value)(value[i] for i in indices.tolist())
    return value


def _camera_local_gt(gt1, gt2):
    to_camera1 = inv(gt1['camera_pose'])
    return geotrf(to_camera1, gt1['pts3d']), geotrf(to_camera1, gt2['pts3d'])


def _surface_normals(points, valid, focal, edge_tau=4.0):
    """Finite-difference normals and detached validity/discontinuity mask."""
    points = points.float()
    valid = valid.bool() & torch.isfinite(points).all(dim=-1)
    center = points
    left, right = torch.roll(points, 1, 2), torch.roll(points, -1, 2)
    up, down = torch.roll(points, 1, 1), torch.roll(points, -1, 1)
    dx = right - left
    dy = down - up
    normal = torch.linalg.cross(dx, dy, dim=-1)
    normal = F.normalize(normal, dim=-1, eps=1e-6)
    # Choose the camera-facing sign (camera center is the origin).
    toward_camera = -center
    normal = torch.where(((normal * toward_camera).sum(-1, keepdim=True) < 0), -normal, normal)

    batch, height, width, _ = points.shape
    interior = torch.ones((batch, height, width), dtype=torch.bool, device=points.device)
    interior[:, 0] = interior[:, -1] = False
    interior[:, :, 0] = interior[:, :, -1] = False
    neighborhood_valid = (valid & torch.roll(valid, 1, 2) & torch.roll(valid, -1, 2) &
                          torch.roll(valid, 1, 1) & torch.roll(valid, -1, 1) & interior)
    spacing = torch.stack(((right - center).norm(dim=-1), (left - center).norm(dim=-1),
                           (down - center).norm(dim=-1), (up - center).norm(dim=-1)), dim=-1).amax(-1)
    depth = center[..., 2].abs().clamp_min(1e-6)
    footprint = depth / focal.reshape(batch, 1, 1).clamp_min(1e-6)
    edge_valid = neighborhood_valid & (spacing < edge_tau * footprint)
    normal_valid = edge_valid & (normal.norm(dim=-1) > 1e-5)
    return normal, normal_valid.detach()


class LADUSt3RTrainingObjective(nn.Module):
    """DUSt3R geometry plus camera-local normal, albedo, and shading losses.

    Optional per-view fields are ``metric_scale`` (B,), ``albedo`` (B,H,W,3),
    ``lights``/``light_params`` as (B,K,6) or a dict with positions,
    intensities, and optional mask, and ``photo_rgb``/``linear_rgb`` (B,H,W,3).
    When RGB targets are omitted, normalized DUSt3R input images are mapped to
    [0,1] as the fixed photometric encoding.
    """
    def __init__(self, geometric_criterion=None, lambda_geo=1.0, lambda_nrm=0.1,
                 lambda_shade=0.01, lambda_rho=0.1, alpha_conf=0.2,
                 alpha_reliability=0.01, eps_rho=1e-3, eps_psi=1e-3,
                 eps_reliability=1e-6, rho_max=1.0, renderer_gain=1.0,
                 attenuation='inverse_square', attenuation_coefficients=(0.0, 0.0, 1.0),
                 edge_tau=4.0, shade_warmup_steps=1000, shade_ramp_steps=1000):
        super().__init__()
        self.geometric_criterion = geometric_criterion or ConfLoss(
            Regr3D(L21, norm_mode='avg_dis'), alpha=alpha_conf)
        self.lambda_geo = lambda_geo
        self.lambda_nrm = lambda_nrm
        self.lambda_shade = lambda_shade
        self.lambda_rho = lambda_rho
        self.alpha_conf = getattr(self.geometric_criterion, 'alpha', alpha_conf)
        self.alpha_reliability = alpha_reliability
        self.eps_rho = eps_rho
        self.eps_psi = eps_psi
        self.eps_reliability = eps_reliability
        self.rho_max = rho_max
        self.renderer_gain = renderer_gain
        self.attenuation = attenuation
        self.attenuation_coefficients = attenuation_coefficients
        self.edge_tau = edge_tau
        self.shade_warmup_steps = shade_warmup_steps
        self.shade_ramp_steps = max(1, shade_ramp_steps)
        self.register_buffer('training_step', torch.zeros((), dtype=torch.long), persistent=False)

    def set_step(self, step):
        self.training_step.fill_(int(step))

    def _shade_weight(self):
        progress = (int(self.training_step) - self.shade_warmup_steps) / self.shade_ramp_steps
        return self.lambda_shade * min(max(progress, 0.0), 1.0)

    def _geometry(self, gt1, gt2, pred1, pred2, metric):
        batch = metric.numel()
        metric_loss = pred1['pts3d'].sum() * 0
        metric_count = int(metric.sum())
        if metric_count:
            ids = metric.nonzero(as_tuple=False).flatten()
            a, b = _subbatch(gt1, ids, batch), _subbatch(gt2, ids, batch)
            p, q = _subbatch(pred1, ids, batch), _subbatch(pred2, ids, batch)
            target1, target2 = _camera_local_gt(a, b)
            valid1, valid2 = a['valid_mask'].bool(), b['valid_mask'].bool()
            pred_points = (p['pts3d'], q['pts3d_in_other_view'])
            targets = (target1, target2)
            masks = (valid1, valid2)
            preds = (p, q)
            losses = []
            for points, target, mask, prediction in zip(pred_points, targets, masks, preds):
                residual = torch.linalg.vector_norm(points.float() - target.float(), dim=-1)
                conf = prediction['conf'].float().clamp_min(1e-6)
                per_pixel = conf * residual - self.alpha_conf * conf.log()
                losses.append(_mask_mean(per_pixel, mask))
            metric_loss = sum(losses)

        legacy_mask = ~metric
        if legacy_mask.any():
            ids = legacy_mask.nonzero(as_tuple=False).flatten()
            legacy_loss, _ = self.geometric_criterion(
                _subbatch(gt1, ids, batch), _subbatch(gt2, ids, batch),
                _subbatch(pred1, ids, batch), _subbatch(pred2, ids, batch))
            metric_loss = metric_loss + legacy_loss
        return metric_loss

    @staticmethod
    def _lights(view):
        lights = view.get('lights', view.get('light_params'))
        if lights is None:
            return None
        if isinstance(lights, dict):
            positions = lights['positions']
            intensities = lights['intensities']
            mask = lights.get('mask')
        else:
            positions, intensities, mask = lights[..., :3], lights[..., 3:], None
        if positions.ndim == 2:
            positions, intensities = positions.unsqueeze(0), intensities.unsqueeze(0)
        if mask is None:
            mask = torch.ones(positions.shape[:2], dtype=torch.bool, device=positions.device)
        else:
            mask = mask.bool()
        return positions.float(), intensities.float().clamp_min(0), mask

    def _normals(self, view, prediction):
        points = prediction['pts3d'].float()
        gt_points, _ = _camera_local_gt(view, view)
        intrinsics = view['camera_intrinsics'].float()
        focal = (intrinsics[:, 0, 0].abs() + intrinsics[:, 1, 1].abs()) * 0.5
        valid = view['valid_mask'].bool()
        pred_normal, pred_mask = _surface_normals(points, valid, focal, self.edge_tau)
        gt_normal, gt_mask = _surface_normals(gt_points.float(), valid, focal, self.edge_tau)
        return pred_normal, gt_normal, (pred_mask & gt_mask).detach()

    def _albedo_loss(self, gt, prediction):
        target = gt.get('albedo')
        if target is None:
            return prediction['albedo'].sum() * 0
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        mask = gt['valid_mask'].bool() & torch.isfinite(target).all(dim=-1)
        target_valid = gt.get('albedo_valid')
        if target_valid is not None:
            target_valid = torch.as_tensor(target_valid, device=mask.device, dtype=torch.bool).reshape(-1)
            mask &= target_valid[:, None, None]
        residual = (torch.log(prediction['albedo'].float().clamp_min(0) + self.eps_rho) -
                    torch.log(target.clamp_min(0) + self.eps_rho)).abs().sum(dim=-1)
        return _mask_mean(residual, mask)

    def _photo_target(self, view):
        target = view.get('photo_rgb', view.get('linear_rgb'))
        if target is None:
            target = (view['img'].float().permute(0, 2, 3, 1) + 1.0) * 0.5
        elif target.ndim == 3:
            target = target.unsqueeze(0)
        if target.ndim == 4 and target.shape[1] == 3:
            target = target.permute(0, 2, 3, 1)
        target = target.float()
        # Linear renderer radiance is not display RGB and commonly exceeds 1.
        # Preserve its dynamic range for the differentiable image-formation loss.
        linear = view.get('photo_rgb_linear')
        if linear is None:
            return target.clamp(0, 1)
        linear = torch.as_tensor(linear, device=target.device, dtype=torch.bool).reshape(-1)
        if linear.numel() == 1 and target.shape[0] > 1:
            linear = linear.expand(target.shape[0])
        linear = linear[:, None, None, None]
        return torch.where(linear, target.clamp_min(0), target.clamp(0, 1))

    def _shade_loss(self, gt, prediction, normals, normal_mask):
        zero = prediction['pts3d'].sum() * 0
        light_data = self._lights(gt)
        if light_data is None:
            return zero, zero.detach(), zero.detach()
        points = prediction['pts3d'].float()
        positions, intensities, light_mask = light_data
        batch = points.shape[0]
        if positions.shape[0] == 1 and batch > 1:
            positions, intensities, light_mask = (x.expand(batch, *x.shape[1:])
                                                  for x in (positions, intensities, light_mask))
        if positions.shape[1] != intensities.shape[1]:
            raise ValueError('light positions and intensities must have the same light count')
        shading = torch.zeros_like(points)
        for light_idx in range(positions.shape[1]):
            delta = positions[:, light_idx, None, None, :] - points
            distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-5)
            direction = delta / distance[..., None]
            cosine = (normals * direction).sum(dim=-1).clamp_min(0)
            if self.attenuation == 'inverse_square':
                attenuation = distance.square().reciprocal()
            elif self.attenuation == 'constant':
                attenuation = torch.ones_like(distance)
            elif self.attenuation == 'quadratic':
                c0, c1, c2 = self.attenuation_coefficients
                attenuation = (c0 + c1 * distance + c2 * distance.square()).clamp_min(1e-6).reciprocal()
            else:
                raise ValueError(f'unknown attenuation model: {self.attenuation}')
            active = light_mask[:, light_idx].float()[:, None, None]
            term = intensities[:, light_idx, None, None, :] * (
                active * attenuation * cosine)[..., None]
            shading = shading + term
        shading = (self.renderer_gain / torch.pi) * prediction['albedo'].float() * shading

        target = self._photo_target(gt)
        residual = torch.sqrt((shading - target).square() + self.eps_psi ** 2).sum(dim=-1)
        reliability = prediction['photometric_reliability'].float().clamp(0, 1)
        loss_map = reliability * residual - self.alpha_reliability * torch.log(
            reliability + self.eps_reliability)
        valid = gt['valid_mask'].bool() & normal_mask & light_mask.any(dim=1)[:, None, None]
        valid &= torch.isfinite(target).all(dim=-1)
        photo_valid = gt.get('photo_valid')
        if photo_valid is not None:
            photo_valid = torch.as_tensor(photo_valid, device=valid.device, dtype=torch.bool).reshape(-1)
            valid &= photo_valid[:, None, None]
        linear = gt.get('photo_rgb_linear')
        if linear is None:
            valid &= target.amax(dim=-1) < 0.99
        else:
            linear = torch.as_tensor(linear, device=valid.device, dtype=torch.bool).reshape(-1)
            if linear.numel() == 1 and valid.shape[0] > 1:
                linear = linear.expand(valid.shape[0])
            valid &= linear[:, None, None] | (target.amax(dim=-1) < 0.99)
        valid &= target.mean(dim=-1) > 0.01
        return _mask_mean(loss_map, valid), _mask_mean(reliability, valid), valid.float().mean()

    def forward(self, gt1, gt2, pred1, pred2, **kwargs):
        batch = pred1['pts3d'].shape[0]
        metric_value = gt1.get('metric_scale', gt1.get('is_metric', gt1.get(
            'metric', gt2.get('metric_scale', gt2.get('is_metric', gt2.get('metric'))))))
        metric = _as_batch_mask(metric_value, batch, pred1['pts3d'].device)
        geo = self._geometry(gt1, gt2, pred1, pred2, metric)
        zero = pred1['pts3d'].sum() * 0

        lit1 = self._lights(gt1)
        lit = metric.clone()
        if lit1 is None:
            lit.zero_()
        else:
            lit &= lit1[2].any(dim=1)

        normal_loss = zero
        shade_loss = zero
        reliability_mean = zero.detach()
        active_fraction = zero.detach()
        if lit.any():
            ids = lit.nonzero(as_tuple=False).flatten()
            ref_gt = _subbatch(gt1, ids, batch)
            ref_pred = _subbatch(pred1, ids, batch)
            pred_normal, gt_normal, normal_mask = self._normals(ref_gt, ref_pred)
            normal_map = 1.0 - (pred_normal * gt_normal).sum(dim=-1).clamp(-1, 1)
            normal_loss = _mask_mean(normal_map, normal_mask)
            shade_loss, reliability_mean, active_fraction = self._shade_loss(
                ref_gt, ref_pred, pred_normal, normal_mask)

        albedo_loss = zero
        if lit.any():
            ids = lit.nonzero(as_tuple=False).flatten()
            lit_gt1, lit_gt2 = _subbatch(gt1, ids, batch), _subbatch(gt2, ids, batch)
            lit_pred1, lit_pred2 = _subbatch(pred1, ids, batch), _subbatch(pred2, ids, batch)
            albedo_loss = 0.5 * (self._albedo_loss(lit_gt1, lit_pred1) +
                                 self._albedo_loss(lit_gt2, lit_pred2))
        shade_weight = self._shade_weight()
        total = (self.lambda_geo * geo + self.lambda_nrm * normal_loss +
                 shade_weight * shade_loss + self.lambda_rho * albedo_loss)
        details = {
            'loss_geo': float(geo.detach()),
            'loss_normal': float(normal_loss.detach()),
            'loss_shade': float(shade_loss.detach()),
            'loss_albedo': float(albedo_loss.detach()),
            'shade_weight': float(shade_weight),
            'reliability_mean': float(reliability_mean.detach()),
            'photo_active_fraction': float(active_fraction.detach()),
        }
        return total, details
