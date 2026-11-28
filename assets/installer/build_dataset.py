"""
build_dataset.py -- convert gen_samples.py output into per-view .npz files and
verify that the proposal's Lambertian shading surrogate matches the renderer.

    pip install numpy OpenEXR pillow
    python build_dataset.py --root ./la_data --check --preview

Per-view .npz fields (all camera-local, OpenCV convention, metres):
    rgb      HxWx3 float32  linear radiance (renderer output, unclamped)
    depth    HxW   float32  z-depth, 0 where invalid
    pointmap HxWx3 float32  camera-local 3D points  x = z * K^-1 [u,v,1]
    normal   HxWx3 float32  GT camera-local normals (diagnostics only)
    albedo   HxWx3 float32  diffuse colour GT
    mask     HxW   bool     foreground (alpha > 0.5 and depth valid)
    inst     HxW   int32    instance id (0 = floor/unknown)
    K (3x3), c2w (4x4, OpenCV), light_pos (Kx3, camera frame), light_E (Kx3)
DUSt3R-style wrappers need only: rgb, depth, K, c2w (pointmap is for the shading loss).
"""
import argparse
import glob
import json
import os

import numpy as np

# --------------------------------------------------------------------------- #
# EXR reading (multilayer, channel names matched by substring)
# --------------------------------------------------------------------------- #


def read_exr_channels(path):
    import OpenEXR  # lazy: only needed for real data
    f = OpenEXR.InputFile(path)
    hdr = f.header()
    dw = hdr["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    out = {}
    for name in hdr["channels"].keys():
        raw = f.channel(name)
        dt = np.float16 if len(raw) == H * W * 2 else np.float32
        out[name] = np.frombuffer(raw, dtype=dt).reshape(H, W).astype(np.float32)
    return out


def get_pass(chans, key, comps):
    """Collect channels of pass `key` ordered by `comps` (e.g. 'RGB', 'XYZ', 'V')."""
    cand = {n: a for n, a in chans.items() if f".{key}." in n or n.startswith(key + ".")}
    if not cand:
        raise KeyError(f"pass '{key}' not found; channels = {sorted(chans)}")
    arrs = []
    for c in comps:
        hit = [a for n, a in cand.items() if n.split(".")[-1] == c]
        if not hit:  # fallback: sorted order
            hit = [a for _, a in sorted(cand.items())[len(arrs):len(arrs) + 1]]
        arrs.append(hit[0])
    return np.stack(arrs, -1) if len(arrs) > 1 else arrs[0]


# --------------------------------------------------------------------------- #
# Geometry / shading (pure numpy -> unit-testable without Blender)
# --------------------------------------------------------------------------- #


def backproject(depth, K):
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)[None, :] + 0.5      # pixel centres
    v = np.arange(H, dtype=np.float32)[:, None] + 0.5
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], -1).astype(np.float32)


def normals_world_to_cam(n_world, c2w_cv):
    R = c2w_cv[:3, :3]
    return (n_world @ R).astype(np.float32)                # = R^T n


def lambert_unit(X, N, albedo, light_pos, light_E):
    """Proposal's S without the gain g, A(r)=1/r^2:  (rho/pi) * sum_k E_k cos / r^2."""
    acc = np.zeros_like(X)
    for l, E in zip(light_pos, light_E):
        d = l[None, None, :] - X
        r = np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-6)
        cos = np.clip((N * d / r).sum(-1, keepdims=True), 0, None)
        acc += E[None, None, :] * cos / r ** 2
    return albedo / np.pi * acc


def fit_gain(rgb, pred_unit, valid):
    """Per-pixel ratio rgb/pred_unit -> robust global gain g and spread."""
    m = valid[..., None] & (pred_unit > 1e-4) & (rgb > 1e-4)
    ratio = (rgb[m] / pred_unit[m])
    if ratio.size == 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - med)))
    return med, mad / max(med, 1e-12), int(ratio.size)


# --------------------------------------------------------------------------- #
# Scene processing
# --------------------------------------------------------------------------- #


def process_scene(scene_dir, out_root, check, preview, gains):
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    K = np.array(meta["K"], dtype=np.float32)
    out_dir = os.path.join(out_root, os.path.basename(scene_dir))
    os.makedirs(out_dir, exist_ok=True)

    for v in meta["views"]:
        ch = read_exr_channels(os.path.join(scene_dir, v["file"]))
        rgb = get_pass(ch, "Combined", "RGB")
        alpha = get_pass(ch, "Combined", "A")
        depth = get_pass(ch, "Depth", "V")
        n_w = get_pass(ch, "Normal", "XYZ")
        
        # NOTE 1
        # albedo = get_pass(ch, "DiffCol", "RGB")
        # inst = get_pass(ch, "IndexOB", "X") if any("IndexOB" in n for n in ch) else np.zeros_like(alpha)
        albedo = get_pass(ch, "Diffuse Color", "RGB")
        inst = get_pass(ch, "Object Index", "X") if any("Object Index" in n for n in ch) else np.zeros_like(alpha)
        # E NOTE 1
        
        c2w = np.array(v["c2w_cv"], dtype=np.float32)
        mask = (alpha > 0.5) & (depth > 0) & (depth < 1e5)
        depth = np.where(mask, depth, 0.0).astype(np.float32)
        X = backproject(depth, K) * mask[..., None]
        N = normals_world_to_cam(n_w, c2w) * mask[..., None]
        L = np.array([l["light_cam_cv"] for l in v["lights"]], dtype=np.float32)
        E = np.array([l["E_rgb"] for l in v["lights"]], dtype=np.float32)

        name = os.path.splitext(v["file"])[0]
        np.savez_compressed(os.path.join(out_dir, name + ".npz"),
                            rgb=rgb.astype(np.float32), depth=depth, pointmap=X, normal=N,
                            albedo=albedo.astype(np.float32), mask=mask, inst=inst.astype(np.int32),
                            K=K, c2w=c2w, light_pos=L, light_E=E)

        if check:
            # keep well-lit, un-saturated, front-facing pixels; shadows remain as outliers
            pred = lambert_unit(X, N, albedo, L, E)
            cosmask = mask & (rgb.max(-1) < 0.98 * rgb.max()) & (pred.max(-1) > 0)
            g, spread, n = fit_gain(rgb, pred, cosmask)
            gains.append((g, spread, n))

        if preview:
            from PIL import Image
            def t(a): return (np.clip(a, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            d = depth / max(depth.max(), 1e-6)
            tiles = [t(rgb / max(np.percentile(rgb[mask], 99), 1e-6) if mask.any() else rgb),
                     t(albedo), ((N * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8),
                     np.repeat((d[..., None] * 255).astype(np.uint8), 3, -1)]
            Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(out_dir, name + "_preview.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="gen_samples.py --out directory")
    ap.add_argument("--out", default="", help="default: <root>/npz")
    ap.add_argument("--check", action="store_true", help="fit shading gain g vs renderer")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "npz")
    gains = []
    for sd in sorted(glob.glob(os.path.join(a.root, "scene_*"))):
        process_scene(sd, out, a.check, a.preview, gains)
    if gains:
        g = np.array([x[0] for x in gains])
        sp = np.array([x[1] for x in gains])
        print(f"fitted gain g: median {np.nanmedian(g):.5f}, across-view std {np.nanstd(g):.5f}; "
              f"median within-view relative MAD {np.nanmedian(sp):.3f}")
        print("Interpretation: g should be ~constant across views and light powers; "
              "a small MAD means the 1/r^2 Lambertian surrogate matches the renderer.")


if __name__ == "__main__":
    main()
