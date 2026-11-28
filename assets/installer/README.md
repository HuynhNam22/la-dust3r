```bash
& "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b -P "C:\Study\CapProj\Dataset3R\Synthetic\gen_samples.py" -- --out "C:\Study\CapProj\Dataset3R\Synthetic\data" --light-mode world --n-scenes 10 --views 6 --res 224 --fov-deg 60 --radius-range 2.0,3.5 --az-span-deg 75 --el-range-deg 5,80 --n-objects 5,10 --world-radius-range 1.5,3.0 --world-el-range-deg 20,75 --power-range 200,800 --n-lights 1
```

```bash
python build_dataset.py --root "C:\Study\CapProj\Dataset3R\Synthetic\data" --check --preview
```

`build_dataset.py` writes `npz/scene_*/view_*.npz` under the generated root.
Train with the repository dataset expression, for example:

```text
LADUSt3RNPZDataset(root='C:/path/to/data', split='train', resolution=224)
```

The loader uses one fixed `image_gain` to convert linear renderer radiance to
the display image fed to DUSt3R. The shading objective receives the original
unclamped linear `rgb` values, metric depth, camera-local lights, and albedo.
