"""
gen_samples.py -- LA-DUSt3R Level-2 synthetic sample generator (Blender / Cycles).

Run headless (Blender 3.6 - 4.x):
    blender -b -P gen_samples.py -- --out ./la_data --n-scenes 20 --views 6

What it renders, per view (one multilayer EXR, linear float32):
    Combined (RGB+A), Depth (z-depth, metres), Normal (world space),
    DiffCol (diffuse colour = albedo GT), IndexOB (instance id)
What it records, per scene (meta.json):
    K, camera-to-world pose (OpenCV convention), light position in the
    camera frame (OpenCV convention), light colour/power/RGB intensity E.

Level-2 settings enforced here:
    * every material is a pure Diffuse BSDF (Lambertian), or glTF assets with
      specular/metallic zeroed
    * world background is black (no ambient)
    * direct light only (max_bounces = 0 by default)
    * point lights (radius 0), all lights on at once; --light-mode selects the frame:
        random : camera-attached, random per-view offset and power
        fixed  : camera-attached, rigid rig (same offset/power every view)
        world  : WORLD-FIXED light, same world position and power in every view of a
                 scene; cameras move, so the camera-frame light position changes per view
    * 'Standard' view transform, linear EXR output (no tone mapping)

Camera convention in meta.json: OpenCV (x right, y down, z forward).
Blender's camera looks down -Z with +Y up, so we flip y and z when converting.
"""
import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Vector

# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--res", type=int, default=512, help="square image size")
    p.add_argument("--fov-deg", type=float, default=50.0, help="horizontal FOV")
    p.add_argument("--seed", type=int, default=0)
    # lights
    p.add_argument("--light-mode", choices=["random", "fixed", "world"], default="random",
                   help="random: camera-attached, per-view offset+power (tests H3); "
                        "fixed: camera-attached rigid rig; world: world-fixed light")
    p.add_argument("--light-world", type=str, default="",
                   help="world mode: explicit light position 'x,y,z' in metres "
                        "(default: sampled once per scene near the cameras' side)")
    p.add_argument("--world-radius-range", type=str, default="1.5,2.5",
                   help="world mode: sampled light distance from scene origin (m)")
    p.add_argument("--world-el-range-deg", type=str, default="30,70",
                   help="world mode: sampled light elevation range")
    p.add_argument("--world-az-span-deg", type=float, default=30.0,
                   help="world mode: light azimuth within +/- this of the camera base azimuth")
    p.add_argument("--fixed-offset", type=str, default="0.10,0.0,0.0",
                   help="camera-frame (OpenCV) offset in metres for fixed rig")
    p.add_argument("--offset-xy", type=float, default=0.15, help="random x,y range (+/-) m")
    p.add_argument("--offset-z", type=float, default=0.05, help="random z range (+/-) m")
    p.add_argument("--power-range", type=str, default="",
                   help="Blender Watts; default 500,1500 camera-attached, 200,600 world")
    p.add_argument("--n-lights", type=int, default=1, help="simultaneous lights per view")
    p.add_argument("--tint", type=float, default=0.0, help="random colour tint amplitude")
    p.add_argument("--no-shadow", action="store_true")
    # scene / cameras
    p.add_argument("--radius-range", type=str, default="2.0,3.0")
    p.add_argument("--az-span-deg", type=float, default=50.0,
                   help="views spread +/- this around a base azimuth (keeps pair overlap)")
    p.add_argument("--el-range-deg", type=str, default="15,60")
    p.add_argument("--n-objects", type=str, default="2,5")
    p.add_argument("--no-floor", action="store_true")
    p.add_argument("--assets", type=str, default="", help="dir of .glb/.gltf to use instead of primitives")
    # render
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--bounces", type=int, default=0, help="0 = direct light only")
    p.add_argument("--device", default="CPU", help="CPU | CUDA | OPTIX | METAL | HIP")
    return p.parse_args(argv)


def pair(s):
    a, b = [float(x) for x in s.split(",")]
    return a, b


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def diffuse_material(rgb, name="mat"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])
    return m


def rand_albedo(rng):
    # linear-space albedo in [0.05, 0.95]; GT albedo == this value
    return [rng.uniform(0.05, 0.95) for _ in range(3)]


'''
def add_primitive(rng, idx):
    kind = rng.choice(["cube", "sphere", "torus", "monkey", "cone", "cylinder"])
    s = rng.uniform(0.25, 0.5)
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2 * s)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=s, segments=64, ring_count=32)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=s, minor_radius=s * 0.35)
    elif kind == "monkey":
        bpy.ops.mesh.primitive_monkey_add(size=2 * s)
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=s, depth=2 * s, vertices=64)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=s * 0.8, depth=2 * s, vertices=64)
    ob = bpy.context.active_object
    ob.location = (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), s * rng.uniform(0.6, 1.2))
    ob.rotation_euler = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    if kind != "cube":
        bpy.ops.object.shade_smooth()
    albedo = rand_albedo(rng)
    ob.data.materials.append(diffuse_material(albedo))
    ob.pass_index = idx
    return ob, albedo
'''

def add_primitive(rng, idx):
    generators = {
        "cube": lambda s: bpy.ops.mesh.primitive_cube_add(size=2*s),
        "sphere": lambda s: bpy.ops.mesh.primitive_uv_sphere_add(
            radius=s, segments=32, ring_count=24
        ),
        "icosphere": lambda s: bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=3, radius=s
        ),
        "torus": lambda s: bpy.ops.mesh.primitive_torus_add(
            major_radius=s, minor_radius=s*0.3
        ),
        "monkey": lambda s: bpy.ops.mesh.primitive_monkey_add(size=2*s),
        "cone": lambda s: bpy.ops.mesh.primitive_cone_add(
            radius1=s, depth=2*s, vertices=32
        ),
        "cylinder": lambda s: bpy.ops.mesh.primitive_cylinder_add(
            radius=s*0.8, depth=2*s, vertices=32
        ),
        "prism": lambda s: bpy.ops.mesh.primitive_cylinder_add(
            vertices=3, radius=s, depth=2*s
        ),
        "hex_prism": lambda s: bpy.ops.mesh.primitive_cylinder_add(
            vertices=6, radius=s, depth=2*s
        ),
        "pyramid": lambda s: bpy.ops.mesh.primitive_cone_add(
            vertices=4, radius1=s, radius2=0, depth=2*s
        ),
        #"star": lambda s: bpy.ops.mesh.primitive_star_add(
        #    points=5, outer_radius=s, inner_radius=s*0.45
        #),
        #"gear": lambda s: bpy.ops.mesh.primitive_gear_add(
        #    number_of_teeth=12, radius=s, width=s*0.2
        #),
    }

    kind = rng.choice(list(generators.keys()))
    s = rng.uniform(0.25, 0.5)

    generators[kind](s)

    ob = bpy.context.active_object

    # Random position
    ob.location = (
        rng.uniform(-0.6, 0.6),
        rng.uniform(-0.6, 0.6),
        s * rng.uniform(0.6, 1.2)
    )

    # Random orientation
    ob.rotation_euler = [
        rng.uniform(0, 2 * math.pi) for _ in range(3)
    ]

    # Smooth shading for curved objects
    if kind not in ("cube", "prism", "hex_prism", "pyramid", "star", "gear"):
        for poly in ob.data.polygons:
            poly.use_smooth = True

    albedo = rand_albedo(rng)
    ob.data.materials.append(diffuse_material(albedo))
    ob.pass_index = idx

    return ob, albedo


def add_floor(rng):
    bpy.ops.mesh.primitive_plane_add(size=6.0)
    ob = bpy.context.active_object
    albedo = rand_albedo(rng)
    ob.data.materials.append(diffuse_material(albedo, "floor"))
    ob.pass_index = 0  # alpha, not index, marks foreground
    return ob, albedo


def make_lambertian_gltf(ob):
    """Zero specular/metallic on imported glTF materials (best effort)."""
    for slot in ob.material_slots:
        m = slot.material
        if not (m and m.use_nodes):
            continue
        for n in m.node_tree.nodes:
            if n.type != "BSDF_PRINCIPLED":
                continue
            for k, v in (("Metallic", 0.0), ("Roughness", 1.0),
                         ("Specular IOR Level", 0.0), ("Specular", 0.0),
                         ("Coat Weight", 0.0), ("Sheen Weight", 0.0)):
                if k in n.inputs and not n.inputs[k].is_linked:
                    n.inputs[k].default_value = v


def import_asset(path, rng, idx):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        return None
    # normalise to ~0.8 m max extent, sit on z=0
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    scale = 0.8 / max((hi - lo).x, (hi - lo).y, (hi - lo).z, 1e-6)
    root = bpy.data.objects.new("asset_root", None)
    bpy.context.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    root.scale = (scale,) * 3
    root.location = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -lo.z * scale)
    for o in meshes:
        make_lambertian_gltf(o)
        o.pass_index = idx
    return root


def build_scene(rng, args):
    objects = []
    if args.assets:
        files = [f for f in os.listdir(args.assets) if f.lower().endswith((".glb", ".gltf"))]
        f = rng.choice(files)
        import_asset(os.path.join(args.assets, f), rng, 1)
        objects.append({"id": 1, "kind": "asset", "file": f, "albedo": None})
    else:
        n = rng.randint(int(args.n_objects.split(",")[0]), int(args.n_objects.split(",")[1]))
        for i in range(n):
            ob, albedo = add_primitive(rng, i + 1)
            objects.append({"id": i + 1, "kind": "primitive", "name": ob.name, "albedo": albedo})
    if not args.no_floor:
        _, albedo = add_floor(rng)
        objects.append({"id": 0, "kind": "floor", "albedo": albedo})
    return objects


def setup_render(sc, args):
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = sc.render.resolution_y = args.res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01          # ~box filter: keeps colour/depth pixel-aligned
    sc.cycles.max_bounces = args.bounces
    for k in ("diffuse_bounces", "glossy_bounces", "transmission_bounces", "volume_bounces"):
        setattr(sc.cycles, k, args.bounces)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = "sRGB"

    # black world => no ambient
    w = bpy.data.worlds.new("world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0
    sc.world = w

    # outputs
    #sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    #sc.render.image_settings.color_depth = "32"
    #sc.render.image_settings.exr_codec = "ZIP"
    #vl = bpy.context.view_layer
    #vl.use_pass_combined = True
    #vl.use_pass_z = True
    #vl.use_pass_normal = True
    #vl.use_pass_diffuse_color = True
    #vl.use_pass_object_index = True

    # outputs

    # XXX 1
    #sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    #sc.render.image_settings.color_depth = "32"
    #sc.render.image_settings.exr_codec = "ZIP"
    # E XXX 1

    # NOTE 1
    # outputs: Blender 5.3 multilayer OpenEXR
    img = sc.render.image_settings
    img.media_type = "MULTI_LAYER_IMAGE"
    img.file_format = "OPEN_EXR_MULTILAYER"
    img.color_depth = "32"
    img.exr_codec = "ZIP"
    img.use_exr_interleave = True  # Use legacy interleaved channels
    # E NOTE 1

    # XXX 2
    #vl = bpy.context.view_layer
    #vl.use_pass_combined = True
    #vl.use_pass_z = True
    #vl.use_pass_normal = True
    #vl.use_pass_diffuse_color = True
    #vl.use_pass_object_index = True
    # E XXX 2

    # NOTE 2
    # Explicitly configure the view layer belonging to this scene.
    vl = bpy.context.view_layer #vl = sc.view_layers[0]

    required_passes = {
        "use_pass_combined": True,
        "use_pass_z": True,
        "use_pass_normal": True,
        "use_pass_diffuse_color": True,
        "use_pass_object_index": True,
    }

    for attr, enabled in required_passes.items():
        if not hasattr(vl, attr):
            raise RuntimeError(f"Blender does not support view-layer property: {attr}")
        setattr(vl, attr, enabled)

        print(f"[render] {attr}: {getattr(vl, attr)}")
    # E NOTE 2

    if args.device.upper() != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = args.device.upper()
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    else:
        sc.cycles.device = "CPU"


def make_camera(sc, args):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(args.fov_deg)
    cam_data.clip_start, cam_data.clip_end = 0.05, 50.0
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def make_lights(sc, cam, n, no_shadow, attach):
    lights = []
    for k in range(n):
        ld = bpy.data.lights.new(f"light{k}", type="POINT")
        ld.shadow_soft_size = 0.0            # true point light
        if no_shadow:
            if hasattr(ld, "use_shadow"):
                ld.use_shadow = False
            elif hasattr(ld, "cycles"):
                ld.cycles.cast_shadow = False
        lo = bpy.data.objects.new(f"light{k}", ld)
        sc.collection.objects.link(lo)
        if attach:
            lo.parent = cam                  # camera-attached; parent-inverse = identity
        lights.append(lo)
    return lights


# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

FLIP = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))  # Blender cam <-> OpenCV cam


def sample_cam_pose(rng, args, az0):
    r = rng.uniform(*pair(args.radius_range))
    az = az0 + math.radians(rng.uniform(-args.az_span_deg, args.az_span_deg))
    el = math.radians(rng.uniform(*pair(args.el_range_deg)))
    pos = Vector((r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el)))
    tgt = Vector((rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.1, 0.4)))
    rot = (tgt - pos).normalized().to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(pos) @ rot


def sample_light_offsets(rng, args):
    """Offsets in OpenCV camera coordinates (x right, y down, z forward)."""
    offs = []
    for k in range(args.n_lights):
        if args.light_mode == "fixed":
            o = [float(x) for x in args.fixed_offset.split(",")]
            if k > 0:                         # spread extra lights symmetrically
                o[0] = -o[0] if k % 2 else o[0]
                o[1] = o[1] + 0.05 * (k // 2)
        else:
            o = [rng.uniform(-args.offset_xy, args.offset_xy),
                 rng.uniform(-args.offset_xy, args.offset_xy),
                 rng.uniform(-args.offset_z, args.offset_z)]
        offs.append(o)
    return offs


def power_range(args):
    if args.power_range:
        return pair(args.power_range)
    return (200.0, 600.0) if args.light_mode == "world" else (500.0, 1500.0)


def sample_light_power(rng, args):
    lo, hi = power_range(args)
    if args.light_mode == "fixed":
        return math.sqrt(lo * hi)             # constant mid power for the fixed rig
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_world_lights(rng, args, az0):
    """World-fixed light position(s), sampled once per scene. The light sits on the same
    side as the cameras (azimuth near az0) so most views are front-lit."""
    if args.light_world:
        base = Vector([float(x) for x in args.light_world.split(",")])
    else:
        r = rng.uniform(*pair(args.world_radius_range))
        az = az0 + math.radians(rng.uniform(-args.world_az_span_deg, args.world_az_span_deg))
        el = math.radians(rng.uniform(*pair(args.world_el_range_deg)))
        base = Vector((r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el)))
    pos = [base]
    for k in range(1, args.n_lights):         # extra simultaneous lights, spread around the first
        pos.append(base + Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2))))
    return pos


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def render_scene(scene_idx, args):
    rng = random.Random(args.seed * 100003 + scene_idx)
    sc = reset_scene()
    objects = build_scene(rng, args)
    setup_render(sc, args)
    cam = make_camera(sc, args)
    world_mode = args.light_mode == "world"
    lights = make_lights(sc, cam, args.n_lights, args.no_shadow, attach=not world_mode)

    scene_dir = os.path.join(args.out, f"scene_{scene_idx:05d}")
    os.makedirs(scene_dir, exist_ok=True)
    f_px = (args.res / 2.0) / math.tan(math.radians(args.fov_deg) / 2.0)
    K = [[f_px, 0.0, args.res / 2.0], [0.0, f_px, args.res / 2.0], [0.0, 0.0, 1.0]]
    az0 = rng.uniform(0, 2 * math.pi)

    world_light_pos, world_power, world_colors = None, None, None
    if world_mode:                            # fixed for the whole scene
        world_light_pos = sample_world_lights(rng, args, az0)
        world_power = sample_light_power(rng, args)
        world_colors = []
        for lo, pos in zip(lights, world_light_pos):
            col = [1.0, 1.0, 1.0]
            if args.tint > 0:
                col = [min(1.0, max(0.0, 1.0 + rng.uniform(-args.tint, args.tint))) for _ in range(3)]
            lo.location = pos
            lo.data.color = col
            lo.data.energy = world_power
            world_colors.append(col)

    views = []
    for v in range(args.views):
        cam.matrix_world = sample_cam_pose(rng, args, az0)
        light_entries = []
        if world_mode:
            # light stays where it is; only the camera moved
            for lo, col in zip(lights, world_colors):
                light_entries.append({"offset_cv": None, "power_W": world_power,
                                      "color": col, "E_rgb": [world_power * c for c in col]})
        else:
            offs = sample_light_offsets(rng, args)
            power = sample_light_power(rng, args)
            for lo, (ox, oy, oz) in zip(lights, offs):
                col = [1.0, 1.0, 1.0]
                if args.tint > 0:
                    col = [min(1.0, max(0.0, 1.0 + rng.uniform(-args.tint, args.tint))) for _ in range(3)]
                lo.data.color = col
                lo.data.energy = power
                lo.location = (ox, -oy, -oz)     # OpenCV offset -> Blender camera-local
                light_entries.append({"offset_cv": [ox, oy, oz], "power_W": power,
                                      "color": col, "E_rgb": [power * c for c in col]})
        bpy.context.view_layer.update()

        c2w_cv = cam.matrix_world @ FLIP
        w2c_cv = c2w_cv.inverted()
        # sanity: light position in camera frame recovered from world transform
        for lo, ent in zip(lights, light_entries):
            ent["light_cam_cv"] = list(w2c_cv @ lo.matrix_world.translation)
            ent["light_world"] = list(lo.matrix_world.translation)

        name = f"view_{v:02d}"
        sc.render.filepath = os.path.join(scene_dir, name)
        bpy.ops.render.render(write_still=True)
        views.append({"file": name + ".exr",
                      "c2w_cv": [list(r) for r in c2w_cv],
                      "lights": light_entries})

    meta = {
        "scene_index": scene_idx, "resolution": args.res, "K": K,
        "convention": "OpenCV camera (x right, y down, z forward); K assumes continuous "
                      "pixel coords (cx = W/2, pixel centre = j + 0.5); depth = z-depth in metres",
        "light_mode": args.light_mode,
        "light_world_fixed": [list(p) for p in world_light_pos] if world_mode else None,
        "bounces": args.bounces, "samples": args.samples,
        "blender_version": bpy.app.version_string, "objects": objects, "views": views,
        "passes": ["Combined", "Depth", "Normal(world)", "DiffCol", "IndexOB"],
    }
    with open(os.path.join(scene_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[gen] scene {scene_idx} done -> {scene_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.start_index, args.start_index + args.n_scenes):
        render_scene(i, args)


if __name__ == "__main__":
    main()