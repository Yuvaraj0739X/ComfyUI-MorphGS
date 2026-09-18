"""
Convert a rigged character (Mixamo .fbx export, or a SkinTokens/glTF-rigged .glb -- anything
Blender can import with an armature + skinned mesh) into MorphGS's expected character format:
    <out_dir>/mesh.obj
    <out_dir>/rigging/mesh_ori_rig.txt   (RigNet-compatible: joints/root/hier/skin, matching
                                          MorphGS's own RigModel.load_rig_txt)

Run inside Blender (headless):
    blender --background --python mesh_to_morphgs.py -- <input.fbx|input.glb> <out_dir> [target_height]

Notes from building the original Mixamo-only version of this script against a real
AI-generated + Mixamo-rigged character:
  - Unit scale is corrected against measured height rather than assuming a fixed 100x/0.01x
    factor, since different exporters bake the cm<->m conversion differently.
  - AI-generated meshes can carry small disconnected components (e.g. unwelded accessory
    geometry) whose cotangent-Laplacian block is degenerate and crashes MorphGS's
    Cholesky-based skinning-weight smoothing; this script welds near-coincident vertices and
    drops any connected component below MIN_COMPONENT_VERTS before export.
"""
import bpy
import os
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
out_dir = argv[1]
target_height = float(argv[2]) if len(argv) > 2 else 1.6

MIN_COMPONENT_VERTS = 50
WELD_DISTANCE = 0.0001  # in final (post-scale-fix) units, so this is sub-millimeter at human scale

os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "rigging"), exist_ok=True)
obj_out_path = os.path.join(out_dir, "mesh.obj")
rig_out_path = os.path.join(out_dir, "rigging", "mesh_ori_rig.txt")

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(input_path)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=input_path)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=input_path)
else:
    raise ValueError(f"Unsupported input format: {ext} (expected .fbx, .glb, or .gltf)")

mesh_obj = next(o for o in bpy.data.objects if o.type == 'MESH')
armature_obj = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

print(f"Mesh: {mesh_obj.name}, {len(mesh_obj.data.vertices)} verts")
print(f"Armature: {armature_obj.name}, {len(armature_obj.data.bones)} bones")

# --- Apply all transforms so mesh + armature share one consistent world frame ---
for o in bpy.data.objects:
    o.select_set(False)
armature_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

dims = mesh_obj.dimensions
print(f"Mesh dimensions after transform apply (X,Y,Z): {dims.x:.4f}, {dims.y:.4f}, {dims.z:.4f}")

# --- Unit sanity fix: exporters can bake a stray unit-conversion factor into the armature's
# object scale (either direction), so correct against measured height rather than assuming a
# fixed 100x/0.01x direction. Only the PARENT (armature) scale is set here -- mesh_obj is
# parented to it, so setting both would double-apply the factor through the parent chain.
height = dims.z  # Blender is Z-up internally; this becomes Y after the -Z-forward/Y-up obj export remap
if height < 1e-6:
    raise RuntimeError(f"Degenerate mesh height ({height}); aborting before writing bad data.")
scale_fix = target_height / height
print(f"Measured height={height:.6f}, target={target_height}, applying scale_fix={scale_fix:.6f}")
armature_obj.scale = (scale_fix, scale_fix, scale_fix)
bpy.ops.object.select_all(action='DESELECT')
armature_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
print(f"Mesh dimensions after scale fix: {mesh_obj.dimensions.x:.4f}, {mesh_obj.dimensions.y:.4f}, {mesh_obj.dimensions.z:.4f}")

# --- Mesh cleanup: weld near-coincident vertices and drop tiny disconnected fragments ---
# A small disconnected component's cotangent-Laplacian block is degenerate, which crashes
# MorphGS's Cholesky-based skinning-weight smoothing (heat_diffusion_smoothing) with a
# "matrix not positive definite" error. Weld first (fixes true seam cracks at this point's
# real-world scale), then drop any remaining fragment below MIN_COMPONENT_VERTS -- these
# are debris, not meaningful body geometry. Vertex groups (skin weights) survive both
# separate() and join() since Blender carries per-vertex data along with the vertices.
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=WELD_DISTANCE)
bpy.ops.mesh.separate(type='LOOSE')
bpy.ops.object.mode_set(mode='OBJECT')

pieces = [o for o in bpy.data.objects if o.type == 'MESH' and (o == mesh_obj or o.name.startswith(mesh_obj.name))]
kept = [o for o in pieces if len(o.data.vertices) >= MIN_COMPONENT_VERTS]
dropped = [o for o in pieces if o not in kept]
print(f"Connected components after weld: {len(pieces)} total, "
      f"keeping {len(kept)} (>={MIN_COMPONENT_VERTS} verts), "
      f"dropping {len(dropped)} fragments with vertex counts {[len(o.data.vertices) for o in dropped]}")
if not kept:
    raise RuntimeError("Mesh cleanup dropped every connected component -- MIN_COMPONENT_VERTS is too high.")

for o in dropped:
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.object.select_all(action='DESELECT')
for o in kept:
    o.select_set(True)
bpy.context.view_layer.objects.active = kept[0]
if len(kept) > 1:
    bpy.ops.object.join()
mesh_obj = bpy.context.view_layer.objects.active
print(f"Mesh after cleanup: {len(mesh_obj.data.vertices)} verts, {len(mesh_obj.data.polygons)} faces")

# --- Export mesh.obj (-Z forward, Y up) ---
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.wm.obj_export(
    filepath=obj_out_path,
    export_selected_objects=True,
    export_uv=True,
    export_normals=True,
    export_materials=False,
    forward_axis='NEGATIVE_Z',
    up_axis='Y',
)
print(f"Exported: {obj_out_path}")

# --- Build joint list (depth-first from root) + parent indices ---
bones = armature_obj.data.bones
root_bones = [b for b in bones if b.parent is None]
if len(root_bones) != 1:
    print(f"WARNING: expected 1 root bone, found {len(root_bones)}: {[b.name for b in root_bones]}")

joint_names = []
parent_indices = []
joint_positions = []
world_mat = armature_obj.matrix_world


def visit(bone, parent_idx):
    idx = len(joint_names)
    joint_names.append(bone.name)
    parent_indices.append(parent_idx)
    head_world = world_mat @ bone.head_local
    # obj export axis remap: obj_x=blender_x, obj_y=blender_z, obj_z=-blender_y
    joint_positions.append((head_world.x, head_world.z, -head_world.y))
    for child in bone.children:
        visit(child, idx)


visit(root_bones[0], -1)
print(f"Collected {len(joint_names)} joints")

# --- Build per-vertex skin weights from vertex groups ---
vg_names = [vg.name for vg in mesh_obj.vertex_groups]
vg_index_to_joint_index = {i: joint_names.index(n) for i, n in enumerate(vg_names) if n in joint_names}

mesh_data = mesh_obj.data
n_verts = len(mesh_data.vertices)
skin_lines = []
unmapped_groups = set()
zero_weight_verts = 0
for v in mesh_data.vertices:
    entries = []
    for g in v.groups:
        if g.weight <= 1e-6:
            continue
        if g.group in vg_index_to_joint_index:
            entries.append((joint_names[vg_index_to_joint_index[g.group]], g.weight))
        else:
            unmapped_groups.add(vg_names[g.group] if g.group < len(vg_names) else f"idx{g.group}")
    if not entries:
        zero_weight_verts += 1
        continue
    total = sum(w for _, w in entries)
    if total > 1e-8:
        entries = [(n, w / total) for n, w in entries]
    parts = [f"skin {v.index}"]
    for n, w in entries:
        parts.append(f"{n} {w:.6f}")
    skin_lines.append(" ".join(parts))

print(f"Vertices with skin weights: {len(skin_lines)}/{n_verts}, zero-weight verts: {zero_weight_verts}")
if unmapped_groups:
    print(f"WARNING: vertex groups with no matching joint (ignored): {unmapped_groups}")

# --- Write mesh_ori_rig.txt in MorphGS's exact save_rig_txt format ---
with open(rig_out_path, 'w') as f:
    for name, pos in zip(joint_names, joint_positions):
        f.write(f"joints {name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    f.write("\n")

    root_idx = parent_indices.index(-1)
    f.write(f"root {joint_names[root_idx]}\n")
    f.write("\n")

    for i, pi in enumerate(parent_indices):
        if pi != -1:
            f.write(f"hier {joint_names[pi]} {joint_names[i]}\n")
    f.write("\n")

    for line in skin_lines:
        f.write(line + "\n")

print(f"Exported: {rig_out_path}")

# --- Persist conversion metadata for the animation-bake step (MorphGS: Export Animated Mesh),
# which needs to apply the exact same scale correction and axis remap to the ORIGINAL rigged
# file so its bone rest positions line up with what training actually saw in mesh_ori_rig.txt.
import json
meta_path = os.path.join(out_dir, "rigging", "conversion_meta.json")
with open(meta_path, "w") as f:
    json.dump({
        "source_file": os.path.basename(input_path),
        "scale_fix": scale_fix,
        "target_height": target_height,
        "obj_export_axis_remap": "blender(x,y,z) -> obj(x, z, -y)",
    }, f, indent=2)
print(f"Exported: {meta_path}")

print("=== DONE ===")
