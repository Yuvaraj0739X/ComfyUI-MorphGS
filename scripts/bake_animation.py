"""
Bake a MorphGS-trained pose sequence (from extract_pose_sequence.py) onto the ORIGINAL
rigged character file as standard keyframed bone animation, and export an animated
FBX/GLB -- a real animated 3D asset, not just a rendered video.

Coordinate-space note (the part that's easy to get subtly wrong): mesh_ori_rig.txt only
stores joint REST POSITIONS (mesh_to_morphgs.py's conversion never captured each bone's
rest ORIENTATION, since MorphGS's own rig format doesn't need it for point-based skinning).
MorphGS's learned per-joint rotation is a rotation DELTA expressed in the pipeline's
world/rig-space axes, applied around the parent joint's rest position -- confirmed by the
network's own zero-initialization: at the start of training rot_params~=0, so bone_Ts is
exactly identity for every joint, i.e. the rest pose is exactly reproduced when no motion has
been learned yet. That means the correct target orientation for each bone at each frame is:
    R_target_blender = (R_delta, converted into Blender axes) @ R_rest_blender
where R_rest_blender is pulled from the ORIGINAL rigged file's own armature (loaded here),
not from mesh_ori_rig.txt.

Run inside Blender (headless):
    blender --background --python bake_animation.py -- <original_rigged.fbx|.glb> \
        <pose_sequence.npz> <conversion_meta.json> <fps> <output.fbx|.glb>
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv
argv = argv[argv.index("--") + 1:]
original_rigged_path = argv[0]
npz_path = argv[1]
meta_path = argv[2]
fps = float(argv[3])
output_path = argv[4]

# --- Load pose sequence produced by extract_pose_sequence.py ---
data = np.load(npz_path, allow_pickle=True)
bone_Ts = data["bone_Ts"]  # (num_frames, NJ, 4, 4), MorphGS rig-space (matches mesh_ori_rig.txt axes)
joint_names = [str(n) for n in data["joint_names"]]
num_frames = int(data["num_frames"])

with open(meta_path) as f:
    meta = json.load(f)
scale_fix = float(meta["scale_fix"])

# --- Axis conversion: mesh_to_morphgs.py wrote joint positions as
# obj_pos = (blender_x, blender_z, -blender_y). As a matrix P such that morphgs_vec = P @ blender_vec:
P = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)
P_T = P.T  # P is a signed permutation matrix, so P^-1 == P^T


def morphgs_to_blender_matrix(M_m):
    """Convert a 4x4 (rotation+translation) from MorphGS rig-space into Blender space."""
    R_m = M_m[:3, :3]
    t_m = M_m[:3, 3]
    R_b = P_T @ R_m @ P
    t_b = P_T @ t_m
    M_b = np.eye(4)
    M_b[:3, :3] = R_b
    M_b[:3, 3] = t_b
    return M_b


# --- Import the ORIGINAL rigged file and apply the exact same scale correction used during
# the mesh.obj/mesh_ori_rig.txt conversion, so bone rest positions line up with the trained
# joint positions. (Orientation is unaffected by uniform scale, but position must match.)
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(original_rigged_path)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=original_rigged_path)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=original_rigged_path)
else:
    raise ValueError(f"Unsupported input format: {ext}")

mesh_obj = next(o for o in bpy.data.objects if o.type == 'MESH')
armature_obj = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

for o in bpy.data.objects:
    o.select_set(False)
armature_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

armature_obj.scale = (scale_fix, scale_fix, scale_fix)
bpy.ops.object.select_all(action='DESELECT')
armature_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

print(f"Applied scale_fix={scale_fix}")

# --- Capture each bone's REST orientation (Blender-space, armature-space == world-space
# after the transform_apply above), which mesh_ori_rig.txt never stored. ---
rest_matrices = {}
for bone in armature_obj.data.bones:
    m = np.array(bone.matrix_local)  # armature-space rest matrix, 4x4
    rest_matrices[bone.name] = m

missing = [n for n in joint_names if n not in rest_matrices]
if missing:
    raise RuntimeError(f"Joint names from the pose sequence not found in this armature: {missing}")

# --- Set up pose bones for quaternion keyframing ---
bpy.ops.object.mode_set(mode='POSE')
for name in joint_names:
    pb = armature_obj.pose.bones[name]
    pb.rotation_mode = 'QUATERNION'

scene = bpy.context.scene
scene.render.fps = int(round(fps))
scene.frame_start = 0
scene.frame_end = num_frames - 1

for f in range(num_frames):
    scene.frame_set(f)
    for j, name in enumerate(joint_names):
        pb = armature_obj.pose.bones[name]
        M_target_morphgs = bone_Ts[f, j]
        M_target_blender = morphgs_to_blender_matrix(M_target_morphgs)

        R_rest = rest_matrices[name][:3, :3]
        R_delta = M_target_blender[:3, :3]
        R_target = R_delta @ R_rest

        # BUG FIX: M_target_blender's translation column is where the COORDINATE ORIGIN maps
        # to under this affine transform -- NOT where the joint itself ends up. Those are only
        # the same thing when a joint's rest position happens to be at the origin (true for
        # the root, false for every other joint -- which is exactly why single-joint/root-only
        # tests looked fine while every child joint collapsed toward the origin). The correct
        # posed position requires applying the full transform to the joint's own REST position.
        t_rest = rest_matrices[name][:3, 3]
        t_target = M_target_blender[:3, :3] @ t_rest + M_target_blender[:3, 3]

        M_pose = Matrix.Identity(4)
        for r in range(3):
            for c in range(3):
                M_pose[r][c] = float(R_target[r, c])
        M_pose[0][3], M_pose[1][3], M_pose[2][3] = (float(t_target[0]), float(t_target[1]), float(t_target[2]))

        pb.matrix = M_pose
        # Update the depsgraph after EACH bone, not once after the whole per-frame loop --
        # otherwise a child bone's matrix-basis gets computed against its parent's stale
        # (pre-this-frame) pose_mat, since Blender doesn't necessarily propagate a pose_bone
        # matrix assignment to its children synchronously. Confirmed via a diagnostic: posing
        # only the root joint (no parent-dependency chain involved) rendered correctly, while
        # posing the full hierarchy without an update between each bone produced a shredded,
        # exploded mesh -- exactly the signature of children reading a stale parent transform.
        bpy.context.view_layer.update()
    for name in joint_names:
        pb = armature_obj.pose.bones[name]
        pb.keyframe_insert(data_path="location", frame=f)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=f)

    if f % 10 == 0 or f == num_frames - 1:
        print(f"Keyframed frame {f}/{num_frames - 1}")

bpy.ops.object.mode_set(mode='OBJECT')

# --- Export ---
bpy.ops.object.select_all(action='DESELECT')
armature_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj

out_ext = os.path.splitext(output_path)[1].lower()
if out_ext == ".fbx":
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        path_mode='COPY',
        embed_textures=True,
    )
elif out_ext in (".glb", ".gltf"):
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB' if out_ext == ".glb" else 'GLTF_EMBEDDED',
        use_selection=True,
        export_animations=True,
        export_frame_range=False,
    )
else:
    raise ValueError(f"Unsupported output format: {out_ext}")

print(f"Exported animated mesh: {output_path}")
print("=== DONE ===")
