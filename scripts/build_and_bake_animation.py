"""
Bake a MorphGS-trained pose sequence onto a character by building the skinned armature
directly from mesh.obj + mesh_ori_rig.txt, instead of requiring the character's original
externally-rigged file.

mesh_ori_rig.txt is a full RigNet-format rig -- it already stores joint rest positions,
parent/child hierarchy, AND per-vertex skin weights ("skin <vertex_idx> <joint> <weight> ...").
That's everything needed to reconstruct a working skinned armature from scratch, so this path
works for ANY MorphGS character, including ones (like MorphGS's own bundled demo characters)
that were never converted from a Blender-rigged source file and have no such file on disk.

Coordinate-space note: MorphGS's own mesh.obj/rig-file convention is Y-up (confirmed
empirically -- across both mesh_to_morphgs.py-converted characters and MorphGS's own bundled
demo characters, the joint-position spread is always largest along Y), not Blender's native
Z-up. Unlike bake_animation.py (which bakes onto an EXTERNAL rigged file already living in
Blender's Z-up space, so it converts MorphGS-space deltas into that space), everything here --
mesh vertices, joint rest positions, and bone_Ts -- starts out in MorphGS's raw Y-up space and
must be converted into Blender's Z-up space before use. No scale correction is needed here,
though (unlike bake_animation.py's scale_fix): there's no external file's own unit convention
to reconcile with, since mesh + rig + pose sequence are all already in one consistent scale.

mesh.obj is parsed by hand (not via Blender's OBJ importer) specifically to avoid any
importer-side axis-convention guessing -- the raw (x, y, z) triples in the file must land in
Blender unchanged, since mesh_ori_rig.txt's joint positions are in that exact same raw space.

Run inside Blender (headless):
    blender --background --python build_and_bake_animation.py -- <mesh.obj> <mesh_ori_rig.txt> \
        <pose_sequence.npz> <fps> <output.fbx|.glb>
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv
argv = argv[argv.index("--") + 1:]
mesh_obj_path = argv[0]
rig_path = argv[1]
npz_path = argv[2]
fps = float(argv[3])
output_path = argv[4]


# MorphGS rig-space (Y-up) -> Blender space (Z-up): morphgs_vec = P @ blender_vec, the same
# convention bake_animation.py uses, so morphgs_to_blender = P_T = P.T (P is an orthogonal
# signed permutation matrix, so its inverse is its transpose).
P = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)
P_T = P.T


def to_blender_vec(v):
    return tuple(P_T @ np.array(v, dtype=np.float64))


def to_blender_matrix(M_m):
    R_b = P_T @ M_m[:3, :3] @ P
    t_b = P_T @ M_m[:3, 3]
    M_b = np.eye(4)
    M_b[:3, :3] = R_b
    M_b[:3, 3] = t_b
    return M_b


def parse_obj(path):
    """Minimal, tolerant OBJ parser -- only cares about "v " and "f " lines (ignoring
    vt/vn/o/g/usemtl/etc, and whatever material library mesh.obj references, since neither
    UVs nor materials affect skinning or posing). Handles both OBJ index conventions: normal
    1-based absolute indices, and negative indices (relative to the vertex count so far at
    that point in the file) -- some OBJ exporters emit the latter, and MorphGS's own mesh.obj
    isn't guaranteed to always be produced by the same tool for every character.
    """
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    idx = int(p.split("/")[0])
                    face.append(idx - 1 if idx > 0 else len(verts) + idx)
                if len(face) >= 3:
                    faces.append(face)
    return verts, faces


def parse_rig(path):
    joints_name = []
    joints_pos = []
    bones = []  # (parent_name, child_name), file order, includes a "root root" self-loop line
    root_name = None
    skin = {}  # vertex_idx -> [(joint_name, weight), ...]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if tokens[0] == "joints":
                joints_name.append(tokens[1])
                joints_pos.append((float(tokens[2]), float(tokens[3]), float(tokens[4])))
            elif tokens[0] == "root":
                root_name = tokens[1]
            elif tokens[0] == "hier":
                bones.append((tokens[1], tokens[2]))
            elif tokens[0] == "skin":
                v_idx = int(tokens[1])
                pairs = []
                rest = tokens[2:]
                for i in range(0, len(rest), 2):
                    pairs.append((rest[i], float(rest[i + 1])))
                skin[v_idx] = pairs

    return joints_name, joints_pos, bones, root_name, skin


verts, faces = parse_obj(mesh_obj_path)
joints_name, joints_pos, bones, root_name, skin = parse_rig(rig_path)
print(f"Parsed mesh: {len(verts)} verts, {len(faces)} faces")
print(f"Parsed rig: {len(joints_name)} joints, {len(bones)} hier entries, {len(skin)} skinned verts")

children_of = {}
for parent, child in bones:
    if parent == child:
        continue  # the root's self-loop marker line, not a real parent/child edge
    children_of.setdefault(parent, []).append(child)

# --- Build the mesh object from the raw parsed vertices/faces (no Blender OBJ importer, so
# there is no axis-convention guessing between this and the rig file's own raw coordinates). ---
bpy.ops.wm.read_factory_settings(use_empty=True)
verts_blender = [to_blender_vec(v) for v in verts]
mesh_data = bpy.data.meshes.new("MorphGSMesh")
mesh_data.from_pydata(verts_blender, [], faces)
mesh_data.update()
mesh_obj = bpy.data.objects.new("MorphGSMesh", mesh_data)
bpy.context.collection.objects.link(mesh_obj)

# --- Build the armature: one bone per joint, head = joint rest position, tail = average of
# its children's positions (or a small fixed offset for leaf joints, since a zero-length bone
# is invalid in Blender) -- the tail placement only affects bone display, not the posing math
# below, which sets each pose bone's full armature-space matrix directly every frame. ---
joint_pos_by_name = {name: to_blender_vec(pos) for name, pos in zip(joints_name, joints_pos)}
arm_data = bpy.data.armatures.new("MorphGSArmature")
arm_obj = bpy.data.objects.new("MorphGSArmature", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj

bpy.ops.object.mode_set(mode='EDIT')
edit_bones = {}
for name in joints_name:
    eb = arm_data.edit_bones.new(name)
    head = np.array(joint_pos_by_name[name])
    kids = children_of.get(name, [])
    if kids:
        tail = np.mean([joint_pos_by_name[k] for k in kids], axis=0)
        if np.linalg.norm(tail - head) < 1e-6:
            tail = head + np.array([0.0, 0.01, 0.0])
    else:
        tail = head + np.array([0.0, 0.01, 0.0])
    eb.head = tuple(head)
    eb.tail = tuple(tail)
    edit_bones[name] = eb

for parent, child in bones:
    if parent == child:
        continue
    edit_bones[child].parent = edit_bones[parent]
    edit_bones[child].use_connect = False

bpy.ops.object.mode_set(mode='OBJECT')

rest_matrices = {b.name: np.array(b.matrix_local) for b in arm_data.bones}

# --- Vertex groups + skin weights, straight from mesh_ori_rig.txt's own "skin" lines. ---
vgroups = {name: mesh_obj.vertex_groups.new(name=name) for name in joints_name}
for v_idx, pairs in skin.items():
    for joint_name, weight in pairs:
        if weight > 0:
            vgroups[joint_name].add([v_idx], weight, 'REPLACE')

mesh_obj.parent = arm_obj
mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
mod.object = arm_obj

# --- Load the trained pose sequence and keyframe it. Already in this exact same rig-space --
# no axis remap or scale correction needed, unlike bake_animation.py. ---
data = np.load(npz_path, allow_pickle=True)
bone_Ts = data["bone_Ts"]
pose_joint_names = [str(n) for n in data["joint_names"]]
num_frames = int(data["num_frames"])

missing = [n for n in pose_joint_names if n not in rest_matrices]
if missing:
    raise RuntimeError(f"Pose sequence joints not found in built armature: {missing}")

bpy.ops.object.mode_set(mode='POSE')
for name in pose_joint_names:
    arm_obj.pose.bones[name].rotation_mode = 'QUATERNION'

scene = bpy.context.scene
scene.render.fps = int(round(fps))
scene.frame_start = 0
scene.frame_end = num_frames - 1

for f in range(num_frames):
    scene.frame_set(f)
    for j, name in enumerate(pose_joint_names):
        pb = arm_obj.pose.bones[name]
        M_target = to_blender_matrix(bone_Ts[f, j])

        R_rest = rest_matrices[name][:3, :3]
        R_target = M_target[:3, :3] @ R_rest

        t_rest = rest_matrices[name][:3, 3]
        t_target = M_target[:3, :3] @ t_rest + M_target[:3, 3]

        M_pose = Matrix.Identity(4)
        for r in range(3):
            for c in range(3):
                M_pose[r][c] = float(R_target[r, c])
        M_pose[0][3], M_pose[1][3], M_pose[2][3] = (float(t_target[0]), float(t_target[1]), float(t_target[2]))

        pb.matrix = M_pose
        bpy.context.view_layer.update()
    for name in pose_joint_names:
        pb = arm_obj.pose.bones[name]
        pb.keyframe_insert(data_path="location", frame=f)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=f)

    if f % 10 == 0 or f == num_frames - 1:
        print(f"Keyframed frame {f}/{num_frames - 1}")

bpy.ops.object.mode_set(mode='OBJECT')

# --- Export ---
bpy.ops.object.select_all(action='DESELECT')
arm_obj.select_set(True)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj

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
