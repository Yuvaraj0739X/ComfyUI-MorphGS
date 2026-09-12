"""
Bake a MorphGS-trained pose sequence onto a character by building the skinned armature
directly from mesh.obj + mesh_ori_rig.txt, instead of requiring the character's original
externally-rigged file.

mesh_ori_rig.txt is a full RigNet-format rig -- it already stores joint rest positions,
parent/child hierarchy, AND per-vertex skin weights ("skin <vertex_idx> <joint> <weight> ...").
That's everything needed to reconstruct a working skinned armature from scratch, so this path
works for ANY MorphGS character, including ones (like MorphGS's own bundled demo characters)
that were never converted from a Blender-rigged source file and have no such file on disk.

Skin weights: pass resolve_skinning_weights.py's output as the optional 6th argument to use
smoothed weights matching what MorphGS actually trained with (see that script's docstring --
some characters' configs apply heat-diffusion smoothing to the raw rig-file weights, which is
invisible at rest pose but causes severe distortion under real motion if skipped). Falls back
to the raw "skin" lines in mesh_ori_rig.txt, 1:1 by vertex index, if omitted.

Coordinate-space note: MorphGS's own mesh.obj/rig-file convention is Y-up (confirmed
empirically -- across both mesh_to_morphgs.py-converted characters and MorphGS's own bundled
demo characters, the joint-position spread is always largest along Y), not Blender's native
Z-up. Unlike bake_animation.py (which bakes onto an EXTERNAL rigged file already living in
Blender's Z-up space, so it converts MorphGS-space deltas into that space), everything here --
mesh vertices, joint rest positions, and bone_Ts -- starts out in MorphGS's raw Y-up space and
must be converted into Blender's Z-up space before use. No scale correction is needed here,
though (unlike bake_animation.py's scale_fix): there's no external file's own unit convention
to reconcile with, since mesh + rig + pose sequence are all already in one consistent scale.

mesh.obj is parsed by hand (not via Blender's OBJ importer, and not via trimesh -- trimesh's
OBJ loader can expand vertex count for per-face-corner UV coordinates, which does NOT match
mesh_ori_rig.txt's raw per-"v"-line vertex indexing that the "skin" lines assume) to avoid any
importer-side vertex-reindexing or axis-convention guessing: the raw (x, y, z) triples in the
file must land in Blender unchanged, one Blender vertex per raw "v" line, since
mesh_ori_rig.txt's joint positions and skin weights are both in that exact same raw space.
UVs are still read from the file's "vt" lines and each face's own v/vt corner indices (a
per-loop attribute, independent of the per-vertex position indexing used for topology/skin
weights, so reading it doesn't reintroduce the trimesh vertex-count mismatch), and a material
is built from the referenced .mtl's "map_Kd" image if present, so textured characters (like
MorphGS's own bundled chickenDC/moose1DOG/spot) keep their appearance through this path too.

Run inside Blender (headless):
    blender --background --python build_and_bake_animation.py -- <mesh.obj> <mesh_ori_rig.txt> \
        <pose_sequence.npz> <fps> <output.fbx|.glb> [resolved_skinning_weights.npz]
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
resolved_weights_path = argv[5] if len(argv) > 5 else None


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
    """Minimal, tolerant OBJ parser. Vertex/face topology (used for both mesh geometry and
    skin-weight indexing) only ever comes from "v "/"f " lines' vertex-position index, matching
    resolve_skinning_weights.py's own parser exactly. UVs are read separately, from "vt" lines
    plus each face's own per-corner v/vt index pairs (a genuinely per-loop attribute in OBJ,
    independent of vertex-position indexing, so reading it doesn't affect vertex count/topology
    at all). Triangulates any face with more than 3 vertices fan-style. Handles both OBJ index
    conventions: normal 1-based absolute indices, and negative indices (relative to the vertex/
    UV count so far at that point in the file).

    Also handles trimesh's own OBJ-with-vertex-colors extension ("v x y z r g b", 7 fields
    instead of 4) -- confirmed on MorphGS's own bundled spot demo character, which has no
    material/UV data at all and encodes its yellow/gray coloring this way instead.
    """
    verts = []
    vertex_colors = []  # (r, g, b) per vertex, only populated if "v" lines carry 7 fields
    uvs = []
    faces = []  # list of vertex-index triples
    face_uvs = []  # list of uv-index triples (or None per face if that face has no vt data)
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >= 7:
                    vertex_colors.append((float(parts[4]), float(parts[5]), float(parts[6])))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                face_uv = []
                for p in parts:
                    tokens = p.split("/")
                    v_idx = int(tokens[0])
                    face.append(v_idx - 1 if v_idx > 0 else len(verts) + v_idx)
                    if len(tokens) >= 2 and tokens[1]:
                        vt_idx = int(tokens[1])
                        face_uv.append(vt_idx - 1 if vt_idx > 0 else len(uvs) + vt_idx)
                    else:
                        face_uv = None
                if len(face) >= 3:
                    faces.append(face[:3])
                    face_uvs.append(face_uv[:3] if face_uv else None)
                    for extra in range(3, len(face)):
                        faces.append([face[0], face[extra - 1], face[extra]])
                        face_uvs.append(
                            [face_uv[0], face_uv[extra - 1], face_uv[extra]] if face_uv else None
                        )
    vertex_colors = vertex_colors if len(vertex_colors) == len(verts) else []
    return verts, faces, uvs, face_uvs, vertex_colors


def parse_mtl_texture(mesh_obj_path):
    """Find the referenced .mtl file (via mesh.obj's own "mtllib" line, falling back to the
    same basename with a .mtl extension) and return the absolute path of its "map_Kd" image,
    resolved relative to mesh.obj's own directory, or None if there's no material/texture."""
    mesh_dir = os.path.dirname(os.path.abspath(mesh_obj_path))
    mtl_name = None
    with open(mesh_obj_path) as f:
        for line in f:
            if line.startswith("mtllib"):
                mtl_name = line.split(maxsplit=1)[1].strip()
                break
    candidates = []
    if mtl_name:
        candidates.append(os.path.join(mesh_dir, mtl_name))
    candidates.append(os.path.splitext(mesh_obj_path)[0] + ".mtl")

    for mtl_path in candidates:
        if not os.path.isfile(mtl_path):
            continue
        with open(mtl_path) as f:
            for line in f:
                if line.strip().startswith("map_Kd"):
                    tex_name = line.split(maxsplit=1)[1].strip()
                    tex_path = os.path.join(mesh_dir, tex_name)
                    return tex_path if os.path.isfile(tex_path) else None
    return None


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


verts, faces, uvs, face_uvs, vertex_colors = parse_obj(mesh_obj_path)
texture_path = parse_mtl_texture(mesh_obj_path)
joints_name, joints_pos, bones, root_name, skin = parse_rig(rig_path)
print(f"Parsed mesh: {len(verts)} verts, {len(faces)} faces, {len(uvs)} UVs, "
      f"{len(vertex_colors)} vertex colors")
print(f"Texture: {texture_path}")
print(f"Parsed rig: {len(joints_name)} joints, {len(bones)} hier entries, {len(skin)} skinned verts")

if resolved_weights_path:
    resolved = np.load(resolved_weights_path, allow_pickle=True)
    resolved_weights = resolved["weights"]  # (NV, NJ), same vertex order as parse_obj above
    resolved_joint_names = [str(n) for n in resolved["joint_names"]]
    if resolved_joint_names != joints_name or resolved_weights.shape[0] != len(verts):
        raise RuntimeError(
            f"resolved_skinning_weights.npz doesn't match this mesh/rig: "
            f"{resolved_weights.shape[0]} verts/{len(resolved_joint_names)} joints vs "
            f"{len(verts)} verts/{len(joints_name)} joints expected."
        )
    print(f"Using resolved (smoothed) skinning weights from {resolved_weights_path}")
else:
    resolved_weights = None

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

# --- UVs: a per-face-corner (loop) attribute, set directly from each face's own v/vt index
# pairs -- independent of the vertex-position indexing above, so this doesn't reintroduce any
# vertex-count mismatch with the skin weights. ---
if uvs and all(fu is not None for fu in face_uvs):
    uv_layer = mesh_data.uv_layers.new(name="UVMap")
    for poly in mesh_data.polygons:
        face_uv_indices = face_uvs[poly.index]
        for loop_idx, uv_idx in zip(poly.loop_indices, face_uv_indices):
            uv_layer.data[loop_idx].uv = uvs[uv_idx]
else:
    print("No UV data found in mesh.obj -- exported mesh will have no texture coordinates.")

# --- Per-vertex color, if mesh.obj used trimesh's "v x y z r g b" extension instead of a
# UV+image texture (confirmed on MorphGS's own bundled spot demo character). A genuinely
# per-vertex attribute, so it's set directly from vertex index, no loop/corner indirection
# needed (unlike UVs above). ---
if vertex_colors:
    color_attr = mesh_data.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
    for i, (r, g, b) in enumerate(vertex_colors):
        color_attr.data[i].color = (r, g, b, 1.0)

# --- Material: a UV+image texture from the .mtl file's "map_Kd" if present, else vertex
# colors if present, else left untextured. ---
if texture_path:
    mat = bpy.data.materials.new(name="MorphGSMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    mesh_data.materials.append(mat)
    print(f"Applied material with texture: {texture_path}")
elif vertex_colors:
    mat = bpy.data.materials.new(name="MorphGSMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    color_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = "Col"
    mat.node_tree.links.new(color_node.outputs["Color"], bsdf.inputs["Base Color"])
    mesh_data.materials.append(mat)
    print("Applied material from per-vertex colors (no UV/image texture in source)")
else:
    print("No texture or vertex colors found -- exported mesh will be untextured.")

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
            tail = head + np.array([0.0, 0.0, 0.01])
    else:
        tail = head + np.array([0.0, 0.0, 0.01])
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

# --- Vertex groups + skin weights: resolved (smoothed) weights if provided, else the raw
# "skin" lines from mesh_ori_rig.txt directly. ---
vgroups = {name: mesh_obj.vertex_groups.new(name=name) for name in joints_name}
if resolved_weights is not None:
    nz_v, nz_j = np.nonzero(resolved_weights > 1e-5)
    for v_idx, j_idx in zip(nz_v.tolist(), nz_j.tolist()):
        vgroups[joints_name[j_idx]].add([v_idx], float(resolved_weights[v_idx, j_idx]), 'REPLACE')
else:
    for v_idx, pairs in skin.items():
        for joint_name, weight in pairs:
            if weight > 0:
                vgroups[joint_name].add([v_idx], weight, 'REPLACE')

mesh_obj.parent = arm_obj
mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
mod.object = arm_obj
mod.use_deform_preserve_volume = True

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
