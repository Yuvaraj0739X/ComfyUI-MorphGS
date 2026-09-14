# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

"""
Pinocchio-based Auto-Rigging for MorphGS
========================================
Alternative rigging method using pynocchio (Pinocchio auto-rigging).
Supports HumanSkeleton and HorseSkeleton templates.

Usage:
    python src/utils/rigging_pinocchio.py --mesh=/path/to/mesh.obj --sk_type=quad
    python src/utils/rigging_pinocchio.py --mesh=/path/to/mesh.obj --sk_type=human
"""

import argparse
import os
import subprocess
import numpy as np
import open3d as o3d
import pynocchio as pyn
from pynocchio import auto_rig, Mesh, Vector3, Points, skeletons
from sklearn.neighbors import NearestNeighbors


# Manifold tool paths (built from MagicPose4D/Manifold/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
MANIFOLD_BIN = os.path.join(_PROJECT_ROOT, "MagicPose4D", "Manifold", "build", "manifold")
SIMPLIFY_BIN = os.path.join(_PROJECT_ROOT, "MagicPose4D", "Manifold", "build", "simplify")

# Preprocessing parameters
MANIFOLD_FACES = 20000
SIMPLIFY_FACES = 5000
SIMPLIFY_RATIO = 0.2


# ==============================================================================
# Mesh Preprocessing
# ==============================================================================

def manifold_remesh(input_obj, output_obj, target_faces=MANIFOLD_FACES):
    """Make a mesh watertight/manifold using the Manifold tool."""
    print(f"  [Manifold] {os.path.basename(input_obj)} -> {os.path.basename(output_obj)}")
    result = subprocess.run(
        [MANIFOLD_BIN, input_obj, output_obj, str(target_faces)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Manifold failed: {result.stderr}")
    return True


def simplify_mesh(input_obj, output_obj, target_faces=SIMPLIFY_FACES, ratio=SIMPLIFY_RATIO):
    """Simplify a mesh for rigging."""
    print(f"  [Simplify] {os.path.basename(input_obj)} -> {os.path.basename(output_obj)}")
    result = subprocess.run(
        [SIMPLIFY_BIN, '-i', input_obj, '-o', output_obj,
         '-m', '-c', '1e-2', '-f', str(target_faces), '-r', str(ratio)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Simplify failed: {result.stderr}")
    return True


def preprocess_mesh(input_obj, cache_dir):
    """
    Preprocess mesh: manifold remesh then simplify.
    Returns (simple_path, remesh_path), using cached versions if available.
    """
    os.makedirs(cache_dir, exist_ok=True)
    remesh_path = os.path.join(cache_dir, "mesh_remesh.obj")
    simple_path = os.path.join(cache_dir, "mesh_simple.obj")

    if not os.path.exists(remesh_path):
        manifold_remesh(input_obj, remesh_path)
    else:
        print(f"  Using cached: {os.path.basename(remesh_path)}")

    if not os.path.exists(simple_path):
        simplify_mesh(remesh_path, simple_path)
    else:
        print(f"  Using cached: {os.path.basename(simple_path)}")

    return simple_path, remesh_path


# ==============================================================================
# Pinocchio Rigging
# ==============================================================================

def run_pinocchio_rig(mesh_path, sk_type='quad'):
    """
    Run Pinocchio auto-rigging on a (simplified) mesh.

    Args:
        mesh_path: path to OBJ mesh (should be manifold/simplified)
        sk_type: 'human' or 'quad'
    Returns:
        embedding: (N_joints, 3) joint positions
        parent_indices: parent index array (-1 for root)
        skin_weights: (N_verts, N_bones) skinning weights
        joint_names: list of joint name strings
    """
    print(f"  [Pinocchio] Rigging {os.path.basename(mesh_path)} (skeleton={sk_type})")

    raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(raw_mesh.vertices)
    triangles = np.asarray(raw_mesh.triangles)
    print(f"  Mesh: {len(vertices)} vertices, {len(triangles)} faces")

    # Build pynocchio mesh
    points = Points()
    for v in vertices:
        points.append(Vector3(float(v[0]), float(v[1]), float(v[2])))

    tri_indices = pyn.Indices()
    for tri in triangles:
        for idx in tri:
            tri_indices.append(int(idx))

    # Select skeleton template
    if sk_type == 'human':
        skeleton = skeletons.HumanSkeleton()
    elif sk_type == 'quad':
        skeleton = skeletons.HorseSkeleton()
    else:
        skeleton = skeletons.FileSkeleton(sk_type)

    skeleton.scale(0.7)
    parent_indices = np.array(skeleton.parent_indices)

    # Auto-rig
    mesh = Mesh(points, tri_indices)
    attach = auto_rig(skeleton, mesh)

    embedding = np.array(attach.embedding)
    skin_weights = attach.getAllWeights()  # (N_verts_simple, N_bones)

    # Generate joint names
    n_joints = len(embedding)
    joint_names = [f"joint_{i}" for i in range(n_joints)]

    print(f"  Joints: {n_joints}, Bones: {np.sum(parent_indices != -1)}")
    print(f"  Skin weights: {skin_weights.shape}")

    return embedding, parent_indices, skin_weights, joint_names


def transfer_skinning_weights(src_verts, src_weights, tgt_verts):
    """Transfer skinning weights from source to target mesh via nearest-neighbor."""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src_verts)
    _, indices = nbrs.kneighbors(tgt_verts)
    return src_weights[indices.flatten()]


# ==============================================================================
# Save to _rig.txt format (compatible with RigModel.load_rig_txt)
# ==============================================================================

def save_rig_txt(filename, joint_names, joint_positions, parent_indices, skinning_weights):
    """
    Save rigging data in MorphGS's _rig.txt format.

    Args:
        filename: output file path
        joint_names: list of joint name strings
        joint_positions: (N_joints, 3) numpy array
        parent_indices: (N_joints,) parent index array (-1 for root)
        skinning_weights: (N_verts, N_joints) numpy array
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Find root index
    root_idx = np.where(parent_indices == -1)[0][0]

    # Build bones list: (parent, child)
    bones = []
    for i, pi in enumerate(parent_indices):
        if pi != -1:
            bones.append((pi, i))

    with open(filename, 'w') as f:
        # Joints
        for name, pos in zip(joint_names, joint_positions):
            f.write(f"joints {name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        f.write("\n")

        # Root
        f.write(f"root {joint_names[root_idx]}\n")
        f.write("\n")

        # Hierarchy
        for parent_idx, child_idx in bones:
            f.write(f"hier {joint_names[parent_idx]} {joint_names[child_idx]}\n")
        f.write("\n")

        # Skinning weights
        # Pinocchio outputs weights per bone (N_bones = N_joints - 1)
        # We need to map bone weights to joint weights
        # parent_indices[i] != -1 means joint i is a child, corresponding to bone (parent_i, i)
        # Pinocchio weights are indexed by bone order
        n_verts = skinning_weights.shape[0]

        # bone_to_child_joint mapping
        bone_joints = [i for i in range(len(parent_indices)) if parent_indices[i] != -1]

        for v_idx in range(n_verts):
            weights = skinning_weights[v_idx]
            active = [(b, w) for b, w in enumerate(weights) if w > 1e-6]
            if not active:
                continue

            parts = [f"skin {v_idx}"]
            for bone_idx, w in active:
                # Map bone index to child joint name
                if bone_idx < len(bone_joints):
                    j_name = joint_names[bone_joints[bone_idx]]
                else:
                    j_name = joint_names[bone_idx]
                parts.append(f"{j_name} {w:.6f}")
            f.write(" ".join(parts) + "\n")

    print(f"  Saved rig: {filename}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def rig_mesh(mesh_path, sk_type='quad', output_dir=None):
    """
    Full rigging pipeline: preprocess → Pinocchio auto-rig → save _rig.txt.

    Args:
        mesh_path: path to original target mesh .obj
        sk_type: 'human' or 'quad'
        output_dir: output directory (default: {mesh_dir}/rigging_pinocchio/)
    Returns:
        output_path: path to saved _rig.txt
    """
    mesh_dir = os.path.dirname(mesh_path)
    if output_dir is None:
        output_dir = os.path.join(mesh_dir, "rigging_pinocchio")
    cache_dir = os.path.join(output_dir, "preprocessed")

    print("=" * 60)
    print("Pinocchio Auto-Rigging")
    print(f"  Mesh: {mesh_path}")
    print(f"  Skeleton: {sk_type}")
    print("=" * 60)

    # 1. Preprocess mesh
    print("\n[1/4] Preprocessing mesh...")
    simple_path, remesh_path = preprocess_mesh(mesh_path, cache_dir)
    simple_verts = np.asarray(o3d.io.read_triangle_mesh(simple_path).vertices)

    # 2. Run Pinocchio
    print("\n[2/4] Running Pinocchio auto-rig...")
    embedding, parent_indices, skin_weights_simple, joint_names = run_pinocchio_rig(
        simple_path, sk_type=sk_type
    )

    # 3. Transfer skinning weights to original mesh 
    # IMPORTANT: Use trimesh with process=False to match MorphGS's mesh loading (mesh_utils.load_mesh).
    # OBJ files can have different vertex counts depending on the loader (UV seam splitting).
    print("\n[3/4] Transferring skinning weights to original mesh...")
    import trimesh
    orig_mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
    orig_verts = np.asarray(orig_mesh.vertices)
    skin_weights_orig = transfer_skinning_weights(simple_verts, skin_weights_simple, orig_verts)
    print(f"  Original mesh (trimesh): {len(orig_verts)} vertices")
    print(f"  Transferred weights: {skin_weights_orig.shape}")

    # 4. Save
    print("\n[4/4] Saving rig file...")
    output_path = os.path.join(output_dir, "mesh_ori_rig.txt")
    save_rig_txt(output_path, joint_names, embedding, parent_indices, skin_weights_orig)

    print("\n" + "=" * 60)
    print("Rigging complete!")
    print(f"  Output: {output_path}")
    print(f"  Joints: {len(embedding)}, Bones: {np.sum(parent_indices != -1)}")
    print(f"  Vertices with weights: {skin_weights_orig.shape[0]}")
    print("=" * 60)

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pinocchio Auto-Rigging for MorphGS")
    parser.add_argument('--mesh', required=True, help="Path to target mesh .obj file")
    parser.add_argument('--sk_type', default='quad', choices=['human', 'quad'],
                        help="Skeleton type: 'human' or 'quad' (default: quad)")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory (default: {mesh_dir}/rigging_pinocchio/)")
    args = parser.parse_args()

    rig_mesh(args.mesh, sk_type=args.sk_type, output_dir=args.output_dir)
