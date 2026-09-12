"""
Resolve the ACTUAL per-vertex skinning weights MorphGS used during training for a given
character/experiment -- not just the raw "skin" lines in mesh_ori_rig.txt.

model/RigModel.py's Rig class doesn't necessarily use the rig file's raw skin weights as-is:
depending on the experiment's config, it applies heat-diffusion Laplacian smoothing whenever
model.smooth_w > 0 (default 10 unless a character's config sets it to 0). This is invisible
at rest pose (all transforms near-identity) but causes severe mesh distortion under real
motion if skipped, since a raw diffuse many-joint weight vector blends very differently once
smoothed. Confirmed empirically: MorphGS's own bundled chickenDC (smooth_w=30) and moose1DOG
(smooth_w=10) demo characters both hit this; spot (smooth_w=0, explicitly disabled) did not.

Note: this deliberately does NOT construct a full Rig instance, which loads the mesh via
trimesh -- trimesh's OBJ loader can expand vertex count for per-face-corner UV coordinates
(confirmed on chickenDC: 9933 raw vertices become 59310), which does NOT match
mesh_ori_rig.txt's "skin <vertex_idx>" lines (which index the raw .obj "v" line order, one
entry per vertex, exactly matching build_and_bake_animation.py's own OBJ parser). Using
Rig's resolved vertices/weights together was tried and made things WORSE (a fully shattered
mesh), confirming the skin data is authored against the raw, unexpanded vertex list. So this
script instead calls Rig's underlying (already-static, mesh-instance-independent) Laplacian
math directly against the SAME raw vertex/face parse build_and_bake_animation.py uses,
smoothing the raw per-vertex weight matrix in place rather than rebuilding the mesh itself.

Run inside the MorphGS conda environment (needs `src` on sys.path):
    python resolve_skinning_weights.py <mesh.obj> <mesh_ori_rig.txt> <experiment_config.yaml> \
        <base_config.yaml> <output.npz>
"""
import os
import sys

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import yaml


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = deep_merge(base.get(k), v) if k in base else v
        return merged
    return override if override is not None else base


def parse_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()[1:]
                face = []
                for tok in p:
                    idx = int(tok.split("/")[0])
                    face.append(idx - 1 if idx > 0 else len(verts) + idx)
                if len(face) >= 3:
                    faces.append(face[:3])  # triangulate fan-style if a face has >3 verts
                    for extra in range(3, len(face)):
                        faces.append([face[0], face[extra - 1], face[extra]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def parse_rig_skin(path, joints_name, NV):
    name_to_idx = {n: i for i, n in enumerate(joints_name)}
    NJ = len(joints_name)
    weights = np.zeros((NV, NJ), dtype=np.float64)
    with open(path) as f:
        for line in f:
            if not line.startswith("skin"):
                continue
            tokens = line.split()
            v_idx = int(tokens[1])
            rest = tokens[2:]
            for i in range(0, len(rest), 2):
                jn, w = rest[i], float(rest[i + 1])
                if jn in name_to_idx:
                    weights[v_idx, name_to_idx[jn]] += w
    return weights


def parse_joint_names(rig_path):
    names = []
    with open(rig_path) as f:
        for line in f:
            t = line.split()
            if t and t[0] == "joints":
                names.append(t[1])
    return names


def heat_diffusion_smoothing(verts, faces, weights, lambd):
    """Exact reimplementation of model.RigModel.Rig.heat_diffusion_smoothing, decoupled from
    a Rig instance (that method only ever uses self.vertices/self.faces to build the
    Laplacian, and is otherwise pure math on `weights`)."""
    from model.RigModel import Rig

    L = Rig._build_cotangent_laplacian(verts, faces)
    NV = len(verts)
    diag = np.asarray(L.diagonal()).flatten()
    D_inv = sparse.diags(1.0 / diag.clip(min=1e-8))
    L_rw = D_inv @ L
    A = sparse.eye(NV, format='csr') + lambd * L_rw
    factor = splinalg.factorized(A.tocsc())

    smoothed = np.zeros_like(weights, dtype=np.float64)
    for j in range(weights.shape[1]):
        smoothed[:, j] = factor(weights[:, j].astype(np.float64))

    smoothed = np.clip(smoothed, 0.0, None)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    smoothed /= row_sums
    return smoothed


def main():
    mesh_path, rig_path, exp_config_path, base_config_path, out_path = sys.argv[1:6]

    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f) or {}
    exp_cfg = {}
    if exp_config_path and os.path.isfile(exp_config_path):
        with open(exp_config_path) as f:
            exp_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge(base_cfg, exp_cfg)
    model_cfg = cfg.get("model", {}) or {}

    smooth_w = float(model_cfg.get("smooth_w", 10))
    calculate_skinning_w = bool(model_cfg.get("calculate_skinning_w", False))
    hybrid_skinning_w = bool(model_cfg.get("hybrid_skinning_w", False))
    if calculate_skinning_w or hybrid_skinning_w:
        raise NotImplementedError(
            "This character's config enables calculate_skinning_w/hybrid_skinning_w, which "
            "aren't supported by this export path yet (no characters tested so far use them). "
            "Refusing to silently produce a wrong result -- please report this character."
        )

    print(f"Resolved config: smooth_w={smooth_w}")

    verts, faces = parse_obj(mesh_path)
    joints_name = parse_joint_names(rig_path)
    weights = parse_rig_skin(rig_path, joints_name, len(verts))

    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.clip(row_sums, 1e-8, None)

    if smooth_w > 0:
        weights = heat_diffusion_smoothing(verts, faces, weights, lambd=smooth_w)

    np.savez(out_path, weights=weights.astype(np.float32), joint_names=np.array(joints_name))
    print(f"Resolved skinning weights: {out_path} ({weights.shape[0]} verts, {weights.shape[1]} joints)")


if __name__ == "__main__":
    main()
