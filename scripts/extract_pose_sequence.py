"""
Extract the per-frame skeletal pose MorphGS learned for a trained experiment, as absolute
per-joint world-space transforms (in MorphGS's rig-space convention -- the same convention
mesh_ori_rig.txt's joint positions use).

This runs the SAME code MorphGS's own training loop uses for animating the skeleton
(model/AnimationField.py's SimpleDeformNet + articulation_utils.calc_rec_abs_T_fast), just
run in inference-only mode against a saved deform-network checkpoint, once per frame,
instead of once per training iteration. Reusing MorphGS's own classes for this (rather than
re-deriving the FK math independently) removes any risk of subtly getting the rotation
composition wrong.

Run inside the MorphGS conda environment (needs `src` on sys.path):
    python extract_pose_sequence.py <mesh_ori_rig.txt> <deform_checkpoint.pth> <num_frames> <output.npz>
"""
import json
import sys
import os

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


def parse_rig(rig_path):
    """Minimal joints/root/hier parser -- ignores skin lines entirely, since this script
    only needs joint rest positions and hierarchy, not per-vertex skin weights.

    Handles two rig-file conventions seen in the wild: mesh_to_morphgs.py's converter emits
    an explicit "root <name>" directive line, while MorphGS's own bundled rig files instead
    mark the root via a "hier <root> <root>" self-loop and have no "root" line at all. Both
    are normalized here to a plain root_idx, with the self-loop entry dropped from the real
    parent/child edge list (it isn't one, and its presence would throw off the sequential
    child-index assumption build_parent_tables relies on).
    """
    joints_name = []
    joints_pos = []
    bones = []  # (parent_name, child_name), real edges only -- self-loop root markers dropped
    root_name = None

    with open(rig_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if tokens[0] == "joints":
                joints_name.append(tokens[1])
                joints_pos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
            elif tokens[0] == "root":
                root_name = tokens[1]
            elif tokens[0] == "hier":
                parent, child = tokens[1], tokens[2]
                if parent == child:
                    if root_name is None:
                        root_name = parent
                    continue
                bones.append((parent, child))

    if root_name is None:
        root_name = joints_name[0]  # convention: the root joint is always declared first

    name_to_idx = {n: i for i, n in enumerate(joints_name)}
    root_idx = name_to_idx[root_name]
    bones_idx = [(name_to_idx[p], name_to_idx[c]) for p, c in bones]
    joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
    return joints_name, joints_pos, bones_idx, root_idx


def build_parent_tables(joints_pos, bones_idx, root_idx):
    """Exact replica of RigModel.py's parent_indices/parent_joint_ex construction (the
    ancestor-chain table calc_rec_abs_T_fast's matrix_chain_product needs), so the FK
    composition here matches training bit-for-bit."""
    NJ = len(joints_pos)
    parent_joint_dict = {c: p for p, c in bones_idx}

    parent_indices_lists = [[root_idx]]
    for i in range(len(bones_idx)):
        j = i + 1
        inds = []
        while j >= 0:
            inds.append(j)
            j = parent_joint_dict.get(j, -1)
        parent_indices_lists.append(inds[::-1])

    max_depth = max(len(x) for x in parent_indices_lists)
    parent_indices = torch.zeros((len(parent_indices_lists), max_depth), dtype=torch.long) - 1
    for i, inds in enumerate(parent_indices_lists):
        parent_indices[i, :len(inds)] = torch.tensor(inds, dtype=torch.long)

    parent_joint_ex = torch.tensor(
        [parent_joint_dict.get(i, 0) for i in range(NJ)], dtype=torch.long
    )
    return parent_indices, parent_joint_ex


def main():
    rig_path, ckpt_path, num_frames, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

    from utils.articulation_utils import calc_rec_abs_T_fast, Rodrigues
    from model.AnimationField import get_embedder, SimpleDeformNet

    joints_name, joints_pos, bones_idx, root_idx = parse_rig(rig_path)
    NJ = len(joints_name)
    parent_indices, parent_joint_ex = build_parent_tables(joints_pos, bones_idx, root_idx)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["deform_net_state_dict"]
    assert state_dict["joint_rotation.weight"].shape[0] == NJ * 3, (
        f"Checkpoint has {state_dict['joint_rotation.weight'].shape[0] // 3} joints, "
        f"but rig file has {NJ}. Wrong rig/checkpoint pairing."
    )

    embed_time_fn, time_input_ch = get_embedder(6, 1)
    deform_net = SimpleDeformNet(NJ, time_input_ch, device="cpu")  # tiny MLP, CPU is plenty and avoids a CUDA dependency here
    deform_net.load_state_dict(state_dict)
    deform_net.eval()

    bone_Ts_all = np.zeros((num_frames, NJ, 4, 4), dtype=np.float32)
    with torch.no_grad():
        for f in range(num_frames):
            t = torch.tensor([f / num_frames], dtype=torch.float32)
            t_embed = embed_time_fn(t.unsqueeze(0))
            global_t, rot_params = deform_net(t_embed)
            global_t = global_t[0]
            rot_params = rot_params[0]  # (NJ, 3) axis-angle

            R_t, _ = Rodrigues(rot_params)
            bone_Ts = calc_rec_abs_T_fast(R_t, joints_pos, parent_joint_ex, parent_indices)  # (NJ,4,4)

            # calc_rec_abs_T_fast composes rotation-about-parent-pivot only; MorphGS's training
            # loop adds the separately-predicted global root translation on top (see main.py's
            # joints_warped = joints_warped_rel + global_t) -- replicate that here too, so the
            # exported pose matches what was actually rendered, not just the rotation component.
            bone_Ts_frame = bone_Ts.clone()
            bone_Ts_frame[:, :3, 3] += global_t

            bone_Ts_all[f] = bone_Ts_frame.numpy()

    np.savez(out_path, bone_Ts=bone_Ts_all, joint_names=np.array(joints_name), num_frames=num_frames)
    print(f"Exported pose sequence: {out_path} ({num_frames} frames, {NJ} joints)")


if __name__ == "__main__":
    main()
