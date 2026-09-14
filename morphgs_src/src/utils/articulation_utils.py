# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import torch


def Rodrigues(rvec, eps=1e-8):
    theta = torch.linalg.norm(rvec, dim=-1)
    axis = rvec / theta.clamp_min(eps).unsqueeze(-1)

    costh = torch.cos(theta)
    sinth = torch.sin(theta)
    one_minus_costh = 2.0 * torch.sin(theta * 0.5) ** 2

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = torch.stack((
        x * x + (1 - x * x) * costh,
        x * y * one_minus_costh - z * sinth,
        x * z * one_minus_costh + y * sinth,
        x * y * one_minus_costh + z * sinth,
        y * y + (1 - y * y) * costh,
        y * z * one_minus_costh - x * sinth,
        x * z * one_minus_costh - y * sinth,
        y * z * one_minus_costh + x * sinth,
        z * z + (1 - z * z) * costh,
    ), dim=1).view(-1, 3, 3)

    return R, theta


def matrix_chain_product(matrix_chain: torch.Tensor) -> torch.Tensor:
    chain_len = matrix_chain.shape[1]
    if chain_len == 1:
        return matrix_chain
    sub_a = matrix_chain_product(matrix_chain[:, :chain_len // 2])
    sub_b = matrix_chain_product(matrix_chain[:, chain_len // 2:])
    return sub_a @ sub_b


def calc_rec_abs_T_fast(
    R_t: torch.Tensor,
    joints: torch.Tensor,
    parent_joint_ex,
    parent_indices,
    pivot_mode: str = "parent",
) -> torch.Tensor:
    if pivot_mode == "joint":
        joints_p = joints
    elif pivot_mode == "parent":
        joints_p = joints[parent_joint_ex]
    else:
        raise ValueError(f"Unsupported pivot_mode: {pivot_mode}")

    T = joints_p[..., None] + R_t @ -joints_p[..., None]
    M = torch.cat((R_t, T), -1)
    hom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=R_t.device)
    hom_rows = hom_row[None, None].repeat(R_t.shape[0], 1, 1)

    M_bones = torch.cat((M, hom_rows), -2)
    M_bones = torch.cat((torch.eye(4, device=R_t.device)[None], M_bones), 0)
    M_paths = M_bones[parent_indices + 1]
    return matrix_chain_product(M_paths)[:, 0]


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def calc_skinning_weights(
    joints_pos,
    skinning_weights,
    theta_weight=0.1,
    merging_mat=None,
    eps=1e-6,
    device="cuda",
    apply_softmax=False,
):
    if not apply_softmax:
        return skinning_weights

    if not isinstance(theta_weight, torch.Tensor):
        theta_weight = torch.tensor([theta_weight], device=device)

    theta_weight = torch.max(torch.tensor([eps], device=device), theta_weight).to(device)
    weights = torch.softmax(skinning_weights / theta_weight, dim=-1).permute(1, 0)

    if merging_mat is None:
        joint_count = len(weights)
        merging_mat = torch.zeros(joint_count, joint_count, joint_count, device=device)
        flat_merging_rules = torch.arange(0, len(joints_pos), device=device)
        for i in range(weights.shape[0]):
            mask = flat_merging_rules == i
            merging_mat[i] = torch.eye(joint_count, device=device) * mask

    merged_weights = torch.bmm(
        merging_mat,
        weights.unsqueeze(0).repeat(len(weights), 1, 1),
    ).sum(1)
    return merged_weights.permute(1, 0)
