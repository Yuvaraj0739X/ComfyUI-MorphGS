# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os

import numpy as np
import trimesh
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.spatial import cKDTree

import torch


def load_rig_txt(filename, NV):
    # read txt file and parse
    joints_name, joints_pos, bones = [], [], []
    skinning_weights = None
    root_idx = 0
    fixed_joint_names = []
    joint_aliases = {}

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()

            if tokens[0] == 'joints':
                if "dup" in tokens[1]:
                    joint_aliases[tokens[1]] = tokens[1].split('dup')[0][:-1]
                    continue
                joints_name.append(tokens[1])
                joints_pos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
            elif len(tokens) == 5 and tokens[0] not in {'skin', 'hier', 'fixed_joint'}:
                # Compact skeleton format: <joint> x y z <parent|-1>. Skinning
                # weights are optional; Rig falls back to heat skinning when
                # none are provided.
                joint_name = tokens[0]
                if "dup" in joint_name:
                    joint_aliases[joint_name] = joint_name.split('dup')[0][:-1]
                    continue
                joints_name.append(joint_name)
                joints_pos.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                parent_name = tokens[4]
                parent_name = joint_aliases.get(parent_name, parent_name)
                if parent_name == "-1":
                    root_idx = len(joints_name) - 1
                else:
                    if 'dup' in parent_name:
                        parent_name = parent_name.split('dup')[0][:-1]
                    parent_id = joints_name.index(parent_name)
                    child_id = len(joints_name) - 1
                    if parent_id != child_id:
                        bones.append([parent_id, child_id])

            elif tokens[0] == 'skin':
                if skinning_weights is None:
                    NJ = len(joints_name)
                    skinning_weights = torch.zeros(NV, NJ, dtype=torch.float32)
                vertex_id = int(tokens[1])
                joint_names = [tokens[i] for i in range(2, len(tokens), 2)]
                for i in range(len(joint_names)):  
                    joint_names[i] = joint_aliases.get(joint_names[i], joint_names[i])
                    if 'dup' in joint_names[i]:
                        joint_names[i] = joint_names[i].split('dup')[0][:-1]
                joint_ids = [joints_name.index(jn) for jn in joint_names]
                weights = [float(tokens[i]) for i in range(3, len(tokens), 2)]
                for joint_id, weight in zip(joint_ids, weights):
                    skinning_weights[vertex_id, joint_id] += weight
            
            elif tokens[0] == 'hier':
                parent_joint = tokens[1]
                child_joint = tokens[2]
                parent_joint = joint_aliases.get(parent_joint, parent_joint)
                child_joint = joint_aliases.get(child_joint, child_joint)
                if 'dup' in parent_joint:
                    parent_joint = parent_joint.split('dup')[0][:-1]
                if 'dup' in child_joint:
                    child_joint = child_joint.split('dup')[0][:-1]
                parent_id = joints_name.index(parent_joint)
                child_id = joints_name.index(child_joint)
                if parent_id == child_id:
                    continue
                bones.append([parent_id, child_id])
                
            elif tokens[0] == 'root':
                root_joint = tokens[1]
                for jn in joints_name:
                    if jn == root_joint:
                        root_idx = joints_name.index(jn)
                        break

            elif tokens[0] == 'fixed_joint':
                fixed_joint_names.extend(tokens[1:])

    joints_pos = torch.tensor(joints_pos, dtype=torch.float32)

    if root_idx != 0:
        NJ = len(joints_name)

        # Re-root the directed hierarchy. Reindexing joints alone is insufficient;
        # edges must also be reoriented so descendants inherit parent motion.
        adjacency = [[] for _ in range(NJ)]
        for p, c in bones:
            adjacency[p].append(c)
            adjacency[c].append(p)

        rerooted_bones = []
        visited = [False] * NJ
        queue = [root_idx]
        visited[root_idx] = True
        head = 0
        while head < len(queue):
            parent = queue[head]
            head += 1
            for child in adjacency[parent]:
                if visited[child]:
                    continue
                visited[child] = True
                rerooted_bones.append([parent, child])
                queue.append(child)

        if len(rerooted_bones) != len(bones):
            raise RuntimeError(
                f"Failed to re-root rig hierarchy for {filename}: "
                f"expected {len(bones)} edges, got {len(rerooted_bones)}."
            )

        new_order = [root_idx] + [i for i in range(NJ) if i != root_idx]
        index_map = {old: new for new, old in enumerate(new_order)}

        joints_name = [joints_name[i] for i in new_order]
        joints_pos = joints_pos[new_order]

        if skinning_weights is not None:
            skinning_weights = skinning_weights[:, new_order]

        bones = [[index_map[p], index_map[c]] for p, c in rerooted_bones]

        root_idx = 0

    fixed_joint_indices = []
    for name in fixed_joint_names:
        if 'dup' in name:
            name = name.split('dup')[0][:-1]
        if name not in joints_name:
            raise ValueError(f"Unknown fixed_joint '{name}' in {filename}")
        fixed_joint_indices.append(joints_name.index(name))

    return joints_name, joints_pos, bones, skinning_weights, root_idx, sorted(set(fixed_joint_indices))


def save_rig_txt(filename, joints_name, joints_pos, bones, skinning_weights, root_idx):
    
    if isinstance(joints_pos, torch.Tensor):
        joints_pos = joints_pos.detach().cpu().numpy().tolist()
    
    if skinning_weights is not None and isinstance(skinning_weights, torch.Tensor):
        skinning_weights = skinning_weights.detach().cpu().numpy()

    with open(filename, 'w') as f:
        # 1. Write joints
        # Format: joints [name] [x] [y] [z]
        for name, pos in zip(joints_name, joints_pos):
            f.write(f"joints {name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        f.write("\n") # Section separator for readability

        # 2. Write root
        # Format: root [name]
        if 0 <= root_idx < len(joints_name):
            root_name = joints_name[root_idx]
            f.write(f"root {root_name}\n")
        f.write("\n")

        # 3. Write hierarchy (bones)
        # Format: hier [parent_name] [child_name]
        for parent_idx, child_idx in bones:
            parent_name = joints_name[parent_idx]
            child_name = joints_name[child_idx]
            f.write(f"hier {parent_name} {child_name}\n")
        f.write("\n")

        # 4. Write skinning weights
        # Format: skin [vertex_id] [joint_name] [weight] [joint_name] [weight] ...
        if skinning_weights is not None:
            num_verts, num_joints = skinning_weights.shape
            
            for v_idx in range(num_verts):
                # Get weights for this vertex
                weights = skinning_weights[v_idx]
                
                # Find indices where weight is effectively non-zero
                # (Optional: threshold to remove negligible weights like 1e-8)
                active_indices = [i for i, w in enumerate(weights) if w > 1e-6]
                
                if not active_indices:
                    continue
                
                line_parts = [f"skin {v_idx}"]
                for j_idx in active_indices:
                    j_name = joints_name[j_idx]
                    w = weights[j_idx]
                    line_parts.append(f"{j_name} {w:.6f}")
                
                f.write(" ".join(line_parts) + "\n")

    print(f"Successfully saved rig data to {filename}")


class Rig:
    def __init__(
        self,
        mesh,
        rig_filename,
        calculate_skinning_w=False,
        hybrid_skinning_w=False,
        hybrid_skinning_calculated_joints=None,
        smooth_w=0,
        device='cuda',
        skinning_method='lbs',
        fk_pivot_mode='parent',
    ):
        """
            Contains the rigging information of a mesh.
            Args:
                mesh: trimesh object
                rig_filename (str): path to the rigging file
                calculate_skinning_w (bool): whether to calculate skinning weights from vertices
                hybrid_skinning_w (bool): keep rig weights except vertices whose calculated
                    dominant joint is listed in hybrid_skinning_calculated_joints
                device (str): device to use
            Members:
                vertices (torch.tensor): NV x 3, vertices of the mesh
                joints_name (list): list of joint names
                joints_pos (torch.tensor): NJ x 3, joint positions
                bones (list): list of bone indices
                skinning_weights (torch.tensor): NV x NJ, skinning weights
        """
        if not isinstance(mesh, trimesh.Trimesh):
            self.vertices = mesh
        else:
            self.mesh = mesh
            self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
            self.faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
            self.eps = torch.tensor(1e-6)
        self.skinning_method = str(skinning_method).lower()
        self.fk_pivot_mode = str(fk_pivot_mode).lower()
        if self.skinning_method not in ("lbs", "dqs"):
            print(f"[Warning] Unknown skinning_method={self.skinning_method}. Falling back to 'lbs'.")
            self.skinning_method = "lbs"
        if self.fk_pivot_mode not in ("parent", "joint"):
            print(f"[Warning] Unknown fk_pivot_mode={self.fk_pivot_mode}. Falling back to 'parent'.")
            self.fk_pivot_mode = "parent"

        skinning_weights = None
        self.root_idx = 0
        self.fixed_joint_indices = []
        try:
            if not os.path.exists(rig_filename):
                raise FileNotFoundError(f"Rig file does not exist: {rig_filename}")
            if rig_filename.endswith('.txt'):
                joints_name, joints_pos, bones, skinning_weights, root_idx, fixed_joint_indices = load_rig_txt(rig_filename, len(self.vertices))
                self.root_idx = root_idx
                self.joints_name = joints_name
                self.joints_pos = joints_pos.to(device)
                self.bones = bones
                self.fixed_joint_indices = fixed_joint_indices
                if skinning_weights is not None:
                    self.skinning_weights = skinning_weights.to(device)
                    
            elif rig_filename.endswith('.npz'):
                data = np.load(rig_filename)
                self.joints_pos = torch.from_numpy(data['nodes']).to(device)
                parents = torch.from_numpy(data['parents'])
                self.skinning_weights = torch.from_numpy(data['skinning_weight']).to(device)
                bones = []
                for i, p in enumerate(parents):
                    if p != -1:
                        bones.append((p.item(), i))
                self.bones = bones
                self.joints_name = np.arange(len(self.joints_pos))
            else:
                raise ValueError('Unsupported rig file format')
        except Exception as e:
            raise RuntimeError(f"Failed to load rig file '{rig_filename}': {e}") from e

        self.parents = [-1] * len(self.joints_name)
        self.children = [[] for _ in range(len(self.joints_name))]
        for bone in self.bones:
            parent, child = bone
            self.parents[child] = parent
            self.children[parent].append(child)

        parent_joints, child_joints = zip(*self.bones)
        self.joints_connection = torch.tensor([parent_joints, child_joints], dtype=torch.long, device=device)

        parent_joint_dict = {b[1]: b[0] for b in self.bones}
        child_joints_dict = {k: [] for k in range(len(self.joints_pos))}
        for k in parent_joint_dict.keys():
            parent_k = parent_joint_dict[k]
            child_joints_dict[parent_k].append(k)
        parent_indices = [[self.root_idx]]
        for i in range(len(self.bones)):
            j = i + 1
            inds = []
            while j >= 0:
                inds += [j]
                j = parent_joint_dict.get(j, -1)
            parent_indices += [inds[::-1]]
        max_depth = np.max([len(x) for x in parent_indices])
        self.parent_indices = torch.zeros((len(parent_indices), max_depth), dtype=torch.long, device=device) - 1
        for i,inds in enumerate(parent_indices):
            self.parent_indices[i,:len(inds)] = torch.from_numpy(np.array(inds)).to(device, dtype=self.parent_indices.dtype)
        self.parent_joint_ex = torch.from_numpy(np.array([parent_joint_dict.get(i, 0) for i in range(len(parent_indices))])).to(device, dtype=self.parent_indices.dtype)

        self.joints_depth = (self.parent_indices != -1).sum(dim=1)

        if skinning_weights is not None:
            self.skinning_weights = self._fill_missing_skinning_weights(self.skinning_weights)

        original_skinning_weights = self.skinning_weights.clone() if skinning_weights is not None else None
        calculated_skinning_weights = None
        need_calculated_skinning = calculate_skinning_w or hybrid_skinning_w or original_skinning_weights is None
        if need_calculated_skinning:
            calculated_skinning_weights = self.calc_skinning_weights_heat(initial_heat=1.0)

        if hybrid_skinning_w and original_skinning_weights is not None:
            if smooth_w:
                lambd = float(smooth_w) if smooth_w > 1 else 10.0
                calculated_np = self.heat_diffusion_smoothing(calculated_skinning_weights.cpu().numpy(), lambd=lambd)
                calculated_skinning_weights = torch.tensor(calculated_np, dtype=torch.float32, device=device)
            self.skinning_weights = self._mix_skinning_weights(
                original_skinning_weights,
                calculated_skinning_weights,
                hybrid_skinning_calculated_joints,
            )
        elif calculated_skinning_weights is not None:
            self.skinning_weights = calculated_skinning_weights
            if smooth_w:
                lambd = float(smooth_w) if smooth_w > 1 else 10.0
                skinning_weights_np = self.heat_diffusion_smoothing(self.skinning_weights.cpu().numpy(), lambd=lambd)
                self.skinning_weights = torch.tensor(skinning_weights_np, dtype=torch.float32, device=device)
        elif smooth_w:
            lambd = float(smooth_w) if smooth_w > 1 else 10.0
            skinning_weights_np = self.heat_diffusion_smoothing(self.skinning_weights.cpu().numpy(), lambd=lambd)
            self.skinning_weights = torch.tensor(skinning_weights_np, dtype=torch.float32, device=device)

    def _resolve_joint_ids(self, joint_specs):
        if joint_specs is None:
            return []
        joint_ids = []
        for joint in joint_specs:
            if isinstance(joint, str) and not joint.isdigit():
                if joint not in self.joints_name:
                    raise ValueError(f"Unknown joint name in hybrid_skinning_calculated_joints: {joint}")
                joint_ids.append(self.joints_name.index(joint))
            else:
                joint_id = int(joint)
                if joint_id < 0 or joint_id >= len(self.joints_name):
                    raise ValueError(f"Joint index out of range in hybrid_skinning_calculated_joints: {joint_id}")
                joint_ids.append(joint_id)
        return sorted(set(joint_ids))

    def _fill_missing_skinning_weights(self, skinning_weights):
        row_sums = skinning_weights.sum(dim=1, keepdim=True)
        valid = row_sums.squeeze(1) > 1e-8
        missing = ~valid
        if not bool(missing.any()):
            return skinning_weights / row_sums.clamp_min(1e-8)
        if not bool(valid.any()):
            return skinning_weights

        vertices_np = self.vertices.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        missing_np = missing.detach().cpu().numpy()
        tree = cKDTree(vertices_np[valid_np])
        _, nearest_valid = tree.query(vertices_np[missing_np], k=1)
        valid_indices = np.flatnonzero(valid_np)

        filled_weights = skinning_weights.clone()
        filled_weights[missing] = skinning_weights[torch.tensor(valid_indices[nearest_valid], device=skinning_weights.device)]
        filled_weights = filled_weights / filled_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        print(
            f"[Rig] Filled {int(missing.sum().item())}/{len(missing)} missing skinning rows "
            "from nearest weighted vertices."
        )
        return filled_weights

    def _mix_skinning_weights(self, original_weights, calculated_weights, calculated_joints):
        calculated_joint_ids = self._resolve_joint_ids(calculated_joints)
        if not calculated_joint_ids:
            print("[Rig] Hybrid skinning enabled with no calculated joints; using original skinning weights.")
            return original_weights

        dominant_calculated_joints = torch.argmax(calculated_weights, dim=1)
        calculated_joint_ids_tensor = torch.tensor(calculated_joint_ids, device=calculated_weights.device)
        use_calculated = (dominant_calculated_joints[:, None] == calculated_joint_ids_tensor[None, :]).any(dim=1)
        original_valid = original_weights.sum(dim=1) > 1e-8
        use_calculated = use_calculated | ~original_valid

        mixed_weights = torch.where(use_calculated[:, None], calculated_weights, original_weights)
        row_sums = mixed_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mixed_weights = mixed_weights / row_sums
        joint_names = [self.joints_name[i] for i in calculated_joint_ids]
        print(
            f"[Rig] Hybrid skinning: using calculated weights for "
            f"{int(use_calculated.sum().item())}/{len(use_calculated)} vertices "
            f"with dominant joints {joint_names}."
        )
        return mixed_weights

    @staticmethod
    def _build_cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
        NV = len(vertices)
        row, col, data = [], [], []
        for f in faces:
            for local_i in range(3):
                vi = f[local_i]
                vj = f[(local_i + 1) % 3]
                vk = f[(local_i + 2) % 3]
                a = vertices[vj] - vertices[vi]
                b = vertices[vk] - vertices[vi]
                cross_len = np.linalg.norm(np.cross(a, b))
                dot = np.dot(a, b)
                cot = dot / (cross_len + 1e-6)
                row.append(vj); col.append(vk); data.append(-0.5 * cot)
                row.append(vk); col.append(vj); data.append(-0.5 * cot)

        L = sparse.coo_matrix((data, (row, col)), shape=(NV, NV)).tocsr()
        diag_vals = -np.asarray(L.sum(axis=1)).flatten()
        L = L + sparse.diags(diag_vals)
        return L.tocsr()

    @staticmethod
    def _build_vertex_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        NV = len(vertices)
        area_sum = np.zeros(NV, dtype=np.float64)
        for f in faces:
            vi, vj, vk = f
            e1 = vertices[vj] - vertices[vi]
            e2 = vertices[vk] - vertices[vi]
            cross_len = np.linalg.norm(np.cross(e1, e2))
            area_sum[vi] += cross_len
            area_sum[vj] += cross_len
            area_sum[vk] += cross_len
        return 1.0 / (1e-10 + area_sum)

    @staticmethod
    def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray):
        ab = b - a
        l2 = np.dot(ab, ab)
        if l2 < 1e-12:
            return np.linalg.norm(p - a, axis=1), np.repeat(a[None], len(p), axis=0)
        t = np.clip(((p - a) @ ab) / l2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.linalg.norm(p - proj, axis=1), proj

    def _compute_bone_dist_and_vis(self, vertices: np.ndarray, joints: np.ndarray, bones: list):
        NV = len(vertices)
        NB = len(bones)
        bone_dist = np.full((NV, NB), np.inf, dtype=np.float64)
        for j, (pi, ci) in enumerate(bones):
            a = joints[pi]
            b = joints[ci]
            dists, _ = self._point_to_segment_dist(vertices, a, b)
            bone_dist[:, j] = dists

        closest = np.argmin(bone_dist, axis=1)
        min_dist = bone_dist[np.arange(NV), closest]
        bone_vis = np.ones((NV, NB), dtype=bool)
        return bone_dist, bone_vis, closest, min_dist

    def calc_skinning_weights_heat(self, initial_heat: float = 1.0) -> torch.Tensor:
        device = self.vertices.device
        verts_np = self.vertices.detach().cpu().numpy().astype(np.float64)
        faces_np = self.faces.detach().cpu().numpy()
        joints_np = self.joints_pos.detach().cpu().numpy().astype(np.float64)
        bones = self.bones

        NV = len(verts_np)
        NB = len(bones)

        L = self._build_cotangent_laplacian(verts_np, faces_np)
        D = self._build_vertex_area(verts_np, faces_np)
        bone_dist, bone_vis, closest, min_dist = self._compute_bone_dist_and_vis(verts_np, joints_np, bones)

        closest_vis = bone_vis[np.arange(NV), closest]
        H = np.zeros(NV, dtype=np.float64)
        H[closest_vis] = initial_heat / (1e-8 + min_dist[closest_vis]) ** 2

        A = -L + sparse.diags(H * D)
        try:
            from sksparse.cholmod import cholesky
            factor = cholesky(A.tocsc())
            solve_fn = lambda rhs: factor(rhs)
        except ImportError:
            factor = splinalg.factorized(A.tocsc())
            solve_fn = factor

        raw_weights = np.zeros((NV, NB), dtype=np.float64)
        for j in range(NB):
            rhs = np.zeros(NV, dtype=np.float64)
            vis_and_close = bone_vis[:, j] & (bone_dist[:, j] <= min_dist * 1.00001)
            rhs[vis_and_close] = (H * D)[vis_and_close]
            w = solve_fn(rhs)
            np.clip(w, 0.0, 1.0, out=w)
            raw_weights[:, j] = w

        row_sums = raw_weights.sum(axis=1, keepdims=True)
        zero_rows = row_sums.flatten() < 1e-10
        if zero_rows.any():
            raw_weights[zero_rows, closest[zero_rows]] = 1.0
            row_sums[zero_rows] = 1.0
        raw_weights /= row_sums

        # Map bone-wise heat weights to joint-wise skinning weights.
        # Each bone influences its child joint transform, not "bone index + 1".
        weights_full = np.zeros((NV, len(self.joints_pos)), dtype=np.float64)
        for bone_idx, (_, child_idx) in enumerate(bones):
            weights_full[:, child_idx] = raw_weights[:, bone_idx]
        return torch.tensor(weights_full, dtype=torch.float32, device=device)

    def heat_diffusion_smoothing(self, weights: np.ndarray, lambd: float = 1.0) -> np.ndarray:
        verts_np = self.vertices.detach().cpu().numpy().astype(np.float64)
        faces_np = self.faces.detach().cpu().numpy()
        NV = len(verts_np)

        L = self._build_cotangent_laplacian(verts_np, faces_np)
        diag = np.asarray(L.diagonal()).flatten()
        D_inv = sparse.diags(1.0 / diag.clip(min=1e-8))
        L_rw = D_inv @ L
        A = sparse.eye(NV, format='csr') + lambd * L_rw

        # A = I + lambd * D_inv @ L is asymmetric by construction (D_inv @ L != (D_inv @ L)^T
        # in general, since D_inv is diagonal and only symmetric matrix multiplication commutes
        # with diagonal scaling on both sides, not one). sksparse.cholmod.cholesky requires a
        # symmetric positive-definite matrix; on irregular meshes (e.g. AI-generated geometry
        # with widely varying triangle sizes -- confirmed here via diag(L) spanning ~190x) this
        # asymmetric operator is not actually positive-definite even after symmetrizing, so
        # Cholesky can fail with CholmodNotPositiveDefiniteError depending on mesh geometry.
        # The system is still non-singular (confirmed via residual-checked LU solve), so use a
        # general sparse LU factorization unconditionally instead of Cholesky for this matrix.
        factor = splinalg.factorized(A.tocsc())
        solve_fn = factor

        smoothed = np.zeros_like(weights, dtype=np.float64)
        for j in range(weights.shape[1]):
            smoothed[:, j] = solve_fn(weights[:, j].astype(np.float64))

        smoothed = np.clip(smoothed, 0.0, None)
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
        smoothed /= row_sums
        return smoothed.astype(np.float32)

    
    def dist_batch(self, p, a, b):
        assert len(a) == len(b), "Same batch size needed for a and b"

        p = p[None, :, :]
        s = b - a
        w = p - a[:, None, :]
        ps = (w * s[:, None, :]).sum(-1)
        res = torch.zeros((a.shape[0], p.shape[1]), dtype=p.dtype, device=p.device)

        ps_smaller_mask = ps <= 0
        lower_mask = torch.where(ps_smaller_mask)
        res[lower_mask] += torch.norm(w[lower_mask], dim=-1)

        l2 = (s * s).sum(-1)
        ps_mask = ~ps_smaller_mask

        temp_mask_l2 = ps >= l2[:, None]
        upper_mask = torch.where(ps_mask & temp_mask_l2)
        res[upper_mask] += torch.norm(p[0][upper_mask[1]] - b[upper_mask[0]], dim=-1)

        within_mask = torch.where(ps_mask & ~temp_mask_l2)
        res[within_mask] += torch.norm(
            p[0][within_mask[1]] - (a[within_mask[0]] + (ps[within_mask] / l2[within_mask[0]]).unsqueeze(-1) * s[within_mask[0]]), dim=-1)

        return res

    def _weights_from_bones(self, pcd, joints, bones, add_noise=False, noise_var=0, val=1, add_zero_weight=False):
        joints = torch.tensor(joints)
        bone_distances = self.dist_batch(
            pcd,
            torch.cat([joints[bone[0]].unsqueeze(0) for bone in bones], dim=0),
            torch.cat([joints[bone[1]].unsqueeze(0) for bone in bones], dim=0)
            )
        bone_argmin = torch.argmin(bone_distances, axis=0)
        weights = torch.zeros((len(bone_argmin), len(bones)))
        weights[torch.arange(len(bone_argmin)), bone_argmin] = val
    
        if add_zero_weight:
            weights = torch.cat([torch.zeros((len(weights), 1), device=weights.device), weights], dim=-1)
        
        if add_noise:
            weights = weights + torch.randn_like(weights, device=weights.device) * noise_var

        return weights.to(pcd.device)
    
    
    def get_lbs_parts(self, sampled_indices=None):
        skinning_weights = self.skinning_weights if sampled_indices is None else self.skinning_weights[sampled_indices]
        dominant_joint_indices = torch.argmax(skinning_weights, dim=1)
        lbs_parts = []
        for i in range(len(self.joints_name)):
            lbs_parts.append(torch.where(dominant_joint_indices == i)[0])

        return lbs_parts, dominant_joint_indices
    
