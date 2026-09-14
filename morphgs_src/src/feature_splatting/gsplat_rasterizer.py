"""
Drop-in replacement for Inria/GRAPHDECO's `diff_gaussian_rasterization` (non-commercial
research license), backed by gsplat (Apache-2.0, nerfstudio-project/gsplat), so MorphGS's
Gaussian-splatting rendering core no longer depends on non-commercially-licensed code.

Exposes the same `GaussianRasterizationSettings` / `GaussianRasterizer` class-based API
that src/feature_splatting/gaussian_renderer/__init__.py already calls (5-output "latent"
convention: rendered_image, rendered_feat, alpha, rendered_depth, radii), so the call site
does not need to change -- only the import.

Verified against MorphGS's actual usage:
  - compute_cov3D_python and convert_SHs_python both default to False, so the exercised
    path is always scales+rotations+SH (cov3D_precomp is explicitly unsupported here).
  - MorphGS's Camera class stores world_view_transform/full_proj_transform *transposed*
    (row-vector convention, matching the original 3DGS reference implementation) --
    un-transposing recovers the standard column-vector world-to-camera matrix gsplat expects.
  - gsplat defaults to packed=True, which drops culled Gaussians instead of zero-padding
    them; packed=False is forced everywhere so `radii` stays densely aligned with the
    input Gaussian ordering, matching what visibility_filter = radii > 0 requires.
"""
import torch
import gsplat


class GaussianRasterizationSettings:
    def __init__(self, image_height, image_width, tanfovx, tanfovy, bg, scale_modifier,
                 viewmatrix, projmatrix, sh_degree, campos, prefiltered, debug):
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg = bg
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.campos = campos
        self.prefiltered = prefiltered
        self.debug = debug


class GaussianRasterizer(torch.nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def _camera_matrices(self):
        rs = self.raster_settings
        device = rs.viewmatrix.device
        dtype = rs.viewmatrix.dtype
        # Undo MorphGS/Inria's stored transpose to recover the standard column-vector
        # world-to-camera matrix gsplat's `viewmats` expects.
        viewmat = rs.viewmatrix.transpose(0, 1).contiguous().unsqueeze(0)  # (1,4,4)
        fx = rs.image_width / (2.0 * rs.tanfovx)
        fy = rs.image_height / (2.0 * rs.tanfovy)
        cx = rs.image_width / 2.0
        cy = rs.image_height / 2.0
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        ).unsqueeze(0)  # (1,3,3)
        return viewmat, K

    @staticmethod
    def _wire_means2D_grad(means2D, means2d_gs):
        """
        MorphGS's densification heuristic (add_densification_stats) reads
        viewspace_point_tensor.grad, expecting the caller-supplied `means2D` placeholder to
        have been populated as a side effect of the rasterizer's backward pass -- this is how
        Inria's original CUDA rasterizer works (means2D is threaded through the forward
        computation as an additive zero-perturbation so its gradient equals d(Loss)/d(2D proj)).
        gsplat computes its own internal projected-means tensor (meta["means2d"]) instead and
        never touches the caller's `means2D`, so `.grad` would otherwise stay None forever.
        Replicate the same externally-visible behavior via a backward hook that copies the
        gradient flowing through gsplat's internal means2d onto the external placeholder.
        """
        if not means2D.requires_grad or not means2d_gs.requires_grad:
            return

        def _hook(grad):
            # Hook must be registered on the tensor that actually appears in gsplat's live
            # computation graph (means2d_gs itself) -- registering it on a slice/index of that
            # tensor creates a dead-end node with no downstream consumers, which never
            # receives a gradient at all.
            g_flat = grad[0] if grad.dim() == 3 else grad  # (N,2)
            with torch.no_grad():
                if means2D.grad is None:
                    means2D.grad = torch.zeros_like(means2D)
                means2D.grad[:, :2] += g_flat.to(means2D.dtype)
            return None

        means2d_gs.register_hook(_hook)

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, features=None,
                scales=None, rotations=None, cov3D_precomp=None):
        rs = self.raster_settings

        if cov3D_precomp is not None:
            raise NotImplementedError(
                "gsplat_rasterizer shim: cov3D_precomp path is not implemented "
                "(MorphGS's default pipe.compute_cov3D_python=False never exercises it). "
                "Set pipe.compute_cov3D_python=False to use scales+rotations instead."
            )
        if scales is None or rotations is None:
            raise ValueError("gsplat_rasterizer shim requires both `scales` and `rotations`.")

        viewmat, K = self._camera_matrices()
        opac = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
        quats = rotations

        if colors_precomp is not None:
            rgb_colors = colors_precomp.unsqueeze(0) if colors_precomp.dim() == 2 else colors_precomp
            sh_degree_arg = None
        else:
            rgb_colors = shs.unsqueeze(0) if shs.dim() == 3 else shs  # (1, N, K, 3)
            sh_degree_arg = rs.sh_degree

        render_rgb, render_alpha, meta = gsplat.rasterization(
            means=means3D, quats=quats, scales=scales, opacities=opac,
            colors=rgb_colors, viewmats=viewmat, Ks=K,
            width=int(rs.image_width), height=int(rs.image_height),
            sh_degree=sh_degree_arg,
            render_mode="RGB",
            packed=False,
        )
        self._wire_means2D_grad(means2D, meta["means2d"])

        # Manual alpha-composite over the requested background, rather than relying on
        # gsplat's own `backgrounds` kwarg (its expected shape differs across render_modes
        # in ways not worth depending on for a single flat RGB background color).
        bg = rs.bg if rs.bg is not None else torch.zeros(3, device=render_rgb.device, dtype=render_rgb.dtype)
        composited = render_rgb[0] + (1.0 - render_alpha[0]) * bg
        rendered_image = composited.permute(2, 0, 1)  # (3,H,W)
        alpha = render_alpha[0].permute(2, 0, 1)  # (1,H,W)

        radii_full = meta["radii"]  # (1, N, 2) -- per-axis screen-space radius, dense (packed=False)
        radii = radii_full[0].amax(dim=-1)  # (N,) matching Inria's single-scalar-per-Gaussian convention

        rendered_depth = None
        render_depth, _, _ = gsplat.rasterization(
            means=means3D, quats=quats, scales=scales, opacities=opac,
            colors=rgb_colors, viewmats=viewmat, Ks=K,
            width=int(rs.image_width), height=int(rs.image_height),
            sh_degree=sh_degree_arg,
            render_mode="ED",
            packed=False,
        )
        rendered_depth = render_depth[0].permute(2, 0, 1)  # (1,H,W)

        rendered_feat = None
        if features is not None:
            feat_in = features.unsqueeze(0) if features.dim() == 2 else features  # (1, N, D)
            render_feat, _, feat_meta = gsplat.rasterization(
                means=means3D, quats=quats, scales=scales, opacities=opac,
                colors=feat_in, viewmats=viewmat, Ks=K,
                width=int(rs.image_width), height=int(rs.image_height),
                render_mode="RGB",
                packed=False,
            )
            self._wire_means2D_grad(means2D, feat_meta["means2d"])
            rendered_feat = render_feat[0].permute(2, 0, 1)  # (D,H,W)

        return rendered_image, rendered_feat, alpha, rendered_depth, radii
