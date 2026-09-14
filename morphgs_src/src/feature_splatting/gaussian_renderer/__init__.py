# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.
#
# Rendering backend: gsplat (Apache-2.0), via the compatibility shim in ../gsplat_rasterizer.py.
# This file previously called Inria/GRAPHDECO'''s non-commercial diff_gaussian_rasterization;
# only the import changed, the render() logic below is unmodified.

import torch
import math
from ..gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh
from pytorch3d.transforms import quaternion_multiply
from ..gsplat_rasterizer import (
    GaussianRasterizationSettings as LatentGaussianRasterizationSettings,
    GaussianRasterizer as LatentGaussianRasterizer,
)
try:
    from diff_gaussian_rasterization_gauhuman import (
        GaussianRasterizationSettings as GauHumanGaussianRasterizationSettings,
        GaussianRasterizer as GauHumanGaussianRasterizer,
    )
    _HAS_GAUHUMAN_RASTERIZER = True
except Exception:
    GauHumanGaussianRasterizationSettings = None
    GauHumanGaussianRasterizer = None
    _HAS_GAUHUMAN_RASTERIZER = False


def render(viewpoint_camera,
            pc : GaussianModel,
            pipe,
            bg_color : torch.Tensor,
            scaling_modifier = 1.0,
            override_color = None,
            render_features = False,
            render_gaussian_idx = False,
            precomp_xyz = None,
            d_rotation = None,
            d_rotation_bias = None,
            use_feature_sh=False,
            render_for_depth=False,
            value=0.95,
            rasterizer_backend="latent",
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    xyz = pc.get_xyz if precomp_xyz is None else precomp_xyz
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Default to the latent-splatting rasterizer backend.
    raster_settings_cls = LatentGaussianRasterizationSettings
    rasterizer_cls = LatentGaussianRasterizer
    if str(rasterizer_backend).lower() == "gauhuman":
        if not _HAS_GAUHUMAN_RASTERIZER:
            raise ImportError(
                "rasterizer_backend='gauhuman' requested, but diff_gaussian_rasterization_gauhuman is not installed."
            )
        raster_settings_cls = GauHumanGaussianRasterizationSettings
        rasterizer_cls = GauHumanGaussianRasterizer

    raster_settings = raster_settings_cls(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = rasterizer_cls(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points

    opacity = pc.get_opacity
    if render_for_depth:
        opacity = torch.ones(pc.get_xyz.shape[0], 1, device=pc.get_xyz.device) * value

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, d_rotation=None if type(d_rotation) is float else d_rotation, gs_rot_bias=d_rotation_bias)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation_bias(d_rotation)
        if d_rotation_bias is not None:
            rotations = quaternion_multiply(d_rotation_bias, rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
        else:
            shs = pc.get_features  # (N, 16 ,3)
    else:
        colors_precomp = override_color
    
    backend = str(rasterizer_backend).lower()
    if backend not in {"latent", "gauhuman"}:
        raise ValueError(f"Unknown rasterizer_backend: {rasterizer_backend}")

    # Get view-independent features for latent backend only.
    distill_feats = None
    if backend == "latent" and use_feature_sh:
        distill_feats = pc.get_distill_features

    # Support both latent (5 outputs) and gauhuman-style (4 outputs) rasterizer returns.
    if backend == "latent":
        rendered_image, rendered_feat, alpha, rendered_depth, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            features = distill_feats,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
    else:
        gauhuman_out = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        if len(gauhuman_out) == 4:
            rendered_image, radii, rendered_depth, alpha = gauhuman_out
            rendered_feat = None
        elif len(gauhuman_out) == 5:
            # Some environments may still return latent-style tuple.
            rendered_image, rendered_feat, alpha, rendered_depth, radii = gauhuman_out
        else:
            raise RuntimeError(f"Unexpected rasterizer output length: {len(gauhuman_out)}")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_alpha": alpha,
            "render_depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
