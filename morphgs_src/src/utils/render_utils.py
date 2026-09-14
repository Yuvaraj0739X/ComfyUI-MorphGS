# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import warnings
from typing import Union

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    BlendParams,
    HardFlatShader,
    HardPhongShader,
    PointLights,
    Materials,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.shader import HardDepthShader
from pytorch3d.structures import Meshes

warnings.filterwarnings("ignore", message="Bin size was too small in the coarse rasterization phase")


class DepthShader(HardDepthShader):
    def forward(self, fragments, meshes, **kwargs):
        return fragments.zbuf


class Renderer(nn.Module):
    def __init__(self, image_size: int, black_bg: bool = True, faces_per_pixels: int = 100, background_color=None):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.color_bg = background_color if background_color is not None else ((1, 1, 1) if black_bg else (255, 255, 255))
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=self.color_bg)
        self.raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * self.blend_params.sigma,
            faces_per_pixel=faces_per_pixels,
        )
        self.raster_settings_vis = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=100000,
        )

    def get_classical_texture(self, vertices: torch.Tensor) -> torch.Tensor:
        mesh_color = torch.tensor([0.0, 172.0 / 255.0, 223.0 / 255.0], dtype=torch.float32, device=vertices.device)
        return torch.ones_like(vertices) * mesh_color

    @torch.no_grad()
    def render_visualization(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        cameras: PerspectiveCameras,
        lights=None,
        texture: Union[torch.Tensor, None] = None,
        materials=None,
    ) -> torch.Tensor:
        device = vertices.device

        if texture is None:
            texture = self.get_classical_texture(vertices)

        if lights is None:
            lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        if materials is None:
            materials = Materials(
                device=device,
                ambient_color=((0.55, 0.55, 0.55),),
                diffuse_color=((0.75, 0.75, 0.75),),
                specular_color=((0.35, 0.35, 0.35),),
                shininess=48,
            )

        mesh = Meshes(verts=vertices, faces=faces, textures=TexturesVertex(verts_features=texture))
        vis_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights, materials=materials, blend_params=self.blend_params),
        )

        images = vis_renderer(mesh)
        images[..., 3] *= 255.0
        return images.permute(0, 3, 1, 2)

    def render_depth(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras, return_visibility=False, debug=False) -> torch.Tensor:
        device = vertices.device
        mesh = Meshes(verts=vertices, faces=faces)
        depth_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis)

        fragments = depth_rasterizer(mesh)
        depth_maps = fragments.zbuf[..., 0]
        depth_maps = depth_maps.clamp(0, float('inf'))
        valid_mask = fragments.pix_to_face[..., 0] >= 0
        depth_maps[depth_maps == float('inf')] = 0.0

        if not return_visibility:
            return depth_maps, valid_mask

        pix_to_face = fragments.pix_to_face[..., 0]
        B, V = vertices.shape[:2]
        visibility_masks = torch.zeros((B, V), dtype=torch.bool, device=device)

        for b in range(B):
            face_ids = pix_to_face[b].reshape(-1)
            face_ids = face_ids[face_ids >= 0]
            visible_face_ids = torch.unique(face_ids)
            vtx_idxs = faces[b][visible_face_ids]
            visibility_masks[b, vtx_idxs.reshape(-1)] = True

        return depth_maps, visibility_masks

def calculate_bounding_box(vertices):
    min_coords, _ = torch.min(vertices, dim=0)
    max_coords, _ = torch.max(vertices, dim=0)

    center = (min_coords + max_coords) / 2
    max_dim = torch.max(max_coords - min_coords).item()
    return center, max_dim


def render_mesh_with_pytorch3d(
        mesh_path, output_dir, image_size=(256, 256), debug=False, use_texture=False, 
        texture_path=None, render_normal=False
    ):
    """
    Render a 3D Mesh using PyTorch3D with a fixed camera following render_animation.
    Also saves depth buffers and camera parameters (Projection * View matrices).

    Args:
        mesh_path (str): Path to the mesh file (OBJ format).
        output_dir (str): Path to save the rendered image.
        image_size (tuple): Resolution of the rendered image (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)

    color_dir = f"{output_dir}/color"
    depth_dir = f"{output_dir}/depth"
    cameras_dir = f"{output_dir}/cameras"
    normal_dir = f"{output_dir}/normal"
    depth_buffers_file = os.path.join(output_dir, "depth_buffers.npy")
    projection_matrix_file = os.path.join(output_dir, "projection_matrix.npy")
    visibility_masks_file = os.path.join(output_dir, "visibility_masks.npy")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Load Mesh ###
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
    if texture_path is not None:
        mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
        mesh.visual.material.image = Image.open(texture_path).convert('RGBA')

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        
    # Check whether UV coordinates exist (TextureVisuals object)
    if hasattr(mesh.visual, 'uv'):
        uv_coords = mesh.visual.uv
    else:
        # No UVs: create zero UVs matching the vertex count
        uv_coords = np.zeros((mesh.vertices.shape[0], 2))

    # Check for an image texture
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
        vertex_colors = query_from_uv(uv_coords, np.array(mesh.visual.material.image))
    else:
        # No texture: use sky blue, RGB (135, 206, 235) normalized to [0, 1]
        sky_blue = np.array([135/255.0, 206/255.0, 235/255.0])
        vertex_colors = np.tile(sky_blue, (mesh.vertices.shape[0], 1))
            
    if use_texture:
        vertex_colors = query_from_uv(mesh.visual.uv, np.array(mesh.visual.material.image))
        textures = torch.tensor(vertex_colors, dtype=torch.float32, device=device) * 255.0  # (N, 3)

    depth_buffers = []
    visiblity_masks = []
    projection_matrix_all = []

    angle_interval = 20
    distance = 0
    elevs = [20]
    cnt = 0

    for i, cam_pos in enumerate(range(360 // angle_interval)):
        for elev in elevs:
            mesh_center, max_dim = calculate_bounding_box(vertices)
            if i == 0:
                distance = max_dim * 1.0

            R, T = look_at_view_transform(
                dist=distance,
                elev=elev,           
                azim=i * angle_interval, 
                at=mesh_center.unsqueeze(0),  # shape (1,3)
                device=device
            )
            
            cameras = PerspectiveCameras(
                device=device,
                R=R,
                T=T,
                image_size=torch.tensor(image_size, dtype=torch.float32, device=device),
                in_ndc=True
            )[0]
        
            renderer = Renderer(image_size[0], black_bg=False)  

            rgb = renderer.render_visualization(
                vertices.unsqueeze(0),   # (1, V, 3)
                faces.unsqueeze(0),      # (1, F, 3)
                cameras,
                texture=None if not use_texture else textures.unsqueeze(0),
            )
            rgb = rgb.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (N, H, W, 3)
            
            depth, visibility_mask = renderer.render_depth(
                vertices.unsqueeze(0),
                faces.unsqueeze(0),
                cameras,
                return_visibility=True,
                debug=debug
            )
            
            depth = depth.squeeze(0).cpu().numpy()  
            depth_buffers.append(depth)             
            visibility_mask = visibility_mask.squeeze(0).cpu().numpy()  # (1, N)
            visiblity_masks.append(visibility_mask)  

            # Render normals
            if render_normal:
                mesh_tmp = Meshes(verts=[vertices], faces=[faces])
                vn = mesh_tmp.verts_normals_packed()
                vn_color = (vn + 1) / 2
                norm_textures = TexturesVertex(vn_color.unsqueeze(0))  # (1, V, 3)
                mesh = Meshes(verts=[vertices], faces=[faces], textures=norm_textures)

                # Rasterizer + HardFlatShader
                raster_settings = RasterizationSettings(
                    image_size=image_size[0],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    bin_size=0
                )
                renderer_norm = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                            raster_settings=raster_settings),
                    shader=HardFlatShader(device=device, cameras=cameras)
                )
                norm = renderer_norm(mesh)[0, ..., :3]           # (H,W,3) float [0,1]
                norm_img = (norm * 255).byte().cpu().numpy()     # uint8 for saving
                

            # ---------- (3) Projection * View Matrix ----------
            # Manually construct the view matrix
            view_matrix_manual = torch.eye(4, device=cameras.R.device).repeat(cameras.R.shape[0], 1, 1)
            view_matrix_manual[:, :3, :3] = cameras.R
            view_matrix_manual[:, :3, 3] = cameras.T.squeeze(-1)

            # Manually construct the projection matrix
            projection_matrix_manual = torch.zeros((cameras.R.shape[0], 4, 4), device=cameras.R.device)
            projection_matrix_manual[:, 0, 0] = cameras.focal_length[:, 0]
            projection_matrix_manual[:, 1, 1] = cameras.focal_length[:, 1]
            projection_matrix_manual[:, 0, 2] = cameras.principal_point[:, 0]
            projection_matrix_manual[:, 1, 2] = cameras.principal_point[:, 1]
            projection_matrix_manual[:, 2, 2] = -(100.0 + 0.1) / (100.0 - 0.1)  # Near/Far clipping
            projection_matrix_manual[:, 2, 3] = -(2 * 100.0 * 0.1) / (100.0 - 0.1)
            projection_matrix_manual[:, 3, 2] = -1  # Perspective division

            # Compute the combined Projection * View matrix
            combined_matrix_manual = projection_matrix_manual.bmm(view_matrix_manual)
            combined_matrix_manual = combined_matrix_manual.squeeze(0).cpu().numpy()
            projection_matrix_all.append(combined_matrix_manual)

            # ---- save images ----
            os.makedirs(color_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(cameras_dir, exist_ok=True)
            
            plt.imsave(f"{color_dir}/{cnt:02d}.png", rgb.squeeze(0))
            plt.imsave(f"{depth_dir}/{cnt:02d}_depth.png", depth, cmap='gray')

            if render_normal:
                os.makedirs(normal_dir, exist_ok=True)
                plt.imsave(f"{normal_dir}/{cnt:02d}.png", norm_img)
            plt.close()  

            # ---- save camera parameters ---      
            torch.save(cameras, f"{cameras_dir}/{cnt:02d}_camera.pth")
            cnt += 1
        
    # ---------- save depth_buffers, projection_matrix ----------
    depth_buffers = np.stack(depth_buffers, axis=0)               # (num_views, H, W)
    projection_matrix_all = np.stack(projection_matrix_all, axis=0)  # (num_views, 4, 4)
    visibility_masks = np.stack(visiblity_masks, axis=0)           # (num_views, N)
    
    # Save files
    np.save(depth_buffers_file, depth_buffers)
    np.save(projection_matrix_file, projection_matrix_all)
    np.save(visibility_masks_file, visibility_masks)
    
    print(f"✅ Rendered images, depths and data saved to {output_dir}")
    
    return color_dir, depth_dir, normal_dir, cameras_dir, depth_buffers_file, projection_matrix_file, visibility_masks_file



def project_points_3d_to_2d(points_3d, viewport_size, camera_params):
    # camera is in the form of PerspectiveCameras
    points_2d_pixel = []
    for i in range(len(camera_params)):    
        # Extract camera parameters
        camera = camera_params[i]
        H, W = viewport_size

        # 3D -> 2D projection
        if hasattr(camera, 'transform_points_screen'):
            # PyTorch3D PerspectiveCameras
            camera = camera.to('cuda')
            screen_points = camera.transform_points_screen(points_3d.clone(), image_size=(H, W))
        else:
            # Dnerf dataset
            full_proj = camera['transform_matrix']
            pts_h   = torch.cat([points_3d, torch.ones_like(points_3d[...,:1])], dim=-1)    
            pts_uv = pts_h @ full_proj
            pts_uv = pts_uv[..., :2] / pts_uv[..., -1:]
            pts_uv = (pts_uv + 1) / 2 * torch.tensor([W, H], device=pts_uv.device, dtype=pts_uv.dtype)
            screen_points = pts_uv
            
        # screen_points: (BATCH, NUM_VERTICES, 3), last dim = (x, y, depth)
        points_2d = screen_points[..., :2]  # (BATCH, NUM_VERTICES, 2), keep only pixel coordinates
        points_2d_pixel.append(points_2d)
    
    points_2d_pixel = torch.stack(points_2d_pixel, dim=0)
    return points_2d_pixel





#### Draw Skeleton ####
def project_point_to_image_plane(points: torch.Tensor, pose: torch.Tensor, intrinsic: torch.Tensor):
    """
    Project 3D points to a batch of image planes.
    Args:
        points: (N, 3)
        pose: (B, 4, 4)
        intrinsic: (B, 3, 3)
    """
    if points.ndim == 2:
        points = torch.repeat_interleave(points.unsqueeze(0), len(pose), 0)  # (B, N, 3)
    pose = pose.inverse()  # (B, 4, 4)

    points = torch.bmm(pose[:, :3, :3], points.transpose(1, 2)).transpose(1, 2) + pose[:, :3, 3:].transpose(1, 2)  # (B, N, 3)
    points = torch.bmm(intrinsic.to(points.device), points.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)
    
    depths = points[:, :, 2]  # (B, N)
    depths_with_eps = points[:, :, 2:]
    
    points = points[:, :, :2] / depths_with_eps  # (B, N, 2)

    return points, depths  # (B, N, 2)


def project_points_2d(points_3d, view):
    if isinstance(view, list):
        pose, K = [],[]
        for v in view:
            pose.append(v.extrinsic)
            K.append(v.intrinsic)
        pose = torch.stack(pose, dim=0)
        K = torch.stack(K, dim=0)
    else:
        pose = view.extrinsic.unsqueeze(0)
        K = view.intrinsic.unsqueeze(0)
    points_2d, depths = project_point_to_image_plane(points_3d, pose, K.to(torch.float32))

    occluded_mask=None

    return points_2d, occluded_mask


def draw_skeleton(joints_2d, bones, view, joint_mask=None, joint_texts=None, color=None, mask=None,
                  draw_joints=True, line_color="black", line_width=2):
    num_joints = joints_2d.shape[1]
    cmap = plt.get_cmap('hsv')
    colors = [tuple(int(255 * c) for c in cmap(i / num_joints)[:3]) for i in range(num_joints)]

    image = Image.new("RGB", (view.image_width, view.image_height), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(image)
    font_size = 25
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for joints in joints_2d:
        if bones is not None:
            for bone in bones:
                if bone[0] >= len(joints) or bone[1] >= len(joints):
                    continue
                x1, y1 = joints[bone[0]]
                x2, y2 = joints[bone[1]]
                if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
                    continue
                draw.line((x1, y1, x2, y2), fill=line_color, width=line_width)

        if not draw_joints:
            continue

        for i, joint in enumerate(joints):
            if joint_mask is None or joint_mask[i]:
                x, y = joint
                draw.ellipse((x-2, y-2, x+2, y+2), fill=colors[i], outline='black')
                if joint_texts:
                    text = joint_texts[i]
                    # if text is string
                    if isinstance(text, str):
                        draw.text((x + 6, y - 6), text, font=font, fill='black')
                    # if text is list
                    elif isinstance(text, list):
                        for j, txt in enumerate(text):
                            draw.text((x + 6 + font_size * j, y - 6), txt, font=font,fill='black')
    return image



def render_titled_grid(stacked_vis, panel_h, panel_w, row_titles, header_h=24):
    """
    Render a titled grid image from a stacked tensor.
    """
    if torch.is_tensor(stacked_vis):
        vis_np = stacked_vis.detach().cpu().numpy()
    else:
        vis_np = stacked_vis

    vis_np = np.clip(vis_np, 0.0, 1.0)
    image = Image.fromarray((vis_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    grid_h, grid_w = image.size[1], image.size[0]
    n_rows = max(1, grid_h // panel_h)
    n_cols = max(1, grid_w // panel_w)

    for r in range(min(n_rows, len(row_titles))):
        titles = row_titles[r]
        for c in range(min(n_cols, len(titles))):
            x0 = c * panel_w
            y0 = r * panel_h
            x1 = min((c + 1) * panel_w - 1, grid_w - 1)
            y1 = min(y0 + header_h, grid_h - 1)

            draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
            draw.text((x0 + 6, y0 + 4), str(titles[c]), font=font, fill=(255, 255, 255))

    return image


def query_from_uv(uv_coords, texture_image):
    """
    Query color from texture image using uv coordinates.
    uv: (N, 2)
    texture_image: (H, W, 3)
    """
    tex_h, tex_w = texture_image.shape[:2]
    pixel_x = (uv_coords[:, 0] * tex_w).astype(int)
    pixel_y = ((1 - uv_coords[:, 1]) * tex_h).astype(int)  # flip V axis (OpenGL convention)

    pixel_x = np.clip(pixel_x, 0, tex_w - 1)
    pixel_y = np.clip(pixel_y, 0, tex_h - 1)

    vertex_colors = texture_image[pixel_y, pixel_x, :3]  # (N_vertices, 3)
    vertex_colors = vertex_colors / 255.0

    return vertex_colors
