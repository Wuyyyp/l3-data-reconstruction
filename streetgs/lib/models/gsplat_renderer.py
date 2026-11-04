"""gsplat renderer with optional StreetGS densification-statistics bridge.

StreetGS owns actors, sky and color correction. The screen-space statistics bridge restores StreetGS NDC gradient units and
its per-pixel absolute-gradient L1 statistic; StreetGS retains topology control.
"""
import torch
from lib.config import cfg
from lib.utils.sh_utils import eval_sh


def render_gsplat(camera, pc, convert_SHs_python=None,
                  compute_cov3D_python=None, scaling_modifier=None,
                  override_color=None, white_background=False):
    if cfg.mode == 'train' and not cfg.optim.get('fixed_gaussians', False) and not cfg.render.get('gsplat_densification', False):
        raise RuntimeError('gsplat training requires optim.fixed_gaussians=True or render.gsplat_densification=True.')
    # StreetGaussianModel.get_xyz concatenates a list and raises when visibility
    # selects no background and no actors (e.g. sky-only or an empty actor frame).
    if getattr(pc, 'num_gaussians', None) == 0:
        means = camera.world_view_transform.new_empty((0,3))
    else:
        means = pc.get_xyz
    n = means.shape[0]
    h, w = int(camera.image_height), int(camera.image_width)
    if n == 0:
        return dict(rgb=means.new_full((3,h,w), float(white_background)),
                    acc=means.new_zeros((1,h,w)), depth=means.new_zeros((1,h,w)),
                    radii=means.new_zeros((0,)),
                    visibility_filter=torch.zeros(0, device=means.device, dtype=torch.bool),
                    viewspace_points=None)
    from gsplat import rasterization
    scale = cfg.render.scaling_modifier if scaling_modifier is None else scaling_modifier
    python_sh = cfg.render.convert_SHs_python if convert_SHs_python is None else convert_SHs_python
    python_cov = cfg.render.compute_cov3D_python if compute_cov3D_python is None else compute_cov3D_python
    if python_cov:
        raise NotImplementedError('Phase 3 uses activated scales/quaternions; compute_cov3D_python must be false.')

    # Recover K from the actual projection matrix, including off-centre principal
    # points. Legacy projects to cx-0.5 and samples integer pixels; gsplat projects
    # to cx and samples pixel+0.5. No additional half-pixel shift is required.
    projection = getattr(camera, 'projection_matrix', None)
    if projection is None:  # MiniCam / viewer camera
        projection = torch.linalg.solve(camera.world_view_transform, camera.full_proj_transform)
    p = projection.transpose(0,1)
    K = means.new_zeros((3,3))
    K[0,0], K[1,1] = p[0,0]*w/2, p[1,1]*h/2
    K[0,2], K[1,2] = (p[0,2]+1)*w/2, (p[1,2]+1)*h/2
    K[2,2] = 1
    if abs(float(p[0,1])) > 1e-7:
        raise ValueError('gsplat pinhole adapter does not support nonzero intrinsic skew.')

    features, names, dims = [], [], []
    if cfg.render.render_normal:
        normals = pc.get_normals(camera)
        features.append(normals); names.append('normals'); dims.append(normals.shape[-1])
    if cfg.data.get('use_semantic', False):
        semantic = pc.get_semantic
        features.append(semantic); names.append('semantic'); dims.append(semantic.shape[-1])

    sh_degree = None
    if override_color is not None:
        colors = override_color
    else:
        try:
            colors = pc.get_features
        except AttributeError:
            colors = pc.get_colors(camera.camera_center)
        else:
            degree = pc.active_sh_degree if python_sh else pc.max_sh_degree
            if python_sh or features:
                direction = means - camera.camera_center
                direction = torch.nn.functional.normalize(direction, dim=-1)
                colors = (eval_sh(degree, colors.transpose(1,2), direction)+0.5).clamp_min(0)
            else:
                sh_degree = degree
    if features:
        colors = torch.cat([colors] + features, dim=-1)
    channels = 3 + sum(dims)
    background = means.new_zeros((1,channels))
    background[:,:3] = float(white_background)
    packed = cfg.render.get('gsplat_packed', False)
    collect_stats = cfg.mode == 'train' and cfg.render.get('gsplat_densification', False) and not cfg.optim.get('fixed_gaussians', False)
    rendered, alpha, meta = rasterization(
        means=means, quats=pc.get_rotation, scales=pc.get_scaling*scale,
        opacities=pc.get_opacity.reshape(-1), colors=colors,
        viewmats=camera.world_view_transform.transpose(0,1).contiguous()[None],
        Ks=K[None], width=w, height=h, near_plane=0.2, far_plane=1e10,
        eps2d=0.3, radius_clip=cfg.render.get("gsplat_radius_clip", 0.0), sh_degree=sh_degree,
        packed=packed, sparse_grad=False, absgrad=collect_stats,
        rasterize_mode='classic', render_mode='RGB+D', backgrounds=None,
    )
    image = rendered[0].permute(2,0,1)
    projected_radii = meta['radii'].reshape(-1, 2).max(dim=-1).values
    if packed:
        radii = projected_radii.new_zeros(n)
        radii[meta['gaussian_ids']] = projected_radii
    else:
        radii = projected_radii
    stats = None
    if collect_stats and meta['means2d'].requires_grad:
        meta['means2d'].retain_grad()
        from lib.models.gsplat_screen_stats import GsplatScreenStats
        stats = GsplatScreenStats(meta['means2d'], n, w, h, meta['gaussian_ids'] if packed else None)
    # gsplat 1.5.3's packed wrapper rejects [C,D] backgrounds. Composite
    # the constant RGB background through alpha; leave features/depth at zero.
    rgb = image[:3]
    if white_background:
        rgb = rgb + (1 - alpha[0].permute(2,0,1))
    if cfg.mode != 'train':
        rgb = rgb.clamp(0,1)
    result = dict(rgb=rgb, acc=alpha[0].permute(2,0,1),
                  depth=image[-1:], radii=radii, visibility_filter=radii>0,
                  viewspace_points=stats)
    offset = 3
    for name, dim in zip(names,dims):
        value = image[offset:offset+dim]; offset += dim
        if name == 'normals':
            value = torch.nn.functional.normalize(value,dim=0)
        elif cfg.model.gaussian.get('semantic_mode','logits') == 'probabilities':
            value = torch.log(value/(value.sum(dim=0,keepdim=True)+1e-8)+1e-8)
        result[name] = value
    return result
