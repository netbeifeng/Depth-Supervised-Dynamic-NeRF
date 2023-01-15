import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils.convert import to8b
import imageio
import numpy as np
from utils.getrays import get_rays, get_rays_np, ndc_rays
from utils.helpers import sample_pdf
from utils.depth_prior import compute_samples_around_depth, precompute_quadratic_samples, sample_3sigma
from utils.metrics import compute_metrics
# For testing and evaluation
def render_path(render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, gt_depths=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):

    H, W, fx, fy, cx, cy = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        fx = fx/render_factor
        fy = fy/render_factor
        cx = cx/render_factor
        cy = cy/render_factor
        
    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        save_dir_depth = os.path.join(savedir, "depth")
        save_dir_depth_gt = os.path.join(savedir, "depth_gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)
        if not os.path.exists(save_dir_depth):
            os.makedirs(save_dir_depth)
        if save_also_gt and not os.path.exists(save_dir_depth_gt):
            os.makedirs(save_dir_depth_gt)

    rgbs = []
    rgbs_gt = []
    disps = []
    depths = []
    depths_gt = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        rgb, disp, acc, depth, _ = render(H, W, [fx, fy], chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgb = torch.clamp(rgb,0,1)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())
        if gt_imgs is not None:
            rgbs_gt.append(gt_imgs[i])
        if gt_depths is not None:
            depths_gt.append(gt_depths[i])
        
        if savedir is not None:
            rgb8_estim = to8b(rgbs[-1])
            depth_estim = to8b(depths[-1]/depths[-1].max())
            filename = os.path.join(save_dir_estim, '{:03d}.png'.format(i+i_offset))
            filename_depth = os.path.join(save_dir_depth, '{:03d}.png'.format(i+i_offset))
            imageio.imwrite(filename, rgb8_estim)
            imageio.imwrite(filename_depth, depth_estim)
            if save_also_gt and gt_imgs is not None:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt)
            if save_also_gt and gt_depths is not None:
                depth_gt = to8b(gt_depths[i]/gt_depths[i].max())
                filename = os.path.join(save_dir_depth_gt, '{:03d}.png'.format(i+i_offset))
                imageio.imwrite(filename, depth_gt)

    rgbs_np = np.stack(rgbs, 0) # [B, H, W, 3]
    disps_np = np.stack(disps, 0) # [B, H, W]
    depths_np = np.stack(depths, 0) # [B, H, W]
    if gt_imgs is not None:
        rgbs_gt_np = np.stack(rgbs_gt, 0) # [B, H, W, 3]
    if gt_depths is not None:
        depths_gt_np = np.stack(depths_gt, 0) # [B, H, W]
    
    if gt_imgs is not None and gt_depths is not None:
        depths_np = depths_np.reshape(depths_np.shape[0], depths_np.shape[1], depths_np.shape[2], 1) # [B, H, W, 1]
        depths_gt_np = depths_gt_np.reshape(depths_gt_np.shape[0], depths_np.shape[1], depths_np.shape[2], 1) # [B, H, W, 1]
        
        # Convert all to tensors
        rgbs = torch.from_numpy(rgbs_np).float().cuda() # [B, H, W, 3]
        rgbs_gt = torch.from_numpy(rgbs_gt_np).float().cuda() # [B, H, W, 3]
        depths = torch.from_numpy(depths_np).float().cuda() # [B, H, W, 1]
        depths_gt = torch.from_numpy(depths_gt_np).float().cuda() # [B, H, W, 1]
        
        compute_metrics(rgbs, rgbs_gt, depths, depths_gt)
        # save the computed metrics in txt file
        # if savedir is not None:
        with open(os.path.join(savedir, 'metrics.txt'), 'w') as f:
            f.write(str(compute_metrics(rgbs, rgbs_gt, depths, depths_gt)))
            
    return rgbs_np, disps_np, depths_np

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        depth_range = None
    else:
        # use provided ray batch
        if rays.shape[0] == 2:
            rays_o, rays_d = rays
            depth_range = None
        elif rays.shape[0] == 3:
            rays_o, rays_d, depth_range = rays
        else:
            raise ValueError('rays should have 2 or 3 elements')
        
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # [..., 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float() # [..., 3]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) # [..., 1], [..., 1]
    frame_time = frame_time * torch.ones_like(rays_d[...,:1]) # [..., 1]
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1) # 3 + 3+ 1 + 1 + 1 = 9 [..., 9]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # [..., 12]

    if depth_range is not None:
        depth_range = torch.reshape(depth_range, [-1,3]).float()
        rays = torch.cat([rays, depth_range], -1)   # [..., 15]
    
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
    
def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                depth_guided_sampling=False,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # ray_batch = [..., 15] or [..., 12] or [..., 9]
    # [..., 9] = [rays_o, rays_d, near, far, frame_time]
    # [..., 12] = [rays_o, rays_d, near, far, frame_time, viewdirs]
    # [..., 15] = [rays_o, rays_d, near, far, frame_time, viewdirs, depth_range]
    
    N_rays = ray_batch.shape[0]
    
    # Get Device from ray_batch
    device = ray_batch.device
    
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs, depth_range = None, None
    # viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else 
    if use_viewdirs:
        viewdirs = ray_batch[:,9:12] if ray_batch.shape[-1] > 9 else None
        if ray_batch.shape[-1] > 12:
            depth_range = ray_batch[:,12:15]
    else:
        depth_range = ray_batch[:,9:12] if ray_batch.shape[-1] > 9 else None

    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0, depth_map_0 = None, None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        
        # Stratified sampling
        z_vals = z_vals.expand([N_rays, N_samples]) # [N_rays, N_samples]
        if depth_guided_sampling:
            dist = z_vals[0, 1] - z_vals[0, 0]
                
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        # Coarse to fine strategy
        if N_importance <= 0:
            # pass
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, depth_map_0 = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
            if depth_guided_sampling: # Depth guided sampling
                is_test = depth_range is None
                
                if is_test: # Test time
                    # print("### Depth guided sampling is not supported in test time. ###")
                    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                    # What is the shape of z_vals[...,1:] ? [N_rays, N_samples-1]
                    # What is the shape of z_vals[...,:-1] ? [N_rays, N_samples-1]
                                        
                    z_vals_2 = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
                    z_vals_2 = z_vals_2.detach()
                    
                else: # Training time
                    # print("### Depth guided sampling is used. ###")
                    valid_depth = depth_range[:,0] >= near[0, 0]
                    invalid_depth = valid_depth.logical_not()
                    z_vals_2 = torch.zeros([N_rays, N_importance]) # guided samples
                    
                    # sample around the predicted depth from the first half of samples, if the input depth is invalid
                    z_vals_2[invalid_depth] = compute_samples_around_depth(raw.detach()[invalid_depth], 
                                                                           z_vals[invalid_depth], rays_d[invalid_depth], 
                                                                           N_importance, perturb, dist, near[0, 0], far[0, 0], device)
                    
                    
                    # sample with in 3 sigma of the input depth, if it is valid
                    z_vals_2[valid_depth] = sample_3sigma(depth_range[valid_depth, 1], depth_range[valid_depth, 2], 
                                                          N_importance, perturb == 0., near[0, 0], far[0, 0], device)
                
                z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_2], -1), -1)
            else:  # Hierarchical sampling
                # print("### Hierarchical sampling is used. ###")
                z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
                z_samples = z_samples.detach()
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'depth_map': depth_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        if depth_map_0 is not None:
            ret['depth0'] = depth_map_0
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")
    
    return ret

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map