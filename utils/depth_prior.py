import torch 
import torch.nn.functional as F
import math
import numpy as np
from utils.helpers import sample_pdf

def precompute_quadratic_samples(near, far, num_samples):
    # normal parabola between 0.1 and 1, shifted and scaled to have y range between near and far
    start = 0.1
    x = torch.linspace(0, 1, num_samples)
    c = near
    a = (far - near)/(1. + 2. * start)
    b = 2. * start * a
    return a * x.pow(2) + b * x + c

def compute_weights(device,raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights


def raw2depth(raw, z_vals, rays_d, device):
    weights = compute_weights(device, raw, z_vals, rays_d)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def compute_samples_around_depth(raw, z_vals, rays_d, N_samples, perturb, lower_bound, near, far, device):
    sampling_depth, sampling_std = raw2depth(raw, z_vals, rays_d, device)
    sampling_std = sampling_std.clamp(min=lower_bound)
    depth_min = sampling_depth - sampling_std
    depth_max = sampling_depth + sampling_std
    return sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far, device)

def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far, device):
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)