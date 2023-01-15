import torch
import torch.nn as nn
def mse(pred_depth, target_depth):
    # mask out invalid depth values where target_depth <= 0
    
    mask = target_depth > 0
    pred_depth = pred_depth[mask]
    target_depth = target_depth[mask]
    
    return torch.mean((pred_depth - target_depth)**2)

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)


def gnll(pred_depth, target_depth, z_vals, weights, std):
    mask = target_depth > 0

    pred_mean = pred_depth[mask] # z_hat(r)
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=pred_depth.device, requires_grad=True)
    
    pred_var = ((z_vals[mask] - pred_mean.unsqueeze(-1)).pow(2) * weights[mask]).sum(-1) + 1e-5 # s_hat^2(r)
    pred_std = pred_var.sqrt() # s_hat(r)
    target_mean = target_depth[mask] # z(r)
    
    # Initialize target_std with the same shape as target_mean with values of std
    target_std = torch.zeros_like(target_depth) + std
    target_std = target_std[mask] # s(r)
    
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=pred_depth.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    
    f = nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(mask.shape[0]) * f(pred_mean, target_mean, pred_var)