# from skimage.metrics import structural_similarity
# from lpips import LPIPS
import torch
import torch.nn.functional as F
import math
import lpips
# structural similarity index
class SSIM(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def __init__(self):
        self.model = lpips.LPIPS(net='vgg').cuda()

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
    
# def ssim(pred, target):
#     '''
#         pred: [B, H, W, C]
#         target: [B, H, W, C]
#     '''
#     return structural_similarity(pred.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2))

def psnr(pred, target):
    return 10 * torch.log10(1 / torch.mean((pred - target)**2))

# def lpips(pred, target):
#     '''
#         pred: [H, W, C]
#         target: [H, W, C]
#     '''
#     lpips_alex = LPIPS()
    
#     lpips = lpips_alex(pred.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
    
#     return lpips

# def lpips_test(pred, target):
#     '''
#         pred: [B, H, W, C]
#         target: [B, H, W, C]
#     '''
#     lpips_alex = LPIPS()
    
#     # lpips = lpips_alex(pred.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
#     lpips = lpips_alex(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), normalize=True)[0]
    
#     return lpips

def compute_metrics(pred, target, pred_depth, target_depth):

    # pred [B, H, W, C]
    
    # Make pred = [B, C, H, W]
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    pred_depth = pred_depth.permute(0, 3, 1, 2)
    target_depth = target_depth.permute(0, 3, 1, 2)
    mask = target_depth > 0


    pred_depth = pred_depth[mask]
    target_depth = target_depth[mask]
    
    # Where target_depth is 0, set pred also to 0
    # pred = pred * mask
    # target = target * mask
    
    ssim_dnerf = SSIM()
    lpips_dnerf = LPIPS()
    
    all_ssim = 0.
    all_rmse = 0.
    all_psnr = 0.
    all_lpips = 0.
    
    for i in range(pred.shape[0]):
        single_pred = pred[i]  
        # add batch dim
        single_pred = single_pred.unsqueeze(0)
        
        single_pred_depth = pred_depth[i]
        single_pred_depth = single_pred_depth.unsqueeze(0)
        
        single_target = target[i]
        single_target = single_target.unsqueeze(0)
        
        single_target_depth = target_depth[i]
        single_target_depth = single_target_depth.unsqueeze(0)
        
        single_ssim = ssim_dnerf(single_pred, single_target)
        single_rmse = rmse(single_pred_depth, single_target_depth)
        single_psnr = psnr(single_pred, single_target)
        single_lpips = lpips_dnerf(single_pred, single_target)
        
        all_ssim += single_ssim
        all_rmse += single_rmse
        all_psnr += single_psnr
        all_lpips += single_lpips
        
        # Save single metrics in txt file
        # with open("metrics_single.txt", "a") as f:
        #     f.write(f"SSIM: {single_ssim}, RMSE: {single_rmse}, PSNR: {single_psnr}, LPIPS: {single_lpips}\n")
                     
    # Save all metrics in txt file
    # with open("metrics_all.txt", "a") as f:
    #     f.write(f"SSIM: {all_ssim}, RMSE: {all_rmse}, PSNR: {all_psnr}, LPIPS: {all_lpips}")
    # print("Pred: ", pred.shape)
    # print("Target: ", target.shape)
    # print("Pred Depth: ", pred_depth.shape)
    # print("Target Depth: ", target_depth.shape)
    
    
    res = {
        'ssim': all_ssim / pred.shape[0],
        'rmse': all_rmse / pred.shape[0],
        'psnr': all_psnr / pred.shape[0],
        'lpips': all_lpips / pred.shape[0]
    }
    
    print("SSIM: ", res['ssim'])
    print("RMSE: ", res['rmse'])
    print("PSNR: ", res['psnr'])
    print("LPIPS: ", res['lpips'])
    
    return res 