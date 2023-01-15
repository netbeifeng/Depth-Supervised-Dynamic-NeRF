import os
import imageio
import time

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from datetime import datetime
import numpy as np
from config import config_parser
from model.nerf import create_nerf

from load_torf import load_dino_data
from load_toss import load_toss_data
try:
    from apex import amp
except ImportError:
    pass

from model.renderer import render, render_path
from utils.convert import img2mse, mse2psnr, to8b
from utils.getrays import get_rays, get_rays_np
from utils.logging import log
from utils.loss import mse, gnll
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def train(args):
    # Load data
    log("TRAIN", "Loading data...")
    if args.dataset_type == "torf":
        images, depths, poses, times, render_poses, render_times, hwf, i_split = load_dino_data(args.datadir)
    elif args.dataset_type == "real":
        images, depths, poses, times, render_poses, render_times, hwf, i_split = load_toss_data(args.datadir)
    else:
        raise ValueError("Unknown dataset type: {}".format(args.dataset_type))

    
    
    log("TRAIN", "Data loaded, Length: {}".format(len(images)))
    i_train, i_val, i_test = i_split

    print("TRAIN:", i_train)
    print("VAL:", i_val)
    
    
    near = np.min(depths[depths > 0]) # Real dataset contains 0s
    far = np.max(depths)
    log("TRAIN", "Near: {}, Far: {}".format(near, far))
    
    if args.white_bkgd:
        images = images[...,:3] * images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time >= 0., "time must be bigger than 0"
    assert max_time <= 1., "max time must be smaller than 1"

    # Cast intrinsics to right types
    H, W, fx, fy, cx, cy = hwf
    H, W = int(H), int(W)
    hwf = [H, W, fx, fy, cx, cy]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    
    # Logging
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
            
    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    log("TRAIN", "Logging to {}".format(os.path.join(basedir, expname)))

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    log("TRAIN", "Model created")    
    
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only and args.render_test: # Test Metrics !!!
        print('RENDER ONLY')
        with torch.no_grad():
            print("Testing for ", i_test)
            testsavedir = os.path.join(basedir, expname, 'renderonly_newb_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            # print('test poses shape', render_poses.shape)

            test_poses = torch.Tensor(np.array(poses)[i_test]).to(device)
            test_times = torch.Tensor(np.array(times)[i_test]).to(device)
            
            rgbs, _, depths = render_path(test_poses, test_times, hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], gt_depths=depths[i_test],
                                  savedir=testsavedir, render_factor=args.render_factor, save_also_gt=True)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'depths.mp4'), to8b(depths/np.max(depths)), fps=10, quality=8)
            return

    if args.render_spherical_pose:
        print('RENDER SPHERICAL POSE')
        with torch.no_grad():
            render_poses = torch.Tensor(render_poses).to(device)
            # Initialize render times with args.i_time , shape like render_poses
            render_times = torch.ones(render_poses.shape[0], 1) * args.i_time
            # render_times = torch.Tensor(render_times).to(device)
            assert torch.all(torch.eq(render_times, render_times[0]))
            testsavedir = os.path.join(basedir, expname, 'render_spherical_pose_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            
            rgbs, _, depths = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=None, gt_depths=None,
                                    savedir=testsavedir, render_factor=args.render_factor, save_also_gt=False)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=15, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'depths.mp4'), to8b(depths/np.max(depths)), fps=15, quality=8)
            return
    
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, [fx, fy], p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    depths = torch.Tensor(depths).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    
    N_iters = args.N_iter + 1
    start = start + 1
    
    log("TRAIN", "Starting training from iteration {}".format(start))
    
    all_psnr = []
    all_loss = []
    all_depth_loss = []
    all_img_loss = []
    # all_ssim = []
    # all_lpips = []
    # all_rmse = []
    
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            raise NotImplementedError("Time not implemented")

            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            if i >= args.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                # If need to precrop, then sample from a subset of images
                skip_factor = i / float(args.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])

            
            target = images[img_i]
            target_depth = depths[img_i]
            pose = poses[img_i, :3, :4]
            frame_time = times[img_i]

            if N_rand is not None:
                # Random rays o = origin, d = direction
                rays_o, rays_d = get_rays(H, W, [fx, fy], torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # Randomly sample N_rand rays
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                # target_s is the rgb value at the sampled ray
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_depth_s = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
                if args.use_depth and args.depth_guided_sampling:
                    depth_std = args.depth_guided_sampling_std 
                    
                    _depth = target_depth_s[:, 0] # (N_rand,) 
                    _depth_min = _depth - depth_std # (N_rand,)
                    _depth_max = _depth + depth_std # (N_rand,)
                                                
                    depth_range = torch.stack([_depth, _depth_min, _depth_max], -1) # (N_rand, 3)
                    depth_range = torch.unsqueeze(depth_range, 0) # (1, N_rand, 3)
                    batch_rays = torch.cat([batch_rays, depth_range], 0)   # (3, N_rand, 3)                
                    
        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, [fx, fy], chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        if args.add_tv_loss:
            frame_time_prev = times[img_i - 1] if img_i > 0 else None
            frame_time_next = times[img_i + 1] if img_i < times.shape[0] - 1 else None

            if frame_time_prev is not None and frame_time_next is not None:
                if np.random.rand() > .5:
                    frame_time_prev = None
                else:
                    frame_time_next = None

            if frame_time_prev is not None:
                rand_time_prev = frame_time_prev + (frame_time - frame_time_prev) * torch.rand(1)[0]
                _, _, _, _, extras_prev = render(H, W, [fx, fy], chunk=args.chunk, rays=batch_rays, frame_time=rand_time_prev,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_kwargs_train)

            if frame_time_next is not None:
                rand_time_next = frame_time + (frame_time_next - frame_time) * torch.rand(1)[0]
                _, _, _, _, extras_next = render(H, W, [fx, fy], chunk=args.chunk, rays=batch_rays, frame_time=rand_time_next,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)

        tv_loss = 0
        if args.add_tv_loss:
            if frame_time_prev is not None:
                tv_loss += ((extras['position_delta'] - extras_prev['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_prev['position_delta_0']).pow(2)).sum()
            if frame_time_next is not None:
                tv_loss += ((extras['position_delta'] - extras_next['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_next['position_delta_0']).pow(2)).sum()
            tv_loss = tv_loss * args.tv_loss_weight

        loss = img_loss + tv_loss
        psnr = mse2psnr(img_loss)
        
        depth_loss = 0
        
        if args.depth_loss_type == "MSE":
            depth_loss = mse(depth, target_depth_s.squeeze(-1))
        elif args.depth_loss_type == "GNLL":
            depth_loss = gnll(depth, target_depth_s.squeeze(-1), z_vals=extras['z_vals'], weights=extras['weights'],  std=args.depth_guided_sampling_std)
            # print("depth_loss", depth_loss)
        else:
            raise NotImplementedError
        
        if args.use_depth and args.depth_weight > 0:
            loss += (depth_loss * args.depth_weight)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            if 'depth0' in extras:
                depth_loss0 = mse(extras['depth0'], target_depth_s.squeeze(-1))
                loss = loss + depth_loss0 * args.depth_weight
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        else:
            img_loss0 = 0
            depth_loss0 = 0
        
        # Log every step    
        all_depth_loss.append(depth_loss.item())
        all_img_loss.append(img_loss.item())
        all_loss.append(loss.item())
        
        all_psnr.append(psnr.item())
          
        writer.add_scalar('every_step/loss', loss, global_step)
        writer.add_scalar('every_step/psnr', psnr, global_step)
        writer.add_scalar('every_step/img_loss', img_loss + img_loss0, global_step)
        writer.add_scalar('every_step/depth_loss', depth_loss + depth_loss0, global_step)
        
        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            if args.do_half_precision:
                save_dict['amp'] = amp.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0: # Train
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {img_loss.item()} PSNR: {psnr.item()}"
            if args.use_depth:
                tqdm_txt += f" Loss_depth: {depth_loss.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
                
            tqdm.write(tqdm_txt)
            
            log("PRINT",f"AVG LOSS= {np.mean(np.array(all_loss))}, AVG IMG LOSS= {np.mean(np.array(all_img_loss))}, AVG DEPTH LOSS= {np.mean(np.array(all_depth_loss))}, AVG PSNR= {np.mean(np.array(all_psnr))}")

            writer.add_scalar('avg/loss', np.mean(np.array(all_loss)), i)
            writer.add_scalar('avg/img_loss', np.mean(np.array(all_img_loss)), i)
            writer.add_scalar('avg/depth_loss', np.mean(np.array(all_depth_loss)), i)
            writer.add_scalar('avg/psnr', np.mean(np.array(all_psnr)), i)
            
            all_depth_loss = []
            all_img_loss = []
            all_loss = [] 
            all_psnr = [] 
            
            writer.add_scalar('print/loss', img_loss.item(), i)
            writer.add_scalar('print/psnr', psnr.item(), i)
            if 'rgb0' in extras:
                writer.add_scalar('print/loss0', img_loss0.item(), i)
                writer.add_scalar('print/psnr0', psnr0.item(), i)
            if 'depth0' in extras:
                writer.add_scalar('print/depth_loss0', depth_loss0.item(), i)
            if args.use_depth:
                writer.add_scalar('print/depth_loss', depth_loss.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('print/tv', tv_loss.item(), i)

        del loss, img_loss, psnr, target_s
        if 'rgb0' in extras:
            del img_loss0, psnr0
        if 'depth0' in extras:
            del depth_loss0
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, extras

        # Disable for faster training
        
        # if i%args.i_img==0: # Val
        #     with torch.no_grad():
        #         torch.cuda.empty_cache()
        #         # Log a rendered validation view to Tensorboard, validation set is fixed
        #         img_i=np.random.choice(i_val)
        #         target = images[img_i]
        #         target_depth = depths[img_i]
        #         pose = poses[img_i, :3,:4]
        #         frame_time = times[img_i]
        #         with torch.no_grad():
        #             rgb, disp, acc, depth, extras = render(H, W, [fx, fy], chunk=args.chunk, c2w=pose, frame_time=frame_time,
        #                                                 **render_kwargs_test)
                
        #         img_loss = img2mse(rgb, target)
        #         depth_loss = mse(depth, target_depth.squeeze(-1))
        #         loss = img_loss + depth_loss * args.depth_weight
        #         psnr = mse2psnr(img_loss)
                
        #         _ssim = ssim(rgb, target)
        #         _lpips = lpips(rgb, target)
        #         _rmse = rmse(depth, target_depth.squeeze(-1))
                
        #         log("VAL", f'Iter: {i} Loss_fine: {loss.item()} Depth_loss: {depth_loss.item()} Img_loss: {img_loss.item()}')
        #         log("VAL", f'PSNR: {psnr.item()} SSIM: {_ssim} RMSE: {_rmse.item()} LPIPS: {_lpips[0, 0, 0]}')
                        
        #         writer.add_scalar('val/loss', loss.item(), i)
        #         writer.add_scalar('val/img_loss', img_loss.item(), i)
        #         writer.add_scalar('val/depth_loss', depth_loss.item(), i)
        #         writer.add_scalar('val/psnr', psnr.item(), i)
        #         writer.add_scalar('val/ssim', _ssim, i) 
        #         writer.add_scalar('val/rmse', _rmse.item(), i)
        #         writer.add_scalar('val/lpips', _lpips[0, 0, 0], i) 
                
        #         writer.add_image('image/rgb_gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
        #         writer.add_image('image/rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
        #         writer.add_image('image/disp', disp.cpu().numpy(), i, dataformats='HW')
                
        #         if args.use_depth:
        #             writer.add_image('image/depth_gt', to8b(target_depth.squeeze(-1).cpu().numpy()/np.max(target_depth.squeeze(-1).cpu().numpy())), i, dataformats='HW')
        #         writer.add_image('image/depth', to8b(depth.cpu().numpy()/np.max(depth.cpu().numpy())), i, dataformats='HW')
        #         writer.add_image('image/acc', acc.cpu().numpy(), i, dataformats='HW')

        #         if 'rgb0' in extras:
        #             writer.add_image('image/rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
        #         if 'depth0' in extras:
        #             writer.add_image('image/depth_rough', to8b(extras['depth0'].cpu().numpy()/np.max(extras['depth0'].cpu().numpy())), i, dataformats='HW')
        #         if 'disp0' in extras:
        #             writer.add_image('image/disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
        #         if 'z_std' in extras:
        #             writer.add_image('image/acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

        #         log("VAL", "finish summary")
        #         writer.flush()

        # if i%args.i_video==0:
        #     # Turn on testing mode
        #     print("Rendering video...")
        #     with torch.no_grad():
        #         savedir = os.path.join(basedir, expname, 'frames_{}_spiral_{:06d}_time/'.format(expname, i))
        #         rgbs, disps = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=None, gt_depths=None, savedir=savedir)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # Test dataset
        # if i%args.i_testset==0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        #     print('Testing poses shape...', poses[i_test].shape)
        #     with torch.no_grad():
        #         render_path(torch.Tensor(poses[i_test]).to(device), torch.Tensor(times[i_test]).to(device),
        #                     hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test].cpu().numpy(),  gt_depths=depths[i_test].cpu().numpy(), savedir=testsavedir)
        #     print('Saved test set')

        global_step += 1


if __name__=='__main__':
    log("MAIN", "Parsing arguments...")
    parser = config_parser()
    args = parser.parse_args()
    setattr(args, 'device', device)
    # print(args.depth_guided_sampling, args.use_depth)
    log("MAIN", f"Device detected: {device}")

    log("MAIN", "Setting up torch tensor type...")    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Train
    log("MAIN", "Training...")
    train(args)

    