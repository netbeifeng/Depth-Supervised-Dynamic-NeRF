import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_toss_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)


    imgs = []
    depths = []
    poses = []
    times = []
    counts = []
    # if s=='train' or testskip==0:
    #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
    # else:
        
    for t, frame in enumerate(meta['frames']):
        fname = os.path.join(basedir, frame['file_path'])
        dname = os.path.join(basedir, frame['depth_frame'])
        imgs.append(imageio.imread(fname))
        depths.append(imageio.imread(dname))
        poses.append(np.array(frame['transform_matrix']))
        cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'])-1)
        times.append(cur_time)
        counts.append(t)

    assert times[0] == 0, "Time must start at 0"

    imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
    
    depths = (np.array(depths)/ 1000).astype(np.float32)  # in meters
    # Add a channel dimension
    depths = np.expand_dims(depths, axis=3)
    
    poses = np.array(poses).astype(np.float32)
    # times = np.array(times).astype(np.float32)

    i_indices = np.arange(len(imgs))
    # i_val = [49, 36, 33,  1, 17, 52]
    # i_val = [ 50, 51, 52, 53, 54]
    # i_test = [49, 36, 33,  1, 17, 52]
    # i_test = [ 50, 51, 52, 53, 54]
    i_val = [17]
    i_test = [17]
    
    # i_train = i_indices - i_test
    i_train = [i for i in i_indices if i not in i_test]
    
    i_split = [i_train, i_val, i_test]
    

    H, W = int(meta['h']), int(meta['w'])
    
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)

    fl_x = float(meta['fl_x'])
    fl_y = float(meta['fl_y'])
    
    cx = float(meta['cx'])
    cy = float(meta['cy'])
    
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, depths, poses, times, render_poses, render_times, [H, W, fl_x, fl_y, cx, cy], i_split


if __name__ == "__main__":
    # Main function
    
    imgs, depths, poses, times, render_poses, render_times, [H, W, fl_x, fl_y, cx, cy], i_split = load_toss_data('data/dynamic_toss', half_res=False, testskip=1)
    
    print(times)
     
    print(imgs[i_split[0]].shape, depths[i_split[0]].shape, poses[i_split[0]].shape)    
    print(imgs[i_split[1]].shape, depths[i_split[1]].shape, poses[i_split[1]].shape)
    print(imgs[i_split[2]].shape, depths[i_split[2]].shape, poses[i_split[2]].shape)
    pass