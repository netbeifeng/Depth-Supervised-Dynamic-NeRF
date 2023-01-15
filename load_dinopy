import numpy as np
import os
import torch
import json

trans_t = lambda t : torch.Tensor([
    [1,0,0,t[0]],
    [0,1,0,t[1]],
    [0,0,1,t[2]],
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

def pose_spherical(theta, phi, t):
    c2w = trans_t(t)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_dino_data(basedir):
    COLOR_PATH = os.path.join(basedir, 'color')
    DEPTH_PATH = os.path.join(basedir, 'depth')
    POSE_PATH = os.path.join(basedir, 'cams/extrinsics.npy')
    INTRINSICS_PATH = os.path.join(basedir, 'cams/intrinsics.npy')
    SPLITS_PATH = os.path.join(basedir, 'splits.json')
    # Get the number of file color folder & depth folder
    color_files = os.listdir(COLOR_PATH)
    depth_files = os.listdir(DEPTH_PATH)
    # Filter the file name with .npy
    color_files = [f for f in color_files if f.endswith('.npy')]
    depth_files = [f for f in depth_files if f.endswith('.npy')]
    
    color_files.sort() 
    depth_files.sort()
    
    # Take 0-62 from color_files
    color_files = color_files[0:62]
    depth_files = depth_files[0:62]
    
    assert len(color_files) == len(depth_files) == 62
    
    N = len(color_files)
    # N = 101
    
    all_imgs = []
    all_depths = []
    all_times = []
    # From 0 to N, step = 1
    # all_indices = np.arange(0, N, 1)
    # Shuffle all_indices
    # np.random.shuffle(all_indices)
    # Divide all_indices to 3 arrays (train, val, test) with ratio 0.8, 0.1, 0.1
    # i_split = np.split(all_indices, [int(.8*N), int(.9*N)])
    
    # Read splits.json
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    
    i_split = sorted(np.array(splits['train'])), sorted(np.array(splits['val'])), sorted(np.array(splits['test']))
    
    # Randomly choose 10 indices from all_indices
    # test = np.random.choice(all_indices, 10, replace=False) 
    
    # i_split = [[0, 121], [], test]
    print("TRAIN SIZE: ", len(i_split[0]))
    print("VAL SIZE: ", len(i_split[1]))
    print("TEST SIZE: ", len(i_split[2]))
    
    # for each file in color folder
    for i in range(N):
        # Load image with numpy
        img = np.load(os.path.join(COLOR_PATH, color_files[i])).astype(np.float32)
        
        # Load depth with numpy
        depth = np.load(os.path.join(DEPTH_PATH, depth_files[i])).astype(np.float32)
        
        all_imgs.append(img)
        all_depths.append(depth)
        
        # time range from 0.0 to 1.0 with two decimal 
        all_times.append(i/60)
    
    # change list to numpy array
    all_imgs = np.array(all_imgs)
    all_depths = np.array(all_depths)
    # Add 1 channel to all_depths at the end (channel = 1)
    all_depths = np.expand_dims(all_depths, -1)
    
    # Load pose
    all_poses = np.load(POSE_PATH)[0:N]
    intrinsics = np.load(INTRINSICS_PATH)
    
    # From intrinsics, get fx, fy, cx, cy
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    
    # focal = (fx + fy) / 2
    
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    H = all_imgs.shape[1]
    W = all_imgs.shape[2]
    
    # if exists test_poses.npy, load it
    if os.path.exists(os.path.join(basedir, 'test_poses.npy')):
        render_poses = np.load(os.path.join(basedir, 'test_poses.npy'))
    else:
        # TODO: find a correct angle to generate poses
        raise NotImplementedError
        # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])
    
    return all_imgs, all_depths, all_poses, all_times, render_poses, render_times, [H, W, fx, fy, cx, cy], i_split
    
if __name__ == '__main__':
    basedir = './data/dino/'
    all_imgs, all_depths, all_poses, all_times, render_poses, render_times, [H, W, fx, fy, cx, cy], i_split = load_dino_data(basedir)
    
    assert all_imgs.shape[0] == all_depths.shape[0]
    assert all_imgs.shape[0] == all_poses.shape[0]
    
    print(i_split)
    print(f"OK, LENGTH: {all_imgs.shape[0]}")
    print(all_times)
    print(all_imgs[i_split[0]].shape, all_depths[i_split[0]].shape, all_poses[i_split[0]].shape)    
    print(all_imgs[i_split[1]].shape, all_depths[i_split[1]].shape, all_poses[i_split[1]].shape)
    print(all_imgs[i_split[2]].shape, all_depths[i_split[2]].shape, all_poses[i_split[2]].shape)
    
    # Sample all_times every 3 elements
    
    # ALL = np.arange(0, 61, 1) # 0 to 100, step = 1
    # # assert len(ALL) == 100
    
    # # TRAIN = ALL[::3]
    
    # # Take 6 elements from ALL randomly without repetition
    # TEST = np.random.choice(ALL, 6, replace=False)
    # VAL = TEST
    # TRAIN = np.setdiff1d(ALL, TEST)
    
    
    # # If 1.0 is not in TRAIN , Append one element to TRAIN 1.0
    # # if 100 not in TRAIN:
    # #     TRAIN = np.append(TRAIN, 100)
    # # print(TRAIN, len(TRAIN))
    
    # # Get other elements except TRAIN in ALL
    # # How to subtract 2 arrays in numpy?
    # # rest = np.setdiff1d(ALL, TRAIN)
    
    # # VAL = rest[::2]
    # # print(VAL, len(VAL))
    
    # # TEST = np.setdiff1d(rest, VAL)
    # # print(TEST, len(TEST))
    
    # # # Save to splits.json
    # splits = {'train': TRAIN.tolist(), 'val': VAL.tolist(), 'test': TEST.tolist()}
    
    # # # FileNotFoundError: [Errno 2] No such file or directory: './data/dino/splits.json'
    # # # How to fix?
     
    # with open(os.path.join(basedir, 'splits.json'), 'w') as f:
    #     json.dump(splits, f)
         