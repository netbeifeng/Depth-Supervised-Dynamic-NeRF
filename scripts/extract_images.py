
import os
import imageio
import numpy as np
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

DATA_DIR = "./data/copier"

COLOR_DIR = os.path.join(DATA_DIR, "color")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")

# Get all files in COLOR_DIR 
color_files = os.listdir(COLOR_DIR)
depth_files = os.listdir(DEPTH_DIR)

# Filter only npy files
color_files = [f for f in color_files if f.endswith(".npy")]
depth_files = [f for f in depth_files if f.endswith(".npy")]

# Sort the files
color_files.sort()
depth_files.sort()

# All images are store in NPY format
# For each file load the image and save it as PNG

all_imgs = []
all_depths = []

# for [color_img, depth_img] in color_files, depth_files:
for color_img, depth_img in zip(color_files, depth_files):
    img = np.load(os.path.join(COLOR_DIR, color_img))
    dep = np.load(os.path.join(DEPTH_DIR, depth_img))
    
    # If not exist create the directory
    if not os.path.exists(os.path.join(COLOR_DIR, "png")):
        os.makedirs(os.path.join(COLOR_DIR, "png"))
    if not os.path.exists(os.path.join(DEPTH_DIR, "png")):
        os.makedirs(os.path.join(DEPTH_DIR, "png"))
    
    imageio.imwrite(os.path.join(COLOR_DIR, "png", color_img.replace("npy", "png")), to8b(img))
    imageio.imwrite(os.path.join(DEPTH_DIR, "png", depth_img.replace("npy", "png")), to8b(dep/np.max(dep)))
    all_imgs.append(img)
    all_depths.append(dep)
    
# Stack all images into a single array
all_imgs = np.stack(all_imgs, axis=0)
all_depths = np.stack(all_depths, axis=0)

# Save to video file
imageio.mimsave(os.path.join(COLOR_DIR, "video.mp4"), to8b(all_imgs), fps=10, quality=8)
imageio.mimsave(os.path.join(DEPTH_DIR, "video.mp4"), to8b(all_depths/np.max(all_depths)), fps=10, quality=8)
        