import glob
import os
import imageio
import numpy as np
# Combine images to video

COLOR_PATH = 'data/dino/color/png'
DEPTH_PATH = 'data/dino/depth/png'

# Get all color and depth images
color_images = sorted(glob.glob(os.path.join(COLOR_PATH, '*.png')))
# sort it by name (1.png, 2.png, 3.png, ...) as numbers
depth_images = sorted(glob.glob(os.path.join(DEPTH_PATH, '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
 
# print(depth_images)
color_images = color_images[1:62]
depth_images = depth_images[1:62]

# Read images by imageio
color_images = [imageio.imread(img) for img in color_images]
depth_images = [imageio.imread(img) for img in depth_images]

# Stack images to numpy array
color_images = np.stack(color_images, axis=0)
depth_images = np.stack(depth_images, axis=0)

# Save images to video
imageio.mimsave('color_cut.mp4', color_images, fps=10, quality=8)
imageio.mimsave('depth_cut.mp4', depth_images, fps=10, quality=8)