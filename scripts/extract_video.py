# Extract video to images with given fps

import imageio 
import os
VIDEO_PATH = './logs/dino_w_depth_low_std_weight_2/renderonly_test_399999/depths.mp4'

vid = imageio.get_reader(VIDEO_PATH,  'ffmpeg' )
fps = vid.get_meta_data()['fps']

 
print(fps)

# Video to images
for i, im in enumerate(vid):
    # Create directory if not exists
    if not os.path.exists(f'./logs/dino_w_depth_low_std_weight_2/renderonly_test_399999/depths'):
        os.makedirs(f'./logs/dino_w_depth_low_std_weight_2/renderonly_test_399999/depths')
        
    imageio.imwrite(f'./logs/dino_w_depth_low_std_weight_2/renderonly_test_399999/depths/{i}.png', im)
     