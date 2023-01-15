import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch

# Calculate diff between two depth maps (pred and target)

GT_DEPTH_PATH = './logs/toss_wo_depth/renderall_test_399999/depth_gt/005.png'
PRED_DEPTH_PATH = './logs/toss_w_depth_MSE_DGS_0.001_0.01/renderonly_test_399999/depth/005.png'
# PRED_DEPTH_PATH = './logs/toss_w_depth_GNLL_DGS_0.001_0.01/renderonly_test_399999/depth/005.png'
# PRED_DEPTH_PATH = './logs/toss_w_depth_MSE__noDGS_0.001/renderonly_test_399999/depth/005.png'


gt_depth = cv2.imread(GT_DEPTH_PATH, cv2.IMREAD_ANYDEPTH)
pred_depth = cv2.imread(PRED_DEPTH_PATH, cv2.IMREAD_ANYDEPTH)

# set pred_depth to 0 where gt_depth is 0 (mask)
pred_depth[gt_depth == 0] = 0
 

# # Display depth maps
plt.imshow(gt_depth)
plt.show()
plt.imshow(pred_depth)
plt.show()
    
# Calculate diff use Color Jet map
diff = np.abs(gt_depth - pred_depth)
diff = diff / np.max(diff)
diff = (diff * 255).astype(np.uint8)
diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
# show diff# cv 
cv2.imwrite('diff.png', diff)