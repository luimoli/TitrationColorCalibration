from colour import colour
import numpy as np
import cv2

from utils import smv_colour
from numpy import genfromtxt

# my_data = 
# gt_xyz = genfromtxt(r"D:\Code\CCM\TitrationColorCalibration\data\gt_xyz_xrite.csv", delimiter=',')
gt_xyz = np.float32(np.loadtxt("./colorchecker_gt/gt_xyz_xrite.csv", delimiter=','))
gt_rgb = smv_colour.XYZ2RGB(gt_xyz, 'bt709')
gt_lab = smv_colour.XYZ2Lab(gt_xyz)
# res = gt_rgb * 255.
# print(gt_rgb)

# res = smv_colour.Lab2XYZ(np.array([96,-0.06,0.07]))
res = smv_colour.xyY2XYZ(np.array([0.3469,0.3608,91.31]))
# print(res)
res = smv_colour.XYZ2Lab(res, 100, 'd65')
# print(res)

# white_patch_gamma = cv2.imread(r'img_corrected_tmp_gamma.png')
# white_patch = cv2.imread(r'img_corrected_tmp.png')
# mean_w_gamma = np.mean(white_patch_gamma, (0,1))
# mean_w = np.mean(white_patch, (0,1))
# print(mean_w_gamma)
# print(mean_w)

mean_w = np.array([216.982, 156.442, 36.3367])
w_xyz = smv_colour.RGB2XYZ((mean_w/255.)**2.2, 'bt709')
w_xyy = smv_colour.XYZ2xyY(w_xyz)
# print(w_xyz)
# print(w_xyy)


# cc_mean_value = np.load('calib\cc_mean_lighthouse_D65_0221.npy')
# cv2.imwrite('./test.png', cc_mean_value[:,::-1])
cc_correct_value = np.load('./cc_correct_rgb.npy')
cc_xyz = smv_colour.RGB2XYZ(cc_correct_value, 'bt709')
cc_xyy = smv_colour.XYZ2xyY(cc_xyz)
# print(cc_xyz)
# print(cc_xyy)

gt = np.float32(np.loadtxt("./colorchecker_gt/gt_xyz_xrite.csv", delimiter=','))
res = gt[:, 1] / cc_xyz[:, :, 1]
print('.')














# img = cv2.imread('./2022-05-26_17-10-18_408.png')
# cv2.imwrite('./2022-05-26_17-10-18_408_gamma.png', ((img/255.) ** (1/2.2)) * 255.)


# img = cv2.imread(r'data\tmp\626b24cc-5994-4ff0-bc83-419143ea9b9d.jpg')
# cv2.imwrite(r'data\tmp\626b24cc-5994-4ff0-bc83-419143ea9b9d_linear.jpg', ((img/255.) ** (2.2)) * 255.)


# a = np.arange(27).reshape((3,3,3))
# print(np.mean(a,(0,1)).shape)
