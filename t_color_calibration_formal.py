from glob import glob
import os
from statistics import mode
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from utils.ImageColorCalibration import ImageColorCalibration
from utils.mcc_detect_color_checker import detect_colorchecker
from utils.evaluate_result import evaluate
from t_calculate_center_cc import get_cc_mean_value

# photo_path = r"image\BK19862AAK00060_230213165459.png"
# image = cv2.imread(photo_path)[:,:,::-1] / 255.
# _, _, cc_mean_value, _ = detect_colorchecker(image)


cc_mean_value = np.load('calib\cc_mean_lighthouse_D65_0221.npy')


model = ImageColorCalibration(src_for_ccm=cc_mean_value, colorchecker_gt_mode=4)
model.setCCM_METHOD('minimize')
model.setColorSpace('linear')
model.setCCM_TYPE('3x3')
model.setCCM_RowSum1(False)
model.run()
print(model.getCCM())
model.save('./calib/TitrationCamera_xrite_ccm_3x3_lighthouse_D65_0221_test.npy') 



calibratedImage = model.infer(cc_mean_value, image_color_space='linear')
deltaC, deltaE00, _ = evaluate(calibratedImage, model.ideal_lab, 'linear', 'deltaC', True)
# img_with_gt = np.clip(img_with_gt, 0, 1)
print('deltaC00:', deltaC)
print('deltaC00 mean:', deltaC.mean())
print('deltaE00:', deltaE00)
print('deltaE00 mean:', deltaE00.mean())


# calibratedImage = model.infer(image, image_color_space='linear')
# deltaC, deltaE00, img_with_gt = evaluate(calibratedImage, model.ideal_lab, 'linear', 'deltaC')
# img_with_gt = np.clip(img_with_gt, 0, 1)
# print('deltaC00, deltaE00 - mean: ', deltaC.mean(), deltaE00.mean())
# print('deltaC00, deltaE00 - max: ', deltaC.max(), deltaE00.max())
# cv2.imwrite("img_ccm.png", calibratedImage[..., ::-1]**(1/2.2)*255.)
# cv2.imwrite('img_ccm_gt.png', img_with_gt[...,::-1]**(1/2.2)*255.)



# for photo_path in glob(r'./data/Xcamera/*.png'):
#     image = cv2.imread(photo_path, -1)[..., ::-1] / 255.
#     _, _, cc_mean_value, _ = detect_colorchecker(image)
#     model = ImageColorCalibration(src_for_ccm=cc_mean_value, colorchecker_gt_mode=1)
#     model.setCCM_METHOD('minimize')
#     model.setColorSpace('linear')
#     model.setCCM_TYPE('3x4')
#     model.setCCM_RowSum1(False)
#     model.run()
#     print(model.getCCM())
#     model.save('./data/Xcamera_calib_xrite_3x4.npy')

#     calibratedImage = model.infer(image, image_color_space='linear')
#     deltaC, deltaE00, img_with_gt = evaluate(calibratedImage, model.ideal_lab, 'linear', 'deltaC')
#     img_with_gt = np.clip(img_with_gt, 0, 1)
#     print('deltaC00, deltaE00: ', deltaC.mean(), deltaE00.mean())
#     cv2.imwrite(photo_path[:-4]+'_ccm.jpg', calibratedImage[..., ::-1]**(1/2.2)*255.)
#     cv2.imwrite(photo_path[:-4]+'_ccm_gt.jpg', img_with_gt[...,::-1]**(1/2.2)*255.)
#     print('....................')
