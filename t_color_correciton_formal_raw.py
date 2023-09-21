import cv2.cv2 as cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.evaluate_result import evaluate
from utils.ImageColorCorrection import ImageColorCorrection
from utils import smv_colour


def correction_test(img_path, img_type, calib_path, img_is_cc=False, result_save_follow=True):
    if img_type == 'uint8':
        image = cv2.imread(img_path)[..., ::-1] / 255.
    elif img_type == 'uint16':
        image = cv2.imread(img_path, -1)[..., ::-1] / 65535.
    elif img_type == 'raw':
        image = np.fromfile(img_path, dtype=np.uint8).reshape(2048, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)[..., ::-1] / 255.
    elif img_type == 'float32':
        image = img_path
    else:
        assert('img_type error!')
    
    calib_dict = np.load(calib_path, allow_pickle=True).item()

    image_color_correction = ImageColorCorrection(calib_dict, "linear")
    corrected_image = image_color_correction.correctImage(image, "linear")

    if result_save_follow:
        if img_type == 'uint16':
            cv2.imwrite(img_path[:-4] + '_corrected_gamma.png', corrected_image[...,::-1]** (1/2.2) * 65535.)
            cv2.imwrite(img_path[:-4] + '_corrected.png', corrected_image[...,::-1] * 65535.)
        else:
            cv2.imwrite(img_path[:-4] + '_corrected_gamma.png', corrected_image[...,::-1]** (1/2.2) * 255.)
            cv2.imwrite(img_path[:-4] + '_corrected.png', corrected_image[...,::-1] * 255.)
    else: 
        # save tmp result
        cv2.imwrite('img_corrected_tmp_gamma.png',corrected_image[...,::-1]** (1/2.2) * 255.)
        cv2.imwrite('img_corrected_tmp.png',corrected_image[...,::-1] * 255.)

    if img_is_cc:
        # set cc_gt_lab #TODO
        # ideal_lab_3 = np.float32(np.loadtxt("./data/real_lab_xrite.csv", delimiter=','))
        made_gt_xyz = np.float32(np.loadtxt("./colorchecker_gt/gt_xyz_xrite.csv", delimiter=',')) / 2.7085751834410250793998466761581
        #2.74769470058882346405
        made_gt_lab = smv_colour.XYZ2Lab(made_gt_xyz/100)

        deltaC, deltaE00, img_with_gt = evaluate(corrected_image, made_gt_lab, 'linear', 'deltaC')
        img_with_gt = np.clip(img_with_gt, 0, 1)
        print('deltaC00, deltaE00 - mean: ', deltaC.mean(), deltaE00.mean())
        print('deltaC00, deltaE00 - max: ', deltaC.max(), deltaE00.max())
        cv2.imwrite('img_ccm_gt_lighthouse_d65.png', img_with_gt[...,::-1]**(1/2.2)*255.)




if __name__ == '__main__':
    img_path = r"test.png"
    img_type = 'uint8'
    calib_path = 'calib\TitrationCamera_xrite_ccm_3x3_lighthouse_D65_0221.npy'

    img_path = np.load(r'calib\cc_mean_lighthouse_D65_0221.npy')
    img_type = 'float32'
    correction_test(img_path, img_type, calib_path, img_is_cc=False, result_save_follow=False)

    # img_path = r'image\24cc_lighthouse_D65_correct\BK19862AAK00060_230214155726.png'
    # image = cv2.imread(img_path)[..., ::-1] / 255.
    
    # ccm_save_path = 'data_calib\TitrationCamera_xrite_ccm_3x3_lighthouse_D65_0221.npy'
    # cct_ccm_dict = np.load(ccm_save_path, allow_pickle=True).item()

    # image_color_correction = ImageColorCorrection(cct_ccm_dict, "linear")
    # # image_color_correction.setMethod("cc")
    # # image_color_correction.doWhiteBalance(wb_image=image_wb)
    # corrected_image = image_color_correction.correctImage(image, "linear")

    # # ideal_lab_3 = np.float32(np.loadtxt("./data/real_lab_xrite.csv", delimiter=','))
    # made_gt_xyz = np.float32(np.loadtxt("./data/gt_xyz_xrite.csv", delimiter=',')) / 2.74769470058882346405
    # made_gt_lab = smv_colour.XYZ2Lab(made_gt_xyz/100)

    # deltaC, deltaE00, img_with_gt = evaluate(corrected_image, made_gt_lab, 'linear', 'deltaC')
    # img_with_gt = np.clip(img_with_gt, 0, 1)
    # print('deltaC00, deltaE00 - mean: ', deltaC.mean(), deltaE00.mean())
    # print('deltaC00, deltaE00 - max: ', deltaC.max(), deltaE00.max())


    # cv2.imwrite('img_corrected_gamma_lighthouse_d65.png',corrected_image[...,::-1]** (1/2.2) * 255.)
    # cv2.imwrite('img_corrected_lighthouse_d65.png',corrected_image[...,::-1] * 255.)
    # # cv2.imwrite('img_corrected_wb.png',corrected_image_wb[...,::-1] * 255.)
    # cv2.imwrite('img_ccm_gt_lighthouse_d65.png', img_with_gt[...,::-1]**(1/2.2)*255.)

    # # cv2.imwrite(img_path[:-4] + '_corrected_gamma.png', corrected_image[...,::-1]** (1/2.2) * 255.)
    # # cv2.imwrite(img_path[:-4] + '_corrected.png', corrected_image[...,::-1] * 255.)



    # plt.figure()
    # plt.imshow(image)
    # plt.figure()
    # plt.imshow(corrected_image ** (1/2.2))
    # plt.show()