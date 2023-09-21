import sys
import cv2
sys.path.append("./utils")
import numpy as np
from misc import gamma, gamma_reverse
import matplotlib.pyplot as plt


class ImageColorCorrection:
    def __init__(self, cct_ccm_dict, ccm_cs):
        """
        Args:
            cct_ccm_dict:
            method: cc:colorchecker  wp:white paper  grey world:grey world
            white_balance_method:
        """
        self.__rgb_gain = np.array([1, 1, 1])
        # self.__ccm = None
        self.__ccm_cs = ccm_cs

        self.__ccm = cct_ccm_dict['ccm']
        self.__rgb_gain = cct_ccm_dict['rgb_gain']


    def apply_wb_and_ccm(self, image, image_color_space):
        image_temp = image.copy()
        print(self.__rgb_gain)
        image_temp = image_temp * self.__rgb_gain[None, None] 
        image_temp = np.clip(image_temp, 0, 1)

        # image_wb_tmp = image_temp.copy()

        if image_color_space.lower() == "srgb" and self.__ccm_cs.lower() == "linear":
            image_temp = gamma(image_temp)
        elif image_color_space.lower() == "linear" and self.__ccm_cs.lower() == "srgb":
            image_temp = gamma_reverse(image_temp)

        # apply ccm
        print(self.__ccm.shape)
        self.__ccm = self.__ccm.T
        # self.__ccm = self.__ccm
        # self.__ccm = self.__ccm.numpy()
        if self.__ccm.shape[0] == 4:
            # image_temp = np.einsum('ic, hwc->hwi', self.__ccm[0:3].T, image_temp) + self.__ccm[3][None, None]
            image_temp = np.einsum('ic, hwc->hwi', self.__ccm[0:3].T, image_temp) + self.__ccm[3][None, None]
        else:
            image_temp = np.einsum('ic, hwc->hwi', self.__ccm.T, image_temp)
        image_temp = np.clip(image_temp, 0, 1)

        if image_color_space.lower() == "srgb" and self.__ccm_cs.lower() == "linear":
            image_temp = gamma_reverse(image_temp)
        elif image_color_space.lower() == "linear" and self.__ccm_cs.lower() == "srgb":
            image_temp = gamma(image_temp)

        return image_temp

    def correctImage(self, image, image_color_space):
        # self.ccm_interpolation(self.__cct)
        corrected_image = self.apply_wb_and_ccm(image, image_color_space)
        return corrected_image
