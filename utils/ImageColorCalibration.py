import os
import numpy as np
import scipy.optimize as optimize

from utils import smv_colour
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils.misc import gamma, gamma_reverse, rgb2lab, lab2rgb

class ImageColorCalibration:
    def __init__(self, src_for_ccm, colorchecker_gt_mode):
        """_summary_

        Args:
            src_for_ccm (arr): src rgb-colorchecker-data(eg.with shape [24,3])
            colorchecker_gt_mode (int): 1: xrite, 2: imatest, 3: 3nh
        """
        self.cc_mean_value = src_for_ccm
        self.rgb_gain = np.array([1, 1, 1])
        self.illumination_gain = 1
        self.cct = None

        self.ideal_lab = None
        self.ideal_linear_rgb = None
        self.setColorChecker_Lab(mode=colorchecker_gt_mode)

        self.__colorspace = 'linear'
        self.__ccm = np.eye(3)
        self.__ccm_method = 'minimize'
        self.__ccm_weight = np.ones((self.ideal_lab.shape[0]))
        self.__ccm_masks = self.checkExposure(src_for_ccm)
        self.__ccm_type = '3x3'
        self.__ccm_metric = 'CIE2000'
        self.__ccm_rowsum1 = True
        self.__ccm_constrain = 0

    def checkExposure(self, arr, thresh=1):
        masks = arr.max(axis=1) < thresh
        if masks.sum() < self.ideal_lab.shape[0]:
            raise ValueError(f'over exposure! sum of ccm_masks is {masks.sum()}')
        return masks

    def setColorChecker_Lab(self, mode):
        """set groundtruth of colorchecker.
        Args:
            mode (int): 1: xrite, 2: imatest, 3: 3nh
        """
        ideal_lab_1 = np.float32(np.loadtxt("./colorchecker_gt/real_lab_xrite.csv", delimiter=',')) # from x-rite
        ideal_lab_2 = np.float32(np.loadtxt("./colorchecker_gt/real_lab_imatest.csv", delimiter=','))  # from imatest
        ideal_lab_3 = np.float32(np.loadtxt("./colorchecker_gt/real_lab_d50_3ns.csv", delimiter=','))  # from 3nh
        made_gt_xyz = np.float32(np.loadtxt("./colorchecker_gt/gt_xyz_xrite.csv", delimiter=',')) / 2.7085751834410250793998466761581
        made_gt_lab = smv_colour.XYZ2Lab(made_gt_xyz/100)
        ideal_lab_dic = {1:ideal_lab_1, 2: ideal_lab_2, 3: ideal_lab_3, 4:made_gt_lab}
        self.ideal_lab = ideal_lab_dic[mode]
        self.ideal_linear_rgb = lab2rgb(self.ideal_lab)

    def setColorSpace(self, space):
        """set the colorspace when calculating CCM.
        Args:
            space (str): ['srgb', 'linear']
        """
        assert space == 'srgb' or space =='linear'
        self.__colorspace = space

    def setCCM_WEIGHT(self, arr):
        """set each color's weight in CCM.
        Args:
            arr (array): the weight for colorcheckers' patches of CCM.
        """
        assert arr.shape[0] == self.ideal_lab.shape[0] and arr.ndim == 1
        self.__ccm_weight = arr

    def getCCM_WEIGHT(self):
        return self.__ccm_weight

    def setCCM_MASKS(self, arr):
        """set masks of CCM calculation.
        Args:
            arr (array): shape:[N], corresponding to the number of colorchecke's patches.
        """
        assert arr.shape[0] == self.ideal_lab.shape[0] and arr.ndim == 1
        self.__ccm_masks = arr

    def getCCM_MASKS(self):
        return self.__ccm_masks


    def setCCM_TYPE(self, type):
        """set the shape of CCM.
        Args:
            type (str): ['3x3', '3x4']
        """
        assert type == '3x3' or type == '3x4'
        self.__ccm_type = type

    def setCCM_METRIC(self, metric):
        """the metric of CCM optimization.
        Args:
            metric (str): ['CIE2000', 'CIE1976]
        """
        assert metric == 'CIE2000' or metric == 'CIE1976'
        self.__ccm_metric = metric

    def setCCM_METHOD(self, method):
        """the method of CCM calculation.
        Args:
            method (str): ['minimize', 'polynominal']
        """
        assert method == 'minimize' or method == 'polynominal'
        self.__ccm_method = method


    def setCCM_RowSum1(self, boolvalue):
        """whether to mantain white balance constrain: the sum of CCM's row is 1.
        Args:
            boolvalue (bool): True or False
        """
        self.__ccm_rowsum1 = boolvalue

    def setCCM_Constrain(self, value):
        """set diagonal value constrain.
        Args:
            value (float): constrain the diagonal of CCM to be less than a value when calculating CCM.
        """
        assert value > 0
        self.__ccm_constrain = value

    def getCCM(self):
        return self.__ccm


    def compute_rgb_gain_from_colorchecker(self, mean_value):
        # assert image.max() <= 1, "image range should be in [0, 1]"
        gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
        rgb_gain = gain[0:3].mean(axis=0)
        return rgb_gain


    def compute_cct_from_white_point(self, white_point):
        xyz = smv_colour.RGB2XYZ(np.float32(white_point), "bt709")
        xyY = smv_colour.XYZ2xyY(xyz)
        cct = smv_colour.xy2CCT(xyY[0:2])
        return cct


    def run(self):
        self.rgb_gain = self.compute_rgb_gain_from_colorchecker(self.cc_mean_value)
        self.cct = self.compute_cct_from_white_point( 1 / self.rgb_gain)

        cc_wb_mean_value = self.cc_mean_value * self.rgb_gain[None]
        self.illumination_gain = (self.ideal_linear_rgb[18:21] / cc_wb_mean_value[18:21]).mean()
        cc_wb_ill_mean_value = self.illumination_gain * cc_wb_mean_value

        if self.__ccm_method == "minimize":
            if self.__colorspace.lower() == "srgb":
                cc_wb_ill_mean_value = gamma(cc_wb_ill_mean_value)
            if self.__ccm_type == '3x4':
                cc_wb_ill_mean_value = np.concatenate((cc_wb_ill_mean_value.copy(), np.ones((cc_wb_ill_mean_value.shape[0], 1))), axis=-1)
            self.__ccm = self.ccm_calculate(cc_wb_ill_mean_value)
        print("self.rgb_gain:", self.rgb_gain)
        print("self.illumination_gain:", self.illumination_gain)



    def apply_ccm(self, img, ccm):
        assert ccm.shape[0] == 3
        if img.ndim == 3:
            img_ccm = np.einsum('hwi,ji->hwj', img, ccm)
        elif img.ndim == 2:
            img_ccm = np.einsum('hi,ji->hj', img, ccm)
        else:
            raise ValueError(img.shape)
        return img_ccm

    def ccm_calculate(self, ccm_rgb_data):
        """[calculate the color correction matrix]
        Args:
            rgb_data ([N*3]): [the RGB data of color_checker]
        Returns:
            [array]: [CCM with shape: 3*3 or 3*4]
        """

        mask_index = np.argwhere(self.__ccm_masks)
        mask_index = np.squeeze(mask_index, -1)
        assert mask_index.ndim == 1
        rgb_data = ccm_rgb_data[mask_index, :].copy()
        ideal_lab = self.ideal_lab[mask_index, :].copy()
        ideal_linear_rgb = lab2rgb(ideal_lab)


        if self.__ccm_method.lower() == 'minimize':
            if self.__ccm_type == '3x3':
                if self.__ccm_rowsum1:
                    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                                            [x[2],1-x[2]-x[3],x[3]],
                                            [x[4],x[5],1-x[4]-x[5]]])
                    x0 = np.zeros((6))

                else:
                    x2ccm=lambda x : np.array([[x[0], x[1],x[2]],
                                            [x[3], x[4], x[5]],
                                            [x[6],x[7],x[8]]])
                    # x0 = np.zeros((9))
                    x0 = np.array([[ideal_linear_rgb[..., 0].mean() / rgb_data[..., 0].mean(), 0, 0],
                                   [0, ideal_linear_rgb[..., 1].mean() / rgb_data[..., 1].mean(), 0],
                                   [0, 0, ideal_linear_rgb[..., 2].mean() / rgb_data[..., 2].mean()]])

            elif self.__ccm_type == '3x4':
                if self.__ccm_rowsum1:
                    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1], x[6]],
                                               [x[2],1-x[2]-x[3],x[3], x[7]],
                                               [x[4],x[5],1-x[4]-x[5], x[8]]])
                    x0 = np.zeros((9))
                else:
                    x2ccm=lambda x : np.array([[x[0], x[1], x[2], x[3]],
                                               [x[4], x[5], x[6], x[7]],
                                               [x[8], x[9], x[10], x[11]]])
                    x0 = np.array([[ideal_linear_rgb[..., 0].mean() / rgb_data[..., 0].mean(), 0, 0, 0],
                                   [0, ideal_linear_rgb[..., 1].mean() / rgb_data[..., 1].mean(), 0, 0],
                                   [0, 0, ideal_linear_rgb[..., 2].mean() / rgb_data[..., 2].mean(), 0]])

        elif self.__ccm_method.lower() == 'polynominal':
            x2ccm=lambda x : np.array([[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]],
                                [x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17]],
                                [x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26]]])
            x0 = np.zeros((27))

        if self.__colorspace.lower() == "linear":
            f_lab=lambda x : rgb2lab(self.apply_ccm(rgb_data, x2ccm(x)))
        elif self.__colorspace.lower() == "srgb":
            f_lab = lambda x: rgb2lab(gamma_reverse(self.apply_ccm(rgb_data, x2ccm(x)), colorspace='sRGB'))

        if self.__ccm_metric == 'CIE1976':
            f_error=lambda x : f_lab(x)- ideal_lab
            f_DeltaE=lambda x : (np.sqrt((f_error(x)**2).sum(axis=1)) * self.__ccm_weight).mean()
        elif self.__ccm_metric == 'CIE2000':
            f_DeltaE=lambda x : ((delta_E_CIE2000(f_lab(x), ideal_lab) * self.__ccm_weight)**2).mean()

        func=lambda x : print('deltaE_00 = ',f_DeltaE(x))

        if self.__ccm_constrain:
            if self.__ccm_type == '3x3':
                if self.__ccm_rowsum1:
                    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] -1 + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: x[2] + x[3] -1 + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: x[4] + x[5] -1 + self.__ccm_constrain})
                else:
                    cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: -x[4] + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: -x[8] + self.__ccm_constrain})

                result=optimize.minimize(f_DeltaE, x0, method='SLSQP', constraints=cons)

            elif self.__ccm_type == '3x4':
                raise ValueError('currently not supported constrain value in 3*4 CCM.')

        else:
            # result=optimize.minimize(f_DeltaE, x0, callback=func, method='Powell')
            result=optimize.minimize(f_DeltaE, x0, method='Powell')

        print('minimize average deltaE00: ', result.fun)
        return x2ccm(result.x)



    def infer(self,
              img,
              image_color_space,
              white_balance=True,
              illumination_gain=True,
              ccm_correction=True):
        """infer the img using the calculated CCM.
            The output img's colorspace keeps with parameter 'image_color_space'.  

        Args:
            img (arr): typically with shape[H,W,3].
            image_color_space (str): ['srgb', 'linear']
            white_balance (bool, optional): . Defaults to True.
            illumination_gain (bool, optional): . Defaults to True.
            ccm_correction (bool, optional): . Defaults to True.

        Returns:
            arr: calibrated image.
        """
        # print('rgb_gain:  ',self.rgb_gain)
        image = img.copy()
        if white_balance:
            assert image.max() <= 1
            image = image * self.rgb_gain[None, None]
            image = np.clip(image, 0, 1)

        if illumination_gain:
            image = image * self.illumination_gain
            image = np.clip(image, 0, 1)

        if ccm_correction:
            if self.__ccm_type == '3x4':
                image = np.concatenate((image.copy(), np.ones((image.shape[0], image.shape[1], 1))), axis=-1)

            if image_color_space.lower() == self.__colorspace.lower():
                image = self.apply_ccm(image, self.__ccm)

            elif image_color_space.lower() == "linear" and self.__colorspace.lower() == "srgb":
                image = gamma(image, "sRGB")
                image = self.apply_ccm(image, self.__ccm)
                image = gamma_reverse(image, "sRGB")

            elif image_color_space.lower() == "srgb" and self.__colorspace.lower() == "linear":
                image = gamma_reverse(image, "sRGB")
                image = self.apply_ccm(image, self.__ccm)
                image = gamma(image, "sRGB")

            image = np.clip(image, 0, 1)

        return image
    
    # def save(self, ccm_save_path):
    #     """save calibration result as a CCM dict for the later correction.
    #     Args:
    #         ccm_save_path (str): npy file path.
    #     """
    #     if os.path.exists(ccm_save_path):
            # cct_ccm_list = np.load(ccm_save_path, allow_pickle=True).item()
    #         cct_ccm_list[self.cct] = self.__ccm
    #         np.save(ccm_save_path, cct_ccm_list)
    #     else:
    #         np.save(ccm_save_path, {self.cct: self.__ccm})

    def save(self, ccm_save_path):
        """save calibration result as a CCM dict for the later correction.
        Args:
            ccm_save_path (str): npy file path.
        """
        # if os.path.exists(ccm_save_path):
        #     cct_ccm_list = np.load(ccm_save_path, allow_pickle=True).item()
        #     cct_ccm_list[self.cct] = self.__ccm
        #     np.save(ccm_save_path, cct_ccm_list)
        # else:
        np.save(ccm_save_path, {'ccm': self.__ccm, 'rgb_gain':self.rgb_gain})


if __name__ == '__main__':
    pass
