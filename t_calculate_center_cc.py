import numpy as np
import cv2
import os
from glob import glob



def get_cc_mean_value(pic_root, h_range, w_range, pic_format):
    """_summary_
    Args:
        pic_root (str): png imgs.
        h_range (tuple): 
        w_range (tuple): 
        pic_format (str): 'uint8', 'uint16'
    Returns:
        arr: [0,1] shape:[24,3]
    """
    path_list_ = glob(pic_root + "\\*.png")
    path_list = sorted(path_list_)
    print(path_list)
    assert len(path_list) == 24
    value_list = []
    for i in range(24):
        # img_path = os.path.join(pic_root, f'{i}.png')
        img_path = path_list[i]
        if pic_format == 'uint8':
            img = cv2.imread(img_path) / 255.
        elif pic_format == 'uint16':
            img = cv2.imread(img_path, -1) / 65535.
        else:
            print(pic_format)
        img = img[:,:,::-1].copy()
        img_center = img[h_range[0]:h_range[1], w_range[0]:w_range[1],:].copy()
        cv2.imwrite(os.path.join(pic_root, f'{i}.jpg'), img_center[:,:,::-1]*255.) # save the patches.
        img_center_mean = np.mean(img_center, (0, 1))
        value_list.append(img_center_mean[None, ...])
    cc_mean_value = np.concatenate(value_list, axis=0)
    return cc_mean_value

if __name__ =='__main__':
    pic_root = 'image\\24c_D65_lighthouse_0221'
    h_range =(140, 390)
    # w_range = (150, 480)
    w_range = (260, 500)
    cc_mean = get_cc_mean_value(pic_root, h_range, w_range, 'uint8')
    print(cc_mean.shape)
    np.save('data_calib\cc_mean_lighthouse_D65_0221.npy', cc_mean)