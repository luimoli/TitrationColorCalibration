import numpy as np
import cv2
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000


lab_two = np.float32(np.loadtxt("./calc_value.csv", delimiter=','))

lab_1 = lab_two[0, :].copy()
lab_2 = lab_two[1, :].copy()

deltaC_00 = delta_C_CIE2000(lab_1, lab_2)
deltaE_00 = delta_E_CIE2000(lab_1, lab_2)
print(deltaC_00)
print(deltaE_00)

