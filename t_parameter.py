import numpy as np


calib = np.load('./calib/TitrationCamera_xrite_ccm_3x3_lighthouse_D65_0221.npy',allow_pickle=True).item() 
print(calib)
with open('parameter/parameter_0221_noon.txt','a') as f:
	for i in calib['rgb_gain']:
		f.write(str(i) + '\n')

with open('parameter/parameter_0221_noon.txt','a') as f:
	for j in range(3):
		for k in range(3):
			content = calib['ccm'][j][k]
			f.write(str(content)+'\n')