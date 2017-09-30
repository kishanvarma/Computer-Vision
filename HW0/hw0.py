import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an color image in grayscale

#convert to hsv,do only on v, then  merge hsv.

img = cv2.imread('k1.jpg', cv2.IMREAD_COLOR)

b,g,r  = cv2.split(img)

"""
histb = cv2.calcHist(b,[0],None,[256],[0,256])
histg = cv2.calcHist(g,[0],None,[256],[0,256])
histr = cv2.calcHist(r,[0],None,[256],[0,256])
"""
histb, binsb = np.histogram(b.flatten() , 256, [0, 256])
histg, binsg = np.histogram(g.flatten() , 256, [0, 256])
histr, binsr = np.histogram(r.flatten() , 256, [0, 256])

cdfb = np.cumsum(histb)
cdf_b = np.ma.masked_equal(cdfb, 0)
cdf_b = (cdf_b - cdf_b.min()) * 255 / (cdf_b.max() - cdf_b.min())
cdf_B_final = np.ma.filled(cdf_b, 0).astype('uint8')

cdfg = np.cumsum(histg)
cdf_g = np.ma.masked_equal(cdfg, 0)
cdf_g = (cdf_g - cdf_g.min()) * 255 / (cdf_g.max() - cdf_g.min())
cdf_G_final = np.ma.filled(cdf_g, 0).astype('uint8')

cdfr = np.cumsum(histr)
cdf_r = np.ma.masked_equal(cdfr, 0)
cdf_r = (cdf_r - cdf_r.min()) * 255 / (cdf_r.max() - cdf_r.min())
cdf_R_final = np.ma.filled(cdf_r, 0).astype('uint8')

B_new = cdf_B_final[b]
G_new = cdf_G_final[g]
R_new = cdf_R_final[r]

final_img = cv2.merge((B_new,G_new,R_new))
cv2.namedWindow('myphoto1', cv2.WINDOW_AUTOSIZE)
cv2.imshow('myphoto1',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
