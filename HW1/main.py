# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   #Convert the image to HSV - one channel is devoted to brightness (Value)
   img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(img)

   histv, binsv = np.histogram(v, 256, [0, 256])
   cdfv = np.cumsum(histv)
   cdf_v = np.ma.masked_equal(cdfv, 0)
   cdf_v = (cdf_v - cdf_v.min()) * 255 / (cdf_v.max() - cdf_v.min())
   cdf_v_final = np.ma.filled(cdf_v, 0).astype('uint8')
   
   v_new = cdf_v_final[v]

   img2 = cv2.merge((h, s, v_new))
   #converting back from hsv to  bgr
   final_img = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
   img_out = final_img # Histogram equalization result
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def lpfilter(img):
   # Getting the frquency transform
   dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = np.fft.fftshift(dft)
   magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

   row, col = img.shape
   m_row, m_col = row/2, col/2

   # mask a 20x20 window of the center of the FT image (the low frequencies).
   mask = np.zeros((row, col, 2), np.uint8)
   mask[m_row - 10:m_row + 10, m_col - 10:m_col + 10] = 1

   # apply mask
   fshift = dft_shift * mask

   # inverse DFT
   f_ishift = np.fft.ifftshift(fshift)
   img_back = cv2.idft(f_ishift)
   img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

   # Normalizing
   img_back = np.divide(img_back * 255, np.max(img_back))

   return img_back

def low_pass_filter(img_in):
	
   # Write low pass filter here
   b, g, r = cv2.split(img_in)

   b_back = lpfilter(b)
   g_back = lpfilter(g)
   r_back = lpfilter(r)

   output_img = cv2.merge((b_back, g_back, r_back))

   img_out = output_img # Low pass filter result
   return True, img_out

def hpfilter(img) :

    #Getting the frequency transform.
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    #coming to centre of the image and making 0
    row, col = img.shape
    m_row, m_col = row / 2, col / 2
    fshift[m_row - 10:m_row + 10, m_col - 10:m_col + 10] = 0

    #applying inverse
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    #normalizing
    img_back = np.divide(img_back * 255, np.max(img_back))

    return img_back

def high_pass_filter(img_in):

   # Write high pass filter here
   b, g, r = cv2.split(img_in)

   b_back = hpfilter(b)
   g_back = hpfilter(g)
   r_back = hpfilter(r)

   corners = cv2.merge((b_back, g_back, r_back))

   img_out = corners # High pass filter result
   
   return True, img_out

def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im),newsize)
    return np.fft.fftshift(dft)

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def deconvolution(img_in):
   
   # Write deconvolution codes here
   #Getting the Gaussian Kernel
   gk = cv2.getGaussianKernel(21, 5)
   gk = gk * gk.T

   imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))
 
   gkf = ft(gk, (img_in.shape[0], img_in.shape[1]))
  
   imconvf = imf / gkf     #Foe deconvolution divide. 
   img = ift(imconvf)
   
   #Scale up the values to get back the clear image
   img_out = img * 255 #np.array(imconvf * 255, dtype = np.uint8)
   #img_out = imconvf # Deconvolution result
   
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH );

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   #cv2.imwrite(output_name3, cv2.convertScaleAbs(output_image3, alpha=255.0))
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def guassian_pyramid(img):
    #create copy of the image
    G = img.copy()
    gp = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def laplacianpyramid(gp):
    #last one in the start
    lp = [gp[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp


def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   #MAking images rectangular 
   img_in1 = img_in1[:, :img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

   gpA = guassian_pyramid(img_in1)
   gpB = guassian_pyramid(img_in2)
   lpA = laplacianpyramid(gpA)
   lpB = laplacianpyramid(gpB)

   # Blending left half and right half in each level
   LS = []
   for la, lb in zip(lpA, lpB):
      rows, cols, dpt = la.shape
      ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
      LS.append(ls)

   # REconstructing using Pyrup.
   ls_ = LS[0]
   for i in xrange(1, 6):
      ls_ = cv2.pyrUp(ls_)
      ls_ = cv2.add(ls_, LS[i])

   img_out = ls_ # Blending result
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
